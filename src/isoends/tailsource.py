"""Per-read poly(A) tails from a source other than the read annotation: a per-read table (CSV, TSV or Parquet with
a read ID and a tail length, negative for no call) or BAMs carrying the `pt:i` tag.

Reads are joined on their UUID, the last 36 characters of IsoForge's `read_name`. A UUID is held as two unsigned
64-bit integers, so the index of a run is about 20 bytes a read and a lookup is a binary search.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .util import LOG, atomic_write_json, atomic_write_parquet

_HEX = np.full(256, 255, dtype=np.uint8)
for _i, _ch in enumerate(b"0123456789abcdef"):
    _HEX[_ch] = _i
for _i, _ch in enumerate(b"ABCDEF"):
    _HEX[_ch] = 10 + _i
_HEX_COLS = np.array([i for i in range(36) if i not in (8, 13, 18, 23)])
_SHIFTS = np.arange(60, -1, -4, dtype=np.uint64)


def _decode_block(chars: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hi, lo, valid) from an (n, 36) uint8 array of UUID text."""
    nib = _HEX[chars[:, _HEX_COLS]]
    valid = (nib != 255).all(axis=1) & (chars[:, 8] == 45) & (chars[:, 13] == 45) & (chars[:, 18] == 45) \
        & (chars[:, 23] == 45)
    nib = nib.astype(np.uint64)
    hi = (nib[:, :16] << _SHIFTS).sum(axis=1, dtype=np.uint64)
    lo = (nib[:, 16:] << _SHIFTS).sum(axis=1, dtype=np.uint64)
    return hi, lo, valid


def uuid_pairs(values, chunk: int = 1 << 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UUID strings (Arrow array, chunked array or array-like) -> (hi, lo, valid). The last 36 characters of each
    value are read, so a prefix is allowed."""
    arr = values if isinstance(values, (pa.Array, pa.ChunkedArray)) else pa.array(np.asarray(values, dtype=object))
    if isinstance(arr, pa.ChunkedArray):
        parts = [uuid_pairs(c, chunk) for c in arr.chunks]
        if not parts:
            e = np.empty(0, np.uint64)
            return e, e.copy(), np.empty(0, bool)
        return tuple(np.concatenate(x) for x in zip(*parts))
    if pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_decode()
    arr = pc.utf8_slice_codeunits(arr.cast(pa.large_string()), -36)
    n = len(arr)
    hi = np.zeros(n, np.uint64)
    lo = np.zeros(n, np.uint64)
    valid = np.zeros(n, bool)
    if n == 0:
        return hi, lo, valid
    offsets = np.frombuffer(arr.buffers()[1], dtype=np.int64)[arr.offset:arr.offset + n + 1]
    data = np.frombuffer(arr.buffers()[2], dtype=np.uint8) if arr.buffers()[2] is not None else np.empty(0, np.uint8)
    lengths = np.diff(offsets)
    nulls = np.asarray(arr.is_null())
    full = (lengths == 36) & ~nulls
    if full.all() and offsets[-1] - offsets[0] == 36 * n:
        block = data[offsets[0]:offsets[0] + 36 * n].reshape(n, 36)
        for s in range(0, n, chunk):
            h, l_, v = _decode_block(block[s:s + chunk])
            hi[s:s + chunk], lo[s:s + chunk], valid[s:s + chunk] = h, l_, v
        return hi, lo, valid
    idx = np.flatnonzero(full)
    if len(idx):
        rows = np.stack([data[offsets[i]:offsets[i] + 36] for i in idx])
        h, l_, v = _decode_block(rows)
        hi[idx], lo[idx], valid[idx] = h, l_, v
    return hi, lo, valid


@dataclass
class TailIndex:
    """Tails (nt) and mapping qualities of a run's reads, sorted by UUID for lookup."""
    hi: np.ndarray
    lo: np.ndarray
    tail: np.ndarray        # int16, >= 0; reads without a call are not stored
    mapq: np.ndarray        # int16, -1 when unknown
    source: str = ""
    n_rows: int = 0         # rows read from the source
    n_no_call: int = 0      # rows without a tail call
    n_duplicates: int = 0   # further rows of a read already stored

    @staticmethod
    def from_arrays(hi, lo, tail, mapq, **meta) -> "TailIndex":
        """Sort by (hi, lo) and keep one row per read (the best mapping quality)."""
        order = np.lexsort((-mapq.astype(np.int32), lo, hi))
        hi = hi[order]
        lo = lo[order]
        tail = tail[order]
        mapq = mapq[order]
        del order
        keep = np.ones(len(hi), bool)
        if len(hi) > 1:
            keep[1:] = (hi[1:] != hi[:-1]) | (lo[1:] != lo[:-1])
        meta["n_duplicates"] = meta.get("n_duplicates", 0) + int((~keep).sum())
        return TailIndex(hi[keep], lo[keep], tail[keep], mapq[keep], **meta)

    def __len__(self) -> int:
        return len(self.hi)

    def lookup(self, hi: np.ndarray, lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(tail, mapq) for each query read; -1 where the read has no tail call."""
        tail = np.full(len(hi), -1, np.int16)
        mapq = np.full(len(hi), -1, np.int16)
        if len(self.hi) == 0 or len(hi) == 0:
            return tail, mapq
        order = np.argsort(hi, kind="stable")
        hs = hi[order]
        left = np.empty(len(hi), np.int64)
        right = np.empty(len(hi), np.int64)
        left[order] = np.searchsorted(self.hi, hs, side="left")
        right[order] = np.searchsorted(self.hi, hs, side="right")
        del hs, order
        width = right - left
        one = width == 1
        pos = left[one]
        hit = self.lo[pos] == lo[one]
        idx = np.flatnonzero(one)[hit]
        tail[idx] = self.tail[pos[hit]]
        mapq[idx] = self.mapq[pos[hit]]
        for i in np.flatnonzero(width > 1):
            j = left[i] + np.searchsorted(self.lo[left[i]:right[i]], lo[i])
            if j < right[i] and self.lo[j] == lo[i]:
                tail[i], mapq[i] = self.tail[j], self.mapq[j]
        return tail, mapq

    def save(self, path: Path) -> None:
        meta = dict(source=self.source, n_rows=self.n_rows, n_no_call=self.n_no_call, n_duplicates=self.n_duplicates)
        atomic_write_json(meta, Path(path).with_suffix(".json"))
        atomic_write_parquet(pd.DataFrame({"hi": self.hi, "lo": self.lo, "tail": self.tail, "mapq": self.mapq}), path)

    @staticmethod
    def load(path: Path) -> "TailIndex":
        df = pd.read_parquet(path)
        meta = json.loads(Path(path).with_suffix(".json").read_text())
        return TailIndex(df.hi.to_numpy(np.uint64), df.lo.to_numpy(np.uint64), df["tail"].to_numpy(np.int16),
                         df.mapq.to_numpy(np.int16), source=str(meta.get("source", "")),
                         n_rows=int(meta.get("n_rows", 0)), n_no_call=int(meta.get("n_no_call", 0)),
                         n_duplicates=int(meta.get("n_duplicates", 0)))


def _batches(path: Path, columns: list[str], block_mb: int = 64, string_columns=()):
    """Record batches of the requested columns from a CSV/TSV (streamed) or a Parquet file. `string_columns` are read
    as text."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        have = set(pf.schema_arrow.names)
        yield from pf.iter_batches(batch_size=1 << 20, columns=[c for c in columns if c in have])
        return
    import pyarrow.csv as pv
    delim = "\t" if path.suffix in (".tsv", ".txt") or path.name.endswith((".tsv.gz", ".txt.gz")) else ","
    with pv.open_csv(path, read_options=pv.ReadOptions(block_size=block_mb << 20),
                     parse_options=pv.ParseOptions(delimiter=delim)) as probe:
        have = set(probe.schema.names)
    types = {c: pa.string() for c in string_columns if c in have}
    reader = pv.open_csv(path, read_options=pv.ReadOptions(block_size=block_mb << 20),
                         parse_options=pv.ParseOptions(delimiter=delim),
                         convert_options=pv.ConvertOptions(include_columns=[c for c in columns if c in have],
                                                           column_types=types))
    yield from reader


def _as_int(col, default: int = -1) -> np.ndarray:
    """A numeric column as float64 (nulls -> default)."""
    a = col if pa.types.is_floating(col.type) else pc.cast(col, pa.float64(), safe=False)
    return np.asarray(a.fill_null(default).to_numpy(zero_copy_only=False), dtype=np.float64)


def build_index_from_table(path: Path, cols: dict) -> TailIndex:
    """The tail index of a per-read table, streamed once. cols: {"read_id": ..., "tail": ..., "mapq": ...}."""
    want = [cols["read_id"], cols["tail"]] + ([cols["mapq"]] if cols.get("mapq") else [])
    his, los, tails, mapqs = [], [], [], []
    n_rows = n_no_call = n_bad = 0
    for batch in _batches(path, want, string_columns=[cols["read_id"]]):
        names = batch.schema.names
        if cols["read_id"] not in names or cols["tail"] not in names:
            raise ValueError(f"{path}: needs columns {cols['read_id']!r} and {cols['tail']!r} (tail_columns in the "
                             "config)")
        tail = _as_int(batch.column(cols["tail"]))
        mapq = _as_int(batch.column(cols["mapq"])) if cols.get("mapq") in names else np.full(len(tail), -1.0)
        tail_i = np.where(np.isnan(tail), -1, tail).astype(np.int64)
        mapq_i = np.where(np.isnan(mapq), -1, mapq).astype(np.int64)
        n_rows += len(tail_i)
        ok = tail_i >= 0
        n_no_call += int((~ok).sum())
        hi, lo, valid = uuid_pairs(batch.column(cols["read_id"]).filter(pa.array(ok)))
        n_bad += int((~valid).sum())
        his.append(hi[valid])
        los.append(lo[valid])
        tails.append(np.minimum(tail_i[ok][valid], 32767).astype(np.int16))
        mapqs.append(np.clip(mapq_i[ok][valid], -1, 32767).astype(np.int16))
    if n_bad:
        LOG.warning("%s: %d read IDs are not UUIDs and were skipped", path, n_bad)

    def cat(xs, dt):
        out = np.concatenate(xs) if xs else np.empty(0, dt)
        xs.clear()
        return out
    return TailIndex.from_arrays(cat(his, np.uint64), cat(los, np.uint64), cat(tails, np.int16),
                                 cat(mapqs, np.int16), source=str(path), n_rows=n_rows, n_no_call=n_no_call)


def build_index_from_bams(bams, tag: str = "pt") -> TailIndex:
    """The tail index of BAMs carrying a tail tag (primary alignments and unmapped reads)."""
    import pysam
    his, los, tails, mapqs = [], [], [], []
    n_rows = n_no_call = 0
    for bam in bams:
        names, t_, m_ = [], [], []
        with pysam.AlignmentFile(str(bam), check_sq=False) as fh:
            for r in fh.fetch(until_eof=True):
                if r.is_secondary or r.is_supplementary:
                    continue
                n_rows += 1
                if not r.has_tag(tag) or r.get_tag(tag) < 0:
                    n_no_call += 1
                    continue
                names.append(r.query_name)
                t_.append(r.get_tag(tag))
                m_.append(-1 if r.is_unmapped else r.mapping_quality)
        hi, lo, valid = uuid_pairs(names)
        his.append(hi[valid])
        los.append(lo[valid])
        tails.append(np.minimum(np.asarray(t_, np.int64)[valid], 32767).astype(np.int16))
        mapqs.append(np.asarray(m_, np.int16)[valid])
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.empty(0, dt))
    return TailIndex.from_arrays(cat(his, np.uint64), cat(los, np.uint64), cat(tails, np.int16),
                                 cat(mapqs, np.int16), source=",".join(map(str, bams)), n_rows=n_rows,
                                 n_no_call=n_no_call)
