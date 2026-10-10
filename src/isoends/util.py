"""Small helpers: file fingerprints and hashes, atomic writes, logging and the step log the report reads."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("isoends")


def setup_logging(verbose: bool = False) -> None:
    if LOG.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s isoends %(levelname)s %(message)s", "%H:%M:%S"))
    LOG.addHandler(h)
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)


def file_fingerprint(path, extra: str = "") -> str:
    """A short key that changes when the file (path, size, modification time) or `extra` changes."""
    p = Path(path)
    st = p.stat()
    return hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{extra}".encode()).hexdigest()[:16]


def combined_fingerprint(paths, extra: str = "") -> str:
    parts = [file_fingerprint(p) for p in paths if p is not None]
    return hashlib.sha1(("|".join(parts) + "|" + extra).encode()).hexdigest()[:16]


def sha256(path, cache_dir: Path | None = None, block: int = 1 << 24) -> str:
    """SHA-256 of a file, remembered in <cache_dir>/hashes.json against its fingerprint so that large inputs are
    hashed once."""
    p = Path(path)
    store = Path(cache_dir) / "hashes.json" if cache_dir else None
    key = file_fingerprint(p)
    known = {}
    if store and store.exists():
        try:
            known = json.loads(store.read_text())
        except json.JSONDecodeError:
            known = {}
    if key in known:
        return known[key]
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    digest = h.hexdigest()
    if store:
        store.parent.mkdir(parents=True, exist_ok=True)
        known[key] = digest
        tmp = store.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(known, indent=0))
        os.replace(tmp, store)
    return digest


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


class StepLog:
    """Counts at each step of an analysis, written to <outdir>/logs/<stage>_steps.json for the report."""

    def __init__(self, outdir: Path, stage: str):
        self.path = Path(outdir) / "logs" / f"{stage}_steps.json"
        self.records: list[dict] = []
        self.t0 = time.time()

    def add(self, step: str, n=None, unit: str = "", **extra) -> None:
        rec = dict(step=step, unit=unit, n=n, **extra)
        self.records.append(rec)
        LOG.info("%s%s: %s", f"[{unit}] " if unit else "", step, n if n is not None else "")

    def warn(self, message: str, unit: str = "") -> None:
        self.records.append(dict(step="WARNING", unit=unit, n=None, message=message))
        LOG.warning("%s%s", f"[{unit}] " if unit else "", message)

    def save(self) -> None:
        atomic_write_json({"records": self.records, "seconds": round(time.time() - self.t0, 1)}, self.path)
