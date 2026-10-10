"""Exact summaries of values held as counts: (group, value, count) rows instead of one row per read.

The quantiles are those of NumPy / pandas with linear interpolation (the median of an even number of values is the
mean of the two middle ones), computed without expanding the counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _group_codes(groups):
    """(codes, index builder) for one key or a DataFrame of keys. Each column is factorised and the codes combined
    into one integer."""
    if not isinstance(groups, pd.DataFrame):
        codes, uniques = pd.factorize(pd.Series(groups), sort=False)
        return codes, lambda first: pd.Index(np.asarray(uniques)[codes[first]]) if len(first) else pd.Index([])
    keys = list(groups.columns)
    parts = [pd.factorize(groups[c], sort=False) for c in keys]
    combined = np.zeros(len(groups), dtype=np.int64)
    for c, u in parts:
        combined = combined * max(len(u), 1) + c
    codes, _ = pd.factorize(combined, sort=False)

    def build(first):
        if not len(first):
            return pd.MultiIndex.from_tuples([], names=keys)
        return pd.MultiIndex.from_arrays([np.asarray(u)[c[first]] for c, u in parts], names=keys)
    return codes, build


def grouped_quantiles(groups, values, counts, qs=(0.25, 0.5, 0.75), extra: dict | None = None) -> pd.DataFrame:
    """Per group: n (sum of counts), mean and the requested quantiles of `values` weighted by `counts`.

    groups: any labels (or a DataFrame of several key columns); values: numbers; counts: non-negative ints.
    extra: {name: counts} further count vectors over the same rows (e.g. reads with MAPQ >= 1), each giving
    n_<name> and q50_<name> from the same sort. Returns a DataFrame indexed by group (a MultiIndex for several
    keys) with columns n, mean, q<100q> [, n_<name>, q50_<name>]."""
    codes, build = _group_codes(groups)
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    extra = {k: np.asarray(v, dtype=np.int64) for k, v in (extra or {}).items()}
    keep = counts > 0
    codes, values, counts = codes[keep], values[keep], counts[keep]
    extra = {k: v[keep] for k, v in extra.items()}
    cols = ["n", "mean"] + [f"q{int(round(q * 100))}" for q in qs]
    for k in extra:
        cols += [f"n_{k}", f"q50_{k}"]
    if len(codes) == 0:
        return pd.DataFrame(columns=cols, index=build(np.empty(0, int)), dtype=float)
    order = np.lexsort((values, codes))
    g, v = codes[order], values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]

    def quantiles(c, qs_):
        cum = np.cumsum(c)
        n = np.add.reduceat(c, starts)
        before = cum[starts] - c[starts]
        res = {}
        for q in qs_:
            h = (n - 1) * q
            lo = np.floor(h).astype(np.int64)
            hi = np.ceil(h).astype(np.int64)
            ok = n > 0
            vlo = v[np.searchsorted(cum, np.where(ok, before + lo, 0), side="right").clip(max=len(v) - 1)]
            vhi = v[np.searchsorted(cum, np.where(ok, before + hi, 0), side="right").clip(max=len(v) - 1)]
            res[q] = np.where(ok, vlo + (vhi - vlo) * (h - lo), np.nan)
        return n, res

    c = counts[order]
    n, res = quantiles(c, qs)
    out = {"n": n, "mean": np.add.reduceat(v * c, starts) / n}
    for q in qs:
        out[f"q{int(round(q * 100))}"] = res[q]
    for k, e in extra.items():
        ne, re_ = quantiles(e[order], (0.5,))
        out[f"n_{k}"] = ne
        out[f"q50_{k}"] = re_[0.5]
    first_rows = order[starts]
    kept_idx = np.flatnonzero(keep)
    return pd.DataFrame(out, index=build(kept_idx[first_rows]))
