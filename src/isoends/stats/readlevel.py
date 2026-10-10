"""Read-level tests from counts per tail value. Descriptive only: reads of one sample are not independent
replicates of the condition, so these P values are not used for inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mann_whitney_counts(a: pd.DataFrame, b: pd.DataFrame, key: str = "gene", value: str = "tail",
                        count: str = "n_reads") -> pd.DataFrame:
    """Two-sided Mann-Whitney (Wilcoxon rank-sum) test per `key` between the reads of a and of b, each given as
    counts per value (key, value, count). Normal approximation with the tie correction and a continuity correction,
    as scipy.stats.mannwhitneyu(method="asymptotic"). Returns key, n_a, n_b, median_a, median_b, u, z, p."""
    d = pd.concat([a[[key, value, count]].assign(_s=0), b[[key, value, count]].assign(_s=1)], ignore_index=True)
    d = d[d[count] > 0]
    w = d.pivot_table(index=[key, value], columns="_s", values=count, aggfunc="sum", fill_value=0)
    for c in (0, 1):
        if c not in w:
            w[c] = 0
    w = w.sort_index()
    ca, cb = w[0].to_numpy(float), w[1].to_numpy(float)
    keys = w.index.get_level_values(0)
    codes, uniq = pd.factorize(keys, sort=False)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    lengths = np.diff(np.r_[starts, len(cb)])
    cum_b = np.cumsum(cb)
    below_b = cum_b - cb - np.repeat(cum_b[starts] - cb[starts], lengths)
    u = np.add.reduceat(ca * (below_b + 0.5 * cb), starts)
    n1 = np.add.reduceat(ca, starts)
    n2 = np.add.reduceat(cb, starts)
    t = ca + cb
    tie = np.add.reduceat(t ** 3 - t, starts)
    n = n1 + n2
    mu = n1 * n2 / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
        z = (np.abs(u - mu) - 0.5) / sigma
        p = np.where((n1 > 0) & (n2 > 0) & (sigma > 0), 2 * stats.norm.sf(np.clip(z, 0, None)), np.nan)
    vals = w.index.get_level_values(1).to_numpy(float)

    def med(c):
        cum = np.cumsum(c)
        tot = np.add.reduceat(c, starts)
        base = cum[starts] - c[starts]
        ilo = np.searchsorted(cum, base + np.floor((tot - 1) / 2), side="right").clip(max=len(vals) - 1)
        ihi = np.searchsorted(cum, base + np.ceil((tot - 1) / 2), side="right").clip(max=len(vals) - 1)
        return np.where(tot > 0, (vals[ilo] + vals[ihi]) / 2, np.nan)
    return pd.DataFrame({key: np.asarray(uniq), "n_a": n1.astype(int), "n_b": n2.astype(int), "median_a": med(ca),
                         "median_b": med(cb), "u": u, "z": np.sign(u - mu) * np.clip(z, 0, None), "p": p})
