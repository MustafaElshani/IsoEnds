"""Multiple testing, t-tests over replicates, the bootstrap and Wilcoxon signed-rank test over units.

Every function takes one value per unit of replication (a replicate, a replicate pair or a unit), never per read.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values. Missing P values stay missing and are not counted in the family."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def paired_t(control, treated) -> dict:
    """Two-sided paired t-test of treated vs control over matched pairs (one value per pair on each side).

    Returns n_pairs, the means, mean_change (treated - control), sd_change, t and p. The test is undefined (t, p
    NaN) with fewer than two pairs or when every pair changed by exactly the same amount."""
    c = np.asarray(control, float)
    t = np.asarray(treated, float)
    ok = ~(np.isnan(c) | np.isnan(t))
    c, t = c[ok], t[ok]
    d = t - c
    n = len(d)
    out = dict(n_pairs=n, mean_control=float(c.mean()) if n else np.nan,
               mean_treated=float(t.mean()) if n else np.nan, mean_change=float(d.mean()) if n else np.nan,
               sd_change=float(d.std(ddof=1)) if n > 1 else np.nan, t=np.nan, p=np.nan)
    if n > 1 and d.std(ddof=1) > 0:
        res = stats.ttest_rel(t, c)
        out["t"], out["p"] = float(res.statistic), float(res.pvalue)
    return out


def welch_t(control, treated) -> dict:
    """Two-sided Welch t-test of treated vs control over unpaired replicates (one value per sample).

    Returns n_control, n_treated, the means, mean_change (treated - control), t and p. The test is undefined (t, p
    NaN) with fewer than two values in a condition or no variance."""
    c = np.asarray(control, float)
    t = np.asarray(treated, float)
    c, t = c[~np.isnan(c)], t[~np.isnan(t)]
    out = dict(n_control=len(c), n_treated=len(t), mean_control=float(c.mean()) if len(c) else np.nan,
               mean_treated=float(t.mean()) if len(t) else np.nan, t=np.nan, p=np.nan)
    out["mean_change"] = out["mean_treated"] - out["mean_control"]
    if len(c) > 1 and len(t) > 1 and (np.ptp(c) > 0 or np.ptp(t) > 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = stats.ttest_ind(t, c, equal_var=False)
        out["t"], out["p"] = float(res.statistic), float(res.pvalue)
    return out


def condition_t(control, treated, paired: bool) -> dict:
    """Treated vs control over the replicates of one unit: paired t-test (values aligned by pair) or Welch t-test.

    Returns n_control, n_treated, mean_control, mean_treated, mean_change, t and p."""
    if not paired:
        return welch_t(control, treated)
    r = paired_t(control, treated)
    return dict(n_control=r["n_pairs"], n_treated=r["n_pairs"], mean_control=r["mean_control"],
                mean_treated=r["mean_treated"], mean_change=r["mean_change"], t=r["t"], p=r["p"])


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10_000, level: float = 0.95,
                 seed: int = 1) -> tuple[float, float, float]:
    """(estimate, lower, upper): percentile bootstrap CI of `stat` over the given units (NaN dropped)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return float(stat(v)), np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = stat(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(stat(v)), float(lo), float(hi)


def wilcoxon_paired(a, b=None, min_n: int = 5) -> dict:
    """Two-sided Wilcoxon signed-rank test over units: a vs b paired by unit, or a vs 0 when b is None.

    Returns n (units with both values), median_difference, n_greater / n_less (units where a > b / a < b) and p
    (NaN below `min_n` units, where the smallest attainable two-sided P is already >= 0.0625)."""
    a = np.asarray(a, float)
    b = np.zeros_like(a) if b is None else np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    out = dict(n=int(ok.sum()), median_difference=float(np.median(d)) if len(d) else np.nan,
               n_greater=int((d > 0).sum()), n_less=int((d < 0).sum()), p=np.nan)
    if len(d) >= min_n and np.any(d != 0):
        out["p"] = float(stats.wilcoxon(d).pvalue)
    return out


def direction_counts(effect, q, alpha: float = 0.05) -> tuple[int, int]:
    """(n up, n down) among tests with q < alpha."""
    effect = np.asarray(effect, float)
    q = np.asarray(q, float)
    sig = q < alpha
    return int((sig & (effect > 0)).sum()), int((sig & (effect < 0)).sum())


def family_row(family: str, analysis: str, test: str, unit_of_replication: str, table: pd.DataFrame,
               effect: str, q: str = "q", alpha: float = 0.05, units: str = "", note: str = "") -> dict:
    """One row of tests_summary.tsv for a family of tests adjusted together."""
    t = table[table[q].notna()] if len(table) and q in table else table.iloc[0:0]
    up, down = direction_counts(t[effect], t[q], alpha) if len(t) else (0, 0)
    return dict(family=family, analysis=analysis, test=test, unit_of_replication=unit_of_replication,
                n_tested=int(len(t)), n_q_below_alpha=up + down, n_up=up, n_down=down, alpha=alpha,
                units=units, note=note)
