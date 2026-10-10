"""Cochran-Mantel-Haenszel test over replicate-pair strata, in closed form and vectorised over features.

For feature i and stratum (replicate pair) k, the 2 x 2 table is

                 event   other
    control       a       b
    treated       c       d

With n = a + b + c + d, E[a] = (a + b)(a + c) / n and Var(a) = (a + b)(c + d)(a + c)(b + d) / (n^2 (n - 1)). The
continuity-corrected statistic X^2 = (|sum_k (a_k - E[a_k])| - 0.5)^2 / sum_k Var(a_k) is referred to chi-square
with 1 df. Strata without reads are skipped. When |sum (a - E[a])| < 0.5 the statistic is 0 (P = 1).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def cmh(a, b, c, d, correction: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(chi2, P) per feature. a, b, c, d: arrays of shape (features, strata) or (strata,)."""
    a, b, c, d = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (a, b, c, d))
    n = a + b + c + d
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.where(n > 0, n, np.nan)
        row1, col1 = a + b, a + c
        expected = row1 * col1 / nn
        var = row1 * (nn - row1) * col1 * (nn - col1) / (nn ** 2 * np.where(n > 1, n - 1, np.nan))
        num = np.abs(np.nansum(a - expected, axis=1))
        if correction:
            num = np.clip(num - 0.5, 0, None)
        den = np.nansum(var, axis=1)
        chi2 = np.where(den > 0, num ** 2 / den, np.nan)
    return chi2, stats.chi2.sf(chi2, 1)


def mantel_haenszel_or(a, b, c, d) -> np.ndarray:
    """The Mantel-Haenszel common odds ratio of the event, treated vs control, per feature:
    sum(c b / n) / sum(a d / n) (> 1: the event is more common with the treatment)."""
    a, b, c, d = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (a, b, c, d))
    n = a + b + c + d
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.where(n > 0, n, np.nan)
        return np.nansum(c * b / nn, axis=1) / np.nansum(a * d / nn, axis=1)
