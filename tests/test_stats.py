"""The tests against their references: statsmodels, SciPy and R limma."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from isoends.stats import bh, bootstrap_ci, cmh, condition_t, paired_t, welch_t, wilcoxon_paired
from isoends.stats.limma import (brent_fmin, logmdigamma, moderated_one_sample_python, moderated_two_sample_python,
                                 trigamma_inverse)

from .conftest import needs_r


def test_bh_matches_statsmodels_and_keeps_nan():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500) ** 3
    p[::37] = np.nan
    q = bh(p)
    ok = ~np.isnan(p)
    assert np.isnan(q[~ok]).all()
    np.testing.assert_allclose(q[ok], sm.multipletests(p[ok], method="fdr_bh")[1], rtol=1e-12)


def test_cmh_matches_statsmodels_stratified_table():
    ct = pytest.importorskip("statsmodels.stats.contingency_tables")
    rng = np.random.default_rng(1)
    n_feat, k = 300, 3
    a, b, c, d = (rng.poisson(lam, (n_feat, k)).astype(float) + 1 for lam in (20, 30, 35, 25))
    chi2, p = cmh(a, b, c, d)
    for i in range(n_feat):
        tables = np.array([[[a[i, j], b[i, j]], [c[i, j], d[i, j]]] for j in range(k)]).transpose(1, 2, 0)
        res = ct.StratifiedTable(tables).test_null_odds(correction=True)
        s = abs(np.sum(a[i] - (a[i] + b[i]) * (a[i] + c[i]) / (a[i] + b[i] + c[i] + d[i])))
        if s > 0.5:
            assert chi2[i] == pytest.approx(res.statistic, rel=1e-10)
            assert p[i] == pytest.approx(res.pvalue, rel=1e-8)


def test_cmh_skips_empty_strata_and_clips_small_statistics():
    chi2, p = cmh([[10, 0]], [[10, 0]], [[10, 0]], [[10, 0]])
    assert chi2[0] == 0 and p[0] == 1
    chi2, p = cmh([[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]])
    assert np.isnan(chi2[0])


def test_cmh_detects_a_consistent_shift_only():
    a, b, c, d = [[30, 30, 30]], [[70, 70, 70]], [[60, 60, 60]], [[40, 40, 40]]
    assert cmh(a, b, c, d)[1][0] < 1e-10
    a, b, c, d = [[30, 60]], [[70, 40]], [[60, 30]], [[40, 70]]
    assert cmh(a, b, c, d)[1][0] > 0.5


def test_paired_t_and_bootstrap_and_wilcoxon():
    r = paired_t([100, 110, 90], [80, 95, 72])
    assert r["n_pairs"] == 3
    assert r["mean_change"] == pytest.approx(-17.666, abs=1e-3)
    assert r["p"] == pytest.approx(stats.ttest_rel([80, 95, 72], [100, 110, 90]).pvalue)
    assert np.isnan(paired_t([1, 2], [2, 3])["p"])
    est, lo, hi = bootstrap_ci(np.arange(10.0), n_boot=2000, seed=1)
    assert lo < est < hi and est == 4.5
    assert bootstrap_ci(np.arange(10.0), n_boot=2000, seed=1) == (est, lo, hi)
    w = wilcoxon_paired([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0])
    assert w["n_greater"] == 6 and w["p"] == pytest.approx(0.03125)
    assert np.isnan(wilcoxon_paired([1, 2, 3], None)["p"])
    w = welch_t([100, 110, 90, np.nan], [80, 95, 72])
    assert w["n_control"] == 3 and w["mean_change"] == pytest.approx(-17.666, abs=1e-3)
    assert w["p"] == pytest.approx(stats.ttest_ind([80, 95, 72], [100, 110, 90], equal_var=False).pvalue)
    assert np.isnan(welch_t([1], [2, 3])["p"])
    assert condition_t([100, 110, 90], [80, 95, 72], paired=True)["p"] == pytest.approx(r["p"])
    assert condition_t([100, 110, 90], [80, 95, 72], paired=False)["p"] == pytest.approx(w["p"])


def test_special_functions():
    from scipy import special
    x = np.array([0.3, 1.0, 4.9, 5.0, 50.0, 5000.0])
    np.testing.assert_allclose(logmdigamma(x), np.log(x) - special.digamma(x), rtol=1e-9)
    y = np.array([0.01, 0.5, 2.0, 30.0])
    np.testing.assert_allclose(special.polygamma(1, trigamma_inverse(y)), y, rtol=1e-8)
    assert brent_fmin(lambda v: (v - 0.7) ** 2, 0.5, 0.9998) == pytest.approx(0.7, abs=1e-4)


def _matrix(n, k, na, seed):
    rng = np.random.default_rng(seed)
    sd = np.sqrt(rng.gamma(2, 1, n)) * 5
    X = rng.normal(rng.normal(0, 2, n)[:, None], sd[:, None], (n, k))
    X[:15] *= 8
    if na:
        X[rng.random((n, k)) < na] = np.nan
    return pd.DataFrame(X, index=[f"g{i}" for i in range(n)])


def test_moderated_t_is_a_t_test_when_the_prior_is_uninformative():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(0, 1, (200, 3)) * rng.uniform(0.1, 20, (200, 1)) ** 2)
    res = moderated_one_sample_python(X, robust=False)
    t_ord = stats.ttest_1samp(X.to_numpy(), 0, axis=1)
    assert np.corrcoef(res.t, t_ord.statistic)[0, 1] > 0.9


@needs_r
@pytest.mark.parametrize("k,na", [(3, 0.0), (3, 0.15), (22, 0.2), (22, 0.0)])
@pytest.mark.parametrize("robust", [False, True])
def test_moderated_t_matches_limma(k, na, robust):
    from isoends.stats.limma import moderated_one_sample_r
    X = _matrix(1500, k, na, seed=k + int(na * 100))
    a = moderated_one_sample_python(X, robust=robust)
    b = moderated_one_sample_r(X, robust=robust).reindex(a.index)
    for col, tol in (("t", 1e-8), ("p", 1e-7), ("df_prior", 1e-7), ("s2_prior", 1e-8), ("q", 1e-7)):
        rel = np.nanmax(np.abs(a[col] - b[col]) / np.maximum(np.abs(b[col]), 1e-300))
        assert rel < tol, (col, rel)


def test_two_sample_moderated_t_reduces_to_student_t_without_a_prior():
    rng = np.random.default_rng(4)
    A, B = pd.DataFrame(rng.normal(0, 1, (2, 4))), pd.DataFrame(rng.normal(1, 1, (2, 5)))
    res = moderated_two_sample_python(A, B)
    ref = stats.ttest_ind(B.to_numpy(), A.to_numpy(), axis=1)
    np.testing.assert_allclose(res.t, ref.statistic)
    np.testing.assert_allclose(res.p, ref.pvalue)
    assert list(res.n_control) == [4, 4] and list(res.df_residual) == [7, 7]


@needs_r
@pytest.mark.parametrize("na", [0.0, 0.15])
@pytest.mark.parametrize("robust", [False, True])
def test_two_sample_moderated_t_matches_limma(na, robust):
    from isoends.stats.limma import moderated_two_sample_r
    X = _matrix(1500, 7, na, seed=7 + int(na * 100))
    A, B = X.iloc[:, :3], X.iloc[:, 3:] + 4
    a = moderated_two_sample_python(A, B, robust=robust)
    b = moderated_two_sample_r(A, B, robust=robust)
    assert a.index.equals(b.index)
    for col, tol in (("mean", 1e-8), ("t", 1e-8), ("p", 1e-7), ("df_prior", 1e-7), ("s2_prior", 1e-8), ("q", 1e-7)):
        rel = np.nanmax(np.abs(a[col] - b[col]) / np.maximum(np.abs(b[col]), 1e-300))
        assert rel < tol, (col, rel)


def test_mann_whitney_counts_matches_scipy():
    from scipy.stats import mannwhitneyu

    from isoends.stats.readlevel import mann_whitney_counts
    rng = np.random.default_rng(3)
    rows_a, rows_b, raw = [], [], {}
    for g, shift in (("G1", 0), ("G2", 6), ("G3", -15)):
        a = rng.integers(20, 120, 300)
        b = rng.integers(20, 120, 250) + shift
        raw[g] = (a, b)
        for arr, rows in ((a, rows_a), (b, rows_b)):
            v, c = np.unique(arr, return_counts=True)
            rows.append(pd.DataFrame({"gene": g, "tail": v, "n_reads": c}))
    res = mann_whitney_counts(pd.concat(rows_a), pd.concat(rows_b)).set_index("gene")
    for g, (a, b) in raw.items():
        ref = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        assert res.u[g] == pytest.approx(ref.statistic)
        assert res.p[g] == pytest.approx(ref.pvalue, rel=1e-10)
        assert res.median_a[g] == np.median(a) and res.median_b[g] == np.median(b)


def test_mann_whitney_counts_one_side_missing():
    from isoends.stats.readlevel import mann_whitney_counts
    a = pd.DataFrame({"gene": ["G1", "G2"], "tail": [50, 60], "n_reads": [10, 10]})
    b = pd.DataFrame({"gene": ["G1"], "tail": [40], "n_reads": [10]})
    res = mann_whitney_counts(a, b).set_index("gene")
    assert np.isnan(res.p["G2"]) and res.p["G1"] < 1e-3
