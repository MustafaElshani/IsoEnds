"""Moderated t-tests (limma: lmFit, then eBayes), in Python and through R.

Each feature has one value per unit of replication (a replicate pair, a sample, or a unit such as a cell line).
The one-sample test asks whether a feature's mean differs from zero; the two-sample test whether its mean differs
between control and treated samples. Missing values are allowed.

Each feature's residual variance s_i^2 on d_i df is squeezed towards a prior s_0^2 with d_0 prior df,
s~_i^2 = (d_0 s_0^2 + d_i s_i^2) / (d_0 + d_i), and the moderated t is referred to t on d_0 + d_i df, capped at the
pooled residual df (Smyth 2004). With robust=True the prior df are feature-specific (Phipson et al. 2016).

The hyperparameter estimation follows limma 3.62, including its choice of estimator: fitFDist (or fitFDistRobustly)
when all features have the same residual df, fitFDistUnequalDF1 when they differ. R's `optimize()` is ported.
"""
from __future__ import annotations

import functools
import math
import shutil
import subprocess
import tempfile
import warnings
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

from .basic import bh

EPS = np.finfo(float).eps


def logmdigamma(x):
    """log(x) - digamma(x), as statmod::logmdigamma (asymptotic series, accurate for large x)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    ok = ~np.isnan(x) & (x > 0)
    if not ok.any():
        return out
    z = x[ok]
    res = np.empty_like(z)
    small = z < 5
    if small.any():
        s = z[small]
        res[small] = (np.log(s / (s + 5)) + logmdigamma(s + 5) + 1 / s + 1 / (s + 1) + 1 / (s + 2) + 1 / (s + 3)
                      + 1 / (s + 4))
    big = ~small
    if big.any():
        zz = z[big]
        u = 1 / zz ** 2
        tail = u * (-1 / 12 + u * (1 / 120 + u * (-1 / 252 + u * (1 / 240 + u * (-1 / 132 + u * (
            691 / 32760 + u * (-1 / 12 + 3617 * u / 8160)))))))
        res[big] = 1 / (2 * zz) - tail
    out[ok] = res
    return out


def trigamma_inverse(x):
    """Inverse of the trigamma function (limma::trigammaInverse, Newton iteration)."""
    x = np.atleast_1d(np.asarray(x, dtype=float)).copy()
    y = np.full_like(x, np.nan)
    ok = ~np.isnan(x) & (x >= 0)
    huge = ok & (x > 1e7)
    tiny = ok & (x < 1e-6)
    y[huge] = 1 / np.sqrt(x[huge])
    y[tiny] = 1 / x[tiny]
    mid = ok & ~huge & ~tiny
    if mid.any():
        xm = x[mid]
        ym = 0.5 + 1 / xm
        for _ in range(51):
            tri = special.polygamma(1, ym)
            dif = tri * (1 - tri / xm) / special.polygamma(2, ym)
            ym = ym + dif
            if np.max(-dif / ym) < 1e-8:
                break
        y[mid] = ym
    return y


def brent_fmin(f, ax: float, bx: float, tol: float = EPS ** 0.25) -> float:
    """Minimise f on [ax, bx]: R's `optimize()` (Brent_fmin in src/library/stats/src/optimize.c), line for line."""
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(EPS)
    a, b = ax, bx
    v = a + c * (b - a)
    w = x = v
    d = e = 0.0
    fx = f(x)
    fv = fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def _r_trimmed_mean(z: np.ndarray, trim: float) -> float:
    """R's mean(z, trim = trim)."""
    n = len(z)
    if trim >= 0.5:
        return float(np.median(z))
    lo = int(math.floor(n * trim))
    s = np.sort(z)
    return float(s[lo:n - lo].mean())


def _f_logsf(x, d1, d2):
    return stats.f.logsf(x, d1, d2) if np.all(np.isfinite(d2)) else stats.chi2.logsf(x * d1, d1)


def fit_fdist(x, df1) -> dict:
    """limma:::fitFDist without a covariate: moment estimates of the prior scale and df."""
    x = np.asarray(x, dtype=float)
    df1 = np.broadcast_to(np.asarray(df1, dtype=float), x.shape).copy()
    n = len(x)
    if n == 0:
        return dict(scale=np.nan, df2=np.nan)
    if n == 1:
        return dict(scale=float(x[0]), df2=0.0)
    ok = np.isfinite(df1) & (df1 > 1e-15) & np.isfinite(x) & (x > -1e-15)
    nok = int(ok.sum())
    if nok == 1:
        return dict(scale=float(x[ok][0]), df2=0.0)
    x, df1 = x[ok], df1[ok]
    x = np.maximum(x, 0)
    m = float(np.median(x))
    if m == 0:
        warnings.warn("More than half of residual variances are exactly zero: eBayes unreliable")
        m = 1.0
    x = np.maximum(x, 1e-5 * m)
    e = np.log(x) + logmdigamma(df1 / 2)
    emean = float(e.mean())
    evar = float(np.sum((e - emean) ** 2) / (nok - 1))
    evar -= float(np.mean(special.polygamma(1, df1 / 2)))
    if evar > 0:
        df2 = float(2 * trigamma_inverse(evar)[0])
        s20 = float(np.exp(emean - logmdigamma(df2 / 2)))
    else:
        df2 = np.inf
        s20 = float(np.mean(x))
    return dict(scale=s20, df2=df2)


def _gauss_quad_uniform(n: int = 128):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (nodes + 1) / 2, weights / 2


def fit_fdist_robustly(x, df1, winsor_tail_p=(0.05, 0.1)) -> dict:
    """limma:::fitFDistRobustly without a covariate (the legacy robust estimator, equal residual df)."""
    x = np.asarray(x, dtype=float).copy()
    df1 = np.broadcast_to(np.asarray(df1, dtype=float), x.shape).copy()
    n = len(x)
    if n < 2:
        return dict(scale=np.nan, df2=np.nan, df2_shrunk=np.full(n, np.nan))
    if n == 2:
        fit = fit_fdist(x, df1)
        return dict(fit, df2_shrunk=np.full(n, fit["df2"]))
    ok = ~np.isnan(x) & np.isfinite(df1) & (df1 > 1e-6)
    if not ok.all():
        fit = fit_fdist_robustly(x[ok], df1[ok], winsor_tail_p)
        shrunk = np.empty(n)
        shrunk[ok] = fit["df2_shrunk"]
        shrunk[~ok] = fit["df2"]
        return dict(scale=fit["scale"], df2=fit["df2"], df2_shrunk=shrunk)
    m = float(np.median(x))
    if m <= 0:
        raise ValueError("Variances are mostly <= 0")
    x[x < m * 1e-12] = m * 1e-12
    nonrobust = fit_fdist(x, df1)
    wtp = np.resize(np.asarray(winsor_tail_p, dtype=float), 2)
    prob = np.array([wtp[0], 1 - wtp[1]])
    if np.all(wtp < 1 / n):
        return dict(nonrobust, df2_shrunk=np.full(n, nonrobust["df2"]))
    df1max = float(df1.max())
    i = df1 < (df1max - 1e-14)
    if i.any():
        s = nonrobust["scale"]
        f = x[i] / s
        df2 = nonrobust["df2"]
        pupper = stats.f.logsf(f, df1[i], df2)
        plower = stats.f.logcdf(f, df1[i], df2)
        up = pupper < plower
        f[up] = stats.f.isf(np.exp(pupper[up]), df1max, df2)
        f[~up] = stats.f.ppf(np.exp(plower[~up]), df1max, df2)
        x[i] = f * s
    df1s = df1max
    z = np.log(x)
    ztrend = _r_trimmed_mean(z, wtp[1])
    zresid = z - ztrend
    zrq = np.quantile(zresid, prob)
    zwins = np.clip(zresid, zrq[0], zrq[1])
    zwmean = float(zwins.mean())
    zwvar = float(np.mean((zwins - zwmean) ** 2) * n / (n - 1))
    gnodes, gweights = _gauss_quad_uniform(128)

    def linkfun(v):
        return v / (1 + v)

    def linkinv(v):
        return v / (1 - v) if v < 1 else np.inf

    def moments(d2):
        if np.isinf(d2):
            fq = stats.chi2.ppf([wtp[0], 1 - wtp[1]], df1s) / df1s
        else:
            fq = stats.f.ppf([wtp[0], 1 - wtp[1]], df1s, d2)
        zq = np.log(fq)
        q = linkfun(fq)
        nodes = q[0] + (q[1] - q[0]) * gnodes
        fnodes = nodes / (1 - nodes)
        znodes = np.log(fnodes)
        dens = (stats.chi2.pdf(fnodes * df1s, df1s) * df1s if np.isinf(d2) else stats.f.pdf(fnodes, df1s, d2))
        fvals = dens / (1 - nodes) ** 2
        q21 = q[1] - q[0]
        mm = q21 * np.sum(gweights * fvals * znodes) + np.sum(zq * wtp)
        vv = q21 * np.sum(gweights * fvals * (znodes - mm) ** 2) + np.sum((zq - mm) ** 2 * wtp)
        return mm, vv

    mom_mean, mom_var = moments(np.inf)
    funval_inf = math.log(zwvar / mom_var)
    if funval_inf <= 0:
        ztc = ztrend + zwmean - mom_mean
        s20 = float(np.exp(ztc))
        fstat = np.exp(z - ztc)
        tail_p = stats.chi2.sf(fstat * df1s, df1s)
        r = stats.rankdata(fstat)
        pno = np.minimum(tail_p / ((n - r + 0.5) / n), 1)
        shrunk = np.full(n, np.inf)
        o_ = pno < 1
        if o_.any():
            shrunk[o_] = pno[o_] * n * df1s
            o = np.argsort(tail_p, kind="stable")
            shrunk[o] = np.maximum.accumulate(shrunk[o])
        return dict(scale=s20, df2=np.inf, df2_shrunk=shrunk)

    def fun(v):
        _, var_ = moments(linkinv(v))
        return math.log(zwvar / var_)

    if np.isinf(nonrobust["df2"]):
        return dict(nonrobust, df2_shrunk=np.full(n, nonrobust["df2"]))
    rbx = linkfun(nonrobust["df2"])
    funval_low = fun(rbx)
    if funval_low >= 0:
        df2 = nonrobust["df2"]
    else:
        from scipy.optimize import brentq

        def fun_bounded(v):
            return funval_inf if v >= 1 else fun(v)

        df2 = linkinv(brentq(fun_bounded, rbx, 1.0, xtol=1e-8))
    mom_mean, _ = moments(df2)
    ztc = ztrend + zwmean - mom_mean
    s20 = float(np.exp(ztc))
    fstat = np.exp(z - ztc)
    log_tail_p = _f_logsf(fstat, df1s, df2)
    r = stats.rankdata(fstat)
    log_emp = np.log(n - r + 0.5) - math.log(n)
    log_pno = np.minimum(log_tail_p - log_emp, 0)
    pno = np.exp(log_pno)
    po = -np.expm1(log_pno)
    if np.any(log_pno < 0):
        min_log_tail_p = float(log_tail_p.min())
        if min_log_tail_p == -np.inf:
            shrunk = pno * df2
        else:
            df2_out = math.log(0.5) / min_log_tail_p * df2
            new_log = float(_f_logsf(fstat.max(), df1s, df2_out))
            df2_out = math.log(0.5) / new_log * df2_out
            shrunk = pno * df2 + po * df2_out
        shrunk = _monotone_shrunk(shrunk, log_tail_p)
    else:
        shrunk = np.full(n, df2)
    return dict(scale=s20, df2=df2, df2_shrunk=shrunk)


def _monotone_shrunk(shrunk: np.ndarray, order_by: np.ndarray) -> np.ndarray:
    """The last step of limma's robust estimators: pool the most outlying features' prior df down to their running
    mean and make the prior df non-decreasing in the order of `order_by` (tail P, most extreme first)."""
    n = len(shrunk)
    o = np.argsort(order_by, kind="stable")
    ordered = shrunk[o].copy()
    m = np.cumsum(ordered) / np.arange(1, n + 1)
    imin = int(np.argmin(m))
    ordered[:imin + 1] = m[imin]
    out = np.empty(n)
    out[o] = np.maximum.accumulate(ordered)
    return out


def fit_fdist_unequal_df1(x, df1, robust: bool = False, prior_weights=None) -> dict:
    """limma:::fitFDistUnequalDF1 without a covariate: weighted maximum-likelihood estimates of the prior scale and df,
    used by limma >= 3.61.8 when the residual df differ between features."""
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    df1 = np.broadcast_to(np.asarray(df1, dtype=float), x.shape).copy()
    pw = None if prior_weights is None else np.asarray(prior_weights, dtype=float).copy()
    na = np.isnan(x)
    if na.any():
        pw = (~na).astype(float) if pw is None else np.where(na, 0.0, pw)
        x[na] = 0.0
    small = df1 < 0.01
    if small.any():
        pw = (~small).astype(float) if pw is None else np.where(small, 0.0, pw)
        df1[small] = 1.0
    informative = x > 0
    if pw is not None:
        informative &= pw != 0
    n_inf = int(informative.sum())
    if n_inf < 2:
        return dict(scale=np.nan, df2=np.nan)
    if n_inf == 2:
        robust, pw = False, None
    m = float(np.median(x[informative]))
    xpos = np.maximum(x, 1e-12 * m)
    z = np.log(xpos)
    d1 = df1 / 2
    e = z + logmdigamma(d1)
    w = 1 / special.polygamma(1, d1)
    if pw is not None:
        w = w * pw
    emean = float(np.sum(w * e) / np.sum(w))
    d1x = d1 * xpos

    def minus_twice_loglik(par):
        d2 = par / (1 - par)
        d2s20 = d2 * math.exp(emean - float(logmdigamma(np.array([d2]))[0]))
        ll = (-(d1 + d2) * np.log1p(d1x / d2s20) - d1 * math.log(d2s20) + special.gammaln(d1 + d2)
              - special.gammaln(d2))
        if pw is not None:
            ll = pw * ll
        return -2.0 * float(np.sum(ll))

    par = brent_fmin(minus_twice_loglik, 0.5, 0.9998)
    d2 = par / (1 - par)
    s20 = float(np.exp(emean - logmdigamma(np.array([d2]))[0]))
    if not robust:
        return dict(scale=s20, df2=2 * d2)
    df2 = 2 * d2
    fstat = x / s20
    right_p = stats.f.sf(fstat, df1, df2)
    left_p = 1 - right_p
    low = left_p < 0.001
    if low.any():
        left_p[low] = stats.f.cdf(fstat[low], df1[low], df2)
    two_sided = 2 * np.minimum(left_p, right_p)
    fdr = bh(two_sided)
    fdr[fdr > 0.3] = 1
    if np.min(fdr) == 1:
        return dict(scale=s20, df2=df2)
    outpw = fit_fdist_unequal_df1(x, df1, robust=False, prior_weights=fdr)
    s20, df2 = outpw["scale"], outpw["df2"]
    r = stats.rankdata(fstat)
    uniform_p = (n - r + 0.5) / n
    pno = np.minimum(right_p / uniform_p, 1)
    if np.min(pno) == 1:
        return outpw
    i = int(np.argmin(right_p))
    min_right_p = right_p[i]
    if min_right_p == 0:
        shrunk = pno * df2
    else:
        df2_out = math.log(0.5) / math.log(min_right_p) * df2
        new_log = float(stats.f.logsf(fstat[i], df1[i], df2_out))
        df2_out = math.log(0.5) / new_log * df2_out
        shrunk = pno * df2 + (1 - pno) * df2_out
    shrunk = _monotone_shrunk(shrunk, right_p)
    return dict(scale=s20, df2=df2, df2_shrunk=shrunk)


def squeeze_var(var, df, robust: bool = False, winsor_tail_p=(0.05, 0.1), legacy: bool | None = None) -> dict:
    """limma::squeezeVar without a covariate: (df_prior, var_prior, var_post, estimator)."""
    var = np.asarray(var, dtype=float).copy()
    df = np.broadcast_to(np.asarray(df, dtype=float), var.shape).copy()
    n = len(var)
    if n == 0:
        raise ValueError("var is empty")
    if n < 3:
        return dict(df_prior=np.zeros(n), var_prior=var, var_post=var, estimator="none (fewer than 3 features)")
    var[df == 0] = 0
    if legacy is None:
        dfp = df[df > 0]
        legacy = bool(len(dfp)) and dfp.min() == dfp.max()
    if legacy:
        if robust:
            fit = fit_fdist_robustly(var, df, winsor_tail_p)
            df_prior = fit["df2_shrunk"]
            estimator = "fitFDistRobustly (legacy, robust)"
        else:
            fit = fit_fdist(var, df)
            df_prior = fit["df2"]
            estimator = "fitFDist (legacy)"
    else:
        fit = fit_fdist_unequal_df1(var, df, robust=robust)
        df_prior = fit.get("df2_shrunk", fit["df2"])
        estimator = "fitFDistUnequalDF1" + (" (robust)" if robust else "")
    df_prior = np.broadcast_to(np.asarray(df_prior, dtype=float), var.shape).copy()
    if np.isnan(df_prior).any():
        raise ValueError("Could not estimate prior df")
    s0 = fit["scale"]
    var_post = _squeeze(var, df, s0, df_prior)
    return dict(df_prior=df_prior, var_prior=s0, var_post=var_post, estimator=estimator)


def _squeeze(var, df, var_prior, df_prior):
    """limma:::.squeezeVar: the posterior variances (infinite prior df give the prior variance)."""
    if np.all(np.isfinite(df_prior)):
        return (df * var + df_prior * var_prior) / (df + df_prior)
    post = np.full(var.shape, float(var_prior))
    fin = np.isfinite(df_prior)
    post[fin] = (df[fin] * var[fin] + df_prior[fin] * var_prior) / (df[fin] + df_prior[fin])
    return post


RESULT_COLUMNS = ["mean", "t", "p", "n_units", "df_residual", "df_prior", "df_total", "s2_prior", "s2_post", "q"]


def _moderated(coef, unscaled, s2, df_res, index, robust: bool, legacy: bool | None, extra: dict) -> pd.DataFrame:
    """Squeeze the variances and form the moderated t, P and q of one coefficient per feature."""
    sq = squeeze_var(s2, df_res, robust=robust, legacy=legacy)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = coef / unscaled / np.sqrt(sq["var_post"])
    df_total = np.minimum(df_res + sq["df_prior"], np.nansum(df_res))
    out = pd.DataFrame({"mean": coef, "t": t, "p": 2 * stats.t.sf(np.abs(t), df_total), **extra,
                        "df_residual": df_res, "df_prior": sq["df_prior"], "df_total": df_total,
                        "s2_prior": sq["var_prior"], "s2_post": sq["var_post"]}, index=index)
    out["q"] = bh(out.p)
    out.attrs["estimator"] = sq["estimator"]
    return out


def _empty_result(index, columns=RESULT_COLUMNS) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns), index=index, dtype=float)


def moderated_one_sample_python(X: pd.DataFrame, robust: bool = True, legacy: bool | None = None) -> pd.DataFrame:
    """Moderated one-sample t-test per row of X (features x units, NaN allowed).

    Returns, indexed like X: mean, t, p, q (BH over the rows tested), n_units, df_residual, df_prior, df_total,
    s2_prior, s2_post. Rows with fewer than 2 values are dropped."""
    M = X.to_numpy(dtype=float)
    n = (~np.isnan(M)).sum(axis=1)
    keep = n >= 2
    M, idx, n = M[keep], X.index[keep], n[keep]
    if len(M) == 0:
        return _empty_result(X.index[:0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(M, axis=1)
        s2 = np.nanvar(M, axis=1, ddof=1)
    return _moderated(mean, 1 / np.sqrt(n), s2, (n - 1).astype(float), idx, robust, legacy, {"n_units": n})


TWO_SAMPLE_COLUMNS = ["mean", "t", "p", "mean_control", "mean_treated", "n_control", "n_treated", "df_residual",
                      "df_prior", "df_total", "s2_prior", "s2_post", "q"]


def _two_sample_rows(control: pd.DataFrame, treated: pd.DataFrame) -> np.ndarray:
    """Rows that can be tested: a value in each group and at least 3 values in all."""
    n1, n2 = control.notna().sum(axis=1).to_numpy(), treated.notna().sum(axis=1).to_numpy()
    return (n1 >= 1) & (n2 >= 1) & (n1 + n2 >= 3)


def moderated_two_sample_python(control: pd.DataFrame, treated: pd.DataFrame, robust: bool = True,
                                legacy: bool | None = None) -> pd.DataFrame:
    """Moderated two-sample t-test per row (features x samples of each condition, NaN allowed), pooled variance.

    Returns, indexed like the inputs: mean (treated - control), t, p, q, mean_control, mean_treated, n_control,
    n_treated, df_residual, df_prior, df_total, s2_prior, s2_post. Rows without a value in each group or with
    fewer than 3 values are dropped."""
    keep = _two_sample_rows(control, treated)
    A, B, idx = control.to_numpy(float)[keep], treated.to_numpy(float)[keep], control.index[keep]
    if len(A) == 0:
        return _empty_result(idx, TWO_SAMPLE_COLUMNS)
    n1, n2 = (~np.isnan(A)).sum(axis=1), (~np.isnan(B)).sum(axis=1)
    m1, m2 = np.nanmean(A, axis=1), np.nanmean(B, axis=1)
    rss = np.nansum((A - m1[:, None]) ** 2, axis=1) + np.nansum((B - m2[:, None]) ** 2, axis=1)
    df_res = (n1 + n2 - 2).astype(float)
    extra = {"mean_control": m1, "mean_treated": m2, "n_control": n1, "n_treated": n2}
    return _moderated(m2 - m1, np.sqrt(1 / n1 + 1 / n2), rss / df_res, df_res, idx, robust, legacy, extra)


@functools.lru_cache(maxsize=None)
def r_limma_available(rscript: str | None = None) -> str | None:
    """Path of an Rscript that can load limma, or None."""
    exe = rscript or shutil.which("Rscript")
    if not exe:
        return None
    try:
        run = subprocess.run([exe, "-e", "suppressMessages(library(limma)); cat(as.character(packageVersion('limma')))"],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return exe if run.returncode == 0 and run.stdout.strip() else None


def r_limma_version(rscript: str) -> str:
    run = subprocess.run([rscript, "-e", "cat(as.character(packageVersion('limma')), R.version.string, sep='|')"],
                         capture_output=True, text=True, timeout=120)
    return run.stdout.strip()


def _run_r(X: pd.DataFrame, robust: bool, rscript: str | None, groups=None) -> tuple[pd.DataFrame, str]:
    """Fit the rows of X with the packaged R script; returns limma's table in the order of X and its version."""
    exe = rscript or r_limma_available()
    if exe is None:
        raise RuntimeError("Rscript with limma not found")
    script = resources.files("isoends").joinpath("R/limma_moderated_t.R")
    with tempfile.TemporaryDirectory() as tmp, resources.as_file(script) as script_path:
        keys = pd.Index([f"f{i}" for i in range(len(X))])
        Xr = X.set_axis(keys, axis=0)
        Xr.columns = [f"u{i}" for i in range(X.shape[1])]
        Xr.rename_axis("feature").to_csv(Path(tmp) / "values.tsv", sep="\t", na_rep="NA", float_format="%.17g")
        cmd = [exe, str(script_path), f"{tmp}/values.tsv", f"{tmp}/out.tsv", str(robust).upper()]
        if groups is not None:
            cmd.append(",".join(str(int(g)) for g in groups))
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"limma failed:\n{run.stderr[-3000:]}")
        res = pd.read_csv(Path(tmp) / "out.tsv", sep="\t", index_col="feature").reindex(keys)
    return res.set_axis(X.index, axis=0), "R limma " + run.stdout.strip()


def _from_r(res: pd.DataFrame, extra: dict) -> pd.DataFrame:
    return pd.DataFrame({"mean": res.logFC, "t": res.t, "p": res["P.Value"], **extra,
                         "df_residual": res.df_residual, "df_prior": res.df_prior, "df_total": res.df_total,
                         "s2_prior": res.s2_prior, "s2_post": res.s2_post, "q": res["adj.P.Val"]}, index=res.index)


def moderated_one_sample_r(X: pd.DataFrame, robust: bool = True, rscript: str | None = None) -> pd.DataFrame:
    """The one-sample test through R limma (lmFit on a design of ones, eBayes, topTable)."""
    X = X[X.notna().sum(axis=1) >= 2]
    if len(X) == 0:
        return _empty_result(X.index)
    res, version = _run_r(X, robust, rscript)
    out = _from_r(res, {"n_units": res.n_units})
    out.attrs["estimator"] = version
    return out


def moderated_two_sample_r(control: pd.DataFrame, treated: pd.DataFrame, robust: bool = True,
                           rscript: str | None = None) -> pd.DataFrame:
    """The two-sample test through R limma (lmFit on a control + treated design, eBayes, topTable)."""
    keep = _two_sample_rows(control, treated)
    control, treated = control[keep], treated[keep]
    if len(control) == 0:
        return _empty_result(control.index, TWO_SAMPLE_COLUMNS)
    X = pd.concat([control, treated], axis=1)
    res, version = _run_r(X, robust, rscript, groups=[0] * control.shape[1] + [1] * treated.shape[1])
    out = _from_r(res, {"mean_control": control.mean(axis=1), "mean_treated": treated.mean(axis=1),
                        "n_control": control.notna().sum(axis=1), "n_treated": treated.notna().sum(axis=1)})
    out.attrs["estimator"] = version
    return out


def _backend(backend: str, rscript: str | None) -> str:
    if backend == "auto":
        return "r" if r_limma_available(rscript) else "python"
    if backend not in ("r", "python"):
        raise ValueError(f"unknown limma backend {backend!r}")
    return backend


def moderated_one_sample(X: pd.DataFrame, robust: bool = True, backend: str = "python",
                         rscript: str | None = None) -> pd.DataFrame:
    """Moderated one-sample t-test per row of X (features x units). backend: "python", "r" or "auto" (R when
    Rscript with limma is found)."""
    if _backend(backend, rscript) == "r":
        return moderated_one_sample_r(X, robust=robust, rscript=rscript)
    return moderated_one_sample_python(X, robust=robust)


def moderated_two_sample(control: pd.DataFrame, treated: pd.DataFrame, robust: bool = True, backend: str = "python",
                         rscript: str | None = None) -> pd.DataFrame:
    """Moderated two-sample t-test per row (features x samples of each condition), treated - control."""
    if _backend(backend, rscript) == "r":
        return moderated_two_sample_r(control, treated, robust=robust, rscript=rscript)
    return moderated_two_sample_python(control, treated, robust=robust)
