# stats_tests.py — Statistical significance tests for forecast comparison
import numpy as np
from scipy import stats


def dm_test(err_model: np.ndarray, err_baseline: np.ndarray, h: int = 1,
            power: int = 2) -> dict:
    """
    Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample correction).

    H0: both forecasts have equal expected loss (loss = |e|**power).
    The loss differential is d_t = L(model) - L(baseline); a NEGATIVE mean
    differential means the model is better.

    Args:
        err_model:    forecast errors of the candidate model (actual - forecast)
        err_baseline: forecast errors of the benchmark
        h:            forecast horizon (the long-run variance uses h-1 lags)
    Returns dict(stat, p_value, mean_diff, model_better, n).
    p_value is two-sided; `p_one_sided` tests H1: model strictly better.
    """
    e1, e2 = np.asarray(err_model, float), np.asarray(err_baseline, float)
    if e1.shape != e2.shape:
        raise ValueError("error arrays must have the same shape")
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    if n < 3:
        return dict(stat=np.nan, p_value=np.nan, p_one_sided=np.nan,
                    mean_diff=float(d.mean()) if n else np.nan,
                    model_better=False, n=n)
    d_bar = d.mean()
    # long-run variance with Bartlett weights up to lag h-1
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        lrv += 2 * (1 - k / h) * cov
    if lrv <= 1e-18:
        return dict(stat=0.0, p_value=1.0, p_one_sided=0.5, mean_diff=float(d_bar),
                    model_better=False, n=n)
    dm = d_bar / np.sqrt(lrv / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    p_two = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    p_one = stats.t.cdf(dm_hln, df=n - 1)          # small when model better
    return dict(stat=float(dm_hln), p_value=float(p_two), p_one_sided=float(p_one),
                mean_diff=float(d_bar), model_better=bool(d_bar < 0 and p_one < 0.05), n=n)


def directional_accuracy_test(actual_ret: np.ndarray, pred_ret: np.ndarray) -> dict:
    """Exact binomial test of H0: directional accuracy = 50%.
    Days with a flat actual or a zero forecast (e.g. the naive model) carry no
    directional call and are excluded; DA is NaN if none remain."""
    a, p = np.asarray(actual_ret, float), np.asarray(pred_ret, float)
    mask = (a != 0) & (p != 0)
    a, p = a[mask], p[mask]
    n = len(a)
    if n == 0:
        return dict(da=np.nan, p_value=np.nan, n=0)
    hits = int(np.sum(np.sign(a) == np.sign(p)))
    res = stats.binomtest(hits, n, 0.5, alternative='greater')
    return dict(da=hits / n, p_value=float(res.pvalue), n=n)


def oos_r2(actual_ret: np.ndarray, pred_ret: np.ndarray) -> float:
    """Out-of-sample R² versus the zero-return (random walk) forecast."""
    a, p = np.asarray(actual_ret, float), np.asarray(pred_ret, float)
    denom = np.sum(a ** 2)
    return float(1 - np.sum((a - p) ** 2) / denom) if denom > 0 else np.nan


def interval_metrics(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     nominal: float = 0.8) -> dict:
    """Prediction-interval coverage (PICP), average width and coverage gap."""
    a, lo, hi = (np.asarray(x, float) for x in (actual, lo, hi))
    covered = (a >= lo) & (a <= hi)
    return dict(picp=float(covered.mean()), nominal=nominal,
                coverage_gap=float(covered.mean() - nominal),
                mean_width=float(np.mean(hi - lo)))
