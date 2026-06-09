"""
metrics.py — agreement / reliability statistics, pure numpy (no scipy/pingouin).

Used first on the synthetic dataset and later, unchanged, on real
MediaPipe-vs-goniometer data. All functions take plain numpy arrays.
"""
from __future__ import annotations
import numpy as np


def error_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Bias (mean signed error), MAE, RMSE of pred vs true (deg)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    m = np.isfinite(pred) & np.isfinite(true)
    d = pred[m] - true[m]
    return {"n": int(m.sum()),
            "bias": float(np.mean(d)),
            "mae": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d * d)))}


def bland_altman(a: np.ndarray, b: np.ndarray) -> dict:
    """Bland-Altman of two measurements: bias = mean(a-b), 95% limits of
    agreement = bias +/- 1.96*SD(a-b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return {"bias": bias, "sd": sd,
            "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def icc21(matrix: np.ndarray) -> float:
    """ICC(2,1), two-way random effects, single measures (Shrout & Fleiss 1979).
    `matrix` is targets (rows) x raters (cols). For method-vs-truth agreement,
    pass a 2-column matrix [truth, method]."""
    X = np.asarray(matrix, float)
    X = X[np.all(np.isfinite(X), axis=1)]
    n, k = X.shape
    grand = X.mean()
    row_m, col_m = X.mean(axis=1), X.mean(axis=0)
    SSR = k * np.sum((row_m - grand) ** 2)          # between targets
    SSC = n * np.sum((col_m - grand) ** 2)          # between raters
    SST = np.sum((X - grand) ** 2)
    SSE = SST - SSR - SSC                           # residual
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    return float((MSR - MSE) / denom) if denom != 0 else float("nan")


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Deterministic bootstrap percentile CI for a statistic of `values`."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([stat(v[i]) for i in idx])
    return float(np.percentile(boots, 100 * alpha / 2)), \
           float(np.percentile(boots, 100 * (1 - alpha / 2)))


def paired_mae_reduction(pred_base, pred_new, true, seed: int = 42) -> dict:
    """MAE reduction (baseline - new) with bootstrap 95% CI on the per-item
    absolute-error difference."""
    true = np.asarray(true, float)
    eb = np.abs(np.asarray(pred_base, float) - true)
    en = np.abs(np.asarray(pred_new, float) - true)
    m = np.isfinite(eb) & np.isfinite(en)
    diff = eb[m] - en[m]                              # >0 means new is better
    lo, hi = bootstrap_ci(diff, np.mean, seed=seed)
    return {"mae_base": float(np.mean(eb[m])), "mae_new": float(np.mean(en[m])),
            "reduction": float(np.mean(diff)), "ci_low": lo, "ci_high": hi}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true = rng.uniform(20, 160, 40)
    good = true + rng.normal(0, 2, 40)        # tight method
    bad = true + rng.normal(0, 12, 40)        # loose method
    print("self-test on synthetic two methods vs truth")
    print("  good:", {k: round(v, 2) for k, v in error_metrics(good, true).items()})
    print("  bad :", {k: round(v, 2) for k, v in error_metrics(bad, true).items()})
    print("  ICC(2,1) good-vs-truth:", round(icc21(np.c_[true, good]), 3))
    print("  ICC(2,1) bad-vs-truth :", round(icc21(np.c_[true, bad]), 3))
    print("  Bland-Altman good-vs-truth:", {k: round(v, 2) for k, v in bland_altman(good, true).items()})
    red = paired_mae_reduction(bad, good, true)
    print("  MAE reduction bad->good:", {k: round(v, 2) for k, v in red.items()})
