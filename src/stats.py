"""Comparación estadística STFT vs CWT (Wilcoxon pareado) y bootstrap CI."""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_paired(
    a: np.ndarray, b: np.ndarray, alternative: str = "two-sided"
) -> dict:
    """Wilcoxon de rangos con signo pareado (uno-a-uno por sujeto)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    diff = a - b
    n = (diff != 0).sum()
    z = (res.statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    rb = z / np.sqrt(n) if n > 0 else 0.0
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "rank_biserial": float(rb),
        "median_diff": float(np.median(diff)),
        "n_paired": int(n),
    }


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    statistic=np.mean,
) -> tuple[float, float, float]:
    """Bootstrap percentil CI para una muestra 1D."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = statistic(values[idx])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(statistic(values)), float(lo), float(hi)


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Devuelve la máscara de hipótesis rechazadas por BH-FDR."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    rejected = p_sorted <= thresh
    if rejected.any():
        last = np.max(np.where(rejected))
        rejected[: last + 1] = True
    out = np.zeros(n, dtype=bool)
    out[order] = rejected
    return out.tolist()
