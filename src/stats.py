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


def delong_test(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> dict:
    """DeLong test pareado para diferencia entre dos AUCs sobre los mismos sujetos.

    Implementación basada en Sun & Xu (2014). Robusto a tamaños pequeños.
    """
    y_true = np.asarray(y_true).astype(int)
    pos = y_true == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Necesitas ambas clases para DeLong")

    def _compute(scores):
        # AUC empírico vía Mann-Whitney U
        s_pos = scores[pos]
        s_neg = scores[neg]
        # Componentes V10 y V01 para varianza
        ranks_neg = np.array([(s_neg < x).sum() + 0.5 * (s_neg == x).sum() for x in s_pos])
        ranks_pos = np.array([(s_pos > x).sum() + 0.5 * (s_pos == x).sum() for x in s_neg])
        V10 = ranks_neg / n_neg
        V01 = ranks_pos / n_pos
        auc = V10.mean()
        return auc, V10, V01

    auc_a, V10_a, V01_a = _compute(np.asarray(score_a, dtype=float))
    auc_b, V10_b, V01_b = _compute(np.asarray(score_b, dtype=float))

    s10 = np.cov(V10_a, V10_b, ddof=1)
    s01 = np.cov(V01_a, V01_b, ddof=1)
    var_diff = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n_pos \
        + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n_neg
    if var_diff <= 0:
        z = 0.0
        pvalue = 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var_diff)
        pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": float(auc_a - auc_b),
        "z": float(z),
        "pvalue": float(pvalue),
    }


def effective_n(n_epochs: int, overlap_ratio: float) -> int:
    """N efectivo por autocorrelación: n_epochs * (1 - overlap_ratio).

    Para epochs de 8 s con paso 1 s (overlap 7/8 = 0.875), N_eff ≈ n_epochs / 8.
    Tests estadísticos con N_eff en lugar de N_epochs evitan p-values inflados.
    """
    return max(1, int(round(n_epochs * (1.0 - overlap_ratio))))


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
