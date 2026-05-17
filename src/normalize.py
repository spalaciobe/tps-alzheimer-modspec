"""Z-score por canal con estadísticos calculados solo en train (anti-leakage)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelStats:
    mean: np.ndarray   # (n_channels,)
    std: np.ndarray    # (n_channels,)


def fit_channel_zscore(X: np.ndarray, eps: float = 1e-8) -> ChannelStats:
    """Calcula media y desviación por canal sobre todos los epochs de train.

    Acumula en float64 para evitar overflow con float16 (max ~65504), aunque
    el input venga en cualquier dtype.

    Args:
        X: (n_epochs, n_channels, F, M).
    """
    axes = (0, 2, 3)
    mean = X.mean(axis=axes, dtype=np.float64)
    std = X.std(axis=axes, dtype=np.float64)
    std = np.where(std < eps, 1.0, std)
    return ChannelStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_channel_zscore(X: np.ndarray, stats: ChannelStats) -> np.ndarray:
    """Z-score por canal. In-place chunked para evitar OOM.

    Si X es float16, hace z-score en float32 (temporal por chunk) y vuelve a fp16
    para preservar el ahorro de RAM. Float16 puro romperia: tras restar la media
    (~10-20 en escala log-power), valores cercanos a 0 pierden precision relativa.
    """
    if X.dtype == np.float16:
        # In-place sobre fp16 directamente no es preciso. Trabajamos por chunk
        # promoviendo cada chunk a fp32, normalizando, y devolviendo a fp16.
        mean = stats.mean[None, :, None, None].astype(np.float32)
        std = stats.std[None, :, None, None].astype(np.float32)
        out = np.empty_like(X)  # fp16
        chunk = 2048
        for i in range(0, X.shape[0], chunk):
            tmp = X[i:i+chunk].astype(np.float32, copy=True)
            tmp -= mean
            tmp /= std
            out[i:i+chunk] = tmp.astype(np.float16)
        return out
    # Camino fp32: in-place chunked
    mean = stats.mean[None, :, None, None].astype(np.float32)
    std = stats.std[None, :, None, None].astype(np.float32)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    else:
        X = X.copy() if not X.flags.writeable else X
    chunk = 2048
    for i in range(0, X.shape[0], chunk):
        X[i:i+chunk] -= mean
        X[i:i+chunk] /= std
    return X
