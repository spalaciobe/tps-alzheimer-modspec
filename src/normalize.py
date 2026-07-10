"""Z-score por canal con estadísticos calculados solo en train (anti-leakage)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelStats:
    mean: np.ndarray   # (n_channels,)
    std: np.ndarray    # (n_channels,)


def fit_channel_zscore(X: np.ndarray, eps: float = 1e-8, chunk: int = 512) -> ChannelStats:
    """Calcula media y desviación por canal sobre todos los epochs de train.

    Acumulación en float64 por CHUNKS de epochs (dos pasadas: media y varianza)
    para (a) evitar overflow con float16 y (b) NO materializar un temporal
    float64 del tamaño completo. `X.std(dtype=np.float64)` sobre un train grande
    (p.ej. (50568, 19, 45, 45)) asignaba ~14.5 GiB de una sola vez → OOM en el
    camino --no-bank. Este cálculo por chunks tiene pico ~chunk·C·F·M·8 bytes
    (~150 MB con chunk=512) y es numéricamente idéntico a numpy.std (ddof=0,
    dos pasadas centrando en la media).

    Args:
        X: (n_epochs, n_channels, F, M).
    """
    n, n_ch = X.shape[0], X.shape[1]
    per = X.shape[2] * X.shape[3]
    count = float(n * per)

    # Pasada 1: media por canal.
    s1 = np.zeros(n_ch, dtype=np.float64)
    for i in range(0, n, chunk):
        xb = np.asarray(X[i:i + chunk], dtype=np.float64)
        s1 += xb.sum(axis=(0, 2, 3))
    mean = s1 / count

    # Pasada 2: varianza por canal (suma de cuadrados centrados).
    m = mean[None, :, None, None]
    s2 = np.zeros(n_ch, dtype=np.float64)
    for i in range(0, n, chunk):
        xb = np.asarray(X[i:i + chunk], dtype=np.float64)
        xb -= m
        xb *= xb
        s2 += xb.sum(axis=(0, 2, 3))
    std = np.sqrt(np.maximum(s2 / count, 0.0))
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
