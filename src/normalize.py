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

    Args:
        X: (n_epochs, n_channels, F, M).
    """
    axes = (0, 2, 3)
    mean = X.mean(axis=axes)
    std = X.std(axis=axes)
    std = np.where(std < eps, 1.0, std)
    return ChannelStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_channel_zscore(X: np.ndarray, stats: ChannelStats) -> np.ndarray:
    mean = stats.mean[None, :, None, None]
    std = stats.std[None, :, None, None]
    return ((X - mean) / std).astype(np.float32)
