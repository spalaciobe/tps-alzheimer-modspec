"""Verifica que la z-norm por canal produce media≈0 y std≈1 en train."""

from __future__ import annotations

import numpy as np

from src.normalize import apply_channel_zscore, fit_channel_zscore


def test_zscore_train_stats():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(50, 19, 45, 45)).astype(np.float32)
    stats = fit_channel_zscore(X)
    Z = apply_channel_zscore(X, stats)
    for c in range(19):
        assert abs(Z[:, c].mean()) < 1e-3
        assert abs(Z[:, c].std() - 1.0) < 1e-3


def test_zscore_no_leakage_apply_with_train_stats():
    """Las estadísticas se calculan en train, NO en test."""
    rng = np.random.default_rng(0)
    Xtr = rng.normal(loc=0.0, scale=1.0, size=(50, 19, 45, 45)).astype(np.float32)
    Xte = rng.normal(loc=10.0, scale=5.0, size=(20, 19, 45, 45)).astype(np.float32)

    stats = fit_channel_zscore(Xtr)
    Z_te = apply_channel_zscore(Xte, stats)
    # Si se hubieran fugado los stats de test la media estaría en ~0, no en ~10
    assert Z_te.mean() > 5.0
