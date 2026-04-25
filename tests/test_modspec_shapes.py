"""Verifica shapes y validez numérica del modulation spectrum."""

from __future__ import annotations

import numpy as np
import pytest

from src.modspec import ModSpecConfig, compute_modulation_spectrum


@pytest.fixture
def synthetic_epoch():
    """Epoch de 8 s a 200 Hz con 19 canales y mezcla de senoidales."""
    rng = np.random.default_rng(0)
    fs = 200
    t = np.arange(8 * fs) / fs
    # Mezcla 10 Hz + 4 Hz modulada en 1 Hz + ruido
    base = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 4 * t) * (1 + 0.5 * np.cos(2 * np.pi * 1 * t))
    epoch = np.stack([base + 0.1 * rng.standard_normal(t.size) for _ in range(19)], axis=0)
    return epoch.astype(np.float32)


def test_stft_shape(synthetic_epoch):
    cfg = ModSpecConfig(method="stft", fs=200)
    M = compute_modulation_spectrum(synthetic_epoch, cfg)
    assert M.shape == (19, 45, 45)
    assert np.isfinite(M).all()


def test_cwt_shape(synthetic_epoch):
    cfg = ModSpecConfig(method="cwt", fs=200)
    M = compute_modulation_spectrum(synthetic_epoch, cfg)
    assert M.shape == (19, 45, 45)
    assert np.isfinite(M).all()


def test_stft_deterministic(synthetic_epoch):
    cfg = ModSpecConfig(method="stft", fs=200)
    M1 = compute_modulation_spectrum(synthetic_epoch, cfg)
    M2 = compute_modulation_spectrum(synthetic_epoch, cfg)
    np.testing.assert_array_equal(M1, M2)


def test_method_unknown(synthetic_epoch):
    cfg = ModSpecConfig(method="hartley", fs=200)
    with pytest.raises(ValueError):
        compute_modulation_spectrum(synthetic_epoch, cfg)
