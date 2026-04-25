"""Verifica que el conteo de epochs coincide con la fórmula esperada."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from src.epoching import expected_n_epochs, make_epochs


def _fake_raw(duration_s: float, fs: int = 200, n_ch: int = 4) -> mne.io.RawArray:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_ch, int(duration_s * fs))).astype(np.float64) * 1e-6
    info = mne.create_info(ch_names=[f"ch{i}" for i in range(n_ch)], sfreq=fs, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


@pytest.mark.parametrize("duration_s", [60.0, 120.0, 480.0])
def test_n_epochs_8s_1s_overlap(duration_s):
    raw = _fake_raw(duration_s)
    epochs = make_epochs(raw, duration_s=8.0, overlap_s=7.0, drop_last_s=0.0)
    expected = expected_n_epochs(duration_s, 8.0, 7.0)
    # MNE puede dejar 1 fuera por bordes; aceptamos ±1
    assert abs(len(epochs) - expected) <= 1


def test_drop_last_reduces_count():
    raw = _fake_raw(60.0)
    full = make_epochs(raw, 8.0, 7.0, drop_last_s=0.0)
    cropped = make_epochs(raw, 8.0, 7.0, drop_last_s=7.0)
    assert len(cropped) < len(full)


def test_min_epochs_8min_record():
    """Paper: ≥460 epochs por sujeto en grabaciones de 8 min."""
    raw = _fake_raw(8 * 60)
    epochs = make_epochs(raw, 8.0, 7.0, drop_last_s=7.0)
    # 480 - 8 - 7 ≈ 465 epochs esperados
    assert len(epochs) >= 460
