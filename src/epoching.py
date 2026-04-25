"""Segmentación en epochs de duración fija con solapamiento."""

from __future__ import annotations

import mne
import numpy as np


def make_epochs(
    raw: mne.io.BaseRaw,
    duration_s: float = 8.0,
    overlap_s: float = 7.0,
    drop_last_s: float = 7.0,
) -> mne.Epochs:
    """Genera epochs de `duration_s` con `overlap_s` segundos de solapamiento.

    El paper de Lopes especifica "8 s con 1 s de solapamiento" en el sentido
    de que el paso entre epochs es 1 s ⇒ overlap_s = duration_s − 1 = 7.

    Antes de epochar se descartan los últimos `drop_last_s` segundos del
    registro para evitar leakage entre folds (siguiendo el paper).
    """
    if drop_last_s > 0:
        tmax = raw.times[-1] - drop_last_s
        if tmax <= 0:
            raise ValueError(f"Registro demasiado corto: {raw.times[-1]:.1f}s")
        raw = raw.copy().crop(tmin=0.0, tmax=tmax)

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=duration_s,
        overlap=overlap_s,
        preload=True,
        verbose="ERROR",
    )
    return epochs


def expected_n_epochs(duration_record_s: float, duration_s: float, overlap_s: float) -> int:
    """Cuántos epochs esperar para una grabación de longitud dada."""
    step = duration_s - overlap_s
    return int(np.floor((duration_record_s - duration_s) / step)) + 1
