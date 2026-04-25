"""Cálculo del modulation spectrum 2D vía STFT o CWT.

Pipeline (Lopes et al. 2023):
    1. Mapeo tiempo-frecuencia X(t, f) — STFT con ventana Hann o CWT-Morlet.
    2. Potencia instantánea P(t, f) = |X(t, f)|^2.
    3. FFT temporal sobre el eje t: M(f, f_mod) = FT_t{P(t, f)}.
    4. Recorte a la región de interés (carrier 0.5-45 Hz, mod 0-22.5 Hz).
    5. Resize a la rejilla objetivo (45×45).
    6. Log-power para estabilidad numérica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt
from scipy.ndimage import zoom
from scipy.signal import stft


@dataclass
class ModSpecConfig:
    method: str                          # "stft" | "cwt"
    fs: int                              # 200 o 500 Hz
    target_shape: tuple[int, int] = (45, 45)
    carrier_range_hz: tuple[float, float] = (0.5, 45.0)
    mod_range_hz: tuple[float, float] = (0.0, 22.5)
    # STFT
    window: str = "hann"
    nperseg: int = 128
    noverlap: int = 64
    # CWT
    wavelet: str = "cmor1.5-1.0"
    n_scales: int = 50
    # Común
    log_power: bool = True
    eps: float = 1e-10


def _stft_tf(
    x: np.ndarray, fs: int, window: str, nperseg: int, noverlap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT ⇒ (f, t, |X|^2)."""
    f, t, Zxx = stft(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        boundary=None, padded=False,
    )
    P = np.abs(Zxx) ** 2
    return f, t, P


def _cwt_tf(
    x: np.ndarray, fs: int, wavelet: str, n_scales: int,
    f_low: float, f_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CWT-Morlet ⇒ (f, t, |X|^2). Escalas log entre f_low y f_high."""
    freqs = np.geomspace(f_low, f_high, n_scales)
    central = pywt.central_frequency(wavelet)
    scales = central * fs / freqs
    coefs, freqs_out = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    P = np.abs(coefs) ** 2
    t = np.arange(P.shape[1]) / fs
    return freqs_out, t, P


def _modspec_from_tf(
    P: np.ndarray, t: np.ndarray, mod_high_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """FFT temporal sobre la potencia instantánea ⇒ (M, f_mod)."""
    n_t = P.shape[1]
    if n_t < 4:
        raise ValueError(f"Muy pocas muestras temporales en T-F: {n_t}")
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    fs_t = 1.0 / dt
    M = np.abs(np.fft.rfft(P, axis=1))
    f_mod = np.fft.rfftfreq(n_t, d=dt)
    keep = f_mod <= mod_high_hz
    return M[:, keep], f_mod[keep]


def _crop_carrier(
    M: np.ndarray, f_carrier: np.ndarray, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    keep = (f_carrier >= low) & (f_carrier <= high)
    return M[keep], f_carrier[keep]


def _resize_2d(M: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    if M.shape == target:
        return M
    zy = target[0] / M.shape[0]
    zx = target[1] / M.shape[1]
    return zoom(M, (zy, zx), order=1)


def compute_modulation_spectrum_single(
    x: np.ndarray, cfg: ModSpecConfig
) -> np.ndarray:
    """Calcula el modulation spectrum 2D de una señal monocanal.

    Args:
        x: array (n_samples,) — un canal de un epoch.
        cfg: configuración.

    Returns:
        Array (F, M) con la rejilla `cfg.target_shape`.
    """
    if cfg.method == "stft":
        f, t, P = _stft_tf(x, cfg.fs, cfg.window, cfg.nperseg, cfg.noverlap)
    elif cfg.method == "cwt":
        f, t, P = _cwt_tf(
            x, cfg.fs, cfg.wavelet, cfg.n_scales,
            cfg.carrier_range_hz[0], cfg.carrier_range_hz[1],
        )
    else:
        raise ValueError(f"method desconocido: {cfg.method}")

    P, f = _crop_carrier(P, f, *cfg.carrier_range_hz)
    M, _f_mod = _modspec_from_tf(P, t, cfg.mod_range_hz[1])
    M = _resize_2d(M, cfg.target_shape)

    if cfg.log_power:
        M = 10.0 * np.log10(M + cfg.eps)
    return M


def compute_modulation_spectrum(
    epoch: np.ndarray, cfg: ModSpecConfig
) -> np.ndarray:
    """Modulation spectrum multicanal.

    Args:
        epoch: array (n_channels, n_samples).
        cfg: configuración.

    Returns:
        Array (n_channels, F, M).
    """
    n_ch = epoch.shape[0]
    out = np.empty((n_ch, *cfg.target_shape), dtype=np.float32)
    for c in range(n_ch):
        out[c] = compute_modulation_spectrum_single(epoch[c], cfg).astype(np.float32)
    return out
