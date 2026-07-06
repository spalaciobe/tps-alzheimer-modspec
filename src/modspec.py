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
    method: str                          # "stft" | "cwt" | "cwt_fair"
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
    mod_env_rate_hz: float | None = None  # tasa objetivo (Hz) de la envolvente
    #   antes de la FFT de modulación. Si se fija, la envolvente de potencia se
    #   decima a esta tasa para que el eje de modulación sea COMPARABLE entre
    #   transformadas con distinta resolución temporal nativa (p.ej. CWT a fs
    #   vs STFT a fs/hop). None = sin decimación (comportamiento original).
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


def _decimate_power_envelope(P: np.ndarray, q: int, axis: int) -> np.ndarray:
    """Downsamplea una envolvente de potencia por factor entero q con
    anti-aliasing (poly-phase FIR, fase cero vía upfirdn/Kaiser).

    ADVERTENCIA: para q grande (p.ej. 64) el FIR anti-alias por defecto tiene
    ~2*10*q+1 taps (~1281). Debe aplicarse sobre la señal COMPLETA (miles de
    muestras), NO sobre un epoch aislado (1600 muestras a 8 s / 200 Hz), donde
    el zero-pad del borde contaminaría gran parte de la ventana. Por eso el fix
    decima el registro completo una sola vez en `compute_modulation_spectrum_subject`.

    `padtype='line'` reduce el transitorio de borde frente al zero-pad. La
    potencia no puede ser negativa, así que se recorta a >=0 el leve undershoot
    del FIR (guarda de validez física).
    """
    if q < 2:
        return P
    from scipy.signal import resample_poly
    out = resample_poly(P, up=1, down=q, axis=axis, padtype="line")
    return np.clip(out, 0.0, None)


def _match_env_rate(
    P: np.ndarray, t: np.ndarray, target_rate_hz: float | None, time_axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Iguala la tasa de muestreo de la envolvente P a `target_rate_hz`.

    Decima P a lo largo de `time_axis` por el factor entero q más cercano y
    reconstruye el vector temporal t con el nuevo paso. Hace COMPARABLE el eje
    de modulación entre transformadas cuya T-F nativa difiere en resolución
    temporal (CWT preserva fs; STFT muestrea a fs/hop). Devuelve (P', t').
    """
    if target_rate_hz is None or len(t) < 2:
        return P, t
    native_rate = 1.0 / float(np.median(np.diff(t)))
    q = int(round(native_rate / target_rate_hz))
    if q < 2:
        return P, t
    P_dec = _decimate_power_envelope(P, q, axis=time_axis)
    n_new = P_dec.shape[time_axis]
    dt_new = q / native_rate  # = 1 / target_rate_hz (por construcción)
    t_new = float(t[0]) + np.arange(n_new) * dt_new
    return P_dec, t_new


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
    elif cfg.method in ("cwt", "cwt_fair"):
        f, t, P = _cwt_tf(
            x, cfg.fs, cfg.wavelet, cfg.n_scales,
            cfg.carrier_range_hz[0], cfg.carrier_range_hz[1],
        )
    else:
        raise ValueError(f"method desconocido: {cfg.method}")

    if cfg.mod_env_rate_hz is not None:
        # La decimación de la envolvente (mod_env_rate_hz, p.ej. cwt_fair) requiere
        # el FIR anti-alias sobre la señal COMPLETA, no sobre un epoch aislado
        # (ver _decimate_power_envelope). Este camino monocanal opera epoch a
        # epoch, así que se prohíbe: usar compute_modulation_spectrum_subject.
        raise ValueError(
            "mod_env_rate_hz no está soportado en compute_modulation_spectrum_single "
            "(decimación por-epoch no fiable). Usa compute_modulation_spectrum_subject."
        )
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


def _full_signal_tf(x: np.ndarray, cfg: ModSpecConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula la T-F de toda la señal de un canal. Devuelve (f, t, P)."""
    if cfg.method == "stft":
        return _stft_tf(x, cfg.fs, cfg.window, cfg.nperseg, cfg.noverlap)
    if cfg.method in ("cwt", "cwt_fair"):
        return _cwt_tf(
            x, cfg.fs, cfg.wavelet, cfg.n_scales,
            cfg.carrier_range_hz[0], cfg.carrier_range_hz[1],
        )
    raise ValueError(f"method desconocido: {cfg.method}")


def compute_modulation_spectrum_subject(
    data: np.ndarray,
    cfg: ModSpecConfig,
    epoch_duration_s: float,
    epoch_step_s: float,
    drop_last_s: float = 0.0,
) -> np.ndarray:
    """Modulation spectrum para todos los epochs de un sujeto, usando una sola T-F.

    Mucho más rápido que aplicar `compute_modulation_spectrum` epoch por epoch
    porque la transformada T-F se computa una vez sobre la señal completa y
    luego se hace slicing por epoch antes de la FFT temporal.

    Args:
        data: array (n_channels, n_samples) — señal completa del sujeto a fs=cfg.fs.
        cfg: ModSpecConfig.
        epoch_duration_s: duración de cada epoch en segundos.
        epoch_step_s: paso entre epochs (1.0 para "8 s con 1 s overlap").
        drop_last_s: descarta los últimos `drop_last_s` segundos antes de epochar.

    Returns:
        Array (n_epochs, n_channels, F, M).
    """
    n_ch, n_samp = data.shape
    fs = cfg.fs
    end_samp = n_samp - int(round(drop_last_s * fs))
    if end_samp <= int(round(epoch_duration_s * fs)):
        raise ValueError("Señal demasiado corta tras drop_last_s")
    data = data[:, :end_samp]

    # T-F una sola vez por canal (canal por canal — pywt.cwt no acepta multicanal aún)
    P_per_ch: list[np.ndarray] = []
    f_carrier: np.ndarray | None = None
    t_axis: np.ndarray | None = None
    for c in range(n_ch):
        f, t, P = _full_signal_tf(data[c], cfg)
        if f_carrier is None:
            f_carrier = f
            t_axis = t
        P_per_ch.append(P)

    P_full = np.stack(P_per_ch, axis=0)                          # (n_ch, F_full, T_tf)
    P_full, f_carrier = _crop_carrier(P_full.transpose(1, 0, 2),
                                      f_carrier,
                                      *cfg.carrier_range_hz)
    P_full = P_full.transpose(1, 0, 2)                           # (n_ch, F, T_tf)

    # Igualar la tasa de la envolvente ANTES de rebanar epochs, para que el eje
    # de modulación sea comparable con otras transformadas (fair comparison).
    # Se decima una sola vez sobre la señal completa (filtrado continuo, sin
    # transitorios por epoch).
    if cfg.mod_env_rate_hz is not None:
        P_full, t_axis = _match_env_rate(P_full, t_axis, cfg.mod_env_rate_hz, time_axis=2)

    # Mapear epochs en muestras → índices en el eje T-F (t_axis)
    epoch_dur_samp = int(round(epoch_duration_s * fs))
    epoch_step_samp = int(round(epoch_step_s * fs))
    epoch_starts = np.arange(0, end_samp - epoch_dur_samp + 1, epoch_step_samp)
    epoch_ends = epoch_starts + epoch_dur_samp

    # Convertir bordes en muestras a bordes en el eje T-F.
    # np.round (no truncar): con dt≈q/native el producto t_axis*fs cae un epsilon
    # por debajo de los múltiplos enteros y astype(int) truncaría -1 muestra en
    # casi cada frontera de epoch.
    t_samples = np.round(t_axis * fs).astype(np.int64)
    n_epochs = len(epoch_starts)

    # Pre-crear out
    out = np.empty((n_epochs, n_ch, *cfg.target_shape), dtype=np.float32)
    mod_high = cfg.mod_range_hz[1]

    for k, (s, e) in enumerate(zip(epoch_starts, epoch_ends)):
        i0 = np.searchsorted(t_samples, s, side="left")
        i1 = np.searchsorted(t_samples, e, side="right")
        if i1 - i0 < 4:
            i1 = min(len(t_axis), i0 + 4)
        P_ep = P_full[:, :, i0:i1]                               # (n_ch, F, T_ep)
        t_ep = t_axis[i0:i1]
        # FFT temporal por (ch, F)
        n_t = P_ep.shape[2]
        dt = float(np.mean(np.diff(t_ep))) if n_t > 1 else 1.0 / fs
        M = np.abs(np.fft.rfft(P_ep, axis=2))
        f_mod = np.fft.rfftfreq(n_t, d=dt)
        keep = f_mod <= mod_high
        M = M[:, :, keep]
        # Resize cada (ch) a target_shape
        zy = cfg.target_shape[0] / M.shape[1]
        zx = cfg.target_shape[1] / M.shape[2]
        if (zy, zx) != (1.0, 1.0):
            from scipy.ndimage import zoom as _zoom
            M = _zoom(M, (1.0, zy, zx), order=1)
        if cfg.log_power:
            M = 10.0 * np.log10(M + cfg.eps)
        out[k] = M.astype(np.float32)

    return out

