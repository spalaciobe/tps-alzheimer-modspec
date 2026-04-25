"""Preproceso EEG: filtro pasabanda, ICA + ICLabel (principal), wICA (ablation), resample."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pywt
from mne.preprocessing import ICA


@dataclass
class PreprocessConfig:
    band_low_hz: float = 0.5
    band_high_hz: float = 45.0
    notch_hz: float | None = 50.0
    resample_hz: int | None = 200
    ica_method: str = "infomax"
    ica_n_components: float = 0.99
    iclabel_threshold: float = 0.8
    iclabel_reject: tuple[str, ...] = (
        "eye blink",
        "muscle artifact",
        "heart beat",
        "line noise",
        "channel noise",
    )
    channels_keep: tuple[str, ...] | None = None
    average_reference: bool = True


def apply_bandpass(raw: mne.io.BaseRaw, low: float, high: float) -> mne.io.BaseRaw:
    return raw.filter(
        l_freq=low,
        h_freq=high,
        method="fir",
        fir_design="firwin",
        phase="zero-double",
        verbose="ERROR",
    )


def apply_notch(raw: mne.io.BaseRaw, freq: float) -> mne.io.BaseRaw:
    return raw.notch_filter(freqs=[freq], method="fir", verbose="ERROR")


def select_and_reorder_channels(
    raw: mne.io.BaseRaw, channels: tuple[str, ...]
) -> mne.io.BaseRaw:
    """Mantiene solo los canales pedidos, en el orden pedido."""
    available = [ch for ch in channels if ch in raw.ch_names]
    raw = raw.pick(available)
    raw = raw.reorder_channels(available)
    return raw


def fit_ica(
    raw: mne.io.BaseRaw,
    n_components: float = 0.99,
    method: str = "infomax",
    seed: int = 0,
) -> ICA:
    fit_params = {"extended": True} if method == "infomax" else None
    ica = ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        random_state=seed,
        max_iter="auto",
    )
    ica.fit(raw, verbose="ERROR")
    return ica


def reject_components_iclabel(
    raw: mne.io.BaseRaw,
    ica: ICA,
    threshold: float,
    reject_classes: tuple[str, ...],
) -> ICA:
    """Marca componentes a excluir según ICLabel."""
    from mne_icalabel import label_components

    labels = label_components(raw, ica, method="iclabel")
    pred_labels = labels["labels"]
    pred_probs = labels["y_pred_proba"]

    exclude = []
    for i, (lbl, prob) in enumerate(zip(pred_labels, pred_probs)):
        if lbl in reject_classes and prob >= threshold:
            exclude.append(i)
    ica.exclude = exclude
    return ica


def apply_ica_iclabel(
    raw: mne.io.BaseRaw, cfg: PreprocessConfig, seed: int = 0
) -> mne.io.BaseRaw:
    ica = fit_ica(raw, cfg.ica_n_components, cfg.ica_method, seed)
    ica = reject_components_iclabel(
        raw, ica, cfg.iclabel_threshold, cfg.iclabel_reject
    )
    raw = ica.apply(raw, verbose="ERROR")
    return raw


def apply_wica(
    raw: mne.io.BaseRaw,
    cfg: PreprocessConfig,
    seed: int = 0,
    wavelet: str = "coif5",
    level: int = 5,
    k_sigma: float = 1.5,
) -> mne.io.BaseRaw:
    """wICA — ablation. Identifica componentes artefacto con ICLabel y umbraliza
    los coeficientes wavelet de detalle, en lugar de descartar la componente.
    """
    from mne_icalabel import label_components

    ica = fit_ica(raw, cfg.ica_n_components, cfg.ica_method, seed)
    labels = label_components(raw, ica, method="iclabel")
    pred_labels = labels["labels"]
    pred_probs = labels["y_pred_proba"]

    sources = ica.get_sources(raw).get_data()  # (n_components, n_samples)
    cleaned = sources.copy()

    for i, (lbl, prob) in enumerate(zip(pred_labels, pred_probs)):
        if lbl in cfg.iclabel_reject and prob >= cfg.iclabel_threshold:
            comp = sources[i]
            coeffs = pywt.wavedec(comp, wavelet, level=level)
            cD = coeffs[1:]
            new_cD = []
            for c in cD:
                sigma = np.median(np.abs(c)) / 0.6745
                t = k_sigma * sigma * np.sqrt(2 * np.log(len(c)))
                new_cD.append(pywt.threshold(c, t, mode="soft"))
            new_coeffs = [coeffs[0]] + new_cD
            rec = pywt.waverec(new_coeffs, wavelet)[: len(comp)]
            cleaned[i] = comp - rec

    mixing = ica.mixing_matrix_ @ ica.pca_components_[: ica.n_components_]
    new_data = mixing.T @ cleaned
    raw_new = raw.copy()
    raw_new._data[:] = new_data[: raw_new._data.shape[0]]
    return raw_new


def preprocess_raw(
    raw: mne.io.BaseRaw,
    cfg: PreprocessConfig,
    seed: int = 0,
    artifact_method: str = "iclabel",
) -> mne.io.BaseRaw:
    """Pipeline principal de preproceso, idéntico a Lopes et al. salvo wICA."""
    if cfg.channels_keep is not None:
        raw = select_and_reorder_channels(raw, cfg.channels_keep)
    if cfg.average_reference:
        raw = raw.set_eeg_reference("average", verbose="ERROR")
    raw = apply_bandpass(raw, cfg.band_low_hz, cfg.band_high_hz)
    if cfg.notch_hz:
        raw = apply_notch(raw, cfg.notch_hz)

    if artifact_method == "iclabel":
        raw = apply_ica_iclabel(raw, cfg, seed)
    elif artifact_method == "wica":
        raw = apply_wica(raw, cfg, seed)
    else:
        raise ValueError(f"artifact_method desconocido: {artifact_method}")

    if cfg.resample_hz is not None:
        raw = raw.resample(cfg.resample_hz, npad="auto", verbose="ERROR")

    return raw
