"""Helpers de visualización para modulation spectrums y mapas de saliencia."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_modspec(
    M: np.ndarray,
    carrier_range: tuple[float, float] = (0.5, 45.0),
    mod_range: tuple[float, float] = (0.0, 22.5),
    title: str | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
) -> plt.Axes:
    """Plot un modulation spectrum 2D.

    Args:
        M: matriz (F_carrier, F_mod). Puede ser una imagen ya en log.
        carrier_range: (low, high) Hz para el eje y.
        mod_range: (low, high) Hz para el eje x.
        title: título opcional.
        ax: eje matplotlib opcional.
        cmap: colormap.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    extent = [mod_range[0], mod_range[1], carrier_range[0], carrier_range[1]]
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, extent=extent)
    ax.set_xlabel("Modulation frequency (Hz)")
    ax.set_ylabel("Carrier frequency (Hz)")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_saliency_overlay(
    M: np.ndarray,
    saliency: np.ndarray,
    carrier_range: tuple[float, float] = (0.5, 45.0),
    mod_range: tuple[float, float] = (0.0, 22.5),
    title: str | None = None,
    alpha: float = 0.5,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Superpone un mapa de saliencia sobre un modulation spectrum."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    extent = [mod_range[0], mod_range[1], carrier_range[0], carrier_range[1]]
    ax.imshow(M, origin="lower", aspect="auto", cmap="gray", extent=extent)
    ax.imshow(
        saliency,
        origin="lower",
        aspect="auto",
        cmap="hot",
        alpha=alpha,
        extent=extent,
    )
    ax.set_xlabel("Modulation frequency (Hz)")
    ax.set_ylabel("Carrier frequency (Hz)")
    if title:
        ax.set_title(title)
    return ax


def annotate_canonical_bands(ax: plt.Axes) -> None:
    """Dibuja líneas horizontales en delta/theta/alpha/beta/gamma."""
    bands = {"δ": 4, "θ": 8, "α": 13, "β": 30}
    for name, freq in bands.items():
        ax.axhline(freq, color="white", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(ax.get_xlim()[1] * 0.98, freq + 0.3, name, color="white",
                fontsize=8, ha="right")
