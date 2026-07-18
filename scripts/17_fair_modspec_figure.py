"""Figura del control DSP en imagenes: espectro de modulacion medio
(STFT | CWT nativa | CWT-fair). Muestra como, al igualar el eje de
modulacion, el 'cono' de la wavelet desaparece y la estructura de
modulacion de la CWT-fair se acerca a la de la STFT.

NOTA honesta: CWT-fair iguala el eje de MODULACION (horizontal), no el de
PORTADORA (vertical). Por eso converge la estructura de modulacion, pero no
necesariamente el patron diagnostico ni el saliency.

Salida: results/figures_multiseed/fair_modspec_cone.png
        (y copia a docs/figures_informe_final/ para el .tex)
"""
from __future__ import annotations
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cache import load_subject
from src.utils.viz import plot_modspec, annotate_canonical_bands

ROOT = Path("D:/Universidad/Maestria/TPS/Proyecto")
DER = ROOT / "data/derivatives"
OUT = ROOT / "results/figures_multiseed"
DOCS = ROOT / "docs/figures_informe_final"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("STFT", "modspec_stft_200"),
    ("CWT nativa", "modspec_cwt_200"),
    ("CWT-fair", "modspec_cwt_fair_200"),
]


def modspec_mean(dirname: str) -> np.ndarray:
    """Media del espectro de modulacion sobre todos los sujetos/epochs/canales."""
    acc = []
    for p in sorted((DER / dirname).glob("*.h5")):
        s = load_subject(p)
        acc.append(s["X"].mean(axis=0).mean(axis=0))  # (F, M)
    return np.mean(acc, axis=0)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (name, dirn) in zip(axes, METHODS):
        m = modspec_mean(dirn)
        plot_modspec(m, title=f"{name}", ax=ax)
        annotate_canonical_bands(ax)
    fig.suptitle(
        "Espectro de modulacion medio: al igualar el eje, el 'cono' de la CWT "
        "desaparece y la CWT-fair se acerca a la STFT",
        fontsize=12,
    )
    fig.tight_layout()
    dst = OUT / "fair_modspec_cone.png"
    fig.savefig(dst, dpi=130)
    plt.close(fig)
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy(dst, DOCS / "fair_modspec_cone.png")
    print("Guardado:", dst, "y copia en", DOCS / "fair_modspec_cone.png")


if __name__ == "__main__":
    main()
