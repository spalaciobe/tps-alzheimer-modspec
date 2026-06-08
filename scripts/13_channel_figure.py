"""Figura: importancia por canal del SVM (mejor modelo).

Barplot horizontal con scores ANOVA F agregados por canal, ordenados de
mayor a menor. Color por región anatómica.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("D:/Universidad/Maestria/TPS/Proyecto")
DATA = BASE / "results/post_experiments/channel_importance.json"
OUT = BASE / "results/figures_multiseed/channel_importance.png"


# Categorización anatómica para colorear
REGION = {
    "Fp1": "frontal", "Fp2": "frontal", "F7": "frontal", "F3": "frontal",
    "Fz": "frontal", "F4": "frontal", "F8": "frontal",
    "T3": "temporal", "T4": "temporal", "T5": "temporal", "T6": "temporal",
    "C3": "central", "Cz": "central", "C4": "central",
    "P3": "parietal", "Pz": "parietal", "P4": "parietal",
    "O1": "occipital", "O2": "occipital",
}
REGION_COLOR = {
    "frontal": "#3b82f6",
    "central": "#10b981",
    "temporal": "#f59e0b",
    "parietal": "#a855f7",
    "occipital": "#ef4444",
}


def main():
    data = json.loads(DATA.read_text())
    agg = data["aggregate"]
    ranking = agg["ranking"]
    channels = [c for c, m, s in ranking]
    means = [m for c, m, s in ranking]
    stds = [s for c, m, s in ranking]
    colors = [REGION_COLOR[REGION[c]] for c in channels]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(channels))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.8, capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(channels)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia ANOVA F (normalizada, media de 3 seeds)")
    ax.set_title("Importancia por canal — SVM vainilla STFT\n(mejor configuración, AUC = 0.856)")

    # Leyenda por región
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, alpha=0.85, label=r.capitalize())
              for r, c in REGION_COLOR.items()]
    ax.legend(handles=legend, loc="lower right", title="Región")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
