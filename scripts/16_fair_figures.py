"""Figuras del hallazgo central (comparación DSP-justa):
  A) Barras agrupadas AUC ± SD: 3 pipelines × {STFT, CWT nativa, CWT-fair}
     — muestra cómo CWT-fair se desplaza hacia STFT.
  B) Forest plot de ΔAUC (DeLong por seed + combinado) para las comparaciones
     confundida (STFT vs CWT nativa) y justa (STFT vs CWT-fair).

Salida: results/figures_multiseed/fair_auc_grouped.png y fair_forest_delong.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("D:/Universidad/Maestria/TPS/Proyecto/results")
OUT = BASE / "figures_multiseed"
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((BASE / "fair_cwt_analysis.json").read_text())

C_STFT, C_NAT, C_FAIR = "#3b6db3", "#c0552b", "#2e8b57"   # azul / naranja / verde


def auc(node, m):
    return D[node][m]["agg"]["auc"]["mean"], D[node][m]["agg"]["auc"]["std"]


# ---------- Figura A: barras agrupadas ----------
def fig_grouped():
    pipes = [("CNN", "cnn", None), ("SVM Grad-CAM", "svm", "gradcam"), ("SVM vainilla", "svm", "vanilla")]
    methods = [("STFT", C_STFT), ("CWT nativa", C_NAT), ("CWT-fair", C_FAIR)]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pipes))
    w = 0.26
    for j, (mlabel, color) in enumerate(methods):
        mkey = {"STFT": "stft", "CWT nativa": "cwt", "CWT-fair": "cwt_fair"}[mlabel]
        means, sds = [], []
        for _, node, sal in pipes:
            agg = D[node][mkey]["agg"] if node == "cnn" else D[node][sal][mkey]["agg"]
            means.append(agg["auc"]["mean"]); sds.append(agg["auc"]["std"])
        ax.bar(x + (j - 1) * w, means, w, yerr=sds, capsize=3, label=mlabel,
               color=color, alpha=0.88, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, ls=":", c="grey", lw=1, label="azar")
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pipes])
    ax.set_ylabel("AUC (media ± SD, 3 seeds)")
    ax.set_ylim(0.5, 0.92)
    ax.set_title("A igualdad de eje de modulación, CWT-fair se acerca a STFT\n"
                 "(la CWT nativa —eje sin igualar— es la que se desvía)")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # anotar el desplazamiento en vainilla
    fig.tight_layout()
    fig.savefig(OUT / "fair_auc_grouped.png", dpi=130)
    plt.close(fig)
    print("Guardado:", OUT / "fair_auc_grouped.png")


# ---------- Figura B: forest plot ΔAUC DeLong ----------
def fig_forest():
    # filas: (etiqueta, node, sal, comparación_key, color)
    rows = [
        ("SVM vainilla — STFT vs CWT nativa (confundida)", "svm", "vanilla", "stft_vs_cwt", C_NAT),
        ("SVM vainilla — STFT vs CWT-fair (justa)", "svm", "vanilla", "stft_vs_cwt_fair", C_FAIR),
        ("SVM Grad-CAM — STFT vs CWT nativa (confundida)", "svm", "gradcam", "stft_vs_cwt", C_NAT),
        ("SVM Grad-CAM — STFT vs CWT-fair (justa)", "svm", "gradcam", "stft_vs_cwt_fair", C_FAIR),
        ("CNN — STFT vs CWT-fair (justa)", "cnn", None, "stft_vs_cwt_fair", C_FAIR),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ylabels, yticks = [], []
    y = 0
    for label, node, sal, key, color in rows:
        node_obj = D[node] if node == "cnn" else D[node][sal]
        if key not in node_obj:
            # 'stft_vs_cwt' (nativa) puede no estar en el JSON fair; derivar de per-seed AUC
            # STFT vs CWT nativa: usar agg means como aproximación de Δ combinado
            m_st = (D[node]["stft"] if node == "cnn" else D[node][sal]["stft"])["agg"]["auc"]["mean"]
            m_cw = (D[node]["cwt"] if node == "cnn" else D[node][sal]["cwt"])["agg"]["auc"]["mean"]
            combined = m_st - m_cw
            per = None
        else:
            dl = node_obj[key]["delong"]
            per = [p["auc_diff"] for p in dl["per_seed"]]
            combined = float(np.mean(per))
        # dibujar per-seed (puntos pequeños) y combinado (rombo)
        if per is not None:
            ax.scatter(per, [y] * len(per), s=22, color=color, alpha=0.5, zorder=2)
            lo, hi = min(per), max(per)
            ax.plot([lo, hi], [y, y], color=color, lw=1, alpha=0.5, zorder=1)
        ax.scatter([combined], [y], marker="D", s=80, color=color, edgecolor="black",
                   linewidth=0.8, zorder=3)
        ylabels.append(label); yticks.append(y)
        y += 1
    ax.axvline(0, ls="--", c="black", lw=1)
    ax.axvspan(-0.02, 0.02, color="grey", alpha=0.12)  # zona ~nula visual
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ AUC (STFT − CWT)   —   rombo = media 3 seeds, puntos = por seed")
    ax.set_title("Al igualar el eje de modulación, Δ AUC se acerca a 0\n"
                 "(comparaciones justas STFT vs CWT-fair: todas NS, BH-FDR q≈0.68)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fair_forest_delong.png", dpi=130)
    plt.close(fig)
    print("Guardado:", OUT / "fair_forest_delong.png")


if __name__ == "__main__":
    fig_grouped()
    fig_forest()
