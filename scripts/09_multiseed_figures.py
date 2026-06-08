"""Figuras finales multi-seed con error bars y boxplots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("D:/Universidad/Maestria/TPS/Proyecto/results")
OUT = BASE / "figures_multiseed"
OUT.mkdir(exist_ok=True)


def bar_with_errors(data: dict, title: str, ylabel: str, out_path: Path):
    """Bar plot con error bars (mean ± std) para múltiples métricas."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = list(data.keys())
    means = [data[m]["mean"] for m in methods]
    stds = [data[m]["std"] for m in methods]
    colors = ["#3b82f6" if "STFT" in m else "#ef4444" for m in methods]
    bars = ax.bar(methods, means, yerr=stds, capsize=8, color=colors, alpha=0.75,
                  edgecolor="black", linewidth=1.2)
    # Anotar valores
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f"{mean:.3f}±{std:.3f}", ha="center", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(means) + max(stds) + 0.1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def boxplot_seeds(data: dict, title: str, ylabel: str, out_path: Path):
    """Boxplot mostrando distribución por seed."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(data.keys())
    values = [data[m]["values"] for m in labels]
    bp = ax.boxplot(values, labels=labels, patch_artist=True, widths=0.6)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor("#3b82f6" if "STFT" in label else "#ef4444")
        patch.set_alpha(0.6)
    for i, vals in enumerate(values):
        ax.scatter([i+1]*len(vals), vals, color="black", zorder=3, s=30)
        for j, v in enumerate(vals):
            ax.text(i+1.1, v, f"s{j}", fontsize=7, va="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    data = json.loads((BASE / "multiseed_analysis.json").read_text())

    # CNN end-to-end
    cnn_acc = {"STFT": data["cnn"]["stft"]["agg"]["accuracy"],
               "CWT":  data["cnn"]["cwt"]["agg"]["accuracy"]}
    cnn_auc = {"STFT": data["cnn"]["stft"]["agg"]["auc"],
               "CWT":  data["cnn"]["cwt"]["agg"]["auc"]}
    bar_with_errors(cnn_acc, "CNN end-to-end — Accuracy (multi-seed, n=3)", "Accuracy", OUT / "cnn_acc_multiseed.png")
    bar_with_errors(cnn_auc, "CNN end-to-end — AUC (multi-seed, n=3)", "AUC", OUT / "cnn_auc_multiseed.png")
    boxplot_seeds(cnn_acc, "CNN end-to-end — Accuracy por seed", "Accuracy", OUT / "cnn_acc_boxplot.png")
    boxplot_seeds(cnn_auc, "CNN end-to-end — AUC por seed", "AUC", OUT / "cnn_auc_boxplot.png")

    # SVM por saliency method
    for sal in ("gradcam", "vanilla"):
        svm_acc = {"STFT": data["svm"][sal]["stft"]["agg"]["accuracy"],
                   "CWT":  data["svm"][sal]["cwt"]["agg"]["accuracy"]}
        svm_auc = {"STFT": data["svm"][sal]["stft"]["agg"]["auc"],
                   "CWT":  data["svm"][sal]["cwt"]["agg"]["auc"]}
        sal_label = "vainilla (paper)" if sal == "vanilla" else "Grad-CAM"
        bar_with_errors(svm_acc, f"SVM con saliency {sal_label} — Accuracy (multi-seed)",
                        "Accuracy", OUT / f"svm_{sal}_acc_multiseed.png")
        bar_with_errors(svm_auc, f"SVM con saliency {sal_label} — AUC (multi-seed)",
                        "AUC", OUT / f"svm_{sal}_auc_multiseed.png")
        boxplot_seeds(svm_acc, f"SVM {sal_label} — Accuracy por seed",
                      "Accuracy", OUT / f"svm_{sal}_acc_boxplot.png")
        boxplot_seeds(svm_auc, f"SVM {sal_label} — AUC por seed",
                      "AUC", OUT / f"svm_{sal}_auc_boxplot.png")

    # Figura resumen master: AUC across all configurations
    fig, ax = plt.subplots(figsize=(10, 5))
    configs = []
    means = []
    stds = []
    colors = []
    configs += ["CNN STFT", "CNN CWT"]
    means += [data["cnn"]["stft"]["agg"]["auc"]["mean"], data["cnn"]["cwt"]["agg"]["auc"]["mean"]]
    stds += [data["cnn"]["stft"]["agg"]["auc"]["std"], data["cnn"]["cwt"]["agg"]["auc"]["std"]]
    colors += ["#3b82f6", "#ef4444"]
    for sal in ("gradcam", "vanilla"):
        sal_label = "vai." if sal == "vanilla" else "G-CAM"
        configs += [f"SVM-{sal_label} STFT", f"SVM-{sal_label} CWT"]
        means += [data["svm"][sal]["stft"]["agg"]["auc"]["mean"],
                  data["svm"][sal]["cwt"]["agg"]["auc"]["mean"]]
        stds += [data["svm"][sal]["stft"]["agg"]["auc"]["std"],
                 data["svm"][sal]["cwt"]["agg"]["auc"]["std"]]
        colors += ["#3b82f6", "#ef4444"]
    x = np.arange(len(configs))
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.75,
           edgecolor="black", linewidth=1)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.015, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("AUC (LOSO 65 sujetos, multi-seed)")
    ax.set_title("AUC across all configurations — mean ± SD over 3 seeds")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Azar")
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.3)
    # Leyenda STFT vs CWT
    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#3b82f6", alpha=0.75, label="STFT"),
              Patch(facecolor="#ef4444", alpha=0.75, label="CWT")]
    ax.legend(handles=legend, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "auc_master_summary.png", dpi=120)
    plt.close()

    print(f"Figuras multi-seed en {OUT}", flush=True)
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
