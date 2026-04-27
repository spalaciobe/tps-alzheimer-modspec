"""Genera figuras finales para el informe: modspec medio por clase,
saliency maps STFT vs CWT, ROC, matrices de confusión."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, roc_curve

from src.cache import load_subject
from src.utils.viz import plot_modspec, plot_saliency_overlay, annotate_canonical_bands


def fig_modspec_class_means(modspec_dir: Path, out: Path, title_prefix: str) -> None:
    """Promedio de modspec sobre todos los epochs de cada clase."""
    AD_acc, HC_acc = [], []
    for p in sorted(modspec_dir.glob("*.h5")):
        d = load_subject(p)
        m = d["X"].mean(axis=0).mean(axis=0)  # (F, M)
        (AD_acc if d["y"] == 1 else HC_acc).append(m)
    AD_mean = np.mean(AD_acc, axis=0)
    HC_mean = np.mean(HC_acc, axis=0)
    diff = AD_mean - HC_mean

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_modspec(HC_mean, title=f"{title_prefix} — HC mean", ax=axes[0])
    annotate_canonical_bands(axes[0])
    plot_modspec(AD_mean, title=f"{title_prefix} — AD mean", ax=axes[1])
    annotate_canonical_bands(axes[1])
    plot_modspec(diff, title=f"{title_prefix} — AD − HC", ax=axes[2], cmap="RdBu_r")
    annotate_canonical_bands(axes[2])
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def fig_saliency_compare(sal_dir_stft: Path, sal_dir_cwt: Path, out: Path) -> None:
    """Mapa de saliency diferencial AD-HC para STFT vs CWT."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for col, (name, sal_dir) in enumerate([("STFT", sal_dir_stft), ("CWT", sal_dir_cwt)]):
        sad = np.load(sal_dir / "saliency_AD.npy")
        shc = np.load(sal_dir / "saliency_HC.npy")
        plot_modspec(sad, title=f"{name} Grad-CAM AD", ax=axes[0, col], cmap="hot")
        annotate_canonical_bands(axes[0, col])
        plot_modspec(sad - shc, title=f"{name} diff AD-HC",
                     ax=axes[1, col], cmap="RdBu_r")
        annotate_canonical_bands(axes[1, col])
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def fig_metrics_comparison(comp_path: Path, out: Path) -> None:
    """Bar chart con accuracy y CI95 para STFT vs CWT."""
    data = json.loads(comp_path.read_text())
    methods = ["stft", "cwt"]
    accs = [data[m]["accuracy"] for m in methods]
    cis = [data[m]["ci95"] for m in methods]
    err_lo = [acc - ci[0] for acc, ci in zip(accs, cis)]
    err_hi = [ci[1] - acc for acc, ci in zip(accs, cis)]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["STFT", "CWT"], accs,
           yerr=[err_lo, err_hi], capsize=10,
           color=["#3498db", "#e67e22"], alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="random")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"AD vs HC — {data['classifier'].upper()} — n={data['n_subjects']}\n"
                 f"Wilcoxon p={data['wilcoxon_correct']['pvalue']:.3f}")
    ax.legend()
    for i, (acc, lo, hi) in enumerate(zip(accs, err_lo, err_hi)):
        ax.text(i, acc + hi + 0.02, f"{acc:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def fig_roc_curves(svm_stft: Path, svm_cwt: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p, color in [("STFT", svm_stft, "#3498db"), ("CWT", svm_cwt, "#e67e22")]:
        d = json.loads((p / "fold_results.json").read_text())
        y_true = np.array([r["true"] for r in d])
        y_score = np.array([r["score"] for r in d])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        from sklearn.metrics import auc as _auc
        a = _auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — SVM con patches saliency-guided")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def fig_confusion_matrices(svm_stft: Path, svm_cwt: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, p) in zip(axes, [("STFT", svm_stft), ("CWT", svm_cwt)]):
        d = json.loads((p / "fold_results.json").read_text())
        y_true = np.array([r["true"] for r in d])
        y_pred = np.array([r["pred"] for r in d])
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["HC", "AD"])
        ax.set_yticks([0, 1], ["HC", "AD"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        ax.set_title(f"{name} — SVM")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--per-fold", action="store_true",
                        help="usar SVM perfold dirs y compare_*_perfold.json")
    parser.add_argument("--saliency-method", choices=["gradcam", "vanilla"], default="gradcam")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    suffix = "_quick" if args.quick else ""
    pf = "_perfold" if args.per_fold else ""
    sal_tag = f"_{args.saliency_method}" if args.saliency_method != "gradcam" else ""

    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]
    res = Path(paths["results"])
    sal = Path(paths["saliency"])
    figs = res / f"figures{suffix}{sal_tag}"
    figs.mkdir(parents=True, exist_ok=True)

    fig_modspec_class_means(
        Path(paths["modspec_root"]) / f"modspec_stft_{args.fs}",
        figs / "modspec_means_stft.png", "STFT")
    fig_modspec_class_means(
        Path(paths["modspec_root"]) / f"modspec_cwt_{args.fs}",
        figs / "modspec_means_cwt.png", "CWT")

    fig_saliency_compare(
        sal / f"stft_{args.fs}_seed{args.seed}{suffix}{sal_tag}",
        sal / f"cwt_{args.fs}_seed{args.seed}{suffix}{sal_tag}",
        figs / "saliency_compare.png")

    for clf in ("svm", "cnn"):
        clf_pf = pf if clf == "svm" else ""
        clf_sal = sal_tag if clf == "svm" else ""
        comp = res / f"compare_{clf}_{args.fs}_seed{args.seed}{suffix}{clf_sal}{clf_pf}.json"
        if comp.exists():
            fig_metrics_comparison(comp, figs / f"compare_{clf}.png")

    svm_stft = res / f"svm_stft_{args.fs}_seed{args.seed}{suffix}{sal_tag}{pf}"
    svm_cwt = res / f"svm_cwt_{args.fs}_seed{args.seed}{suffix}{sal_tag}{pf}"
    if svm_stft.exists() and svm_cwt.exists():
        fig_roc_curves(svm_stft, svm_cwt, figs / "roc_svm.png")
        fig_confusion_matrices(svm_stft, svm_cwt, figs / "confusion_svm.png")

    print(f"Figuras generadas en {figs}")


if __name__ == "__main__":
    main()
