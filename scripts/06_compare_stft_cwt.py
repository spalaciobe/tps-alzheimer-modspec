"""Comparación estadística STFT vs CWT con Wilcoxon pareado por sujeto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.stats import bootstrap_ci, delong_test, wilcoxon_paired
from src.utils.logging import get_logger


def load_fold_scores(results_dir: Path, metric: str = "score") -> dict[str, float]:
    data = json.loads((results_dir / "fold_results.json").read_text())
    return {r["test_subject"]: float(r[metric]) for r in data}


def load_fold_truth(results_dir: Path) -> dict[str, int]:
    data = json.loads((results_dir / "fold_results.json").read_text())
    return {r["test_subject"]: int(r["true"]) for r in data}


def per_subject_correct(results_dir: Path) -> dict[str, int]:
    data = json.loads((results_dir / "fold_results.json").read_text())
    return {r["test_subject"]: int(r["pred"] == r["true"]) for r in data}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=int, choices=[200, 500], default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classifier", choices=["cnn", "svm"], default="cnn")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--per-fold", action="store_true",
                        help="Buscar resultados SVM en directorios _perfold")
    parser.add_argument("--saliency-method", choices=["gradcam", "vanilla"], default="gradcam")
    parser.add_argument("--method-a", choices=["stft", "cwt", "cwt_fair"], default="stft",
                        help="Primer método a comparar (etiqueta 'stft' en el JSON)")
    parser.add_argument("--method-b", choices=["stft", "cwt", "cwt_fair"], default="cwt",
                        help="Segundo método (p.ej. cwt_fair para la comparación justa)")
    args = parser.parse_args()
    suffix = "_quick" if args.quick else ""
    pf = "_perfold" if args.per_fold else ""
    sal_tag = f"_{args.saliency_method}" if args.saliency_method != "gradcam" else ""
    logger = get_logger("compare")

    cfg = yaml.safe_load(open(args.config))
    results_root = Path(cfg["paths"]["results"])

    ma, mb = args.method_a, args.method_b
    if args.classifier == "cnn":
        stft_dir = results_root / f"{ma}_{args.fs}_seed{args.seed}{suffix}"
        cwt_dir = results_root / f"{mb}_{args.fs}_seed{args.seed}{suffix}"
    else:
        stft_dir = results_root / f"svm_{ma}_{args.fs}_seed{args.seed}{suffix}{sal_tag}{pf}"
        cwt_dir = results_root / f"svm_{mb}_{args.fs}_seed{args.seed}{suffix}{sal_tag}{pf}"

    stft_scores = load_fold_scores(stft_dir)
    cwt_scores = load_fold_scores(cwt_dir)
    truth = load_fold_truth(stft_dir)
    common = sorted(set(stft_scores) & set(cwt_scores))

    a = np.array([stft_scores[s] for s in common])
    b = np.array([cwt_scores[s] for s in common])

    # Wilcoxon sobre prob(AD) por sujeto
    w_score = wilcoxon_paired(a, b)

    # Wilcoxon sobre acierto binario por sujeto (0/1)
    a_correct = np.array([int((stft_scores[s] >= 0.5) == truth[s]) for s in common])
    b_correct = np.array([int((cwt_scores[s] >= 0.5) == truth[s]) for s in common])
    w_correct = wilcoxon_paired(a_correct.astype(float), b_correct.astype(float))

    # Bootstrap CI de accuracy por método
    acc_stft, lo_stft, hi_stft = bootstrap_ci(a_correct.astype(float), n_boot=1000, seed=args.seed)
    acc_cwt, lo_cwt, hi_cwt = bootstrap_ci(b_correct.astype(float), n_boot=1000, seed=args.seed)

    # DeLong test sobre AUC pareado
    y_true_arr = np.array([truth[s] for s in common])
    delong = delong_test(y_true_arr, a, b)

    summary = {
        "n_subjects": len(common),
        "classifier": args.classifier,
        "fs": args.fs,
        "method_a": ma,
        "method_b": mb,
        ma: {"accuracy": acc_stft, "ci95": [lo_stft, hi_stft]},
        mb: {"accuracy": acc_cwt, "ci95": [lo_cwt, hi_cwt]},
        "wilcoxon_score": w_score,
        "wilcoxon_correct": w_correct,
        "delong_auc": delong,
    }
    # Sufijo de par de métodos (vacío para el par por defecto stft-vs-cwt → compat).
    pair = "" if (ma, mb) == ("stft", "cwt") else f"_{ma}_vs_{mb}"
    out = Path(cfg["paths"]["results"]) / f"compare_{args.classifier}_{args.fs}_seed{args.seed}{suffix}{sal_tag}{pf}{pair}.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
