"""Entrena el SVM RBF sobre features extraídas con los patches del paso 04, en LOSO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.cache import load_subject
from src.evaluate import compute_metrics
from src.feature_extraction import (
    Patch,
    features_for_dataset,
    select_top_k_features,
)
from src.svm_pipeline import fit_svm_pipeline
from src.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["stft", "cwt"], required=True)
    parser.add_argument("--fs", type=int, choices=[200, 500], default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--svm-config", type=Path, default=Path("configs/svm.yaml"))
    args = parser.parse_args()
    logger = get_logger("svm")

    cfg = yaml.safe_load(open(args.config))
    svm_cfg = yaml.safe_load(open(args.svm_config))
    paths = cfg["paths"]

    modspec_dir = Path(paths["modspec_root"]) / f"modspec_{args.method}_{args.fs}"
    sal_dir = Path(paths["saliency"]) / f"{args.method}_{args.fs}_seed{args.seed}"

    masks = np.load(sal_dir / "patch_masks.npy")
    patches = [Patch(cluster_id=i, mask=masks[i].astype(bool)) for i in range(masks.shape[0])]
    if not patches:
        raise SystemExit("No hay patches; correr 04_extract_saliency_features.py primero")

    h5_paths = sorted(modspec_dir.glob("*.h5"))
    data = {p.stem: load_subject(p) for p in h5_paths}

    fold_results = []
    for test_sid in tqdm(data.keys(), desc="LOSO-SVM"):
        train_sids = [s for s in data if s != test_sid]

        # Features (potencia + ratios) para train y test
        Xtr = np.concatenate([features_for_dataset(data[s]["X"], patches) for s in train_sids], axis=0)
        ytr = np.concatenate([np.full(data[s]["X"].shape[0], data[s]["y"]) for s in train_sids])
        Xte = features_for_dataset(data[test_sid]["X"], patches)
        yte = np.full(Xte.shape[0], data[test_sid]["y"])

        # ANOVA top-24 sobre train
        sel = select_top_k_features(Xtr, ytr, k=svm_cfg["features"]["top_k"])
        Xtr_s = sel.transform(Xtr)
        Xte_s = sel.transform(Xte)

        scaler, clf = fit_svm_pipeline(Xtr_s, ytr, C=svm_cfg["svm"]["C"])
        probs = clf.predict_proba(scaler.transform(Xte_s))
        epoch_pred = probs.argmax(1)
        epoch_metrics = compute_metrics(yte, epoch_pred, y_score=probs[:, 1])

        # Agregación a sujeto
        subj_score = float(probs[:, 1].mean())
        subj_pred = int(subj_score >= 0.5)
        fold_results.append({
            "test_subject": test_sid,
            "true": int(data[test_sid]["y"]),
            "pred": subj_pred,
            "score": subj_score,
            "epoch_metrics": epoch_metrics,
        })

    out_dir = Path(paths["results"]) / f"svm_{args.method}_{args.fs}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fold_results.json"
    out_path.write_text(json.dumps(fold_results, indent=2))

    y_true = np.array([r["true"] for r in fold_results])
    y_pred = np.array([r["pred"] for r in fold_results])
    y_score = np.array([r["score"] for r in fold_results])
    overall = compute_metrics(y_true, y_pred, y_score=y_score)
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2))
    logger.info(f"SVM LOSO: {overall}")


if __name__ == "__main__":
    main()
