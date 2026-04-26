"""SVM RBF sobre features (potencia de patches + ratios) en LOSO.

Memory-efficient: streaming por sujeto, subsample opcional de epochs/sujeto.
"""

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
from src.utils.seed import set_seed


def features_for_h5(p: Path, patches: list[Patch], n_subsample: int | None,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    data = load_subject(p)
    X = data["X"].astype(np.float32)
    y = int(data["y"])
    if n_subsample is not None and X.shape[0] > n_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], n_subsample, replace=False)
        X = X[idx]
    F = features_for_dataset(X, patches)
    return F, np.full(F.shape[0], y, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["stft", "cwt"], required=True)
    parser.add_argument("--fs", type=int, choices=[200, 500], default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--svm-config", type=Path, default=Path("configs/svm.yaml"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs-per-subject", type=int, default=80,
                        help="Subsample por sujeto para acelerar SVM RBF (O(n²)).")
    args = parser.parse_args()
    set_seed(args.seed)
    logger = get_logger("svm")

    cfg = yaml.safe_load(open(args.config))
    svm_cfg = yaml.safe_load(open(args.svm_config))
    paths = cfg["paths"]
    suffix = "_quick" if args.quick else ""

    modspec_dir = Path(paths["modspec_root"]) / f"modspec_{args.method}_{args.fs}"
    sal_dir = Path(paths["saliency"]) / f"{args.method}_{args.fs}_seed{args.seed}{suffix}"

    masks = np.load(sal_dir / "patch_masks.npy")
    patches = [Patch(cluster_id=i, mask=masks[i].astype(bool)) for i in range(masks.shape[0])]
    if not patches:
        raise SystemExit("No hay patches; correr 04_extract_saliency_features.py primero")
    logger.info(f"{len(patches)} patches cargados")

    h5_paths = sorted(modspec_dir.glob("*.h5"))
    logger.info(f"Calculando features de {len(h5_paths)} sujetos (subsample={args.epochs_per_subject})...")

    # Pre-computar features para todos los sujetos UNA vez
    feats_per_sub: dict[str, tuple[np.ndarray, int]] = {}
    for p in tqdm(h5_paths, desc="features"):
        F, y = features_for_h5(p, patches, args.epochs_per_subject, args.seed)
        feats_per_sub[p.stem] = (F, int(y[0]))

    fold_results = []
    for test_sid in tqdm(feats_per_sub.keys(), desc="LOSO-SVM"):
        train_sids = [s for s in feats_per_sub if s != test_sid]
        Xtr = np.concatenate([feats_per_sub[s][0] for s in train_sids], axis=0)
        ytr = np.concatenate([
            np.full(feats_per_sub[s][0].shape[0], feats_per_sub[s][1])
            for s in train_sids
        ])
        Xte = feats_per_sub[test_sid][0]
        yte_label = feats_per_sub[test_sid][1]

        sel = select_top_k_features(Xtr, ytr, k=svm_cfg["features"]["top_k"])
        Xtr_s = sel.transform(Xtr)
        Xte_s = sel.transform(Xte)

        scaler, clf = fit_svm_pipeline(Xtr_s, ytr, C=svm_cfg["svm"]["C"])
        probs = clf.predict_proba(scaler.transform(Xte_s))
        epoch_pred = probs.argmax(1)
        epoch_metrics = compute_metrics(
            np.full(len(epoch_pred), yte_label), epoch_pred, y_score=probs[:, 1]
        )

        subj_score = float(probs[:, 1].mean())
        subj_pred = int(subj_score >= 0.5)
        fold_results.append({
            "test_subject": test_sid,
            "true": yte_label,
            "pred": subj_pred,
            "score": subj_score,
            "epoch_metrics": epoch_metrics,
        })

    out_dir = Path(paths["results"]) / f"svm_{args.method}_{args.fs}_seed{args.seed}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fold_results.json").write_text(json.dumps(fold_results, indent=2))

    y_true = np.array([r["true"] for r in fold_results])
    y_pred = np.array([r["pred"] for r in fold_results])
    y_score = np.array([r["score"] for r in fold_results])
    overall = compute_metrics(y_true, y_pred, y_score=y_score)
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2))
    logger.info(f"SVM LOSO {args.method.upper()}{suffix}: {overall}")


if __name__ == "__main__":
    main()
