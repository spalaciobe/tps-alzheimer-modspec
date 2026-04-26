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
    parser.add_argument("--per-fold-patches", action="store_true",
                        help="Usar patch_masks_fold_NN.npy (anti-leakage). "
                             "Si no, usa patch_masks.npy global (compat).")
    parser.add_argument("--saliency-method", choices=["gradcam", "vanilla"], default="gradcam")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    set_seed(args.seed)
    logger = get_logger("svm")

    cfg = yaml.safe_load(open(args.config))
    svm_cfg = yaml.safe_load(open(args.svm_config))
    paths = cfg["paths"]
    suffix = "_quick" if args.quick else ""
    sal_tag = f"_{args.saliency_method}" if args.saliency_method != "gradcam" else ""
    rt = f"_{args.run_tag}" if args.run_tag else ""

    modspec_dir = Path(paths["modspec_root"]) / f"modspec_{args.method}_{args.fs}{rt}"
    sal_dir = Path(paths["saliency"]) / f"{args.method}_{args.fs}_seed{args.seed}{suffix}{sal_tag}{rt}"

    h5_paths = sorted(modspec_dir.glob("*.h5"))
    sid_to_idx = {p.stem: i for i, p in enumerate(h5_paths)}

    # Cargar patches (per-fold o global)
    per_fold_dir = sal_dir / "per_fold"
    use_per_fold = args.per_fold_patches and per_fold_dir.exists()
    if use_per_fold:
        logger.info(f"Usando patches POR FOLD desde {per_fold_dir}")
    else:
        masks = np.load(sal_dir / "patch_masks.npy")
        patches_global = [Patch(cluster_id=i, mask=masks[i].astype(bool))
                          for i in range(masks.shape[0])]
        if not patches_global:
            raise SystemExit("No hay patches globales; correr 04 primero")
        logger.info(f"Usando {len(patches_global)} patches globales")

    fold_results = []
    for test_sid in tqdm(sorted(sid_to_idx.keys()), desc="LOSO-SVM"):
        fold_idx = sid_to_idx[test_sid]
        if use_per_fold:
            mask_path = per_fold_dir / f"patch_masks_fold{fold_idx:02d}.npy"
            if not mask_path.exists():
                logger.warning(f"sin patches para fold {fold_idx}, skip")
                continue
            masks = np.load(mask_path)
            patches = [Patch(cluster_id=i, mask=masks[i].astype(bool))
                       for i in range(masks.shape[0])]
            if not patches:
                continue
        else:
            patches = patches_global

        # Features por fold (recalculadas con los patches de este fold)
        train_paths = [p for p in h5_paths if p.stem != test_sid]
        Xtr_list, ytr_list = [], []
        for p in train_paths:
            F, y = features_for_h5(p, patches, args.epochs_per_subject, args.seed)
            Xtr_list.append(F)
            ytr_list.append(y)
        Xtr = np.concatenate(Xtr_list, axis=0)
        ytr = np.concatenate(ytr_list)
        Xte, yte = features_for_h5(
            next(p for p in h5_paths if p.stem == test_sid),
            patches, args.epochs_per_subject, args.seed,
        )
        yte_label = int(yte[0])

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
            "fold": fold_idx,
            "test_subject": test_sid,
            "true": yte_label,
            "pred": subj_pred,
            "score": subj_score,
            "epoch_metrics": epoch_metrics,
            "n_patches": len(patches),
        })

    pf_tag = "_perfold" if use_per_fold else ""
    out_dir = Path(paths["results"]) / f"svm_{args.method}_{args.fs}_seed{args.seed}{suffix}{sal_tag}{pf_tag}{rt}"
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
