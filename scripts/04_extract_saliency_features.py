"""Extrae mapas Grad-CAM (o vanilla) agregados por clase y patches.

Estrategia memory-efficient:
- No carga el bank completo (5GB) a RAM.
- Para un fold dado, usa el modelo de ese fold y procesa los sujetos de train
  uno por uno (streaming por archivo HDF5).
- Acumula los mapas saliency en RAM (45×45 floats — despreciable).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cache import load_subject
from src.datasets import ModSpecSubjectDataset
from src.feature_extraction import find_patches
from src.models.cnn import ModSpecCNN
from src.normalize import ChannelStats, fit_channel_zscore
from src.saliency import gradcam_per_class, vanilla_saliency_per_class
from src.utils.logging import get_logger


def stats_from_subjects(h5_paths: list[Path]) -> ChannelStats:
    """Calcula media/std agregando estadísticos suficientes por canal."""
    n_total = 0
    sum_ = None
    sumsq = None
    for p in h5_paths:
        X = load_subject(p)["X"].astype(np.float32)
        if sum_ is None:
            sum_ = X.sum(axis=(0, 2, 3))
            sumsq = (X * X).sum(axis=(0, 2, 3))
        else:
            sum_ += X.sum(axis=(0, 2, 3))
            sumsq += (X * X).sum(axis=(0, 2, 3))
        n_total += X.shape[0] * X.shape[2] * X.shape[3]
    mean = sum_ / n_total
    var = sumsq / n_total - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    return ChannelStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["stft", "cwt"], required=True)
    parser.add_argument("--fs", type=int, choices=[200, 500], default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--cnn-config", type=Path, default=Path("configs/cnn.yaml"))
    parser.add_argument("--svm-config", type=Path, default=Path("configs/svm.yaml"))
    parser.add_argument("--saliency-method", choices=["gradcam", "vanilla"], default="gradcam")
    parser.add_argument("--folds", type=int, default=10,
                        help="Folds usados para promediar (10 ≈ buena estimación, 65 = todos)")
    parser.add_argument("--max-subjects-per-fold", type=int, default=20,
                        help="Sujetos de train usados por fold (subset para acelerar)")
    args = parser.parse_args()
    logger = get_logger("saliency")

    cfg = yaml.safe_load(open(args.config))
    cnn_cfg = yaml.safe_load(open(args.cnn_config))
    svm_cfg = yaml.safe_load(open(args.svm_config))
    paths = cfg["paths"]
    suffix = "_quick" if args.quick else ""
    run_id = f"{args.method}_{args.fs}_seed{args.seed}{suffix}"

    modspec_dir = Path(paths["modspec_root"]) / f"modspec_{args.method}_{args.fs}"
    results_dir = Path(paths["results"]) / run_id
    sal_dir = Path(paths["saliency"]) / run_id
    sal_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(modspec_dir.glob("*.h5"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, {len(h5_paths)} sujetos disponibles")

    n_folds = min(args.folds, len(h5_paths))
    accum = {0: None, 1: None}
    counts = {0: 0, 1: 0}

    sal_fn = gradcam_per_class if args.saliency_method == "gradcam" else vanilla_saliency_per_class
    rng = np.random.default_rng(args.seed)

    for fold_idx in tqdm(range(n_folds), desc="folds"):
        test_path = h5_paths[fold_idx]
        ckpt = results_dir / f"fold{fold_idx:02d}_{test_path.stem}.pt"
        if not ckpt.exists():
            continue

        train_paths = [p for p in h5_paths if p != test_path]
        if args.max_subjects_per_fold and len(train_paths) > args.max_subjects_per_fold:
            picks = rng.choice(len(train_paths), args.max_subjects_per_fold, replace=False)
            train_paths = [train_paths[i] for i in sorted(picks)]

        # Estadísticas streaming
        stats = stats_from_subjects(train_paths)

        # Cargar modelo
        model = ModSpecCNN(
            in_channels=cnn_cfg["model"]["in_channels"],
            conv_filters=tuple(cnn_cfg["model"]["conv_filters"]),
            kernel_size=cnn_cfg["model"]["kernel_size"],
            conv_dropout=cnn_cfg["model"]["conv_dropout"],
            fc_units=tuple(cnn_cfg["model"]["fc_units"]),
            fc_dropout=cnn_cfg["model"]["fc_dropout"],
            fc_negative_slope=cnn_cfg["model"]["fc_negative_slope"],
            num_classes=cnn_cfg["model"]["num_classes"],
        )
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device).eval()

        # Para cada sujeto de train, hacer pasada saliency
        per_class_sum = {0: None, 1: None}
        per_class_n = {0: 0, 1: 0}
        for p in train_paths:
            ds = ModSpecSubjectDataset(p, stats=stats)
            loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
            cls = ds.y
            try:
                m = sal_fn(model, loader, target_class=cls, device=device)
                per_class_sum[cls] = m if per_class_sum[cls] is None else per_class_sum[cls] + m
                per_class_n[cls] += 1
            except RuntimeError:
                pass

        for cls in (0, 1):
            if per_class_sum[cls] is not None and per_class_n[cls] > 0:
                fold_map = per_class_sum[cls] / per_class_n[cls]
                accum[cls] = fold_map if accum[cls] is None else accum[cls] + fold_map
                counts[cls] += 1

    if any(v is None for v in accum.values()):
        raise SystemExit("No se generaron saliency maps")

    sal_AD = accum[1] / max(counts[1], 1)
    sal_HC = accum[0] / max(counts[0], 1)
    np.save(sal_dir / "saliency_AD.npy", sal_AD)
    np.save(sal_dir / "saliency_HC.npy", sal_HC)
    np.save(sal_dir / "saliency_diff.npy", sal_AD - sal_HC)

    diff = sal_AD - sal_HC
    patches = find_patches(
        np.abs(diff), threshold_pct=88.0, n_clusters=4,
        random_state=svm_cfg["clustering"]["random_state"],
    )
    masks = (np.stack([p.mask for p in patches], axis=0)
             if patches else np.zeros((0, *diff.shape), dtype=bool))
    np.save(sal_dir / "patch_masks.npy", masks)

    summary = {
        "saliency_method": args.saliency_method,
        "n_folds_used": min(n_folds, sum(counts.values())),
        "n_patches": len(patches),
        "fold_count": counts,
        "max_subjects_per_fold": args.max_subjects_per_fold,
        "sal_AD_stats": {"min": float(sal_AD.min()), "max": float(sal_AD.max())},
        "sal_HC_stats": {"min": float(sal_HC.min()), "max": float(sal_HC.max())},
    }
    (sal_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Saliency en {sal_dir}: {len(patches)} patches encontrados")


if __name__ == "__main__":
    main()
