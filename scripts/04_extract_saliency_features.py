"""Extrae mapas Grad-CAM y vanilla saliency agregados, y guarda los patches óptimos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.cache import load_subject
from src.datasets import ModSpecConcatDataset
from src.feature_extraction import find_patches
from src.models.cnn import ModSpecCNN
from src.saliency import gradcam_per_class, vanilla_saliency_per_class
from src.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["stft", "cwt"], required=True)
    parser.add_argument("--fs", type=int, choices=[200, 500], default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--cnn-config", type=Path, default=Path("configs/cnn.yaml"))
    parser.add_argument("--svm-config", type=Path, default=Path("configs/svm.yaml"))
    parser.add_argument("--saliency-method", choices=["gradcam", "vanilla"], default="gradcam")
    args = parser.parse_args()
    logger = get_logger("saliency")

    cfg = yaml.safe_load(open(args.config))
    cnn_cfg = yaml.safe_load(open(args.cnn_config))
    svm_cfg = yaml.safe_load(open(args.svm_config))
    paths = cfg["paths"]

    modspec_dir = Path(paths["modspec_root"]) / f"modspec_{args.method}_{args.fs}"
    results_dir = Path(paths["results"]) / f"{args.method}_{args.fs}_seed{args.seed}"
    sal_dir = Path(paths["saliency"]) / f"{args.method}_{args.fs}_seed{args.seed}"
    sal_dir.mkdir(parents=True, exist_ok=True)

    # Cargar todos los HDF5 y agregar saliency con el modelo de cada fold
    h5_paths = sorted(modspec_dir.glob("*.h5"))
    labels = {p.stem: load_subject(p)["y"] for p in h5_paths}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accum = {0: None, 1: None}
    counts = {0: 0, 1: 0}

    for fold_idx, test_path in enumerate(h5_paths):
        ckpt = results_dir / f"fold{fold_idx:02d}_{test_path.stem}.pt"
        if not ckpt.exists():
            logger.warning(f"checkpoint faltante: {ckpt}")
            continue
        train_paths = [p for p in h5_paths if p != test_path]
        ds = ModSpecConcatDataset(train_paths)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

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
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()

        sal_fn = gradcam_per_class if args.saliency_method == "gradcam" else vanilla_saliency_per_class
        for cls in (0, 1):
            try:
                m = sal_fn(model, loader, target_class=cls, device=device)
                accum[cls] = m if accum[cls] is None else accum[cls] + m
                counts[cls] += 1
            except RuntimeError as e:
                logger.warning(f"fold {fold_idx} clase {cls}: {e}")

    sal_AD = accum[1] / max(counts[1], 1)
    sal_HC = accum[0] / max(counts[0], 1)
    np.save(sal_dir / "saliency_AD.npy", sal_AD)
    np.save(sal_dir / "saliency_HC.npy", sal_HC)
    np.save(sal_dir / "saliency_diff.npy", sal_AD - sal_HC)

    # Patches a partir del mapa diferencial — usar config por defecto p=88, K=4
    diff = sal_AD - sal_HC
    patches = find_patches(
        np.abs(diff),
        threshold_pct=88.0,
        n_clusters=4,
        random_state=svm_cfg["clustering"]["random_state"],
    )
    patch_masks = np.stack([p.mask for p in patches], axis=0) if patches else np.zeros((0, *diff.shape))
    np.save(sal_dir / "patch_masks.npy", patch_masks)
    (sal_dir / "summary.json").write_text(json.dumps({
        "saliency_method": args.saliency_method,
        "n_patches": len(patches),
        "fold_count": counts,
    }, indent=2))
    logger.info(f"Saliency y patches guardados en {sal_dir}")


if __name__ == "__main__":
    main()
