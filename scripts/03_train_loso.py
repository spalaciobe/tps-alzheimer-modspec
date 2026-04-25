"""Entrena la CNN con LOSO-CV. Una corrida por (method, fs, seed).

Uso:
    python scripts/03_train_loso.py --method stft --fs 200 --seed 0
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

from src.datasets import (
    ModSpecConcatDataset,
    ModSpecSubjectDataset,
    hold_out_val,
    loso_splits,
)
from src.evaluate import compute_metrics, predict_per_subject
from src.models.cnn import ModSpecCNN
from src.train import TrainConfig, train_cnn
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["stft", "cwt"], required=True)
    parser.add_argument("--fs", type=int, choices=[200, 500], default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--cnn-config", type=Path, default=Path("configs/cnn.yaml"))
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Para debug: limitar número de folds")
    args = parser.parse_args()
    set_seed(args.seed)
    logger = get_logger("train_loso")

    cfg = yaml.safe_load(open(args.config))
    cnn_cfg = yaml.safe_load(open(args.cnn_config))
    paths = cfg["paths"]

    modspec_dir = Path(paths["modspec_root"]) / f"modspec_{args.method}_{args.fs}"
    h5_paths = sorted(modspec_dir.glob("*.h5"))
    if not h5_paths:
        raise SystemExit(f"No hay HDF5 en {modspec_dir}; corre 02_compute_modspec.py")

    # Etiquetas por sujeto
    from src.cache import load_subject
    labels = {p.stem: load_subject(p)["y"] for p in h5_paths}

    splits = loso_splits(h5_paths)
    if args.max_folds:
        splits = splits[: args.max_folds]

    results_dir = Path(paths["results"]) / f"{args.method}_{args.fs}_seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold_idx, (test_path, train_paths) in enumerate(tqdm(splits, desc="LOSO")):
        train_paths_only, val_paths = hold_out_val(train_paths, labels, args.seed + fold_idx)

        train_ds = ModSpecConcatDataset(train_paths_only)
        stats = train_ds.stats
        val_ds = ModSpecConcatDataset(val_paths, stats=stats)
        test_ds = ModSpecSubjectDataset(test_path, stats=stats)

        train_loader = DataLoader(train_ds, batch_size=cnn_cfg["train"]["batch_size"],
                                  shuffle=True, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

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
        tcfg = TrainConfig(
            learning_rate=cnn_cfg["train"]["learning_rate"],
            weight_decay=cnn_cfg["train"]["weight_decay"],
            batch_size=cnn_cfg["train"]["batch_size"],
            epochs=cnn_cfg["train"]["epochs"],
            early_stopping_patience=cnn_cfg["train"]["early_stopping_patience"],
            use_class_weights=cnn_cfg["train"]["use_class_weights"],
        )
        ckpt = results_dir / f"fold{fold_idx:02d}_{test_path.stem}.pt"
        train_cnn(model, train_loader, val_loader, tcfg,
                  train_labels=train_ds.y, ckpt_path=ckpt)

        # Evaluación por epoch sobre el sujeto test
        device = torch.device(tcfg.device)
        model.to(device).eval()
        ys, preds, scores = [], [], []
        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, 1).cpu().numpy()
                ys.append(y.numpy())
                preds.append(probs.argmax(1))
                scores.append(probs[:, 1])
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        scores = np.concatenate(scores)
        epoch_metrics = compute_metrics(ys, preds, y_score=scores)

        # Agregación por sujeto
        subj_pred = predict_per_subject(
            model, test_loader, device,
            aggregation=cnn_cfg["aggregation"]["epoch_to_subject"],
        )

        fold_results.append({
            "fold": fold_idx,
            "test_subject": test_path.stem,
            "true": int(subj_pred["y_true"][0]),
            "pred": int(subj_pred["y_pred"][0]),
            "score": float(subj_pred["y_score"][0]),
            "epoch_metrics": epoch_metrics,
        })
        logger.info(
            f"[fold {fold_idx:02d}] {test_path.stem} "
            f"true={subj_pred['y_true'][0]} pred={subj_pred['y_pred'][0]} "
            f"epoch_acc={epoch_metrics['accuracy']:.3f}"
        )

    out_json = results_dir / "fold_results.json"
    out_json.write_text(json.dumps(fold_results, indent=2))
    logger.info(f"Resultados guardados en {out_json}")


if __name__ == "__main__":
    main()
