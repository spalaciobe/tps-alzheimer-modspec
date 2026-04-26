"""Entrena la CNN con LOSO-CV. Una corrida por (method, fs, seed).

Uso:
    python scripts/03_train_loso.py --method stft --fs 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cache import load_subject
from src.datasets import IndexedDataset, SubjectBank, hold_out_val, loso_splits
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
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Perfil acelerado (batch=32, epochs=10, subsample=100)")
    parser.add_argument("--run-tag", default="",
                        help="Sufijo (ej. 'v2') para directorios paralelos")
    args = parser.parse_args()
    set_seed(args.seed)
    logger = get_logger("train_loso")

    cfg = yaml.safe_load(open(args.config))
    cnn_cfg = yaml.safe_load(open(args.cnn_config))
    paths = cfg["paths"]

    ms_name = f"modspec_{args.method}_{args.fs}"
    if args.run_tag:
        ms_name += f"_{args.run_tag}"
    modspec_dir = Path(paths["modspec_root"]) / ms_name
    h5_paths = sorted(modspec_dir.glob("*.h5"))
    if not h5_paths:
        raise SystemExit(f"No hay HDF5 en {modspec_dir}; corre 02_compute_modspec.py")

    # Precarga TODOS los sujetos a RAM una sola vez
    t0 = time.perf_counter()
    bank = SubjectBank.from_paths(h5_paths)
    logger.info(
        f"SubjectBank cargado en {time.perf_counter()-t0:.1f}s — "
        f"X.shape={bank.X.shape} ({bank.X.nbytes/1e9:.2f} GB)"
    )

    # Perfil quick
    train_overrides: dict = {}
    epochs_per_subject = None
    if args.quick:
        q = cnn_cfg["train"].get("quick", {})
        train_overrides = {
            "batch_size": q.get("batch_size", 32),
            "epochs": q.get("epochs", 10),
            "early_stopping_patience": q.get("early_stopping_patience", 3),
        }
        epochs_per_subject = q.get("epochs_per_subject", 100)
        logger.info(f"Perfil quick: {train_overrides}, epochs/sujeto={epochs_per_subject}")

    splits = loso_splits(h5_paths)
    if args.max_folds:
        splits = splits[: args.max_folds]

    suffix = "_quick" if args.quick else ""
    tag = f"_{args.run_tag}" if args.run_tag else ""
    results_dir = Path(paths["results"]) / f"{args.method}_{args.fs}_seed{args.seed}{suffix}{tag}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resume: cargar fold_results.json existente y saltar folds con .pt presentes
    fold_results_path = results_dir / "fold_results.json"
    fold_results = []
    if fold_results_path.exists():
        try:
            fold_results = json.loads(fold_results_path.read_text())
            logger.info(f"Resume: {len(fold_results)} folds previos cargados de {fold_results_path}")
        except Exception:
            fold_results = []
    done_subjects = {r["test_subject"] for r in fold_results}

    for fold_idx, (test_path, train_paths) in enumerate(tqdm(splits, desc="LOSO")):
        if test_path.stem in done_subjects:
            continue
        train_paths_only, val_paths = hold_out_val(
            train_paths, bank.subject_to_label, args.seed + fold_idx
        )
        train_sids = [p.stem for p in train_paths_only]
        val_sids = [p.stem for p in val_paths]
        test_sid = test_path.stem

        train_mask = bank.mask_for_subjects(train_sids)
        val_mask = bank.mask_for_subjects(val_sids)
        test_mask = bank.mask_for_subjects([test_sid])

        train_ds = IndexedDataset(
            bank, train_mask,
            epochs_per_subject=epochs_per_subject,
            seed=args.seed + fold_idx,
        )
        stats = train_ds.stats
        val_ds = IndexedDataset(bank, val_mask, stats=stats,
                                epochs_per_subject=epochs_per_subject,
                                seed=args.seed + fold_idx)
        test_ds = IndexedDataset(bank, test_mask, stats=stats)

        batch_size = train_overrides.get("batch_size", cnn_cfg["train"]["batch_size"])
        pin_mem = bool(cnn_cfg["train"].get("pin_memory", False)) and torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=False, pin_memory=pin_mem)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                                num_workers=0, pin_memory=pin_mem)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                                 num_workers=0, pin_memory=pin_mem)

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
            batch_size=batch_size,
            epochs=train_overrides.get("epochs", cnn_cfg["train"]["epochs"]),
            early_stopping_patience=train_overrides.get(
                "early_stopping_patience", cnn_cfg["train"]["early_stopping_patience"]
            ),
            use_class_weights=cnn_cfg["train"]["use_class_weights"],
        )
        ckpt = results_dir / f"fold{fold_idx:02d}_{test_sid}.pt"
        t1 = time.perf_counter()
        train_cnn(model, train_loader, val_loader, tcfg,
                  train_labels=train_ds.y, ckpt_path=ckpt)
        train_dt = time.perf_counter() - t1

        device = torch.device(tcfg.device)
        model.to(device).eval()

        # Métricas por epoch sobre el sujeto test
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
            "test_subject": test_sid,
            "true": int(subj_pred["y_true"][0]),
            "pred": int(subj_pred["y_pred"][0]),
            "score": float(subj_pred["y_score"][0]),
            "epoch_metrics": epoch_metrics,
            "train_seconds": round(train_dt, 1),
        })
        # Guardar incrementalmente
        (results_dir / "fold_results.json").write_text(json.dumps(fold_results, indent=2))
        logger.info(
            f"[fold {fold_idx:02d}] {test_sid} true={subj_pred['y_true'][0]} "
            f"pred={subj_pred['y_pred'][0]} score={subj_pred['y_score'][0]:.3f} "
            f"epoch_acc={epoch_metrics['accuracy']:.3f} train={train_dt:.1f}s"
        )

    out_json = results_dir / "fold_results.json"
    out_json.write_text(json.dumps(fold_results, indent=2))
    logger.info(f"Resultados en {out_json}")


if __name__ == "__main__":
    main()
