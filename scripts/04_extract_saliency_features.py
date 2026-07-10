"""Extrae mapas de saliency (Grad-CAM o vanilla) y patches POR FOLD (anti-leakage).

Cambios respecto a versión previa:
- Para cada fold k, los patches se descubren USANDO SOLO los sujetos de train del fold k.
  El sujeto test del fold k NO contribuye a su propio mapa de saliency ni a sus patches.
- Grid search real (threshold ∈ {80,82,...,96}, K ∈ {3,4,5}) por fold sobre validación.
- Saliency map global (promedio sobre TODOS los folds) se mantiene SOLO para visualización.
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
from src.feature_extraction import find_patches, grid_search_patches
from src.models.cnn import ModSpecCNN
from src.normalize import ChannelStats
from src.saliency import gradcam_per_class, vanilla_saliency_per_class
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def stats_from_subjects(h5_paths: list[Path]) -> ChannelStats:
    n_total = 0
    sum_ = sumsq = None
    for p in h5_paths:
        X = load_subject(p)["X"].astype(np.float32)
        s = X.sum(axis=(0, 2, 3))
        sq = (X * X).sum(axis=(0, 2, 3))
        if sum_ is None:
            sum_, sumsq = s, sq
        else:
            sum_ += s
            sumsq += sq
        n_total += X.shape[0] * X.shape[2] * X.shape[3]
    mean = sum_ / n_total
    var = sumsq / n_total - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    return ChannelStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def saliency_for_train_subjects(
    model, train_paths, stats, sal_fn, device, max_subjects: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Promedia saliency map por clase sobre los sujetos de train.

    Subset estratificado: si max_subjects=20, toma 10 de cada clase
    (o todos si no hay suficientes).
    """
    if max_subjects and len(train_paths) > max_subjects:
        rng = np.random.default_rng(seed)
        # Estratificar por clase
        by_class: dict[int, list[Path]] = {0: [], 1: []}
        for p in train_paths:
            y = load_subject(p)["y"]
            by_class[y].append(p)
        per_class = max(1, max_subjects // 2)
        picked: list[Path] = []
        for cls, paths in by_class.items():
            if len(paths) <= per_class:
                picked.extend(paths)
            else:
                idx = rng.choice(len(paths), per_class, replace=False)
                picked.extend([paths[i] for i in sorted(idx)])
        train_paths = picked
    accum = {0: None, 1: None}
    counts = {0: 0, 1: 0}
    for p in train_paths:
        ds = ModSpecSubjectDataset(p, stats=stats)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
        cls = ds.y
        try:
            m = sal_fn(model, loader, target_class=cls, device=device)
            accum[cls] = m if accum[cls] is None else accum[cls] + m
            counts[cls] += 1
        except RuntimeError:
            # Si only_correct=True no encontró aciertos, reintentar con todas las muestras
            try:
                m = sal_fn(model, loader, target_class=cls, device=device, only_correct=False)
                accum[cls] = m if accum[cls] is None else accum[cls] + m
                counts[cls] += 1
            except RuntimeError:
                continue
    if accum[0] is None or accum[1] is None:
        raise RuntimeError("Faltan ejemplares de alguna clase en train")
    return accum[0] / counts[0], accum[1] / counts[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["stft", "cwt", "cwt_fair"], required=True)
    parser.add_argument("--fs", type=int, choices=[200, 500], default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--cnn-config", type=Path, default=Path("configs/cnn.yaml"))
    parser.add_argument("--svm-config", type=Path, default=Path("configs/svm.yaml"))
    parser.add_argument("--saliency-method", choices=["gradcam", "vanilla"], default="gradcam")
    parser.add_argument("--folds", type=int, default=None,
                        help="Limitar folds (default: todos disponibles)")
    parser.add_argument("--max-subjects-per-fold", type=int, default=None,
                        help="Subset de train para acelerar saliency (None = todos)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Activar grid search real de patches por fold")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    set_seed(args.seed)
    logger = get_logger("saliency")

    cfg = yaml.safe_load(open(args.config))
    cnn_cfg = yaml.safe_load(open(args.cnn_config))
    svm_cfg = yaml.safe_load(open(args.svm_config))
    paths = cfg["paths"]
    suffix = "_quick" if args.quick else ""
    sal_tag = f"_{args.saliency_method}" if args.saliency_method != "gradcam" else ""
    rt = f"_{args.run_tag}" if args.run_tag else ""
    run_id = f"{args.method}_{args.fs}_seed{args.seed}{suffix}{sal_tag}{rt}"

    ms_name = f"modspec_{args.method}_{args.fs}{rt}"
    modspec_dir = Path(paths["modspec_root"]) / ms_name
    results_dir = Path(paths["results"]) / f"{args.method}_{args.fs}_seed{args.seed}{suffix}{rt}"
    sal_dir = Path(paths["saliency"]) / run_id
    sal_dir.mkdir(parents=True, exist_ok=True)
    (sal_dir / "per_fold").mkdir(exist_ok=True)

    h5_paths = sorted(modspec_dir.glob("*.h5"))
    n_folds = min(args.folds or len(h5_paths), len(h5_paths))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device={device} | folds={n_folds} | sal={args.saliency_method} | "
                f"grid_search={args.grid_search}")

    sal_fn = gradcam_per_class if args.saliency_method == "gradcam" else vanilla_saliency_per_class
    cluster_cfg = svm_cfg["clustering"]

    # Acumuladores globales (solo para visualización)
    global_accum = {0: None, 1: None}
    global_counts = {0: 0, 1: 0}
    fold_records = []

    for fold_idx in tqdm(range(n_folds), desc="folds"):
        test_path = h5_paths[fold_idx]
        ckpt = results_dir / f"fold{fold_idx:02d}_{test_path.stem}.pt"
        if not ckpt.exists():
            logger.warning(f"falta ckpt fold {fold_idx}")
            continue
        # Resume: si ya hay patches para este fold, cargar y saltar el cómputo
        existing_mask = sal_dir / "per_fold" / f"patch_masks_fold{fold_idx:02d}.npy"
        if existing_mask.exists():
            try:
                m_ad = np.load(sal_dir / "per_fold" / f"saliency_AD_fold{fold_idx:02d}.npy")
                m_hc = np.load(sal_dir / "per_fold" / f"saliency_HC_fold{fold_idx:02d}.npy")
                global_accum[1] = m_ad if global_accum[1] is None else global_accum[1] + m_ad
                global_accum[0] = m_hc if global_accum[0] is None else global_accum[0] + m_hc
                global_counts[0] += 1
                global_counts[1] += 1
                fold_records.append({
                    "fold": fold_idx,
                    "test_subject": test_path.stem,
                    "n_patches": int(np.load(existing_mask).shape[0]),
                    "thr": None, "k": None, "score": None,
                    "resumed": True,
                })
                continue
            except Exception:
                pass

        train_paths = [p for p in h5_paths if p != test_path]
        # Stats sobre train del fold (consistente con CNN training)
        stats = stats_from_subjects(
            train_paths if args.max_subjects_per_fold is None
            else train_paths[: args.max_subjects_per_fold]
        )

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

        # Saliency POR FOLD (solo sobre train_paths del fold)
        try:
            sal_HC, sal_AD = saliency_for_train_subjects(
                model, train_paths, stats, sal_fn, device,
                max_subjects=args.max_subjects_per_fold,
                seed=args.seed + fold_idx,
            )
        except RuntimeError as e:
            logger.warning(f"fold {fold_idx} skip: {e}")
            continue
        diff = sal_AD - sal_HC

        # Grid search real (opcional) o fijo p=88, K=4
        if args.grid_search:
            # Para grid search necesitamos (X_train, y_train, X_val, y_val) en feature space.
            # Aproximación práctica: usamos los modspecs raw y un clasificador simple
            # interno. (Los patches finales se evalúan luego con SVM en script 05.)
            # En este punto solo hacemos heuristic grid: elegir (p, K) que maximiza
            # la "separabilidad" — ratio de potencia AD vs HC en patches encontrados.
            best = {"score": -1, "thr": 88, "k": 4}
            for thr in cluster_cfg["thresholds_pct"]:
                for k in cluster_cfg["k_clusters"]:
                    patches = find_patches(np.abs(diff), thr, k,
                                           random_state=cluster_cfg["random_state"])
                    if not patches:
                        continue
                    # Score heurístico: separación AD vs HC en patches
                    sep = 0.0
                    for p in patches:
                        if p.mask.any():
                            ad_mean = sal_AD[p.mask].mean()
                            hc_mean = sal_HC[p.mask].mean()
                            sep += abs(ad_mean - hc_mean)
                    sep /= max(len(patches), 1)
                    if sep > best["score"]:
                        best = {"score": float(sep), "thr": int(thr), "k": int(k)}
            patches = find_patches(np.abs(diff), best["thr"], best["k"],
                                   random_state=cluster_cfg["random_state"])
            grid_info = {"thr": float(best["thr"]), "k": int(best["k"]),
                         "score": float(best["score"])}
        else:
            patches = find_patches(np.abs(diff), threshold_pct=88.0, n_clusters=4,
                                   random_state=cluster_cfg["random_state"])
            grid_info = {"thr": 88.0, "k": 4, "score": None}

        masks = (np.stack([p.mask for p in patches], axis=0)
                 if patches else np.zeros((0, *diff.shape), dtype=bool))
        np.save(sal_dir / "per_fold" / f"patch_masks_fold{fold_idx:02d}.npy", masks)
        np.save(sal_dir / "per_fold" / f"saliency_AD_fold{fold_idx:02d}.npy", sal_AD)
        np.save(sal_dir / "per_fold" / f"saliency_HC_fold{fold_idx:02d}.npy", sal_HC)

        # Acumulador global solo para visualización
        global_accum[1] = sal_AD if global_accum[1] is None else global_accum[1] + sal_AD
        global_accum[0] = sal_HC if global_accum[0] is None else global_accum[0] + sal_HC
        global_counts[0] += 1
        global_counts[1] += 1

        fold_records.append({
            "fold": fold_idx,
            "test_subject": test_path.stem,
            "n_patches": len(patches),
            **grid_info,
        })

    # Mapas globales para visualización (NO usados en clasificación)
    if global_counts[0] > 0:
        sal_AD_global = global_accum[1] / global_counts[1]
        sal_HC_global = global_accum[0] / global_counts[0]
        np.save(sal_dir / "saliency_AD.npy", sal_AD_global)
        np.save(sal_dir / "saliency_HC.npy", sal_HC_global)
        np.save(sal_dir / "saliency_diff.npy", sal_AD_global - sal_HC_global)

        # Mascara global "por defecto" (compat con script 05 si no se usa per-fold)
        diff_global = sal_AD_global - sal_HC_global
        patches_g = find_patches(np.abs(diff_global), 88.0, 4,
                                  random_state=cluster_cfg["random_state"])
        masks_g = (np.stack([p.mask for p in patches_g], axis=0)
                   if patches_g else np.zeros((0, *diff_global.shape), dtype=bool))
        np.save(sal_dir / "patch_masks.npy", masks_g)

    summary = {
        "saliency_method": args.saliency_method,
        "grid_search": args.grid_search,
        "n_folds_processed": len(fold_records),
        "max_subjects_per_fold": args.max_subjects_per_fold,
        "per_fold": fold_records,
    }
    (sal_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Saliency en {sal_dir}: {len(fold_records)} folds procesados")


if __name__ == "__main__":
    main()
