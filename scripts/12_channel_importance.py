"""Análisis de importancia por canal del SVM (mejor configuración).

Cada feature en el SVM vainilla STFT viene de un (canal × patch). Si se sabe
qué features tienen mayor F-score ANOVA, se puede agregar por canal y ver
qué electrodos son más informativos para el diagnóstico AD vs HC.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_selection import f_classif

from src.cache import load_subject
from src.feature_extraction import Patch, features_for_dataset

BASE = Path("D:/Universidad/Maestria/TPS/Proyecto")
SAL_DIR = BASE / "data/derivatives/saliency"
OUT = BASE / "results/post_experiments"
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]


def load_patches(saliency_dir: Path, fold: int) -> list[Patch]:
    mask_path = saliency_dir / "per_fold" / f"patch_masks_fold{fold:02d}.npy"
    if not mask_path.exists():
        return []
    masks = np.load(mask_path)
    return [Patch(cluster_id=i, mask=masks[i].astype(bool)) for i in range(masks.shape[0])]


def channel_importance_per_fold(method: str = "stft", saliency: str = "vanilla", seed: int = 0) -> dict:
    """Para cada fold, calcula F-score ANOVA de cada feature.

    Las features están ordenadas como [ch0_patch0, ch0_patch1, ..., ch18_patchK, ch0_ratio01, ...].
    Se agrega F-score por canal para identificar electrodos más discriminativos.
    """
    sal_tag = "_vanilla" if saliency == "vanilla" else ""
    saliency_dir = SAL_DIR / f"{method}_200_seed{seed}{sal_tag}"
    modspec_dir = BASE / f"data/derivatives/modspec_{method}_200"
    h5_paths = sorted(modspec_dir.glob("*.h5"))

    # Cargar todos los modspec una vez
    data = {}
    for p in h5_paths:
        data[p.stem] = load_subject(p)

    channel_scores_per_fold = []
    rng = np.random.default_rng(seed)

    for fold_idx in range(65):
        test_sid = h5_paths[fold_idx].stem
        train_paths = [p for p in h5_paths if p.stem != test_sid]

        patches = load_patches(saliency_dir, fold_idx)
        if not patches:
            continue
        n_patches = len(patches)
        n_ch = len(CHANNELS)

        # Calcular features train (subsample 80 epochs)
        Xtr_list, ytr_list = [], []
        for p in train_paths:
            X = data[p.stem]["X"].astype(np.float32)
            if X.shape[0] > 80:
                idx = rng.choice(X.shape[0], 80, replace=False)
                X = X[idx]
            F = features_for_dataset(X, patches)
            Xtr_list.append(F)
            ytr_list.append(np.full(F.shape[0], data[p.stem]["y"]))
        Xtr = np.concatenate(Xtr_list, axis=0)
        ytr = np.concatenate(ytr_list)

        # ANOVA F-score
        f_scores, _ = f_classif(Xtr, ytr)
        # Las primeras n_ch * n_patches features son potencia de patches
        # Las siguientes son ratios: para cada par (i,j) i<j, 1 ratio por canal
        # Estructura: features_for_epoch flatten R (n_ch, n_patch) + ratios (n_ch para cada par)
        n_pow = n_ch * n_patches
        # Score por canal: suma f-scores de las potencias (n_patches por canal) + ratios
        ch_score_pow = np.zeros(n_ch)
        # R está flatten como (n_ch * n_patches)
        for c in range(n_ch):
            ch_score_pow[c] = f_scores[c * n_patches: (c + 1) * n_patches].sum()
        # Ratios: shape (n_ch * n_pairs) donde n_pairs = C(n_patches, 2)
        n_pairs = n_patches * (n_patches - 1) // 2
        ch_score_ratio = np.zeros(n_ch)
        for pair_idx in range(n_pairs):
            start = n_pow + pair_idx * n_ch
            end = start + n_ch
            if end <= len(f_scores):
                ch_score_ratio += f_scores[start:end]
        ch_total = ch_score_pow + ch_score_ratio
        channel_scores_per_fold.append(ch_total)

    if not channel_scores_per_fold:
        return {}
    arr = np.array(channel_scores_per_fold)
    # Promedio entre folds
    mean_score = arr.mean(axis=0)
    std_score = arr.std(axis=0, ddof=1)
    # Normalizar
    mean_norm = mean_score / mean_score.sum() if mean_score.sum() > 0 else mean_score
    return {
        "channel_score_mean": mean_score.tolist(),
        "channel_score_std": std_score.tolist(),
        "channel_score_norm": mean_norm.tolist(),
        "n_folds": len(channel_scores_per_fold),
        "method": method,
        "saliency": saliency,
        "seed": seed,
    }


def main():
    full_result = {}
    print("=== Importancia por canal (SVM vainilla STFT, mejor modelo) ===", flush=True)

    # Solo la mejor configuración por velocidad
    for seed in (0, 1, 2):
        print(f"\nSeed {seed} ...", flush=True)
        r = channel_importance_per_fold("stft", "vanilla", seed)
        if r:
            full_result[f"seed_{seed}"] = r
            sorted_ch = sorted(zip(CHANNELS, r["channel_score_norm"]), key=lambda x: x[1], reverse=True)
            print(f"  Top 5 canales: {sorted_ch[:5]}")

    # Agregado entre seeds
    if "seed_0" in full_result:
        all_scores = np.array([full_result[f"seed_{s}"]["channel_score_norm"]
                               for s in (0, 1, 2) if f"seed_{s}" in full_result])
        agg_mean = all_scores.mean(axis=0)
        agg_std = all_scores.std(axis=0, ddof=1) if len(all_scores) > 1 else np.zeros_like(agg_mean)
        ranking = sorted(zip(CHANNELS, agg_mean, agg_std), key=lambda x: x[1], reverse=True)
        full_result["aggregate"] = {
            "channels": CHANNELS,
            "mean": agg_mean.tolist(),
            "std": agg_std.tolist(),
            "ranking": [(c, float(m), float(s)) for c, m, s in ranking],
        }
        print()
        print("=== RANKING AGREGADO (3 seeds) ===")
        for c, m, s in ranking:
            print(f"  {c:4s}: {m:.4f} +/- {s:.4f}")

    (OUT / "channel_importance.json").write_text(json.dumps(full_result, indent=2))
    print(f"\nGuardado: {OUT / 'channel_importance.json'}")


if __name__ == "__main__":
    main()
