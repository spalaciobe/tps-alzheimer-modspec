"""Selección de features guiada por saliency: umbral + KMeans → patches → potencia + ratios.

Réplica de la sección 2.6 de Lopes et al. 2023 con grid search sobre
threshold ∈ {80,82,...,96}% y K ∈ {3,4,5}, seguido de selección ANOVA top-24.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif


@dataclass
class Patch:
    cluster_id: int
    mask: np.ndarray   # (H, W) booleano


def find_patches(
    saliency_map: np.ndarray,
    threshold_pct: float,
    n_clusters: int,
    random_state: int = 0,
) -> list[Patch]:
    """Umbraliza el saliency map y agrupa los píxeles activos con KMeans."""
    thr = np.percentile(saliency_map, threshold_pct)
    coords = np.argwhere(saliency_map >= thr)
    if len(coords) < n_clusters:
        return []
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(coords)
    patches: list[Patch] = []
    for k in range(n_clusters):
        mask = np.zeros_like(saliency_map, dtype=bool)
        for (i, j), lbl in zip(coords, labels):
            if lbl == k:
                mask[i, j] = True
        patches.append(Patch(cluster_id=k, mask=mask))
    return patches


def patch_powers(modspec_epoch: np.ndarray, patches: list[Patch]) -> np.ndarray:
    """Potencia media en cada patch, por canal.

    Args:
        modspec_epoch: (n_channels, H, W).
        patches: lista de patches.

    Returns:
        Vector (n_channels * n_patches,)
    """
    n_ch = modspec_epoch.shape[0]
    out = np.zeros((n_ch, len(patches)), dtype=np.float32)
    for k, p in enumerate(patches):
        if p.mask.any():
            out[:, k] = modspec_epoch[:, p.mask].mean(axis=1)
    return out.flatten()


def features_for_epoch(modspec_epoch: np.ndarray, patches: list[Patch]) -> np.ndarray:
    """Potencia por patch + ratios entre patches, todo por canal."""
    n_ch = modspec_epoch.shape[0]
    n_p = len(patches)
    R = np.zeros((n_ch, n_p), dtype=np.float32)
    for k, p in enumerate(patches):
        if p.mask.any():
            R[:, k] = modspec_epoch[:, p.mask].mean(axis=1)

    feats = [R.flatten()]
    if n_p >= 2:
        eps = 1e-10
        for i, j in combinations(range(n_p), 2):
            ratio = R[:, i] / (R[:, j] + eps)
            feats.append(ratio.astype(np.float32))
    return np.concatenate(feats)


def features_for_dataset(X: np.ndarray, patches: list[Patch]) -> np.ndarray:
    """Aplica `features_for_epoch` a un batch de epochs."""
    return np.stack([features_for_epoch(x, patches) for x in X], axis=0)


def grid_search_patches(
    saliency_map: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    thresholds_pct: list[float],
    k_clusters: list[int],
    classifier_factory,
    random_state: int = 0,
) -> dict:
    """Grid search sobre (threshold, K) seleccionando por F1 en val.

    `classifier_factory()` debe devolver un clasificador sklearn-like que
    acepte fit / predict tras un MinMaxScaler.
    """
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MinMaxScaler

    best = {"f1": -1.0, "thr": None, "k": None, "patches": None}
    for thr in thresholds_pct:
        for k in k_clusters:
            patches = find_patches(saliency_map, thr, k, random_state)
            if not patches:
                continue
            Ftr = features_for_dataset(X_train, patches)
            Fv = features_for_dataset(X_val, patches)

            scaler = MinMaxScaler(feature_range=(-1, 1)).fit(Ftr)
            Ftr_s = scaler.transform(Ftr)
            Fv_s = scaler.transform(Fv)

            clf = classifier_factory()
            clf.fit(Ftr_s, y_train)
            pred = clf.predict(Fv_s)
            f1 = f1_score(y_val, pred, average="macro", zero_division=0)
            if f1 > best["f1"]:
                best = {"f1": f1, "thr": thr, "k": k, "patches": patches}
    return best


def select_top_k_features(
    X_train: np.ndarray, y_train: np.ndarray, k: int = 24
) -> SelectKBest:
    """ANOVA F-value top-k (Lopes et al. usan k=24)."""
    sel = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    sel.fit(X_train, y_train)
    return sel
