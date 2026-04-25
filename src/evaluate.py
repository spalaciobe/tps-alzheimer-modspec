"""Métricas por epoch y agregación a nivel de sujeto."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    """Accuracy, F1 macro, sensibilidad (AD=1), especificidad (HC=0), AUC."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    return out


@torch.no_grad()
def predict_per_subject(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    aggregation: str = "mean_softmax",
) -> dict:
    """Agrega probabilidades por sujeto y devuelve y_true, y_pred, y_score."""
    model.eval()
    bag_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    bag_label: dict[str, int] = {}
    for batch in loader:
        x, y, sids = batch
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for i, sid in enumerate(sids):
            bag_probs[sid].append(probs[i])
            bag_label[sid] = int(y[i].item())

    sids = list(bag_probs.keys())
    y_true = np.array([bag_label[s] for s in sids], dtype=np.int64)

    if aggregation == "mean_softmax":
        y_score = np.array([np.mean(np.stack(bag_probs[s], axis=0), axis=0)[1] for s in sids])
        y_pred = (y_score >= 0.5).astype(np.int64)
    elif aggregation == "majority_vote":
        y_pred = np.array([
            int(np.mean([np.argmax(p) for p in bag_probs[s]]) >= 0.5) for s in sids
        ])
        y_score = np.array([np.mean([p[1] for p in bag_probs[s]]) for s in sids])
    else:
        raise ValueError(f"agregación desconocida: {aggregation}")

    return {"subject_ids": sids, "y_true": y_true, "y_pred": y_pred, "y_score": y_score}
