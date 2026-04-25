"""Loop de entrenamiento de la CNN con early stopping sobre F1 macro de val."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import NAdam
from torch.utils.data import DataLoader

from .evaluate import compute_metrics
from .models.cnn import ModSpecCNN


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 4
    epochs: int = 50
    early_stopping_patience: int = 10
    use_class_weights: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _class_weights_from_labels(y: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.from_numpy(w)


def train_cnn(
    model: ModSpecCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    train_labels: np.ndarray,
    ckpt_path: Path | None = None,
) -> dict:
    """Entrena `model` y devuelve el historial + best state_dict."""
    device = torch.device(cfg.device)
    model.to(device)

    if cfg.use_class_weights:
        weights = _class_weights_from_labels(train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optim = NAdam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_f1 = -1.0
    best_state: dict | None = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)

        # Eval
        val_loss, val_metrics = _evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_path:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)

    return {"history": history, "best_f1": best_f1}


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, dict]:
    model.eval()
    losses = []
    ys, preds = [], []
    for batch in loader:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item() * x.size(0))
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    n = sum(yi.shape[0] for yi in ys)
    val_loss = float(np.sum(losses)) / max(n, 1)
    metrics = compute_metrics(np.concatenate(ys), np.concatenate(preds))
    return val_loss, metrics
