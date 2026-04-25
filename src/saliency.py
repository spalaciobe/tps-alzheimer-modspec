"""Saliency: Grad-CAM (principal) y vanilla gradient (ablation, fiel a Lopes)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _correct_mask(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    logits = model(x)
    return logits.argmax(1) == y


def gradcam_per_class(
    model: nn.Module,
    loader: DataLoader,
    target_class: int,
    device: torch.device,
    only_correct: bool = True,
) -> np.ndarray:
    """Promedio de mapas Grad-CAM sobre todas las epochs de la clase indicada
    (opcionalmente solo las clasificadas correctamente).

    Returns:
        Array (H, W) — saliency promediada en la rejilla del modspec.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval().to(device)
    target_layers = [model.gradcam_target_layer]
    cam = GradCAM(model=model, target_layers=target_layers)

    accum: np.ndarray | None = None
    n = 0
    for batch in loader:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        mask_class = y == target_class
        if only_correct:
            mask_class = mask_class & _correct_mask(model, x, y)
        if not mask_class.any():
            continue
        x_sel = x[mask_class]
        targets = [ClassifierOutputTarget(target_class)] * x_sel.size(0)
        grayscale = cam(input_tensor=x_sel, targets=targets)  # (B, H, W) en [0,1]
        if accum is None:
            accum = grayscale.sum(axis=0)
        else:
            accum = accum + grayscale.sum(axis=0)
        n += x_sel.size(0)

    if accum is None or n == 0:
        raise RuntimeError(f"No hay muestras de la clase {target_class}")
    return (accum / n).astype(np.float32)


def vanilla_saliency_per_class(
    model: nn.Module,
    loader: DataLoader,
    target_class: int,
    device: torch.device,
    only_correct: bool = True,
) -> np.ndarray:
    """Vanilla gradient (Simonyan et al. 2014) — réplica fiel del paper de Lopes.

    El paper toma `|d output_class / d input|` y promedia sobre canales.
    """
    from captum.attr import Saliency

    model.eval().to(device)
    sal = Saliency(model)

    accum: np.ndarray | None = None
    n = 0
    for batch in loader:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        mask_class = y == target_class
        if only_correct:
            mask_class = mask_class & _correct_mask(model, x, y)
        if not mask_class.any():
            continue
        x_sel = x[mask_class].requires_grad_(True)
        attr = sal.attribute(x_sel, target=target_class, abs=True)
        # Promediar canales EEG → mapa 2D por epoch
        attr2d = attr.mean(dim=1).detach().cpu().numpy()  # (B, H, W)
        if accum is None:
            accum = attr2d.sum(axis=0)
        else:
            accum = accum + attr2d.sum(axis=0)
        n += x_sel.size(0)

    if accum is None or n == 0:
        raise RuntimeError(f"No hay muestras de la clase {target_class}")
    out = accum / n
    # Normalizar a [0,1] como Grad-CAM
    out = (out - out.min()) / (out.ptp() + 1e-12)
    return out.astype(np.float32)


def map_correlation(M1: np.ndarray, M2: np.ndarray) -> float:
    """Correlación de Pearson entre dos saliency maps 2D del mismo shape."""
    a = M1.flatten() - M1.mean()
    b = M2.flatten() - M2.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den < 1e-12:
        return 0.0
    return float((a * b).sum() / den)
