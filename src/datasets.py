"""Datasets PyTorch LOSO-aware sobre los modulation spectrums cacheados.

Para evitar recargar 64 HDF5 en cada fold (lento), usa `SubjectBank`: carga
todos los sujetos UNA vez, y los Datasets filtran por máscara booleana.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import load_subject
from .normalize import ChannelStats, apply_channel_zscore, fit_channel_zscore


@dataclass
class SubjectBank:
    """Carga ÚNICA de todos los modulation spectrums en RAM.

    Atributos:
        X: (N_total, n_ch, F, M) — todos los epochs concatenados.
            Soporta float16 (mitad de RAM, ~3 GB) o float32.
        y: (N_total,) — etiqueta por epoch.
        subject_ids: (N_total,) — sub-XXX por epoch.
        subject_to_label: dict subject_id → etiqueta entera.
        subject_paths: lista ordenada de paths.
    """
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    subject_to_label: dict[str, int]
    subject_paths: list[Path]

    @classmethod
    def from_paths(cls, h5_paths: list[Path], dtype=np.float32) -> "SubjectBank":
        """Carga todos los modspec. `dtype=np.float16` ahorra ~50% RAM
        (útil en Colab Free 12 GB). El z-score y forward del modelo
        convierten back a float32 al usar."""
        Xs, ys, sids = [], [], []
        sub_to_label: dict[str, int] = {}
        for p in h5_paths:
            data = load_subject(p)
            X = data["X"]
            Xs.append(X.astype(dtype, copy=False))
            ys.append(np.full(X.shape[0], int(data["y"]), dtype=np.int64))
            sids.append(np.full(X.shape[0], p.stem, dtype=object))
            sub_to_label[p.stem] = int(data["y"])
        return cls(
            X=np.concatenate(Xs, axis=0),
            y=np.concatenate(ys, axis=0),
            subject_ids=np.concatenate(sids, axis=0),
            subject_to_label=sub_to_label,
            subject_paths=list(h5_paths),
        )

    def mask_for_subjects(self, subject_ids: list[str] | set[str]) -> np.ndarray:
        wanted = set(subject_ids)
        return np.array([sid in wanted for sid in self.subject_ids])


class IndexedDataset(Dataset):
    """Dataset PyTorch sobre arrays ya cargados en RAM, filtrado por máscara.

    `epochs_per_subject`: si se da, subsamplea aleatoriamente N epochs por sujeto.
    """

    def __init__(
        self,
        bank: SubjectBank,
        mask: np.ndarray,
        stats: ChannelStats | None = None,
        epochs_per_subject: int | None = None,
        seed: int = 0,
    ):
        idx = np.where(mask)[0]
        if epochs_per_subject is not None:
            rng = np.random.default_rng(seed)
            kept: list[int] = []
            sids = bank.subject_ids[idx]
            for sid in np.unique(sids):
                pos = idx[sids == sid]
                if len(pos) > epochs_per_subject:
                    pos = rng.choice(pos, epochs_per_subject, replace=False)
                kept.extend(pos.tolist())
            idx = np.array(sorted(kept))

        # Promote a float32 SOLO el subset usado por este fold (no el bank entero)
        X = bank.X[idx].astype(np.float32, copy=False)
        if stats is None:
            stats = fit_channel_zscore(X)
        self.stats = stats
        self.X = apply_channel_zscore(X, stats)
        self.y = bank.y[idx]
        self.subject_ids = bank.subject_ids[idx]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y, str(self.subject_ids[idx])


# --- Compatibilidad: stubs de la API anterior por si algún script los importa --

class ModSpecSubjectDataset(Dataset):
    """Dataset de un solo sujeto leído desde HDF5 (modo legacy)."""

    def __init__(self, h5_path: str | Path, stats: ChannelStats | None = None):
        data = load_subject(h5_path)
        self.X = data["X"].astype(np.float32)
        self.y = int(data["y"])
        self.subject_id = Path(h5_path).stem
        if stats is not None:
            self.X = apply_channel_zscore(self.X, stats)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y, dtype=torch.long)
        return x, y, self.subject_id


class ModSpecConcatDataset(Dataset):
    """Compat: concatena epochs de múltiples sujetos leyendo HDF5 cada vez (legacy)."""

    def __init__(
        self,
        h5_paths: list[Path],
        stats: ChannelStats | None = None,
        epochs_per_subject: int | None = None,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        Xs, ys, sids = [], [], []
        for p in h5_paths:
            data = load_subject(p)
            X = data["X"].astype(np.float32)
            if epochs_per_subject is not None and X.shape[0] > epochs_per_subject:
                idx = rng.choice(X.shape[0], epochs_per_subject, replace=False)
                X = X[idx]
            Xs.append(X)
            ys.append(np.full(X.shape[0], int(data["y"]), dtype=np.int64))
            sids.append(np.full(X.shape[0], p.stem, dtype=object))
        self.X = np.concatenate(Xs, axis=0)
        self.y = np.concatenate(ys, axis=0)
        self.subject_ids = np.concatenate(sids, axis=0)
        if stats is None:
            stats = fit_channel_zscore(self.X)
        self.stats = stats
        self.X = apply_channel_zscore(self.X, stats)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y, str(self.subject_ids[idx])


def loso_splits(subject_paths: list[Path]) -> list[tuple[Path, list[Path]]]:
    paths = list(subject_paths)
    splits = []
    for i, test in enumerate(paths):
        train = paths[:i] + paths[i + 1:]
        splits.append((test, train))
    return splits


def hold_out_val(
    train_paths: list[Path], labels: dict[str, int], seed: int = 0
) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[Path]] = {}
    for p in train_paths:
        by_label.setdefault(labels[p.stem], []).append(p)
    val: list[Path] = []
    for _lbl, group in by_label.items():
        idx = rng.integers(0, len(group))
        val.append(group[idx])
    val_set = set(val)
    train = [p for p in train_paths if p not in val_set]
    return train, val
