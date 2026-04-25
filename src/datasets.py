"""Datasets PyTorch LOSO-aware sobre los modulation spectrums cacheados."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import load_subject
from .normalize import ChannelStats, apply_channel_zscore, fit_channel_zscore


class ModSpecSubjectDataset(Dataset):
    """Carga todos los epochs de un sujeto desde su HDF5."""

    def __init__(
        self,
        h5_path: str | Path,
        stats: ChannelStats | None = None,
    ):
        data = load_subject(h5_path)
        self.X = data["X"].astype(np.float32)
        self.y = int(data["y"])
        self.subject_id = Path(h5_path).stem
        if stats is not None:
            self.X = apply_channel_zscore(self.X, stats)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y, dtype=torch.long)
        return x, y, idx


class ModSpecConcatDataset(Dataset):
    """Concatena epochs de múltiples sujetos (e.g., todos los de train)."""

    def __init__(
        self,
        h5_paths: list[Path],
        stats: ChannelStats | None = None,
    ):
        Xs, ys, sids = [], [], []
        for p in h5_paths:
            data = load_subject(p)
            Xs.append(data["X"].astype(np.float32))
            ys.append(np.full(data["X"].shape[0], int(data["y"]), dtype=np.int64))
            sids.append(np.full(data["X"].shape[0], Path(p).stem, dtype=object))
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
    """Genera (test_subject, train_subjects) para LOSO."""
    paths = list(subject_paths)
    splits = []
    for i, test in enumerate(paths):
        train = paths[:i] + paths[i + 1:]
        splits.append((test, train))
    return splits


def hold_out_val(
    train_paths: list[Path],
    labels: dict[str, int],
    seed: int = 0,
) -> tuple[list[Path], list[Path]]:
    """Separa 1 sujeto de validación del set de train, estratificado por clase."""
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
