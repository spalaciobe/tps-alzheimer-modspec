"""Caché HDF5 de modulation spectrums por sujeto."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import h5py
import numpy as np


def config_hash(cfg: object) -> str:
    """Hash determinista de una config (dataclass o dict)."""
    if is_dataclass(cfg):
        cfg = asdict(cfg)
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def save_subject(
    path: str | Path,
    X: np.ndarray,
    y: int,
    epoch_idx: np.ndarray,
    cfg_hash: str,
    extra: dict | None = None,
) -> None:
    """Guarda los modspec de un sujeto en HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("y", data=np.array([y]))
        f.create_dataset("epoch_idx", data=epoch_idx)
        f.attrs["cfg_hash"] = cfg_hash
        if extra:
            for k, v in extra.items():
                f.attrs[k] = v


def load_subject(path: str | Path) -> dict:
    path = Path(path)
    with h5py.File(path, "r") as f:
        return {
            "X": f["X"][:],
            "y": int(f["y"][0]),
            "epoch_idx": f["epoch_idx"][:],
            "cfg_hash": f.attrs.get("cfg_hash", ""),
        }
