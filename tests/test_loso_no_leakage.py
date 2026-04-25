"""Verifica que LOSO no mezcla sujetos entre train y test."""

from __future__ import annotations

from pathlib import Path

from src.datasets import hold_out_val, loso_splits


def _fake_paths():
    return [Path(f"sub-{i:03d}.h5") for i in range(10)]


def test_loso_disjoint():
    paths = _fake_paths()
    splits = loso_splits(paths)
    assert len(splits) == len(paths)
    for test_path, train_paths in splits:
        assert test_path not in train_paths
        assert len(train_paths) == len(paths) - 1


def test_hold_out_val_disjoint():
    paths = _fake_paths()
    labels = {p.stem: i % 2 for i, p in enumerate(paths)}
    train, val = hold_out_val(paths, labels, seed=0)
    train_set = set(train)
    val_set = set(val)
    assert train_set.isdisjoint(val_set)
    assert len(val_set) >= 1
    # Estratificado: al menos un sujeto por clase en val si ambas existen
    val_labels = {labels[p.stem] for p in val}
    assert val_labels == set(labels.values())
