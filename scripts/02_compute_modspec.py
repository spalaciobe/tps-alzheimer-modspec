"""Calcula el modulation spectrum (STFT o CWT) para todos los sujetos preprocesados.

Uso:
    python scripts/02_compute_modspec.py --method stft --fs 200
    python scripts/02_compute_modspec.py --method cwt --fs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import yaml
from tqdm import tqdm

from src.cache import config_hash, save_subject
from src.io_bids import list_subjects
from src.modspec import ModSpecConfig, compute_modulation_spectrum_subject
from src.utils.logging import get_logger


def build_modspec_config(method: str, fs: int, configs_dir: Path) -> ModSpecConfig:
    cfg_file = configs_dir / f"{method}.yaml"
    d = yaml.safe_load(open(cfg_file))
    if method == "stft":
        nperseg = d["nperseg_at_500hz"] if fs == 500 else d["nperseg_at_200hz"]
        noverlap = d["noverlap_at_500hz"] if fs == 500 else d["noverlap_at_200hz"]
        return ModSpecConfig(
            method="stft", fs=fs,
            target_shape=tuple(d["target_shape"]),
            carrier_range_hz=tuple(d["carrier_range_hz"]),
            mod_range_hz=tuple(d["mod_range_hz"]),
            window=d["window"], nperseg=nperseg, noverlap=noverlap,
            log_power=d["log_power"], eps=d["eps"],
        )
    return ModSpecConfig(
        method="cwt", fs=fs,
        target_shape=tuple(d["target_shape"]),
        carrier_range_hz=tuple(d["carrier_range_hz"]),
        mod_range_hz=tuple(d["mod_range_hz"]),
        wavelet=d["wavelet"], n_scales=d["n_scales"],
        log_power=d["log_power"], eps=d["eps"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["stft", "cwt"], required=True)
    parser.add_argument("--fs", type=int, choices=[200, 500], default=200)
    parser.add_argument("--version", choices=["paper", "dataset"], default="paper")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logger = get_logger("modspec")

    cfg_dict = yaml.safe_load(open(args.config))
    paths = cfg_dict["paths"]
    eps_cfg = cfg_dict["epoching"]

    preproc_dir = Path(paths["preproc_paper" if args.version == "paper" else "preproc_dataset"])
    out_dir = Path(paths["modspec_root"]) / f"modspec_{args.method}_{args.fs}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ms_cfg = build_modspec_config(args.method, args.fs, Path("configs"))
    h = config_hash(ms_cfg)
    logger.info(f"ModSpec config hash = {h}")

    subjects = list_subjects(paths["raw"], tuple(cfg_dict["dataset"]["groups_keep"]))
    if args.limit:
        subjects = subjects[: args.limit]

    for rec in tqdm(subjects):
        fif = preproc_dir / f"{rec.subject_id}_preproc-raw.fif"
        if not fif.exists():
            logger.warning(f"[{rec.subject_id}] no preprocesado, skip")
            continue
        out = out_dir / f"{rec.subject_id}.h5"
        if out.exists():
            continue

        raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
        if args.fs != int(raw.info["sfreq"]):
            raw = raw.resample(args.fs, npad="auto", verbose="ERROR")

        signal = raw.get_data()  # (n_ch, n_samples)
        epoch_step = eps_cfg["duration_s"] - eps_cfg["overlap_s"]
        X = compute_modulation_spectrum_subject(
            signal,
            ms_cfg,
            epoch_duration_s=eps_cfg["duration_s"],
            epoch_step_s=epoch_step,
            drop_last_s=eps_cfg["drop_last_s"],
        )
        if X.shape[0] < eps_cfg["min_epochs_per_subject"]:
            logger.warning(
                f"[{rec.subject_id}] solo {X.shape[0]} epochs (<{eps_cfg['min_epochs_per_subject']})"
            )
        save_subject(
            out, X, rec.label_binary,
            np.arange(X.shape[0]), cfg_hash=h,
            extra={"group": rec.group, "method": args.method, "fs": args.fs},
        )

    logger.info(f"Listo. {len(list(out_dir.glob('*.h5')))} archivos en {out_dir}")


if __name__ == "__main__":
    main()
