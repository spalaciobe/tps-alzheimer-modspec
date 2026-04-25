"""Aplica el pipeline de preproceso a todos los sujetos del ds004504.

Uso:
    python scripts/01_preprocess_all.py --version paper
    python scripts/01_preprocess_all.py --version dataset
    python scripts/01_preprocess_all.py --version paper --artifacts wica
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from src.io_bids import list_subjects, load_raw
from src.preprocess import PreprocessConfig, preprocess_raw
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["paper", "dataset"], default="paper")
    parser.add_argument("--artifacts", choices=["iclabel", "wica"], default="iclabel")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N sujetos (debug)")
    args = parser.parse_args()
    set_seed(args.seed)
    logger = get_logger("preprocess")

    cfg_dict = yaml.safe_load(open(args.config))
    paths = cfg_dict["paths"]
    pp = cfg_dict["preprocess"]

    out_dir = Path(paths["preproc_paper" if args.version == "paper" else "preproc_dataset"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessConfig(
        band_low_hz=pp["band_low_hz"],
        band_high_hz=pp["band_high_hz"],
        notch_hz=pp.get("notch_hz"),
        resample_hz=pp.get("resample_hz"),
        ica_method=pp["ica_method"],
        ica_n_components=pp["ica_n_components"],
        iclabel_threshold=pp["iclabel_threshold"],
        iclabel_reject=tuple(pp["iclabel_reject"]),
        channels_keep=tuple(cfg_dict["dataset"]["channels_19_1020"]),
        average_reference=True,
    )

    subjects = list_subjects(paths["raw"], tuple(cfg_dict["dataset"]["groups_keep"]))
    if args.limit:
        subjects = subjects[: args.limit]
    logger.info(f"Procesando {len(subjects)} sujetos en modo '{args.version}' / '{args.artifacts}'")

    for rec in tqdm(subjects):
        out_path = out_dir / f"{rec.subject_id}_preproc-raw.fif"
        if out_path.exists():
            continue
        try:
            raw = load_raw(rec)
            raw = preprocess_raw(raw, cfg, seed=args.seed, artifact_method=args.artifacts)
            raw.save(out_path, overwrite=True)
        except Exception as e:
            logger.error(f"[{rec.subject_id}] falló: {e}")

    logger.info(f"Listo. Salida en {out_dir}")


if __name__ == "__main__":
    main()
