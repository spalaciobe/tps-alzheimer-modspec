"""Descarga el dataset OpenNeuro ds004504 a data/raw/.

Uso: python scripts/00_download_dataset.py [--target data/raw]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", default="ds004504")
    parser.add_argument("--include-derivatives", action="store_true",
                        help="Incluir versión preprocesada del dataset")
    args = parser.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)

    try:
        import openneuro
    except ImportError as e:
        raise SystemExit(
            "openneuro-py no está instalado. "
            "Activa el env: `mamba activate tps-alzheimer`"
        ) from e

    include = ["sub-*"] if not args.include_derivatives else ["sub-*", "derivatives/*"]
    print(f"Descargando {args.dataset} a {args.target} (include={include})")
    openneuro.download(dataset=args.dataset, target_dir=str(args.target / args.dataset), include=include)
    print("Descarga completa")


if __name__ == "__main__":
    main()
