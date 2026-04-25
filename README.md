# Replicación Lopes et al. 2023 — CNN saliency + EEG modulation spectrum sobre OpenNeuro ds004504

Proyecto del curso *Tópicos en Procesamiento Digital de Señales* (UNAL Medellín, prof. Freddy Bolaños).

## Objetivo

1. **Replicar** el pipeline de Lopes et al. 2023 (*"Using CNN Saliency Maps and EEG Modulation Spectra for Improved and More Interpretable ML-Based Alzheimer's Disease Diagnosis"*, IEEE TNSRE) sobre el dataset público **OpenNeuro ds004504** (Miltiadous et al. 2023) en clasificación binaria AD vs HC.
2. **Comparar STFT vs CWT-Morlet** como etapa de descomposición tiempo-frecuencia para construir el modulation spectrum.

## Setup

```bash
# 1. Crear entorno
mamba env create -f environment.yml
mamba activate tps-alzheimer

# 2. Descargar dataset (~3-4 GB)
python scripts/00_download_dataset.py

# 3. Pipeline completo
python scripts/01_preprocess_all.py --version paper
python scripts/02_compute_modspec.py --method stft --fs 200
python scripts/02_compute_modspec.py --method cwt --fs 200
python scripts/03_train_loso.py --method stft --fs 200 --seed 0
python scripts/03_train_loso.py --method cwt --fs 200 --seed 0
python scripts/04_extract_saliency_features.py
python scripts/05_run_svm.py
python scripts/06_compare_stft_cwt.py
```

## Estructura

- `src/` — módulos del pipeline (carga, preproc, modspec, modelos, train, eval, saliency, features, SVM, stats).
- `scripts/` — entrypoints CLI.
- `configs/` — YAML con rutas, params de modspec, CNN, SVM.
- `notebooks/` — EDA y visualizaciones.
- `tests/` — tests unitarios anti-leakage y de shape.
- `data/` — dataset y derivados (gitignored).
- `results/` — métricas, figuras, checkpoints (gitignored).

## Plan completo

Ver [`docs/plan.md`](docs/plan.md).
