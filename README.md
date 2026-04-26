# Replicación Lopes et al. 2023 — CNN saliency + EEG modulation spectrum sobre OpenNeuro ds004504

Proyecto del curso *Tópicos en Procesamiento Digital de Señales* (UNAL Medellín, prof. Freddy Bolaños).

## Objetivo

1. **Replicar** el pipeline de Lopes et al. 2023 (*"Using CNN Saliency Maps and EEG Modulation Spectra for Improved and More Interpretable ML-Based Alzheimer's Disease Diagnosis"*, IEEE TNSRE) sobre el dataset público **OpenNeuro ds004504** (Miltiadous et al. 2023) en clasificación binaria AD vs HC.
2. **Comparar STFT vs CWT-Morlet** como etapa de descomposición tiempo-frecuencia para construir el modulation spectrum.

## Setup

### Opción A — venv + pip (sin conda)

```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows bash
pip install -r requirements.txt
# Para GPU NVIDIA (CUDA 12.x), reinstalar torch con wheels CUDA:
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Opción B — mamba/conda

```bash
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

## Resultados (quick mode, 2026-04-25)

Ver [`results/RESULTS_quick.md`](results/RESULTS_quick.md). Pipeline end-to-end validado:

- **SVM con patches saliency-guided**:
  - STFT: Acc 0.77, AUC 0.85
  - CWT:  Acc 0.69, AUC 0.74
  - Wilcoxon STFT vs CWT: p = 0.34 (no significativo)

- **CNN end-to-end** (sub-entrenado en quick mode): cerca de azar.

Para el run completo (epochs=50 sin subsample), usar Colab T4 — ver [`docs/optimization_options.md`](docs/optimization_options.md).
