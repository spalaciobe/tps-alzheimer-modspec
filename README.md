# Replicación Lopes et al. 2023 — CNN saliency + EEG modulation spectrum sobre OpenNeuro ds004504

Proyecto del curso *Tópicos en Procesamiento Digital de Señales* (UNAL Medellín, prof. Freddy Bolaños).

## ⚠️ Disclaimer clínico

Este proyecto es una **replicación académica de investigación** sobre el dataset público OpenNeuro ds004504. Los modelos entrenados son **únicamente** para validación metodológica y reproducibilidad de Lopes et al. (2023).

**No están aprobados para uso clínico directo. No realizan diagnósticos médicos válidos.** Cualquier aplicación clínica requiere validación en cohortes externas, aprobación regulatoria (FDA/EMA/INVIMA), consentimiento informado y supervisión médica cualificada.

## 🤖 Uso de IA generativa

La generación inicial de código, configuración del pipeline, debugging de leakage en LOSO-CV y auditoría metodológica fueron asistidos por **Claude Code (Anthropic, 2026)**. Todo el código fue revisado, ejecutado y validado por el autor antes del commit.

## 📜 Licencia

Código bajo **MIT License** (ver [`LICENSE`](LICENSE)). Dataset OpenNeuro ds004504 bajo **CC0 1.0 Universal**.

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

## Resultados finales (full multi-seed, 3 seeds, 2026-06)

Ver [`results/RESULTS_full.md`](results/RESULTS_full.md) y [`docs/INFORME_TFM.md`](docs/INFORME_TFM.md).

**Mejor configuración (SVM vainilla + STFT, paper-faithful):**

| Métrica | Valor (n=3 seeds) |
|---|---|
| Accuracy | **0.764 ± 0.009** |
| F1 macro | **0.762 ± 0.011** |
| AUC | **0.856 ± 0.022** |

Replicación: el paper Lopes 2023 reporta Acc 0.71 ± 0.02 en T2 (N vs AD). Este TFM supera por +5 puntos con anti-leakage estricto y multi-seed.

**Conclusión central**: la hipótesis original (CWT > STFT) NO se confirma. Con saliency vainilla (paper-faithful), STFT supera a CWT con significancia (DeLong p=0.014). Más detalles en [`docs/INFORME_TFM.md`](docs/INFORME_TFM.md).
