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

1. **Replicar** el pipeline de Lopes et al. 2023 (*"Using CNN Saliency Maps and EEG Modulation Spectra for Improved and More Interpretable Machine Learning-Based Alzheimer's Disease Diagnosis"*, Computational Intelligence and Neuroscience 2023, art. 3198066, DOI 10.1155/2023/3198066) sobre el dataset público **OpenNeuro ds004504** (Miltiadous et al. 2023) en clasificación binaria AD vs HC.
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
python scripts/00_download_dataset.py --include-derivatives

# 3. Pipeline completo (single-seed básico)
python scripts/01_preprocess_all.py --version paper
python scripts/02_compute_modspec.py --method stft --fs 200
python scripts/02_compute_modspec.py --method cwt  --fs 200
python scripts/03_train_loso.py --method stft --fs 200 --seed 0
python scripts/03_train_loso.py --method cwt  --fs 200 --seed 0

# 4. Saliency POR FOLD (anti-leakage) — Grad-CAM y vainilla, ambos métodos T-F
python scripts/04_extract_saliency_features.py --method stft --fs 200 --seed 0 --grid-search --max-subjects-per-fold 20
python scripts/04_extract_saliency_features.py --method cwt  --fs 200 --seed 0 --grid-search --max-subjects-per-fold 20
python scripts/04_extract_saliency_features.py --method stft --fs 200 --seed 0 --grid-search --max-subjects-per-fold 20 --saliency-method vanilla
python scripts/04_extract_saliency_features.py --method cwt  --fs 200 --seed 0 --grid-search --max-subjects-per-fold 20 --saliency-method vanilla

# 5. SVM con patches por fold (--per-fold-patches es obligatorio para anti-leakage)
python scripts/05_run_svm.py --method stft --fs 200 --seed 0 --per-fold-patches --epochs-per-subject 80
python scripts/05_run_svm.py --method cwt  --fs 200 --seed 0 --per-fold-patches --epochs-per-subject 80
python scripts/05_run_svm.py --method stft --fs 200 --seed 0 --per-fold-patches --epochs-per-subject 80 --saliency-method vanilla
python scripts/05_run_svm.py --method cwt  --fs 200 --seed 0 --per-fold-patches --epochs-per-subject 80 --saliency-method vanilla

# 6. Comparativas + figuras
python scripts/06_compare_stft_cwt.py --classifier cnn
python scripts/06_compare_stft_cwt.py --classifier svm --per-fold
python scripts/06_compare_stft_cwt.py --classifier svm --per-fold --saliency-method vanilla
python scripts/07_generate_figures.py --per-fold
python scripts/07_generate_figures.py --per-fold --saliency-method vanilla

# 7. Multi-seed (opcional, repetir 3-5 con --seed 1 y --seed 2)
python scripts/08_multiseed_analysis.py   # tabla media ± SD entre seeds + Wilcoxon + DeLong
python scripts/09_multiseed_figures.py    # figuras con error bars y boxplots

# 8. Experimentos post-análisis (correlación saliency, Jaccard, bandas, confounders, género, canales)
python scripts/10_post_experiments.py
python scripts/11_gender_stratified.py
python scripts/12_channel_importance.py
```

> **Importante**: `--per-fold-patches` en el SVM es **obligatorio** para anti-leakage. Sin ese flag, el script lee `patch_masks.npy` global (legacy), lo cual puede introducir leakage al usar patches generados con todos los sujetos.

> **Atajos**: `scripts/run_remaining_v2.sh` encadena todo en serie con resume automático.

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

## Resultados finales (full multi-seed, 3 seeds, 2026-07)

Ver [`results/RESULTS_full.md`](results/RESULTS_full.md) y [`docs/INFORME_TFM.md`](docs/INFORME_TFM.md).

**Mejor configuración (SVM vainilla + STFT, paper-faithful):**

| Métrica | Valor (n=3 seeds) |
|---|---|
| Accuracy | **0.764 ± 0.009** |
| F1 macro | **0.760 ± 0.008** |
| AUC | **0.856 ± 0.022** |

Comparación con paper Lopes 2023: el paper reporta Acc 0.71 ± 0.02 en T2 (N vs AD). Este TFM obtiene cifras en el mismo rango (~+5 puntos) sobre datos públicos independientes, lo cual es **el objetivo de una replicación** — no una comparación numérica estricta (datasets y poblaciones diferentes).

**Conclusión (comparación DSP-justa)**: la comparación ingenua STFT vs CWT estaba **confundida por el eje de modulación** (STFT muestrea la envolvente a fs/hop=3.125 Hz → Nyquist mod. 1.56 Hz; la CWT nativa la preserva a 200 Hz → 100 Hz). Se añade la condición de control **CWT-fair** (CWT con el eje de modulación igualado a la STFT). Resultado: al igualar el eje, la brecha aparente STFT↔CWT **se reduce sustancialmente en ambos métodos de saliency** (vainilla ~64%: CWT-fair AUC 0.828 vs STFT 0.856, DeLong p=0.318; Grad-CAM ~36%: p=0.587), y **ninguna comparación justa sobrevive BH-FDR** (q≈0.68). Es decir, **a igualdad de eje de modulación STFT y CWT-Morlet son estadísticamente indistinguibles**: las diferencias aparentes eran **en gran parte** el artefacto de DSP, no la transformada. *Caveat*: CWT-fair "iguala hacia abajo" (descarta modulaciones >1.56 Hz); es una de dos definiciones de comparación justa (la alternativa, subir la STFT con hop fino, queda como trabajo futuro). Detalles en [`docs/INFORME_TFM.md`](docs/INFORME_TFM.md).
