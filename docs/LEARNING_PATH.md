# Learning path — entender el proyecto desde nivel básico de ingeniería

Esta es una guía progresiva (de fundamentos a detalles del proyecto) en dos pistas paralelas:

- 🎚️ **Pista A — Señales**: del EEG al modulation spectrum.
- 💻 **Pista B — Código**: del Python básico al pipeline LOSO con CNN + SVM.

Cada nivel tiene **objetivo**, **conceptos clave**, **referencias** y **ejercicio práctico** sobre el repo.

> **Cómo usar esta guía**: avanza nivel por nivel. Cada uno asume el anterior. Los ejercicios son optativos pero anclan los conceptos al código real de este proyecto.

---

## 🎚️ Pista A — Señales: de EEG al diagnóstico

### Nivel A1 — ¿Qué es una señal EEG?

**Objetivo**: entender qué mide el EEG y cómo se digitaliza.

**Conceptos**:
- **Electroencefalograma**: voltaje superficial del cuero cabelludo (~µV) producido por actividad sincronizada de poblaciones de neuronas piramidales.
- **Electrodos 10-20**: 19-21 puntos estandarizados (Fp1, F3, Cz, ...). En este proyecto usamos los 19 del ds004504.
- **Frecuencia de muestreo**: ds004504 graba a **500 Hz** (puede capturar hasta 250 Hz por Nyquist; nosotros decidimos resamplear a 200 Hz porque la información clínica de Alzheimer está en 0.5–45 Hz).
- **Resolución bits**: 12 bits ≈ 4096 niveles. No la cambiamos.
- **Bandas canónicas**: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma (>30 Hz).

**En este proyecto**:
- 65 sujetos AD+HC, ~10–13 min de grabación cada uno, ojos cerrados (resting state).
- Etiquetas en `data/raw/ds004504/participants.tsv` (columna `Group`: A=AD, F=FTD, C=Control).

**Ejercicio**: en `notebooks/01_eda_ds004504.ipynb`, mira cómo se carga un sujeto con MNE-Python:
```python
from src.io_bids import list_subjects, load_raw
subs = list_subjects('data/raw/ds004504', groups_keep=('A', 'C'))
raw = load_raw(subs[0])
raw.plot()        # señal cruda
raw.compute_psd().plot()  # densidad espectral
```

**Referencias**: Niedermeyer's *Electroencephalography* (libro clásico); MNE-Python tutorial.

---

### Nivel A2 — Procesamiento básico: filtrado y artefactos

**Objetivo**: limpiar la señal antes de analizarla.

**Conceptos**:
- **Filtro pasa-banda**: deja pasar solo frecuencias de interés. Aquí 0.5–45 Hz porque por debajo hay drift y por encima hay ruido muscular/red.
- **Filtro FIR vs IIR**: usamos FIR (Finite Impulse Response) porque permite **fase cero** (forward+backward), evitando distorsión temporal.
- **Notch 50/60 Hz**: elimina interferencia de red eléctrica.
- **Artefactos**: parpadeos (Fp1/Fp2 grandes a baja frecuencia), EMG (muscular, alta frecuencia), saltos de electrodo.
- **ICA (Independent Component Analysis)**: descompone EEG en componentes estadísticamente independientes; algunas son cerebrales, otras artefactos.
- **ICLabel**: red neuronal pre-entrenada (Pion-Tonachini 2019) que clasifica cada componente ICA en `brain/eye/muscle/heart/line_noise/channel_noise/other`.

**En este proyecto** (`src/preprocess.py`):
1. `apply_bandpass(raw, 0.5, 45)` — FIR fase cero.
2. `apply_notch(raw, 50)` — opcional.
3. `apply_ica_iclabel(raw, cfg)` — ICA Infomax + ICLabel rechaza componentes con probabilidad ≥0.8 de ser artefacto.
4. `raw.resample(200)` — bajamos a 200 Hz (alinea con paper).

**Por qué no wICA** (lo que usa el paper Lopes): wICA aplica thresholding wavelet sobre componentes artefacto en lugar de descartarlas. Es más sofisticado pero requiere ajustar wavelet/umbral. Implementamos ICA+ICLabel (más reproducible) como pipeline principal y wICA como ablation (`apply_wica()`).

**Ejercicio**: en `notebooks/02_preproc_sanity.ipynb`, compara la PSD antes y después del filtrado. Verifica que veas un pico alpha (~10 Hz) más claro en O1/O2 (occipital) en sujetos HC que en AD.

**Referencias**: Bell & Sejnowski 1995 (Infomax ICA); Pion-Tonachini et al. 2019 (ICLabel).

---

### Nivel A3 — Tiempo-frecuencia: STFT y CWT

**Objetivo**: representar cómo cambia el contenido frecuencial a lo largo del tiempo.

**Conceptos**:
- **Transformada de Fourier (FT)**: descompone señal en suma de senoidales. Pierde la dimensión temporal.
- **Short-Time Fourier Transform (STFT)**: aplica FT a ventanas deslizantes. Da mapa `X(t, f)`.
  - Ventana **Hann** (suaviza bordes), nperseg=128 a 200 Hz → resolución frecuencial = 200/128 ≈ 1.56 Hz, resolución temporal = 128/200 = 0.64 s.
  - **Heisenberg**: no puedes tener buena resolución en tiempo Y frecuencia simultáneamente. STFT da resolución **uniforme** fijada por la ventana.
- **Continuous Wavelet Transform (CWT)**: en lugar de ventana fija, usa **wavelet madre** que se escala. Da resolución **adaptativa**: alta resolución frecuencial en bajas frecuencias, alta resolución temporal en altas.
  - **Wavelet Morlet compleja**: una senoidal compleja modulada por una gaussiana. Ideal para EEG por buena localización tiempo-frecuencia.
  - `cmor1.5-1.0`: B=1.5 (bandwidth), C=1.0 (center freq).

**En este proyecto** (`src/modspec.py`):
- `_stft_tf(x, fs=200, window='hann', nperseg=128, noverlap=64)` → matriz STFT.
- `_cwt_tf(x, fs=200, wavelet='cmor1.5-1.0', n_scales=32, f_low=0.5, f_high=45)` → matriz CWT.

**Hipótesis del proyecto**: como AD concentra alteraciones en bajas frecuencias (delta/theta), CWT debería capturar mejor esas modulaciones que STFT (resolución uniforme). Ese es justamente el experimento.

**Ejercicio**: en `notebooks/03_modspec_visualization.ipynb`, plotea STFT vs CWT del mismo epoch. Observa que STFT es una rejilla regular y CWT muestra mayor detalle en bajas Hz.

**Referencias**: Oppenheim & Schafer *Discrete-Time Signal Processing* (Cap. 10 STFT); Mallat *A Wavelet Tour of Signal Processing* (Cap. 4 CWT).

---

### Nivel A4 — Modulation spectrum: la idea clave

**Objetivo**: entender el "espectrograma de espectrogramas" que es el núcleo del paper.

**Conceptos**:
- **Periodicidades de segundo orden**: el EEG no solo tiene frecuencias (alpha 10 Hz), sino también **modulaciones** de esas frecuencias (la potencia alpha sube y baja con cierta cadencia, ~1 Hz).
- **Modulation spectrum** (Atlas & Shamma 2003): aplica FT **dos veces**:
  1. Primera FT (en tiempo) → STFT/CWT, da `X(t, f_carrier)`.
  2. Calcular **potencia instantánea** `P(t, f) = |X(t, f)|²`.
  3. Segunda FT (sobre el eje temporal de la potencia) → `M(f_carrier, f_mod)`.
- **Resultado**: matriz 2D donde
  - Eje vertical: frecuencia portadora (carrier) — ej. "alpha 10 Hz".
  - Eje horizontal: frecuencia de modulación — ej. "alpha que se modula a 1 Hz".
  - Un valor alto en `(10, 1)` significa "potencia alpha que oscila a 1 Hz".

**En el paper de Lopes**: el modulation spectrum tiene **45×45 bins a 1 Hz de resolución**. Los autores observaron que esa representación 2D contiene patrones discriminativos para AD que las bandas canónicas no capturan.

**En este proyecto** (`src/modspec.py:compute_modulation_spectrum_subject`):
1. T-F para todo el sujeto (STFT o CWT) → `P(t, f)`.
2. Para cada epoch (8 s con paso 1 s), `np.fft.rfft(P_epoch, axis=1)` sobre la potencia instantánea.
3. Recorte a carrier 0.5–45 Hz, modulation 0–22.5 Hz.
4. Resize a 45×45 con `scipy.ndimage.zoom`.
5. Log-power `10·log10(|·|+ε)` para estabilidad.

**Ejercicio**: ejecuta `compute_modulation_spectrum` sobre un epoch sintético con dos componentes (10 Hz modulado a 1 Hz + 4 Hz puro). Verifica que el pico esté en `(10, 1)` para el componente modulado.

**Referencias**: Atlas & Shamma 2003 *Joint acoustic and modulation frequency*; Falk et al. 2010 *Spectro-temporal modulation analysis for AD diagnosis*.

---

### Nivel A5 — Epoching y validación cruzada

**Objetivo**: dividir la señal en muestras manejables sin filtrar información entre train/test.

**Conceptos**:
- **Epoching**: cortar la señal en segmentos cortos (8 s aquí) que se procesan independientemente.
- **Solapamiento**: cada nuevo epoch comienza 1 s después del anterior → solapamiento de 7 s. Genera más muestras pero **estas no son independientes** (comparten 7/8 = 87.5% de los datos).
- **LOSO-CV (Leave-One-Subject-Out)**: para evaluar generalización, dejas un sujeto entero como test, entrenas en los otros 64. Repites 65 veces (uno por sujeto).
- **Anti-leakage**: NUNCA mezclar epochs de un mismo sujeto entre train y test (LOSO lo garantiza). NUNCA calcular estadísticos de normalización con datos de test.
- **Drop últimos 7 s**: el paper descarta el último segmento para evitar que el último epoch test "vea" datos del siguiente fold.

**En este proyecto** (`src/datasets.py`):
- `loso_splits()` produce 65 tuplas `(test, [train])`.
- `hold_out_val()` extrae 1 sujeto por clase de train para early stopping.
- `IndexedDataset` filtra el `SubjectBank` por máscara booleana.
- Z-score (`fit_channel_zscore`) **solo** sobre train del fold actual.

**Pitfall importante**: el solapamiento del 87.5% significa que si reportas "N=460 epochs", en realidad tienes ~58 muestras estadísticamente independientes. Tests como Wilcoxon a nivel epoch son sospechosos. **A nivel sujeto (LOSO) está bien**.

**Ejercicio**: lee `tests/test_loso_no_leakage.py` y `tests/test_normalization.py`. Verifica que entiendes por qué cada assert es necesario.

**Referencias**: cualquier libro de ML, Hastie et al. *Elements of Statistical Learning* Cap. 7.

---

### Nivel A6 — CNN, saliency y SVM final

**Objetivo**: clasificador completo del paper.

**Conceptos**:
- **CNN (Convolutional Neural Network)**: capas que aplican filtros aprendidos sobre la imagen 2D del modulation spectrum. Detecta patrones locales (manchas, bordes) que crecen jerárquicamente.
- **Arquitectura** (paper Lopes Tabla 3): input 19×45×45 → 2 conv 3×3 ReLU + maxpool → 3 fully-connected LeakyReLU → output 2 clases. Dropout 85% (alto, para regularizar).
- **Saliency map**: heatmap que indica qué píxeles del input fueron más decisivos para la clasificación.
  - **Vanilla gradient** (paper original): `|∂y/∂x|` evaluado por píxel.
  - **Grad-CAM** (este proyecto): pondera mapas de activación de la última conv por sus gradientes. Más localizado pero da resoluciones más bajas.
- **Patches saliency-guided**: umbralizamos el saliency map (>p%, ej. 88%), aplicamos KMeans (K=4) sobre los píxeles activos, y obtenemos K regiones (patches) discriminativas.
- **SVM RBF**: support vector machine con kernel gaussiano. Usa como features la potencia media en cada patch + ratios entre patches (todo por canal, 19 canales).
  - C=1, γ=1/24 (≈ 1/n_features), MinMaxScaler [-1, 1], top-24 features por ANOVA F-value.
- **Por qué SVM además de CNN?**: el paper argumenta que las features compactas (24 valores) capturan la esencia y dan mejor interpretabilidad clínica.

**En este proyecto**:
- CNN: `src/models/cnn.py:ModSpecCNN`.
- Saliency: `src/saliency.py:gradcam_per_class` y `vanilla_saliency_per_class`.
- Patches y features: `src/feature_extraction.py:find_patches`, `features_for_epoch`.
- SVM: `src/svm_pipeline.py:fit_svm_pipeline`.

**Resultados quick** (`results/RESULTS_quick.md`):
- SVM STFT: Acc 0.77, AUC 0.85.
- SVM CWT: Acc 0.69, AUC 0.74.
- Wilcoxon STFT vs CWT pareado: p=0.34 (no significativo).

**Ejercicio**: lee `notebooks/05_saliency_inspection.ipynb`. Ve los heatmaps Grad-CAM superpuestos al modulation spectrum medio.

**Referencias**: Selvaraju et al. 2017 (Grad-CAM); Cortes & Vapnik 1995 (SVM); Lopes et al. 2023 (paper original).

---

## 💻 Pista B — Código: del Python básico al pipeline completo

### Nivel B1 — Python científico: numpy, scipy, matplotlib

**Objetivo**: dominar los building blocks numéricos.

**Conceptos**:
- **numpy arrays**: arrays N-dimensionales con operaciones vectorizadas (sin bucles for).
- **dtype**: `float32` (mitad de memoria que float64, suficiente para EEG).
- **scipy.signal**: `stft`, `welch`, `butter`, `cwt`. Aquí usamos `stft` y `pywt.cwt`.
- **matplotlib**: `imshow` para spectrograms, `axhline` para anotar bandas.

**En este proyecto**:
- Todos los modspecs son arrays `(n_epochs, 19, 45, 45)` en float32.
- Almacenados en HDF5 con compresión gzip (`src/cache.py`).

**Ejercicio**: lee `src/utils/viz.py:plot_modspec`. Entiende por qué usamos `extent=` para que los ejes muestren Hz reales en lugar de índices.

**Referencias**: McKinney *Python for Data Analysis*; numpy quickstart.

---

### Nivel B2 — Estructura de un proyecto Python: paquetes y configs

**Objetivo**: organizar código de forma escalable.

**Conceptos**:
- **Paquete**: directorio con `__init__.py`. `src/` es nuestro paquete.
- **`pip install -e .`**: instala el paquete en modo editable (cambios al código se reflejan sin reinstalar).
- **Scripts CLI con argparse**: `scripts/01_preprocess_all.py --version paper` parsea argumentos de línea de comandos.
- **Configs YAML**: separar parámetros del código. Cargar con `yaml.safe_load(open('config.yaml'))`.
- **Hash de config**: `cache.py:config_hash` produce un SHA1 corto de la config; se guarda en HDF5 para invalidar caché si los params cambian.

**En este proyecto**:
- `pyproject.toml` declara el paquete y dependencias.
- `configs/` tiene 5 YAMLs: global, stft, cwt, cnn, svm.
- 7 scripts numerados en `scripts/` (00 download → 06 compare).

**Ejercicio**: corre `python scripts/02_compute_modspec.py --help`. Mira cómo argparse genera la documentación automáticamente.

---

### Nivel B3 — MNE-Python: el ecosistema EEG

**Objetivo**: cargar, filtrar, hacer ICA con MNE.

**Conceptos clave de la API**:
- `mne.io.read_raw_eeglab(path)` → `Raw` object.
- `raw.filter(l, h)` — pasa-banda.
- `raw.notch_filter(f)` — elimina interferencia.
- `raw.set_eeg_reference('average')` — re-referenciado.
- `raw.resample(fs_new)` — resampleo con anti-aliasing.
- `mne.preprocessing.ICA(method='infomax').fit(raw)` → ICA.
- `mne.make_fixed_length_epochs(raw, duration, overlap)` → epochs.
- `mne_icalabel.label_components(raw, ica, method='iclabel')` → clasificación de componentes.

**En este proyecto**:
- `src/io_bids.py:load_raw` envuelve `read_raw_eeglab`.
- `src/preprocess.py:preprocess_raw` orquesta filter+ICA+resample.
- `src/epoching.py:make_epochs` envuelve `make_fixed_length_epochs` con drop de últimos 7 s.

**Ejercicio**: en una notebook, carga un sujeto y ejecuta cada paso del preproceso visualizando con `raw.plot()` entre pasos. Identifica visualmente parpadeos (Fp1/Fp2 grandes, baja freq) y verifica que ICA los elimina.

**Referencias**: MNE-Python tutorials (oficial).

---

### Nivel B4 — PyTorch: deep learning desde cero

**Objetivo**: entender Module, Tensor, autograd, training loop.

**Conceptos**:
- **Tensor**: como numpy array, pero con autograd y aceleración GPU.
- **`nn.Module`**: clase base para arquitecturas. Definir `__init__` (capas) y `forward(x)`.
- **`nn.Conv2d(in_ch, out_ch, kernel_size, padding)`**: convolución 2D. `in_ch=19` porque tenemos 19 canales EEG.
- **`nn.MaxPool2d(2)`**: reduce dimensiones a la mitad.
- **`nn.Dropout2d(0.85)`**: anula 85% de los canales aleatoriamente durante entrenamiento (regularización fuerte).
- **`nn.Linear(in, out)`**: capa fully-connected.
- **`nn.LeakyReLU(0.1)`**: activación que permite gradiente pequeño en valores negativos.
- **Optimizer**: `torch.optim.NAdam(model.parameters(), lr=1e-4, weight_decay=1e-2)` actualiza pesos. Nadam = Adam con Nesterov momentum.
- **Loss**: `nn.CrossEntropyLoss(weight=class_weights)` para clasificación con desbalance.
- **AMP (Automatic Mixed Precision)**: usa float16 donde es seguro, float32 donde necesita precisión. ~2× speedup en GPUs Ampere (RTX 3050+).

**En este proyecto**:
- `src/models/cnn.py:ModSpecCNN` implementa la arquitectura del paper.
- `src/train.py:train_cnn` es el training loop con early stopping y AMP.

**Ejercicio**: lee `src/train.py` línea por línea. Identifica:
1. ¿Dónde se hace `model.train()` vs `model.eval()`?
2. ¿Por qué `optim.zero_grad(set_to_none=True)`?
3. ¿Cómo decide el early stopping cuándo parar?

**Referencias**: PyTorch tutorial (oficial); Goodfellow *Deep Learning* libro.

---

### Nivel B5 — DataLoaders y datasets en PyTorch

**Objetivo**: alimentar la GPU eficientemente.

**Conceptos**:
- **`Dataset`**: define `__len__` y `__getitem__(idx)`. Devuelve un sample.
- **`DataLoader`**: orquesta el batch, shuffle, pin_memory, num_workers.
- **`pin_memory=True`**: aloja batches en RAM page-locked → transferencia CPU→GPU más rápida.
- **`num_workers > 0`**: carga batches en procesos paralelos. En Windows con Python 3.13 puede dar problemas; aquí usamos `num_workers=0`.

**En este proyecto** (`src/datasets.py`):
- `SubjectBank.from_paths(h5_paths)` carga TODOS los modspecs a RAM una vez.
- `IndexedDataset(bank, mask, stats)` filtra por máscara booleana (sin recargar).
- Esto evita el cuello de botella de leer 64 archivos HDF5 cada fold (15 min/fold → 4 min/fold).

**Ejercicio**: corre el siguiente script y observa cuánta RAM usa el bank:
```python
from src.datasets import SubjectBank
from pathlib import Path
bank = SubjectBank.from_paths(sorted(Path('data/derivatives/modspec_stft_200').glob('*.h5')))
print(f'X.shape={bank.X.shape}, RAM={bank.X.nbytes/1e9:.2f} GB')
```

---

### Nivel B6 — Validación cruzada y leakage

**Objetivo**: implementar LOSO sin trampas.

**Conceptos**:
- **Leakage**: cualquier información del test que se filtra al train.
  - Ejemplo: calcular media y std globales y aplicarlas — está mal porque el test contribuyó a la media.
  - Solución: `fit_channel_zscore(X_train)` SOLO con train; `apply_channel_zscore(X_test, train_stats)`.
- **Anti-leakage por sujeto** en LOSO: garantizar que `subject_id` de test ∉ train_indices.

**En este proyecto** (`tests/test_loso_no_leakage.py`):
```python
def test_loso_disjoint():
    paths = _fake_paths()
    splits = loso_splits(paths)
    for test_path, train_paths in splits:
        assert test_path not in train_paths
```

**Ejercicio**: introduce intencionalmente un bug (e.g., calcular stats sobre X_train+X_test en `IndexedDataset`) y verifica que `test_zscore_no_leakage_apply_with_train_stats` falla. Luego corrige.

---

### Nivel B7 — Saliency y feature engineering guiado por datos

**Objetivo**: extraer features interpretables del CNN.

**Conceptos**:
- **Grad-CAM**: capa target = última conv. Output: heatmap del tamaño del feature map (luego se interpola).
- **Aggregation**: el saliency promedio sobre TODAS las muestras de train de una clase da el mapa "típico" de esa clase.
- **Diferencial**: `saliency_AD - saliency_HC` resalta regiones discriminativas.
- **KMeans clustering**: agrupa píxeles activos en K patches espaciales contiguos.
- **ANOVA F-value**: por feature mide separabilidad entre clases. `SelectKBest(f_classif, k=24)` toma las 24 mejores.
- **MinMaxScaler [-1, 1]**: escala features para SVM.

**En este proyecto**:
- `src/saliency.py:gradcam_per_class` produce un mapa por clase.
- `src/feature_extraction.py:find_patches` umbraliza + KMeans.
- `src/feature_extraction.py:features_for_epoch` extrae potencias y ratios.
- `src/svm_pipeline.py:fit_svm_pipeline` entrena SVM con MinMax + RBF.

**Ejercicio**: visualiza los 4 patches encontrados:
```python
import numpy as np
masks = np.load('data/derivatives/saliency/stft_200_seed0_quick/patch_masks.npy')
print(masks.shape)  # (4, 45, 45)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for i, m in enumerate(masks): axes[i].imshow(m, origin='lower'); axes[i].set_title(f'Patch {i}')
```

---

### Nivel B8 — Estadística para comparar modelos

**Objetivo**: dar peso estadístico a las afirmaciones.

**Conceptos**:
- **Bootstrap CI95**: muestrea con reemplazo N veces, calcula la métrica, toma percentiles 2.5 y 97.5.
- **Wilcoxon signed-rank pareado**: test no-paramétrico para diferencias entre dos métodos sobre los mismos sujetos.
- **DeLong test**: específico para diferencias en AUC.
- **Multiple comparisons**: si haces muchos tests, corrige (Bonferroni, BH-FDR).
- **Power analysis**: con N=65 sujetos, ¿qué tamaño de efecto puedes detectar?

**En este proyecto** (`src/stats.py`):
- `wilcoxon_paired(a, b)` con z-score y rank-biserial r.
- `bootstrap_ci(values, n_boot=1000, ci=0.95)`.
- `benjamini_hochberg(pvals, alpha=0.05)`.

**Pitfall del proyecto**: el solapamiento del 87.5% entre epochs invalida tests asumiendo IID a nivel epoch. A nivel sujeto (un score por sujeto agregando epochs) está bien.

**Ejercicio**: corre `python scripts/06_compare_stft_cwt.py --quick --classifier svm` y lee `results/compare_svm_200_seed0_quick.json`. Verifica que entiendes los CIs y el p-value reportado.

---

## 🧭 Diagrama de flujo del pipeline completo

```
                   ┌──────────────────────────┐
                   │  ds004504 (.set EEGLAB)  │
                   │  500 Hz, 19 canales      │
                   └─────────────┬────────────┘
                                 │ src/io_bids.py
                                 ▼
                   ┌──────────────────────────┐
                   │ Filter 0.5–45 Hz FIR     │
                   │ Notch 50 Hz              │
                   │ ICA + ICLabel            │  src/preprocess.py
                   │ Resample 200 Hz          │
                   └─────────────┬────────────┘
                                 ▼
                   ┌──────────────────────────┐
                   │ Epochs 8s / paso 1s      │
                   │ Drop últimos 7 s         │  src/epoching.py
                   │ ~470 epochs / sujeto     │
                   └─────────────┬────────────┘
                                 ▼
                   ┌──────────────────────────┐
                   │  T-F → STFT  o  CWT      │
                   │  |X|² → FFT temporal     │  src/modspec.py
                   │  Resize → 19×45×45       │
                   └─────────────┬────────────┘
                                 │ HDF5 cache
                                 ▼
                   ┌──────────────────────────┐
                   │ LOSO-CV (65 folds)       │  scripts/03_train_loso.py
                   │ z-score per fold         │  src/datasets.py
                   │ CNN train (Nadam, AMP)   │  src/train.py + models/cnn.py
                   │ Predict per subject      │  src/evaluate.py
                   └─────────────┬────────────┘
                                 ▼
                   ┌──────────────────────────┐
                   │ Grad-CAM per class       │  src/saliency.py
                   │ Aggregate over folds     │  scripts/04_*
                   │ Diferencial AD - HC      │
                   │ Threshold + KMeans       │
                   │ → patch_masks (45×45)    │
                   └─────────────┬────────────┘
                                 ▼
                   ┌──────────────────────────┐
                   │ Features = potencia      │
                   │ patches + ratios         │  src/feature_extraction.py
                   │ ANOVA top-24             │
                   │ SVM RBF (γ=1/24, C=1)    │  src/svm_pipeline.py
                   └─────────────┬────────────┘
                                 ▼
                   ┌──────────────────────────┐
                   │ Wilcoxon STFT vs CWT     │  src/stats.py
                   │ Bootstrap CI95           │  scripts/06_*
                   │ Figuras (ROC, etc.)      │  scripts/07_*
                   └──────────────────────────┘
```

---

## 📚 Roadmap de lectura recomendada

### Para entender el problema (Alzheimer + EEG)
1. WHO 2023 Dementia fact sheet.
2. Trambaiolli et al. 2011 — primer paper de modulation EEG-AD.
3. Fraga et al. 2013 — caracterización de severidad.

### Para entender el método (modulation spectrum)
1. Atlas & Shamma 2003 — joint acoustic and modulation freq.
2. Falk et al. 2010 — spectro-temporal modulation analysis.
3. Cassani 2020 — patches en modulation spectrum.

### Para entender CNN saliency
1. Simonyan et al. 2014 — vanilla saliency (paper original).
2. Selvaraju et al. 2017 — Grad-CAM.
3. Lopes et al. 2023 — paper que estamos replicando.

### Para entender el dataset
1. Miltiadous et al. 2023 — descriptor del ds004504.

### Para Python científico
1. McKinney *Python for Data Analysis*.
2. MNE-Python tutorials (mne.tools/stable/auto_tutorials).
3. PyTorch tutorial (pytorch.org/tutorials).

### Para estadística
1. Hastie et al. *Elements of Statistical Learning*, Cap. 7 (CV).
2. Efron & Tibshirani *Introduction to the Bootstrap*.

---

## 🛠️ Path práctico — orden sugerido para correr y entender

1. `notebooks/01_eda_ds004504.ipynb` — explora los datos crudos.
2. `notebooks/02_preproc_sanity.ipynb` — filtrado e ICA paso a paso.
3. `notebooks/03_modspec_visualization.ipynb` — STFT vs CWT lado a lado.
4. Lee `src/modspec.py` línea por línea.
5. `notebooks/04_cnn_training_demo.ipynb` — entrena la CNN con datos sintéticos (rápido).
6. Corre el pipeline real: `01_preprocess → 02_compute_modspec → 03_train_loso --quick`.
7. `notebooks/05_saliency_inspection.ipynb` — visualiza Grad-CAM.
8. `04_extract_saliency_features → 05_run_svm → 06_compare → 07_generate_figures`.
9. Lee `results/RESULTS_quick.md` y `docs/AUDIT.md`.
10. Si tienes dudas, vuelve a este learning path en el nivel correspondiente.

---

## ✋ Preguntas que deberías poder responder al terminar

**Señales**:
- ¿Por qué filtramos a 0.5–45 Hz y no a 1–100 Hz?
- ¿Qué diferencia hay entre frecuencia portadora y frecuencia de modulación?
- ¿Por qué 8 s de epoch y no 4 s o 16 s?
- ¿Por qué CWT podría ser mejor que STFT en este problema (hipotéticamente)?

**Código**:
- ¿Por qué cargamos el `SubjectBank` una sola vez en lugar de archivo por archivo cada fold?
- ¿Qué hace `torch.amp.autocast('cuda', dtype=torch.float16)` y por qué es seguro?
- ¿Cómo el `IndexedDataset` evita leakage entre train y test?
- ¿Qué tamaño tiene un modulation spectrum por sujeto y por qué (en GB)?

**Estadística**:
- ¿Por qué Wilcoxon pareado y no t-test?
- ¿Por qué el solapamiento 87.5% es un problema para tests a nivel epoch?
- ¿Qué dice un AUC de 0.85 en términos clínicos?

Si puedes responder estas, **dominas el proyecto a nivel de ingeniería**. 🎉
