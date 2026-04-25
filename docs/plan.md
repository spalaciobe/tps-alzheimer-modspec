# Plan TFM — Replicación Lopes et al. 2023 sobre OpenNeuro ds004504 + comparación STFT vs CWT

## Contexto

Sebastián Palacio (UNAL Medellín, curso Tópicos en Procesamiento Digital de Señales, prof. Freddy Bolaños) propone un proyecto de 10 semanas (mar–may 2026) con dos contribuciones articuladas:

1. **Validar de forma independiente** el pipeline de Lopes et al. 2023 (*"Using CNN Saliency Maps and EEG Modulation Spectra for Improved and More Interpretable ML-Based Alzheimer's Disease Diagnosis"*, IEEE TNSRE), que fue desarrollado sobre un dataset privado de 54 sujetos, replicándolo sobre el dataset público **OpenNeuro ds004504** (Miltiadous et al. 2023).
2. **Comparar STFT vs CWT-Morlet** como etapa de descomposición tiempo-frecuencia para construir el modulation spectrum, hipotetizando que la CWT — con mejor resolución frecuencial en bajas frecuencias — captura mejor las modulaciones diagnósticas de Alzheimer.

El estado del repositorio en `D:\Universidad\Maestria\TPS\Proyecto\` es inicial: sólo PDFs y markdowns del paper y la propuesta, sin código, sin entorno, sin dataset descargado.

**Pregunta de investigación**: ¿Produce la sustitución de STFT por CWT-Morlet un modulation spectrum con mayor poder discriminativo para AD vs HC, y revela regiones de saliencia (Grad-CAM) distintas a las reportadas por Lopes et al.?

---

## Decisiones tomadas con el usuario

| Tema | Decisión |
|---|---|
| Versión del dataset | **RAW principal** (replicar 0.5–45 Hz + ICA + epoching) + **preprocesada como ablation** |
| Eliminación de artefactos | **ICA Infomax + mne-icalabel** principal; **wICA con pywt** como ablation en sem 9 |
| Clasificación final | **CNN end-to-end + SVM con patches/ANOVA** (réplica fiel del paper completo) |
| Resolución modspec / fs | **Principal a 200 Hz**, modspec 45×45 a 1 Hz para STFT y CWT (fiel a Lopes); **ablation a 500 Hz** para ambas transformadas |
| Saliency | **Grad-CAM** (alineado con la propuesta) + **vanilla gradient** como ablation para comparar con Lopes |
| Tareas | **Binaria AD vs HC** (36 AD + 29 CN = 65 sujetos). FTD excluido. |
| Validación | **LOSO-CV** con agregación epoch→sujeto vía promedio de softmax |

---

## A. Estructura del repositorio

Layout bajo `D:\Universidad\Maestria\TPS\Proyecto\`:

```
Proyecto/
├── data/
│   ├── raw/ds004504/                      # BIDS original
│   ├── derivatives/preprocessed_paper/    # FIFs tras filtro+ICA+resample 200 Hz (réplica)
│   ├── derivatives/preprocessed_dataset/  # cache de la versión derivatives/ del ds004504
│   ├── derivatives/epochs/                # epochs 8s/1s overlap por sujeto
│   ├── derivatives/modspec_stft_200/      # tensores 45×45×19 por sujeto, fs=200
│   ├── derivatives/modspec_cwt_200/
│   ├── derivatives/modspec_stft_500/      # ablation
│   ├── derivatives/modspec_cwt_500/       # ablation
│   └── derivatives/saliency/              # mapas Grad-CAM y vanilla
├── configs/
│   ├── config.yaml                        # rutas, banda, epoch
│   ├── stft.yaml / cwt.yaml               # params específicos por método
│   ├── cnn.yaml                           # arquitectura, optim, sched
│   └── svm.yaml                           # umbrales, clusters, ANOVA, SVM
├── src/
│   ├── io_bids.py                         # carga .set + participants.tsv
│   ├── preprocess.py                      # filtro, ICA+ICLabel, wICA, resample
│   ├── epoching.py                        # 8s, 1s overlap, drop últimos 7s
│   ├── modspec.py                         # compute_modulation_spectrum(x, method)
│   ├── normalize.py                       # z-score por canal con stats de train
│   ├── cache.py                           # IO HDF5 con hash de config
│   ├── datasets.py                        # torch Dataset LOSO-aware
│   ├── models/cnn.py                      # ModSpecCNN (réplica Lopes)
│   ├── train.py                           # loop train/val por fold
│   ├── evaluate.py                        # métricas epoch + sujeto
│   ├── saliency.py                        # Grad-CAM y vanilla, agregación
│   ├── feature_extraction.py              # umbral + KMeans + patches + ANOVA
│   ├── svm_pipeline.py                    # SVM RBF (γ=1/24, C=1)
│   ├── stats.py                           # Wilcoxon pareado, bootstrap CI
│   └── utils/{seed.py,logging.py,viz.py}
├── notebooks/
│   ├── 01_eda_ds004504.ipynb
│   ├── 02_preproc_sanity.ipynb
│   ├── 03_modspec_visualization.ipynb
│   ├── 04_cnn_training_demo.ipynb
│   ├── 05_saliency_inspection.ipynb
│   └── 06_results_stats.ipynb
├── scripts/
│   ├── 00_download_dataset.py
│   ├── 01_preprocess_all.py               # --version paper|dataset
│   ├── 02_compute_modspec.py              # --method stft|cwt --fs 200|500
│   ├── 03_train_loso.py                   # --method --fs --seed
│   ├── 04_extract_saliency_features.py
│   ├── 05_run_svm.py
│   └── 06_compare_stft_cwt.py
├── tests/
│   ├── test_modspec_shapes.py
│   ├── test_epoching_overlap.py
│   ├── test_loso_no_leakage.py
│   └── test_normalization.py
├── results/                               # tablas, figuras, checkpoints, manifest
├── environment.yml
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## B. Entorno y dependencias

- **Python 3.11**, gestionado con **mamba/conda**.
- Núcleo EEG: `mne>=1.6`, `mne-bids>=0.14`, `mne-icalabel>=0.6` (para ICLabel), `pyprep>=0.4` (RANSAC opcional).
- DSP: `numpy>=1.26`, `scipy>=1.11`, `pywavelets>=1.5`.
- ML: `scikit-learn>=1.4`, `torch>=2.2`, `torchvision`, `pytorch-grad-cam>=1.5`, `captum` (vanilla saliency).
- Datos / configs: `h5py`, `pandas`, `pyyaml`, `hydra-core`, `tqdm`, `rich`.
- Viz: `matplotlib`, `seaborn`.
- Test / lint: `pytest`, `ruff`, `black`.
- Descarga: `openneuro-py` (más simple que datalad en Windows).

**Dataset**: `openneuro-py download --dataset ds004504 --target-dir data/raw/`. Tamaño esperado ~3–4 GB (raw + derivatives).

**GPU recomendada** pero no obligatoria: con batch=4 y 65 folds LOSO × 2 métodos × 5 semillas el entrenamiento es viable en CPU (~4 días) o ~10–15 h en GPU consumer.

---

## C. Pipeline de datos

### C.1 Carga BIDS y etiquetas
- `mne_bids.read_raw_bids` o `mne.io.read_raw_eeglab(path_set, preload=True)` para los `.set` de EEGLAB.
- Cargar `participants.tsv` para etiquetas `Group ∈ {A, F, C}` y filtrar A (AD) y C (HC). FTD se excluye en el pipeline binario; opcionalmente reservado para experimento 3-clases en anexo.

### C.2 Preproceso versión "paper" (ruta principal)
Sobre la señal RAW:
1. `raw.set_eeg_reference('average')` (re-referenciado promedio común; el paper usa A1/A2 pero ds004504 viene con su propio reference — re-referenciar a promedio para uniformizar).
2. `raw.filter(l_freq=0.5, h_freq=45, method='fir', fir_design='firwin', phase='zero-double')`.
3. `raw.notch_filter(50)` si se observa contaminación de red.
4. **ICA + ICLabel**: `mne.preprocessing.ICA(method='infomax', fit_params={'extended': True}, n_components=0.99)` → `mne_icalabel.label_components(raw, ica, method='iclabel')` → rechazar componentes con `prob > 0.8` en `{eye, muscle, heart, line_noise, channel_noise}` → `ica.apply(raw)`.
5. `raw.resample(200, npad='auto')` (alinear con el paper).
6. Selección/reordenamiento de los 19 canales 10-20 estándar del ds004504.

### C.3 Preproceso versión "dataset" (ablation)
Cargar directamente desde `derivatives/preprocessed/` del ds004504, resamplear a 200 Hz, mismo reordenamiento de canales. Sin más transformaciones.

### C.4 Preproceso wICA (ablation, sem 9)
Implementar en `preprocess.py:apply_wica()`:
- ICA Infomax → marcar componentes artefacto con ICLabel → sobre cada componente marcada, `pywt.wavedec(comp, 'coif5', level=5)` → umbralizar coeficientes de detalle con `k·σ` (k=1.5) → `pywt.waverec` → recomponer.
- Correr sobre 5–10 sujetos elegidos al azar y comparar modulation spectrums vs ICA+ICLabel (correlación 2D, RMSE).

### C.5 Epoching
- `mne.make_fixed_length_epochs(raw, duration=8.0, overlap=7.0)` produce paso de 1 s ⇒ "8 s con 1 s overlap" en sentido del paper (≥460 epochs/sujeto a 8 min). **Confirmar con sanity check** que el conteo de epochs ≈ 8·60 − 8 + 1 ≈ 473 por sujeto.
- Descartar últimos 7 s por sujeto antes de epochar para evitar leakage.

### C.6 Modulation spectrum (`src/modspec.py`)
**Función**: `compute_modulation_spectrum(epoch, method, target_shape=(45, 45), fs=200)` → `(n_channels=19, F=45, M=45)`.

- Entrada: `epoch (19, n_samples)` con `n_samples = 8·fs`.
- **STFT**: `scipy.signal.stft(window='hann', nperseg=128 a 200 Hz / 320 a 500 Hz, noverlap=nperseg//2)` → `X(f, t)` complejo.
- **CWT**: `pywt.cwt(signal, scales, 'cmor1.5-1.0')` con escalas tales que `pywt.scale2frequency` cubra 0.5–45 Hz logarítmicamente (~50 escalas).
- Envolvente: `np.abs(X)**2` (potencia instantánea).
- FFT temporal: `np.fft.rfft` sobre el eje temporal de la potencia → magnitud.
- Recortes: portadora 0.5–45 Hz, modulación 0–22.5 Hz (Nyquist del paso temporal del spectrogram intermedio).
- **Interpolar a `target_shape=(45,45)`** con `scipy.ndimage.zoom` o `scipy.interpolate.RegularGridInterpolator`.
- Salida en **logaritmo** (10·log10(|.|+ε)) para estabilidad numérica antes de la z-norm.

### C.7 Z-normalización
Por canal y sujeto, con **media y std calculadas SOLO sobre epochs de train del fold**. Aplicar al val/test del fold con esos estadísticos. Implementación en `normalize.py` con un `StandardScaler`-like persistido por fold.

### C.8 Caché
HDF5 por sujeto: `derivatives/modspec_<method>_<fs>/sub-XXX.h5` con datasets `X (n_epochs,19,45,45)`, `y` escalar, `epoch_idx`. Hash de `configs/<method>.yaml` en atributo del archivo para invalidación.

---

## D. Arquitectura CNN (réplica fiel de Lopes)

`src/models/cnn.py:ModSpecCNN`:

- **Input**: `(B, 19, 45, 45)` — los 19 canales EEG entran como `in_channels` de Conv2d (igual que el paper, que usa tensor `45×45×20`).
- **Block 1**: `Conv2d(19→32, 3×3, pad=1)` → ReLU → `Dropout2d(0.85)` → `MaxPool2d(2)`.
- **Block 2**: `Conv2d(32→64, 3×3, pad=1)` → ReLU → `Dropout2d(0.85)` → `MaxPool2d(2)`.
- **Flatten** → `Linear(64·11·11 → 128)` → LeakyReLU(0.1) → Dropout(0.85) → `Linear(128 → 64)` → LeakyReLU(0.1) → Dropout(0.85) → `Linear(64 → 2)`.
- **Regularización**: `weight_decay=1e-2` en optimizer (equivalente al L2=1e-2 del paper).
- **Optimizer**: `Nadam(lr=1e-4)`.
- **Loss**: `CrossEntropyLoss(weight=class_weights)` con `class_weights = [n_total/(2·n_AD), n_total/(2·n_HC)]` para el imbalance 36 vs 29.
- **Batch size**: 4 (como el paper).
- **Epochs**: 50, con early stopping (paciencia 10) sobre F1 macro de validación. El paper no usa early stopping pero lo añadimos para evitar overfitting con tan pocos sujetos.
- **Init**: Kaiming He.
- **Reproducibilidad**: `set_seed(seed)`, `torch.use_deterministic_algorithms(True)`.

---

## E. Saliency / interpretabilidad

`src/saliency.py`:

- **Principal — Grad-CAM** (`pytorch_grad_cam.GradCAM`): capa target = última conv del Block 2. Por fold y por clase, promediar mapas Grad-CAM (resize a 45×45) sobre todas las epochs de train clasificadas correctamente. Salida: `saliency_AD_<method>_<fold>.npy`, agregado global como promedio de folds.
- **Ablation — vanilla gradient** (`captum.attr.Saliency`): replica el método de Lopes ("último dense layer + promedio sobre training samples") para comparación directa con figuras del paper.
- **Visualización** (`utils/viz.py`): heatmap superpuesto al modulation spectrum medio de la clase, ejes etiquetados (`f_carrier`, `f_mod`), bandas canónicas anotadas.
- **Comparación STFT vs CWT**: correlación 2D de Pearson entre mapas promedio + visual side-by-side.

---

## F. Selección de features y SVM (réplica fiel)

`src/feature_extraction.py` y `src/svm_pipeline.py`:

1. Mapa diferencial: `D = mean_modspec_AD - mean_modspec_HC` (por canal o promediado entre canales — el paper promedia 20 saliency maps; aquí promediamos los 19 canales).
2. **Grid search** sobre umbral `p ∈ {80, 82, 84, 86, 88, 90, 92, 94, 96}%` (percentil del saliency map) y `K ∈ {3, 4, 5}` clusters → KMeans sobre coordenadas `(f_carrier, f_mod)` salientes → patches `R_i`.
3. Selección óptima `(p*, K*)` por F1 en validación (no en test).
4. Por epoch y canal: features = `{R_i: potencia media en patch i}` ∪ `{R_i/R_j: ratios}`. Vector de dimensión `19·(K + K·(K-1)/2)`.
5. **ANOVA F-value** sobre train → top **24 features** (`SelectKBest(f_classif, k=24)`).
6. `MinMaxScaler` a [-1, 1] (fit sólo en train).
7. **SVM(kernel='rbf', C=1, gamma=1/24)**, sin tuning (consistente con el paper, evalúa features no clasificador).
8. Predicción por sujeto: promedio de `predict_proba` sobre epochs del sujeto → argmax.

---

## G. Validación LOSO

`src/datasets.py` y `src/train.py`:

- 65 sujetos AD+HC ⇒ 65 folds.
- Dentro de cada fold de train: separar 1 sujeto adicional como val (estratificado por clase) para early stopping y selección de hiperparámetros (umbral, K).
- **Anti-leakage** (cubierto por `tests/test_loso_no_leakage.py`):
  - Z-score y MinMax fit SOLO en train.
  - Drop últimos 7 s por sujeto.
  - Verificar `subject_id` test ∉ train indices.
- **Class imbalance**: `class_weights` en CE; opcional undersample epochs AD a la mediana CN; **NO SMOTE** (epochs son temporalmente correlados).
- **Agregación epoch→sujeto**: `argmax(mean(softmax(logits)))` (preferida) y `majority_vote(argmax_epoch)` (reportar ambas).
- **Métricas reportadas**:
  - Por epoch y por sujeto: Accuracy, F1-macro, Sensibilidad (recall AD), Especificidad (recall HC), AUC.
  - Bootstrap 1000× sobre los 65 sujetos para CI 95%.

---

## H. Comparación STFT vs CWT

Diseño factorial:
- Factor 1: `method ∈ {STFT, CWT}`.
- Factor 2: `fs ∈ {200, 500}` (principal vs ablation).
- Factor 3: `seed ∈ {0,1,2,3,4}` para 5 réplicas.
- Mismas particiones LOSO, misma CNN, mismo schedule.

**Test estadístico** (`src/stats.py`):
- Métrica primaria: F1-macro por sujeto (mediana sobre 5 semillas).
- `scipy.stats.wilcoxon(stft_per_subject, cwt_per_subject, alternative='two-sided')` pareado por sujeto.
- Tamaño de efecto: `r = Z/sqrt(N)` (rank-biserial).
- Corrección BH-FDR si se reportan múltiples métricas.

**Análisis cualitativo**: comparar mapas Grad-CAM agregados entre STFT y CWT, reportar correlación y diferencias en bandas (delta vs alpha vs gamma).

---

## I. Cronograma 10 semanas (mar–may 2026)

| Sem | Hito | Entregable |
|---|---|---|
| 1 (mar 2-8) | Setup repo, env, descarga ds004504, EDA | `notebook 01`, `environment.yml`, `README.md` |
| 2 (mar 9-15) | Pipeline preproceso versión "paper" + ablation versión "dataset" | `io_bids.py`, `preprocess.py`, `epoching.py`, `notebook 02`, tests |
| 3 (mar 16-22) | `modspec.py` STFT y CWT a fs=200, caché HDF5 | `modspec.py`, `notebook 03` con figuras de modspec medio por clase |
| 4 (mar 23-29) | CNN + entrenamiento, primer fold de prueba | `models/cnn.py`, `datasets.py`, `train.py`, `notebook 04` |
| 5 (mar 30-abr 5) | LOSO completo STFT @ 200 Hz, métricas + agregación sujeto | `evaluate.py`, primer informe parcial |
| 6 (abr 6-12) | LOSO completo CWT @ 200 Hz, ajustes de hiperparámetros CWT | resultados intermedios CSV |
| 7 (abr 13-19) | Grad-CAM + vanilla, agregación de mapas, visualizaciones | `saliency.py`, `notebook 05` |
| 8 (abr 20-26) | Pipeline SVM completo (patches+ANOVA+RBF), LOSO ambos métodos | `feature_extraction.py`, `svm_pipeline.py` |
| 9 (abr 27-may 3) | Ablations: fs=500, versión "dataset", wICA en 5-10 sujetos. Wilcoxon, bootstrap CI | `notebook 06`, tablas finales, figuras paper-ready |
| 10 (may 4-10) | Redacción TFM, lock de seeds, defensa | informe + presentación + repo limpio |

---

## J. Archivos críticos a crear

Rutas absolutas (orden de implementación por sem 1-4):

- `D:\Universidad\Maestria\TPS\Proyecto\environment.yml`
- `D:\Universidad\Maestria\TPS\Proyecto\configs\config.yaml`
- `D:\Universidad\Maestria\TPS\Proyecto\src\io_bids.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\preprocess.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\epoching.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\modspec.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\normalize.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\datasets.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\models\cnn.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\train.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\evaluate.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\saliency.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\feature_extraction.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\svm_pipeline.py`
- `D:\Universidad\Maestria\TPS\Proyecto\src\stats.py`
- `D:\Universidad\Maestria\TPS\Proyecto\tests\test_loso_no_leakage.py`

---

## K. Verificación / criterios de éxito

**Sanity checks**:
- PSD de canales control con forma 1/f y pico alpha en O1/O2 (`notebook 02`).
- Modulation spectrum HC: pico marcado en `(f_carrier≈10 Hz, f_mod≈0–1 Hz)`. AD: atenuación de ese pico, aumento delta/theta (`notebook 03`).
- Conteo de epochs por sujeto ≥ 460 (alineado con el paper).

**Cifras esperadas (binario AD vs HC, ds004504, LOSO)**:
- CNN end-to-end: **Accuracy 80–90%, F1 0.80–0.88, AUC 0.88–0.95** (binario más fácil que el T1 multiclase del paper).
- SVM con 24 features: **Accuracy 75–85%, F1 0.75–0.83** (referencia paper T2: 71%, T5: 89%).
- Si ambas vías quedan <70% en val ⇒ bug (revisar normalización, leakage, etiquetas).

**Tests unitarios mínimos** (`tests/`):
- `test_modspec_shapes.py`: shape `(19,45,45)` determinista para STFT y CWT, sin NaN.
- `test_epoching_overlap.py`: número de epochs `= floor((duracion - 8)/1) + 1`.
- `test_loso_no_leakage.py`: `set(test_subjects) ∩ set(train_subjects) == ∅`; scaler no vio test.
- `test_normalization.py`: media≈0, std≈1 en train post z-score.

**Reproducibilidad**:
- `set_seed(seed)` global.
- `manifest.json` por run en `results/<run_id>/` con commit hash, config.yaml hash, métricas, métricas por fold.
- `environment.yml` con versiones pineadas al cierre del proyecto.

---

## L. Riesgos y mitigaciones

1. **Definición ambigua de "8 s con 1 s overlap"** — verificar con conteo ≥460 epochs/sujeto; si no cuadra, probar paso=1s (overlap=7s).
2. **Imbalance 36/29** — class_weights + reportar tanto métricas balanceadas (F1 macro, AUC) como sensibilidad/especificidad por separado.
3. **Resolución CWT a 200 Hz**: 200 Hz puede ser justo para CWT en bajas frecuencias — la ablation a 500 Hz es la red de seguridad.
4. **wICA bug** — limitarlo a 5–10 sujetos en sem 9 y comparar contra ICA+ICLabel; no es camino crítico.
5. **Tiempo de cómputo LOSO** — 65 folds × 2 métodos × 2 fs × 5 semillas × 50 epochs = ~50 h GPU, ~5–7 d CPU. Si no hay GPU, reducir a 3 semillas en `fs=500` ablation.
6. **Hiperparámetros wICA y umbral patches** — fijar antes de tocar el test set; documentar en `configs/`.
7. **Condición C (banco de filtros)** — declarada como stretch goal en la propuesta. Mantener fuera del camino crítico; agregar sólo si sem 9 termina antes.
