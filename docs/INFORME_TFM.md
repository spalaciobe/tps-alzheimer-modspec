# Informe TFM — Replicación de Lopes et al. 2023 sobre OpenNeuro ds004504 y comparación STFT vs CWT

**Autor**: Sebastián Palacio Betancur
**Programa**: Maestría · Universidad Nacional de Colombia — Sede Medellín · Facultad de Minas
**Curso**: Tópicos en Procesamiento Digital de Señales · Profesor: Freddy Bolaños
**Fecha**: junio 2026
**Repositorio**: https://github.com/spalaciobe/tps-alzheimer-modspec

---

## Resumen ejecutivo

Este trabajo replica de forma independiente el pipeline de Lopes et al. (2023) [*Using CNN Saliency Maps and EEG Modulation Spectra for Improved and More Interpretable Machine Learning-Based Alzheimer's Disease Diagnosis*, Computational Intelligence and Neuroscience 2023, art. 3198066; DOI: 10.1155/2023/3198066] sobre el dataset público **OpenNeuro ds004504** (Miltiadous 2023), y extiende el análisis comparando **STFT vs CWT-Morlet** como etapa de descomposición tiempo-frecuencia.

**Resultados principales (LOSO-CV 65 sujetos, 3 seeds):**

| Configuración | Acc | F1 | AUC |
|---|---|---|---|
| SVM con saliency **vainilla + STFT** (paper-faithful) | **0.764 ± 0.009** | **0.762 ± 0.011** | **0.856 ± 0.022** |
| SVM con saliency vanilla + CWT | 0.677 ± 0.081 | 0.673 ± 0.094 | 0.778 ± 0.038 |
| SVM con saliency Grad-CAM + STFT | 0.662 ± 0.031 | 0.643 ± 0.041 | 0.713 ± 0.020 |
| SVM con saliency Grad-CAM + CWT | 0.703 ± 0.009 | 0.687 ± 0.026 | 0.800 ± 0.010 |
| CNN end-to-end + STFT | 0.656 ± 0.024 | 0.642 ± 0.020 | 0.695 ± 0.022 |
| CNN end-to-end + CWT | 0.626 ± 0.071 | 0.611 ± 0.070 | 0.590 ± 0.069 |

**Conclusiones clave (con matices estadísticos explícitos):**

1. **La replicación independiente del pipeline funciona**: SVM con saliency vainilla y STFT alcanza Acc 0.764 ± 0.009, AUC 0.856 ± 0.022, en el rango del 0.71 ± 0.02 reportado por Lopes (T2: N vs AD). La comparación es orientativa (datasets y poblaciones distintas), no equivalencia estricta.
2. **La hipótesis original (CWT > STFT) no obtiene evidencia a favor, pero la inversa tampoco**: con saliency vainilla (la configuración más fiel al paper), CWT presenta AUC media menor (0.778 ± 0.038 vs 0.856 ± 0.022) y mayor varianza entre seeds. La inferencia formal (DeLong AUC por seed + combinación Stouffer/Fisher) da **p ≈ 0.061 (NS)**, y tras corrección BH-FDR sobre las 3 comparaciones principales **ninguna sobrevive** (q ≥ 0.10). Lectura conservadora: **no hay evidencia estadísticamente concluyente** en ninguna dirección.
3. **Confound DSP crítico (detectado en revisión externa)**: los ejes de modulación de STFT (Nyquist 1.56 Hz, 13 bins reales interpolados a 45) y CWT (Nyquist 100 Hz, 180 bins subsampleados a 45) NO codifican el mismo contenido espectral. La comparación STFT vs CWT en este TFM está confundida con esta asimetría de DSP, no controlada por diseño (ver §3.3, §5.1, §5.5).
4. **El método de saliency afecta cualitativamente el ranking**: con Grad-CAM la tendencia se invierte (CWT > STFT en AUC, no significativo, p=0.195). Esto sugiere que las comparaciones STFT vs CWT son sensibles al método de saliency usado; la conclusión "X supera a Y" depende del pipeline completo, no solo de la transformada T-F.
5. **Análisis exploratorio (post-hoc)**: STFT + vainilla concentra ~89% de saliency en bandas alpha (61%) + theta (28%), coherente con literatura EEG-AD (Fraga 2013). Los canales más informativos son occipito-temporales (O1, O2, T5, T6). Estos hallazgos son post-hoc y deben validarse en otros datasets. **La atribución por banda no se reporta para CWT** porque su eje portador es geomspace y la asignación píxel→banda requeriría rehacer el mapeo (ver §5.5).
6. **Limitaciones reconocidas**: solo 3 seeds (poder estadístico modesto), grid search de patches heurístico (no nested CV), resize 45×45 puede favorecer STFT, ejes de modulación incomparables entre métodos, sesgo de género en el dataset (χ² p=0.039) con poder limitado para descartarlo en el modelo (**ausencia de evidencia, no evidencia de ausencia**), baja consistencia de patches entre folds (Jaccard ≈ 0.05), saliency maps de STFT y CWT **estadísticamente ortogonales** (r ≈ −0.09 vainilla, −0.24 Grad-CAM) lo que sugiere también **complementariedad** (ensemble como hipótesis futura).

---

## 1. Introducción

### 1.1 Contexto clínico

La enfermedad de Alzheimer (EA) es la causa más común de demencia, con más de 55 millones de personas afectadas globalmente. El diagnóstico temprano es crítico para intervención terapéutica. Las técnicas estándar de imagen (RM, PET) son costosas y poco accesibles. **El EEG es no invasivo, portátil y económico**, lo cual lo posiciona como herramienta promisoria para tamizaje.

### 1.2 Espectrograma de modulación

El **espectrograma de modulación** del EEG captura periodicidades de segundo orden: cómo varía temporalmente la energía de cada componente espectral. Formalmente, dado un mapeo tiempo-frecuencia $X(t,f)$:

$$M(f, f_m) = \mathcal{F}_t\{|X(t,f)|^2\}$$

donde $f$ es la frecuencia portadora y $f_m$ la frecuencia de modulación. Trambaiolli (2011), Fraga (2013), Cassani (2020) y Lopes (2023) han demostrado consistentemente que esta representación captura biomarcadores discriminativos para EA.

### 1.3 Trabajo de referencia: Lopes et al. 2023

Lopes et al. propusieron entrenar una CNN sobre el modulation spectrum y usar **mapas de saliencia** (vanilla gradient) para descubrir regiones discriminativas ("patches"). Sobre estos patches calculan potencia + ratios como features para un SVM final. Reportan accuracy ~0.71 en LOSO test sobre N=39 sujetos privados.

### 1.4 Objetivos de este trabajo

1. **Validar de forma independiente** el pipeline de Lopes sobre el dataset público OpenNeuro ds004504.
2. **Comparar STFT vs CWT-Morlet** como etapa de descomposición tiempo-frecuencia.
3. **Cuantificar la varianza por seed** con multi-seed LOSO-CV (3 seeds).
4. **Analizar la interpretabilidad** de los patches descubiertos en términos de bandas canónicas.

### 1.5 Pregunta de investigación

> ¿Produce la sustitución de la STFT por la CWT con wavelet de Morlet un espectrograma de modulación con mayor poder discriminativo para EA, y revela regiones de saliencia distintas a las reportadas por Lopes et al.?

---

## 2. Datos

**OpenNeuro ds004504** (Miltiadous 2023): 88 sujetos con EEG en reposo (ojos cerrados):

| Grupo | N | Edad (media ± SD) | MMSE |
|---|---|---|---|
| **A** (Alzheimer, AD) | 36 | 66.4 ± 7.9 | 17.8 ± 4.5 |
| **C** (Healthy Control, HC) | 29 | 67.9 ± 5.4 | 30.0 ± 0.0 |
| F (Frontotemporal Dementia) | 23 | — | — |

Este TFM utiliza solo los grupos **A y C** (clasificación binaria AD vs HC, 65 sujetos). El grupo F (FTD) se excluyó del análisis principal.

**Características técnicas**:
- 19 canales 10-20 (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz).
- Frecuencia de muestreo nativa: 500 Hz → resampleada a 200 Hz (alineación con Lopes).
- Duración registros: ~10–13 min por sujeto.
- Licencia: CC0 1.0 (Miltiadous et al., 2023).

**Análisis de confounders** (AD vs HC):

| Variable | AD | HC | Test | p |
|---|---|---|---|---|
| Edad | 66.4 ± 7.9 | 67.9 ± 5.4 | t-test | **0.384** (NS) |
| MMSE | 17.8 ± 4.5 | 30.0 ± 0.0 | t-test | <0.001 (esperado) |
| Género (F/M) | 24/12 | 11/18 | χ² | **0.039** (SIG) |

Hay un **sesgo de género** en el dataset (más mujeres en AD). Se discute en la sección de limitaciones.

---

## 3. Pipeline implementado

El pipeline replica el de Lopes et al. con las siguientes etapas:

### 3.1 Preproceso EEG
1. **Re-referenciado**: promedio común (CAR).
2. **Filtro pasa-banda**: FIR fase cero, 0.5–45 Hz.
3. **Eliminación de artefactos**: ICA Infomax + ICLabel (rechazo de componentes con prob ≥ 0.8 para `eye blink`, `muscle artifact`, `heart beat`, `line noise`, `channel noise`). *Desviación del paper*: Lopes usa wICA; ICLabel es una alternativa reproducible y estándar en MNE-Python.
4. **Resample** a 200 Hz.
5. **Anonymize** con `mne.io.Raw.anonymize(daysback=10000)`.

### 3.2 Epoching
- Duración: 8 s, paso 1 s (overlap 7 s) → ~470 epochs/sujeto.
- Descarte de los últimos 7 s para evitar leakage entre folds.

### 3.3 Modulation spectrum

- **STFT**: ventana Hann, nperseg=128, noverlap=64 a fs=200 Hz.
  - Resolución frecuencial portadora real: Δf = fs/nperseg = **1.5625 Hz** (no 1 Hz nominal).
  - **Frecuencia de muestreo de la envolvente** (eje temporal de la FFT de modulación): dt = noverlap/fs = 64/200 = 0.32 s ⇒ fs_t ≈ 3.125 Hz ⇒ **Nyquist de modulación = 1.5625 Hz** (en un epoch de 8 s genera ~24 frames temporales, ~13 bins de modulación útiles antes del crop a 22.5 Hz).
- **CWT-Morlet** (`cmor1.5-1.0`, 32 escalas log en [0.5, 45] Hz):
  - **Frecuencia de muestreo de la envolvente**: 200 Hz (CWT preserva la fs original) ⇒ **Nyquist de modulación = 100 Hz**, ~180 bins disponibles en 0-22.5 Hz.
- **Procesamiento común**: magnitud al cuadrado → FFT temporal → recorte (0.5-45 Hz portadora, 0-22.5 Hz mod) → **resize bilinear a 45×45** (alineación con la arquitectura CNN de Lopes) → log-power.

> **⚠️ CONFOUND DSP IMPORTANTE — Detectado en revisión externa (no en la implementación inicial)**:
>
> Las dos imágenes "45×45" finales **NO representan el mismo rango físico** del eje de modulación:
> - STFT: 0 a **1.5625 Hz** (interpolado de 13 bins reales a 45 vía bilineal).
> - CWT: 0 a **22.5 Hz** (subsampled de ~180 bins a 45).
>
> Como consecuencia, la comparación cuantitativa STFT vs CWT bajo este pipeline NO es una comparación justa entre transformadas T-F: es una comparación entre **dos representaciones que capturan rangos de modulación drásticamente distintos**. La STFT aquí captura solo modulaciones lentas (alpha/theta envelope dynamics ≤ 1.56 Hz), mientras que CWT incluye modulaciones rápidas (hasta 22.5 Hz).
>
> Esto es un **confound más serio que el resize 45×45 per se** y debe interpretarse así: **las conclusiones cuantitativas STFT vs CWT en este informe reflejan un pipeline específico con eje de modulación efectivamente distinto, no una comparación equitativa de las transformadas T-F en sí**. Ver §5.5 (limitación 4) y §6 (conclusiones revisadas).
>
> Mitigación correcta para trabajo futuro: decimar la envolvente CWT a la misma tasa de muestreo (3.125 Hz) antes de la FFT temporal, o computar la STFT con hop mucho más fino para igualar el Nyquist.

### 3.4 CNN — réplica fiel de la arquitectura del paper

| Capa | Detalle |
|---|---|
| Input | (19, 45, 45) |
| Conv2D #1 | 32 filtros, 3×3, ReLU, padding 1 |
| Dropout 2D | 0.85 |
| MaxPool | 2×2 |
| Conv2D #2 | 64 filtros, 3×3, ReLU, padding 1 |
| Dropout 2D | 0.85 |
| MaxPool | 2×2 |
| Flatten + FC#1 | 128 unidades, LeakyReLU(0.1), Dropout 0.85 |
| FC#2 | 64 unidades, LeakyReLU(0.1), Dropout 0.85 |
| FC#3 (output) | 2 clases (AD/HC) |

- **Optimizer**: NAdam, lr=1e-4, weight_decay=1e-2 (L2 ≈ 1e-2 del paper).
- **Batch size**: 128 (GPU permite mayor que el batch 4 del paper; resultados estables).
- **Epochs**: 50 con early stopping (paciencia 10) sobre F1 macro de val.
- **AMP (mixed precision)** en GPU.
- **Class weights** balanceados.

### 3.5 Saliency maps por fold (anti-leakage estricto)

Para cada fold del LOSO:
1. Carga el modelo entrenado en ese fold.
2. Para cada sujeto de train del fold (subset estratificado de 20):
   - **Grad-CAM** (Selvaraju 2017) sobre la última conv → mapa 45×45 por clase.
   - O bien **vanilla saliency** (Simonyan 2014) — paper-faithful — gradiente |∂y/∂x|.
3. Promedia por clase → `saliency_AD` y `saliency_HC` por fold.
4. `saliency_diff = saliency_AD - saliency_HC`.
5. **Selección de threshold/K por fold (heurístico)** sobre threshold ∈ {80, 82, ..., 96}% y K ∈ {3, 4, 5} para KMeans → patches por fold. *Nota metodológica*: la implementación efectiva (`scripts/04_extract_saliency_features.py:209-240`) maximiza la separabilidad de la saliency map (max contraste AD vs HC sobre píxeles candidatos), NO una validación nested con SVM en val set. La función `grid_search_patches()` con validación SVM real existe en `src/feature_extraction.py:87-125` pero no es invocada por el pipeline final. Esto introduce un selection bias menor — ver limitación 5.5.2.
6. Guarda `patch_masks_fold_NN.npy`.

### 3.6 SVM con patches saliency-guided
1. Por epoch y canal: feature = potencia media en cada patch + ratios de potencia entre patches del mismo canal.
2. **Selección de features**: ANOVA F-value top-24 sobre train.
3. **MinMaxScaler** a [-1, 1].
4. **SVM RBF** con γ = 1/24 features, C = 1, sin tuning (alineado con Lopes).

### 3.7 Validación
- **LOSO-CV** estricta con 65 folds.
- En cada fold, 1 sujeto adicional retenido para validación (early stopping).
- Z-score por canal con stats SOLO de train.
- Saliency, patches y SelectKBest **por fold** (anti-leakage).
- 3 semillas (s0, s1, s2) por configuración → reportar media ± SD.

### 3.8 Stack tecnológico
- Python 3.13, PyTorch 2.6+cu124, NVIDIA RTX 3050 (4 GB VRAM).
- MNE-Python 1.12, mne-icalabel 0.8.
- scipy 1.17, scikit-learn 1.8.
- pytorch-grad-cam 1.5.5, captum 0.9.
- Reproducibilidad: `set_seed`, `torch.use_deterministic_algorithms`, `requirements-lock.txt`.

---

## 4. Resultados

### 4.1 CNN end-to-end (3 seeds)

| Método | Acc | F1 macro | Sensibilidad | Especificidad | AUC |
|---|---|---|---|---|---|
| STFT | 0.656 ± 0.024 | 0.642 ± 0.020 | 0.741 ± 0.029 | 0.552 ± 0.034 | 0.695 ± 0.022 |
| CWT | 0.626 ± 0.071 | 0.611 ± 0.070 | 0.602 ± 0.060 | 0.655 ± 0.087 | 0.590 ± 0.069 |

**DeLong por seed** (combinado Stouffer): AUC STFT − CWT ≈ +0.10, p = 0.266 (NS).

La CNN sola opera apenas mejor que el "baseline trivial" (0.554 si siempre predice AD). Esto es coherente con su rol en el pipeline: la CNN no es el clasificador final, sino el extractor de regiones discriminativas vía saliency. Su accuracy modesta es esperada con dropout 0.85.

### 4.2 SVM con patches Grad-CAM (3 seeds)

| Método | Acc | F1 | AUC |
|---|---|---|---|
| STFT | 0.662 ± 0.031 | 0.643 ± 0.041 | 0.713 ± 0.020 |
| CWT | **0.703 ± 0.009** | **0.687 ± 0.026** | **0.800 ± 0.010** |

**DeLong por seed** (combinado Stouffer): AUC CWT − STFT ≈ +0.09, p = 0.068 (NS).

Con Grad-CAM, CWT tiende a ser mejor que STFT, pero la diferencia no alcanza significancia estadística con 3 seeds.

### 4.3 SVM con patches vainilla (paper-faithful) — **Resultado principal** (3 seeds)

| Método | Acc | F1 | AUC |
|---|---|---|---|
| **STFT** | **0.764 ± 0.009** | **0.762 ± 0.011** | **0.856 ± 0.022** |
| CWT | 0.677 ± 0.081 | 0.673 ± 0.094 | 0.778 ± 0.038 |

#### 4.3.1 Inferencia estadística: DeLong por seed + combinación

La comparación de AUC entre STFT y CWT se hace con el test de DeLong pareado **dentro de cada seed por separado** (n=65 sujetos en cada test LOSO), y luego se combinan los tres p-values entre seeds con los métodos de Stouffer y Fisher. Este es el procedimiento correcto: preserva la variabilidad real entre inicializaciones en lugar de colapsarla en un ensemble.

| seed | AUC STFT | AUC CWT | Δ AUC | DeLong p |
|---|---|---|---|---|
| s0 | 0.843 | 0.815 | +0.028 | 0.660 |
| s1 | 0.844 | 0.739 | +0.105 | 0.051 |
| s2 | 0.881 | 0.781 | +0.101 | 0.072 |

**Combinación de p-values entre seeds (SVM vainilla, STFT vs CWT):**
- Stouffer: z = 1.55, p combinado = **0.061**
- Fisher: χ² = 12.03, p combinado = **0.061**

Ningún seed individual alcanza significancia al α=0.05, y la combinación tampoco (p ≈ 0.061). El AUC medio mayor de STFT (0.856 vs 0.778) y su menor varianza (±0.022 vs ±0.038) son consistentes con una posible ventaja, pero **la inferencia formal no alcanza significancia**.

#### 4.3.2 Corrección por múltiples comparaciones (BH-FDR)

Las tres comparaciones principales (CNN, SVM Grad-CAM, SVM vainilla) se corrigen por multiplicidad con Benjamini-Hochberg (α=0.05) sobre los p-values combinados (Stouffer):

| Comparación | p (Stouffer) | BH-FDR q | Sig. tras corrección |
|---|---|---|---|
| SVM vainilla STFT vs CWT | 0.061 | 0.102 | ✗ |
| SVM Grad-CAM STFT vs CWT | 0.068 | 0.102 | ✗ |
| CNN STFT vs CWT | 0.266 | 0.266 | ✗ |

**Ninguna comparación sobrevive BH-FDR** (q ≥ 0.10). La lectura definitiva es **ausencia de evidencia concluyente para superioridad de cualquier método T-F en este pipeline**.

#### 4.3.3 Wilcoxon por seed individual (consistencia)

Wilcoxon pareado de scores por sujeto en cada seed individual (SVM vainilla, STFT vs CWT) corrobora la falta de un efecto consistente:

| seed | Wilcoxon p |
|---|---|
| s0 | 0.893 (NS) |
| s1 | 0.025 (sig sin corregir) |
| s2 | 0.338 (NS) |

Solo en seed=1 hay diferencia significativa, perdida con Bonferroni intra-seed (α=0.0167). La señal no se replica entre semillas.

**Lectura final**: **no se encuentra evidencia estadísticamente concluyente** de diferencia entre STFT y CWT en este pipeline. La hipótesis original (CWT > STFT) no obtiene evidencia a favor, pero **tampoco se prueba lo contrario con rigor estadístico**.

### 4.4 Comparación con el paper Lopes 2023

| Métrica | Paper Lopes (T2: N vs AD, LOSO test) | Este TFM (SVM vainilla STFT, 3 seeds) |
|---|---|---|
| N sujetos | 39 (20 N + 19 AD1) | 65 (29 HC + 36 AD) |
| Accuracy | 0.71 ± 0.02 | **0.764 ± 0.009** |
| F1 | 0.61 ± 0.02 | **0.762 ± 0.011** |
| AUC | no reportado | **0.856 ± 0.022** |

**El TFM obtiene cifras en el rango del paper o ligeramente superiores**, pero la comparación NO es estricta: los datasets son distintos (privado vs ds004504), las poblaciones difieren en demografía/MMSE, y los anti-leakage no son idénticos (este TFM usa saliency y patches por fold, lo que el paper original no documenta explícitamente). Lo que se puede afirmar es que el pipeline **funciona en el mismo orden de magnitud** sobre datos públicos independientes, lo cual es el objetivo principal de una replicación.

### 4.5 Análisis de saliency: correlación 2D entre métodos

Pearson r entre saliency maps `saliency_diff` (3 seeds, media ± SD):

| Comparación | r |
|---|---|
| **STFT vs CWT** con Grad-CAM | **-0.237 ± 0.119** |
| **STFT vs CWT** con vainilla | **-0.086 ± 0.076** |
| **Grad-CAM vs vainilla** con STFT | +0.114 ± 0.043 |
| **Grad-CAM vs vainilla** con CWT | -0.166 ± 0.106 |

**Interpretación**: los saliency maps STFT y CWT NO son consistentes — son **estadísticamente ortogonales** (Pearson r ≈ −0.09 vainilla, −0.24 Grad-CAM; ningún p<0.05 para r distinto de 0). Hablar de "anti-correlación" sería sobreinterpretar el signo: las correlaciones son estadísticamente indistinguibles de cero. Lo mismo ocurre entre Grad-CAM y vainilla. Esto demuestra que **"el biomarcador descubierto" depende fuertemente de la elección de pipeline**.

### 4.6 Análisis por banda canónica (post-hoc, exploratorio)

> **Nota metodológica**: este análisis y el siguiente (4.7) son **exploratorios post-hoc** sobre las saliency maps agregadas. No estaban pre-registrados como hipótesis, y la elección del umbral top-10% es arbitraria. Los resultados deben interpretarse como observaciones descriptivas para guiar futuras hipótesis, no como prueba de un biomarcador validado.

Proporción de píxeles top-10% saliency en cada banda canónica (media entre seeds):

| Configuración | δ (0.5–4 Hz) | θ (4–8 Hz) | α (8–13 Hz) | β (13–30 Hz) | γ (30–45 Hz) |
|---|---|---|---|---|---|
| **STFT vainilla** | 5.8% | **27.9%** | **61.1%** | 5.3% | 0.0% |
| CWT vainilla | 0.8% | 2.0% | 7.6% | 54.2% | 35.5% |
| STFT Grad-CAM | 0.0% | 0.0% | 0.0% | 0.7% | 99.3% |
| CWT Grad-CAM | 0.0% | 0.0% | 0.0% | 67.5% | 30.9% |

**STFT con saliency vainilla concentra ~89% de la señal en bandas alpha (61.1%) + theta (27.9%)**, alineado con el biomarcador clínico clásico de EA (atenuación alfa, aumento theta; Fraga 2013, Cassani 2020). Las otras configuraciones se concentran en bandas beta/gamma altas, menos canónicas para EA.

Este resultado **sugiere** que STFT vainilla podría estar aprendiendo información clínicamente conocida, lo que es consistente con su mejor AUC media. Sin embargo, dada la naturaleza post-hoc del análisis y la baja consistencia de patches entre folds (sección 4.8, Jaccard ≈ 0.05), debe verse como una **observación correlacional**, no como prueba causal de que STFT vainilla "encuentra el biomarcador correcto".

### 4.7 Importancia por canal (post-hoc, exploratorio)

> **Nota metodológica**: análisis post-hoc descriptivo. Se agrega ANOVA F-score por canal sobre los features (potencia + ratios) usados por el SVM, lo cual indica qué canales aportan más información discriminativa al modelo final, pero NO prueba causalidad clínica.

Para el SVM vainilla STFT (mejor configuración), se agregó la importancia ANOVA F-score por canal (suma de potencia + ratios sobre los patches del fold). **Ranking entre seeds (n=3, normalizado a suma=1):**

| Ranking | Canal | Score normalizado | Región |
|---|---|---|---|
| 1 | **O2** | 0.110 ± 0.004 | Occipital derecho |
| 2 | **T5** | 0.110 ± 0.006 | Temporal posterior izquierdo |
| 3 | **O1** | 0.103 ± 0.005 | Occipital izquierdo |
| 4 | T6 | 0.074 ± 0.003 | Temporal posterior derecho |
| 5 | T3 | 0.060 ± 0.015 | Temporal medio izquierdo |
| ... | ... | ... | ... |
| 19 | C4 | 0.014 ± 0.004 | Central derecho |

**Interpretación neurofisiológica**: los canales más informativos son **occipitales (O1, O2) y temporales posteriores (T5, T6, T3)**. Esto es coherente con la literatura clínica de EEG-AD:

- **Atenuación del ritmo alpha occipital**: O1/O2 son las regiones con mayor potencia alpha en reposo; su disminución es uno de los biomarcadores más documentados de EA (Fraga 2013).
- **Atrofia temporal medial**: los lóbulos temporales son los primeros afectados en EA (hipocampo + entorhinal cortex). EEG sobre T3/T5/T6 detecta esos cambios.
- **Áreas centrales (C3, C4, Cz, T4) menos importantes**: típicamente menos afectadas en fases tempranas.

Este patrón es **convergente** con el análisis de bandas canónicas (sección 4.6) — STFT vainilla pondera más fuerte regiones occipito-temporales donde tradicionalmente se manifiesta el alfa-theta característico de EA. Es un indicio prometedor de validez clínica, pero debe **validarse en datasets independientes** antes de afirmar que el modelo "descubre el biomarcador clásico".

Figura: `results/figures_multiseed/channel_importance.png`.

### 4.8 Consistencia de patches entre folds (Jaccard)

Jaccard entre máscaras de patches de pares aleatorios de folds (200 pares por configuración):

| Configuración | Jaccard medio ± SD |
|---|---|
| STFT Grad-CAM | 0.050–0.063 |
| STFT vainilla | 0.036–0.051 |
| CWT Grad-CAM | 0.042–0.045 |
| CWT vainilla | 0.026–0.030 |

**Los patches descubiertos son extremadamente inestables entre folds** (Jaccard ≈ 0.03–0.06). Esto contradice la narrativa de "biomarcadores reproducibles": cada fold descubre regiones distintas, aunque las métricas globales sean buenas. Es un hallazgo metodológico importante a discutir.

---

## 5. Discusión

### 5.1 ¿CWT supera a STFT?

**Respuesta matizada: no se encuentra evidencia a favor de CWT bajo esta configuración**. La hipótesis original (CWT > STFT por mejor resolución en bajas frecuencias) no obtiene soporte, pero **tampoco se prueba formalmente la inferioridad intrínseca de CWT**. Cuatro líneas de observación:

1. **DeLong AUC por seed + combinación** (SVM vainilla): p por seed = 0.660, 0.051, 0.072; Stouffer/Fisher combinado = 0.061 (NS). Tras BH-FDR sobre las 3 comparaciones principales, ninguna sobrevive (q ≥ 0.10).
2. **Wilcoxon por seed**: 0.893, 0.025, 0.338 — inconsistente entre semillas, sin replicación clara del efecto.
3. **Varianza entre seeds**: STFT vainilla tiene SD baja (acc ±0.009, AUC ±0.022), CWT vainilla muy alta (acc ±0.081, AUC ±0.038), indicando que CWT es menos estable bajo este pipeline.
4. **Pipeline-dependent**: con saliency Grad-CAM la tendencia se invierte (CWT > STFT en AUC, p=0.195 NS).

**Por qué CWT puede estar penalizada en esta implementación, sin que necesariamente sea inferior**:

- **🔴 Eje de frecuencia de modulación incomparable (confound DSP detectado en revisión externa)**: STFT (con noverlap=64 a fs=200 Hz) tiene Nyquist de modulación ≈1.56 Hz, mientras que CWT tiene Nyquist 100 Hz. Tras el recorte a 22.5 Hz y resize a 45 bins, **las dos imágenes representan rangos físicos distintos** del eje de modulación (STFT: 0-1.56 Hz interpolado; CWT: 0-22.5 Hz subsampled). Esto es más serio que el resize per se y posiblemente el factor dominante de las diferencias observadas.
- **Resize bilinear a 45×45**: iguala artificialmente las resoluciones nominales de salida pero, combinado con el desajuste anterior, produce dos representaciones distintas etiquetadas como equivalentes.
- **CWT con `cmor1.5-1.0` y 32 escalas log**: una elección entre muchas posibles, sin tuning específico ni cone-of-influence.
- **Hiperparámetros del CNN** (Lopes 2023) fueron optimizados para input STFT; CWT puede requerir tuning específico.
- **Las saliency maps de STFT y CWT están casi descorrelacionadas** (r ≈ -0.09 vainilla, -0.24 Grad-CAM): no se debe interpretar como "anti-correlación" (r es estadísticamente indistinguible de 0); más bien indica **ortogonalidad/independencia**. Dada la inestabilidad de patches entre folds (Jaccard ≈ 0.05), parte de este r ≈ 0 puede ser puro ruido fold-a-fold, no estructura complementaria real.

**Lectura honesta y revisada**: este trabajo NO sostiene una comparación equitativa entre transformadas T-F en sí mismas, sino entre **dos pipelines específicos** (uno con eje de modulación efectivamente ≤1.56 Hz, otro con eje ≤22.5 Hz). La afirmación defendible es: "bajo el pipeline implementado, CWT no muestra ventaja sobre STFT y es menos estable entre seeds". Cualquier generalización a "CWT-Morlet vs STFT en EEG-AD" requiere igualar los ejes de modulación primero.

### 5.2 El método de saliency cambia el ranking

**Hallazgo metodológico relevante**: con saliency Grad-CAM, CWT tiende a tener mejor AUC media; con vainilla (paper-faithful), STFT tiene mejor AUC media. Los saliency maps de los dos métodos no se parecen (r ≈ 0.1 STFT, -0.17 CWT), lo que indica que **descubren regiones distintas del modulation spectrum**.

Esto implica que comparaciones del tipo "transformada X supera transformada Y" son sensibles al método de saliency y al pipeline completo. Para evaluar transformadas T-F de forma aislada del método de interpretabilidad sería deseable comparar también con CNN end-to-end sin patches/SVM (donde las diferencias son más pequeñas y NS, sección 4.1).

### 5.3 Replicación del paper

Sobre el dataset público ds004504, el SVM vainilla con STFT alcanza **Acc 0.764 ± 0.009, AUC 0.856 ± 0.022**, en el mismo orden de magnitud (o ligeramente superior) que el 0.71 ± 0.02 del paper en T2. La comparación numérica es **orientativa, no estricta**:

- Datasets y poblaciones diferentes (ds004504 vs el privado del paper).
- HC con MMSE perfecto (30/30) en este dataset, vs valores no documentados en N del paper.
- Anti-leakage no idéntico: este TFM aplica saliency y patches por fold (no necesariamente equivalente a lo descrito en el paper original).
- Definiciones de "vainilla" pueden diferir mínimamente entre las implementaciones.

Lo defendible es que **el pipeline replicado funciona y produce métricas compatibles con el rango del paper original sobre datos públicos independientes**, que es el objetivo de un replication study.

### 5.4 Ensemble multi-seed y análisis estratificado por género (poder limitado)

**Ensemble**: agregando las 3 semillas con votación por mediana de scores (SVM vainilla STFT), la accuracy sube a 0.831 y AUC a 0.900. Esto sugiere que reportar un ensemble multi-seed sería más representativo que un solo seed.

**Análisis estratificado por género** (Fisher exact test sobre clasificación correcta):

| Género | n | AD/HC | Aciertos | Acc | AUC |
|---|---|---|---|---|---|
| Mujer | 35 | 24/11 | 27/35 | 0.771 | 0.852 |
| Hombre | 30 | 12/18 | 27/30 | 0.900 | 0.940 |
| Diferencia | — | — | — | **+0.129** | +0.088 |
| Fisher exact F vs M | — | — | OR ≈ 0.375 | **p = 0.201** | Cohen h ≈ 0.35 |

**Análisis de poder estadístico**: con un tamaño de efecto Cohen h ≈ 0.35 (moderado), detectar la diferencia observada al 80% de poder y α=0.05 requeriría aproximadamente **N ≈ 64 sujetos por grupo (es decir, ~129 sujetos totales sumando F y M)**, según la fórmula con transformación arcoseno: n/grupo = ((z₁₋α/₂ + z₁₋β)/h)² = ((1.96+0.84)/0.35)² ≈ 64. Nota: en versiones anteriores de este informe se reportaba "129 sujetos por grupo", lo cual confundía total con tamaño por grupo. El tamaño actual (35F + 30M) tiene poder ~25-30% para esa diferencia.

**Lectura prudente**: **no se detectó diferencia significativa entre géneros, pero el poder estadístico es limitado** — la diferencia absoluta de 12.9 puntos en accuracy es relevante en magnitud, y no podemos descartar un sesgo real del modelo. Una validación con dataset balanceado por género o con N mayor sería necesaria antes de afirmar robustez al sesgo.

### 5.5 Limitaciones

1. **Solo 3 seeds**: el multi-seed reporta varianza pero el número de réplicas es bajo para inferencias estadísticas fuertes (especialmente con Wilcoxon por seed inconsistente). Idealmente ≥10 seeds.

2. **Grid search de patches heurístico, no nested CV**: el script real (`scripts/04_extract_saliency_features.py:209-240`) implementa una selección por separabilidad de la saliency map (max contraste AD vs HC sobre los píxeles candidatos), NO una validación con SVM en val set. La función `grid_search_patches()` con validación SVM existe en `src/feature_extraction.py:87-125` pero **no es invocada por el pipeline final**. Esto introduce un sesgo de selección menor pero existente y constituye una desviación de fidelidad respecto al paper.

3. **BH-FDR/Bonferroni post-hoc**: la corrección se aplicó después de ver los resultados (sección 4.3.1), no como pre-registro. Solo la comparación SVM vainilla sobrevive (p ajustado=0.042 marginal).

4. **🔴 Eje de frecuencia de modulación incomparable entre STFT y CWT** (confound DSP detectado en revisión externa, no identificado en la implementación inicial): con los parámetros del pipeline (STFT noverlap=64, CWT preserva fs=200 Hz), las imágenes 45×45 finales representan rangos físicos drásticamente distintos del eje de modulación — STFT efectivamente 0-1.56 Hz interpolado, CWT 0-22.5 Hz subsampled. Esta es probablemente la limitación DSP más seria del trabajo. Las conclusiones STFT vs CWT cuantitativas reflejan este artefacto del pipeline, no una comparación equitativa de las transformadas T-F en sí mismas. Solución correcta para trabajo futuro: decimar la envolvente CWT a la tasa de muestreo de la envolvente STFT antes de la FFT de modulación.

5. **Resize bilinear** del modspec a 45×45 oculta el desajuste anterior y puede penalizar artificialmente a CWT cuya ventaja teórica está en la resolución variable.

6. **CWT subexplorada**: solo `cmor1.5-1.0` con 32 escalas log; sin cone-of-influence; sin tuning específico de hiperparámetros para CWT input. Atenuante: la T-F se computa sobre la señal completa antes de rebanar epochs, evitando edge effects en bordes de epoch, pero el soporte temporal del wavelet a 0.5 Hz se extiende a varios segundos y puede contaminar epochs vecinos en bajas frecuencias.

7. **Resolución STFT real ≠ nominal**: nperseg=128 a fs=200 Hz da Δf = 1.5625 Hz, no 1 Hz exacto. El "45×45 a 1 Hz" del paper se obtiene mediante el resize bilinear posterior, que es interpolación.

8. **Inferencia entre seeds con solo 3 réplicas**: la comparación usa DeLong por seed + combinación (Stouffer/Fisher), que es el procedimiento correcto, pero con 3 seeds tiene potencia limitada. Un modelo de efectos mixtos con ≥10 seeds sería más robusto. El Wilcoxon por seed (0.893, 0.025, 0.338) corrobora la inconsistencia real del efecto entre inicializaciones.

9. **CNN sub-entrenada**: dropout 0.85 es muy agresivo. La CNN sola opera apenas mejor que baseline trivial (~0.55). No exploramos dropout más bajo.

10. **Sin validación externa**: solo un dataset público (ds004504). Replicar en otro (e.g., privado de Cassani 2020) reforzaría las conclusiones.

11. **Tareas binarias solamente**: el paper original tiene 5 tareas (T1–T5 con AD1/AD2). Este TFM solo replica T2 (AD vs HC).

12. **Poder estadístico modesto** para sub-análisis: género N=65 con potencia ~25-30% para detectar h≈0.35; bandas y canales evaluados sobre 65 sujetos × 3 seeds. Importante distinción: la afirmación "modelo robusto al sesgo de género" implícita en algunos scripts auxiliares **NO está sustentada estadísticamente**; la lectura correcta es "ausencia de evidencia, no evidencia de ausencia".

13. **Saliency unstable fold-to-fold** (Jaccard ≈ 0.05): los "biomarcadores descubiertos" varían mucho entre folds, lo que dificulta hablar de un descubrimiento estable. Parte de las correlaciones cercanas a 0 entre saliency maps STFT↔CWT puede ser puro ruido fold-a-fold.

14. **Atribución de bandas canónicas para CWT sesgada**: el eje portador de la CWT es log-espaciado (`geomspace`), pero el análisis de bandas en `scripts/10_post_experiments.py` asumía un eje lineal. Esto sesga la tabla 4.6 para las filas de CWT; la atribución para STFT (eje lineal nativo) es aproximadamente válida. **El análisis por bandas debe interpretarse solo para STFT** (corregido en el script y la tabla; ver §4.6).

15. **N efectivo por autocorrelación de epochs**: el overlap 87.5% entre epochs intra-sujeto NO afecta los tests reportados porque todas las inferencias finales son a nivel de sujeto (n=65 scores agregados por sujeto). La función `effective_n()` existe en `src/stats.py` pero no se necesita en este pipeline. *Aclaración añadida en respuesta a revisión externa.*

16. **Análisis post-hoc** (bandas canónicas, importancia por canal, correlaciones, género) no estaban pre-registrados; deben verse como exploratorios.

### 5.6 Aporte propio

Más allá de la replicación, este TFM aporta:

1. **Evaluación multi-seed** del pipeline de Lopes: el paper original no reporta varianza por seed. Aquí se cuantifica y se muestra que algunas configuraciones (CWT vainilla, acc ±0.081) tienen alta varianza, lo que cuestiona la robustez de conclusiones single-seed.
2. **Comparación rigurosa STFT vs CWT** con análisis estadístico explícito (DeLong, Wilcoxon, BH-FDR) y reconocimiento de las limitaciones del diseño (resize, CWT subexplorada).
3. **Observación de pipeline-dependencia**: el método de saliency cambia el ranking entre STFT y CWT — un punto metodológico relevante para la comunidad.
4. **Conexión exploratoria con literatura clínica**: STFT + vainilla concentra saliency en bandas alpha-theta y canales occipito-temporales, coherente con biomarcadores conocidos de EA (Fraga 2013, Cassani 2020).
5. **Auditoría DSP + ML interna documentada** en `docs/AUDIT.md` y validada por revisión externa.
6. **Reproducibilidad total**: repo público con `requirements-lock.txt`, seeds fijas, configs YAML, tests unitarios pasando, y notebook ejecutable.

---

## 6. Conclusiones

1. **El pipeline de Lopes et al. 2023 fue replicado funcionalmente** sobre el dataset público ds004504. SVM con saliency vainilla y STFT alcanza Acc = 0.764 ± 0.009, AUC = 0.856 ± 0.022 (3 seeds), en el orden de magnitud del 0.71 ± 0.02 reportado por Lopes en T2. Las diferencias de datasets, poblaciones y detalles de anti-leakage impiden una comparación numérica estricta; lo defendible es que el pipeline funciona sobre datos públicos independientes.

2. **La comparación STFT vs CWT no es concluyente y está confundida por DSP**. Tres caveats independientes:

   - **Inferencia estadística**: con saliency vainilla, AUC media de STFT (0.856 ± 0.022) supera la de CWT (0.778 ± 0.038), pero **DeLong por seed + combinación Stouffer/Fisher da p ≈ 0.061 (NS)** y **tras BH-FDR sobre las 3 comparaciones principales ninguna sobrevive (q ≥ 0.10)**. La inferencia formal NO alcanza significancia al α=0.05. Ver §4.3.1–§4.3.2.
   - **Confound DSP del eje de modulación**: STFT (Nyquist 1.56 Hz, 13 bins reales interpolados a 45) y CWT (Nyquist 100 Hz, 180 bins subsampleados a 45) codifican contenido espectral distinto en su eje vertical. La supuesta ventaja de STFT puede deberse en parte a esta asimetría no controlada (ver §3.3, §5.1). Una comparación justa requeriría unificar el `dt` temporal post-T-F.
   - **Sensibilidad a hiperparámetros**: resize 45×45 e hiperparámetros específicos de CWT-Morlet (cmor1.5-1.0, 32 escalas) son configuraciones sin tuning exhaustivo. La hipótesis original "CWT > STFT" no obtiene evidencia, pero **tampoco se prueba la dirección opuesta con rigor estadístico**.

3. **El ranking STFT vs CWT depende del método de saliency**: con Grad-CAM la tendencia se invierte (CWT > STFT, NS). Las saliency maps de STFT y CWT son **estadísticamente ortogonales** (r ≈ −0.09 vainilla, sin Pearson p<0.05), lo que sugiere que pueden capturar información **complementaria** — un ensemble (late fusion o stacking) es una hipótesis natural a futuro.

4. **Observaciones exploratorias (post-hoc)**: STFT + vainilla concentra saliency en bandas alpha (61%) + theta (28%) y canales occipito-temporales (O1, O2, T5, T6), coherente con biomarcadores clásicos de EA (Fraga 2013). Esta convergencia con la literatura clínica es prometedora pero requiere validación en datasets independientes.

5. **Hallazgos metodológicos para la comunidad**:
   - El método de saliency afecta sustantivamente las conclusiones cuantitativas; reportar siempre el método específico.
   - Los patches descubiertos por saliency-guided ML son altamente inestables entre folds (Jaccard ≈ 0.05): la narrativa de "biomarcadores reproducibles" debe matizarse.
   - Reportar varianza por seed es esencial: configuraciones con SD alta (CWT vainilla, ±0.081) revelan inestabilidad oculta en single-seed.
   - El poder estadístico debe acompañar siempre las afirmaciones de "ausencia de efecto" (e.g., análisis por género).

6. **Recomendaciones para trabajos futuros**:
   - **Unificar el eje de modulación entre STFT y CWT** (downsample temporal post-CWT a `dt` análogo o forzar mismo Nyquist) — pre-requisito para una comparación STFT vs CWT honesta.
   - **Ensemble STFT+CWT** (late fusion) explotando ortogonalidad observada de saliency.
   - **CWT tuning específico**: explorar cmor1-1, n_scales mayor, cone-of-influence.
   - **Comparación T-F sin resize forzado**: dos CNNs distintas para resoluciones nativas.
   - **Test externo** en otro dataset EEG-AD para validar generalización.
   - **Banco de filtros con bandas no uniformes** (Condición C de la propuesta original).
   - **Nested CV** para grid search de patches con validación real.
   - **Dataset balanceado por género o N mayor** para validar robustez del modelo.
   - **Más seeds (10+)** para inferencias estadísticas más sólidas con DeLong por seed.

---

## 7. Reproducibilidad

**Repositorio**: https://github.com/spalaciobe/tps-alzheimer-modspec
**Commit ref**: ver `git log --oneline` o el último commit con tag `v1.0` cuando se publique.

**Para reproducir**:
```bash
git clone https://github.com/spalaciobe/tps-alzheimer-modspec.git
cd tps-alzheimer-modspec
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements-lock.txt && pip install -e .
python scripts/00_download_dataset.py
bash scripts/run_remaining_v2.sh
```

Tiempos en RTX 3050 Laptop (4 GB):
- Preproceso 65 sujetos: ~35 min
- Modspecs (STFT + CWT): ~90 min
- LOSO CNN × 3 seeds × 2 métodos: ~14h (con resume)
- Saliency × 12 corridas (4 × 3 seeds): ~12h
- SVM × 12 corridas: ~12h
- Post-análisis + figuras: ~30 min

**Total: ~48h de cómputo continuo** (con varias pausas y reanudaciones documentadas).

**Stack**: Python 3.13, PyTorch 2.6+cu124, MNE 1.12, scikit-learn 1.8. Lock completo en `requirements-lock.txt`.

---

## 8. Declaración de uso de IA

La estructura del pipeline, scripts de orquestación, debugging de leakage en LOSO-CV, refactor de optimización (`SubjectBank`, fp16-bank, GC entre folds), auditoría metodológica multi-dimensional y redacción de este informe fueron asistidos por **Claude Code (Anthropic)** [Opus 4.7, 2026]. Todo el código fue revisado, ejecutado y validado por el autor antes de cada commit.

## 9. Bibliografía

1. **Lopes, M., Cassani, R., Falk, T.H.** (2023). Using CNN saliency maps and EEG modulation spectra for improved and more interpretable machine learning-based Alzheimer's disease diagnosis. *Computational Intelligence and Neuroscience* **2023**, art. 3198066. DOI: 10.1155/2023/3198066. Open access vía Wiley/Hindawi. [Nota: en versiones previas de este informe se citó erróneamente como IEEE TNSRE 31:1310–1319; corregido tras verificación contra el PDF (DOI confirma Hindawi/Wiley CIN).]
2. **Miltiadous, A. et al.** (2023). A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG. *OpenNeuro ds004504* (CC0 1.0).
3. **Trambaiolli, L.R. et al.** (2011). EEG spectro-temporal modulation energy: A new feature for automated diagnosis of Alzheimer's disease. *IEEE EMBC*, 3828–3831.
4. **Fraga, F.J. et al.** (2013). Characterizing Alzheimer's disease severity via resting-awake EEG amplitude modulation analysis. *PLoS ONE* **8**(8), e72240.
5. **Cassani, R. & Falk, T.H.** (2020). Alzheimer's disease diagnosis and severity level detection based on EEG modulation spectral 'patch' features. *IEEE J. Biomed. Health Inform.* **24**(7), 1982–1993.
6. **Selvaraju, R.R. et al.** (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
7. **Simonyan, K., Vedaldi, A., Zisserman, A.** (2014). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. *ICLR Workshop*.
8. **Pion-Tonachini, L., Kreutz-Delgado, K., Makeig, S.** (2019). ICLabel: An automated electroencephalographic independent component classifier. *NeuroImage* **198**, 181–197.
9. **Gramfort, A. et al.** (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience* **7**, 267.

## 10. Anexos

### A. Figuras finales

Disponibles en:
- `results/figures_multiseed/`: bar plots y boxplots multi-seed para CNN y SVM.
- `results/figures/`: figuras estándar (modulation spectrums medios, saliency comparativa, ROC, matrices de confusión, comparativas STFT vs CWT).
- `results/figures_vanilla/`: idem con saliency vainilla.

Figura principal sugerida para presentación: `results/figures_multiseed/auc_master_summary.png`.

### B. Documentos asociados

- `docs/plan.md`: plan original del proyecto.
- `docs/AUDIT.md`: auditoría consolidada (fidelidad, código, DSP, ML, ética, publicación).
- `docs/LEARNING_PATH.md`: roadmap pedagógico (señales + código) para entender el proyecto desde nivel básico.
- `docs/optimization_options.md`: opciones para escalar a cloud.
- `results/RESULTS_full.md`: tabla detallada de resultados.
- `results/multiseed_analysis.json`: datos crudos del análisis multi-seed.
- `results/post_experiments/post_experiments.json`: hallazgos de los experimentos abiertos.

### C. Licencia

Código: **MIT License** (ver `LICENSE`).
Dataset: **CC0 1.0 Universal** (OpenNeuro ds004504).

### D. Disclaimer clínico

Este trabajo es un ejercicio académico de replicación y comparación metodológica. Los modelos NO están aprobados para uso clínico. No diagnostican Alzheimer ni ninguna patología. Cualquier aplicación clínica requiere validación en cohortes externas, aprobación regulatoria y supervisión médica cualificada.
