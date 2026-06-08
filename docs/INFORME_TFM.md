# Informe TFM — Replicación de Lopes et al. 2023 sobre OpenNeuro ds004504 y comparación STFT vs CWT

**Autor**: Sebastián Palacio Betancur
**Programa**: Maestría · Universidad Nacional de Colombia — Sede Medellín · Facultad de Minas
**Curso**: Tópicos en Procesamiento Digital de Señales · Profesor: Freddy Bolaños
**Fecha**: junio 2026
**Repositorio**: https://github.com/spalaciobe/tps-alzheimer-modspec

---

## Resumen ejecutivo

Este trabajo replica de forma independiente el pipeline de Lopes et al. (2023) [*Using CNN Saliency Maps and EEG Modulation Spectra for Improved and More Interpretable ML-Based Alzheimer's Disease Diagnosis*, IEEE TNSRE] sobre el dataset público **OpenNeuro ds004504** (Miltiadous 2023), y extiende el análisis comparando **STFT vs CWT-Morlet** como etapa de descomposición tiempo-frecuencia.

**Resultados principales (LOSO-CV 65 sujetos, 3 seeds):**

| Configuración | Acc | F1 | AUC |
|---|---|---|---|
| SVM con saliency **vainilla + STFT** (paper-faithful) | **0.764 ± 0.009** | **0.762 ± 0.011** | **0.856 ± 0.022** |
| SVM con saliency vanilla + CWT | 0.677 ± 0.081 | 0.673 ± 0.094 | 0.778 ± 0.038 |
| SVM con saliency Grad-CAM + STFT | 0.662 ± 0.031 | 0.643 ± 0.041 | 0.713 ± 0.020 |
| SVM con saliency Grad-CAM + CWT | 0.703 ± 0.009 | 0.687 ± 0.026 | 0.800 ± 0.010 |
| CNN end-to-end + STFT | 0.656 ± 0.024 | 0.642 ± 0.020 | 0.695 ± 0.022 |
| CNN end-to-end + CWT | 0.626 ± 0.071 | 0.611 ± 0.070 | 0.590 ± 0.069 |

**Conclusiones clave:**

1. **La replicación independiente del pipeline es exitosa**: SVM vainilla con STFT alcanza Acc 0.764, AUC 0.856 — supera el 0.71 ± 0.02 reportado por Lopes (T2: N vs AD), reproducible entre 3 semillas (SD bajísima de ±0.009).
2. **La hipótesis del proyecto (CWT > STFT) NO se confirma**: con la metodología fiel al paper (saliency vainilla), **STFT supera a CWT** con DeLong pooled p=0.014 (significativo).
3. **El método de saliency afecta cualitativamente la conclusión**: con Grad-CAM se ve tendencia inversa (CWT > STFT, no significativo). Esto sugiere que la aparente ventaja de CWT con Grad-CAM puede ser artefacto del método saliency, no una mejora real de la representación T-F.
4. **STFT + vainilla descubre las bandas neurofisiológicamente esperadas** (alpha 61% + theta 28%); el resto de configuraciones se concentra en bandas beta/gamma menos clínicamente conocidas.
5. **Limitaciones detectadas**: sesgo de género en el dataset (χ² p=0.039), baja consistencia de patches entre folds (Jaccard ≈ 0.05), saliency maps de STFT y CWT son sustancialmente distintos (r ≈ -0.2).

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
- **STFT**: ventana Hann, nperseg=128 (1.6 s a 200 Hz), noverlap=64.
- **CWT-Morlet** (`cmor1.5-1.0`): 32 escalas en [0.5, 45] Hz logarítmico.
- Magnitud al cuadrado → FFT temporal → recorte (0.5–45 Hz portadora, 0–22.5 Hz mod) → **resize bilinear a 45×45** (alineación con Lopes) → log-power.

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
5. **Grid search por fold** sobre threshold ∈ {80, 82, ..., 96}% y K ∈ {3, 4, 5} para KMeans → patches por fold.
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

**DeLong pooled** (mediana entre seeds): AUC STFT − CWT = +0.095, p = 0.252 (NS).

La CNN sola opera apenas mejor que el "baseline trivial" (0.554 si siempre predice AD). Esto es coherente con su rol en el pipeline: la CNN no es el clasificador final, sino el extractor de regiones discriminativas vía saliency. Su accuracy modesta es esperada con dropout 0.85.

### 4.2 SVM con patches Grad-CAM (3 seeds)

| Método | Acc | F1 | AUC |
|---|---|---|---|
| STFT | 0.662 ± 0.031 | 0.643 ± 0.041 | 0.713 ± 0.020 |
| CWT | **0.703 ± 0.009** | **0.687 ± 0.026** | **0.800 ± 0.010** |

**DeLong pooled**: AUC CWT − STFT = +0.087, p = 0.195 (NS).

Con Grad-CAM, CWT tiende a ser mejor que STFT, pero la diferencia no alcanza significancia estadística con 3 seeds.

### 4.3 SVM con patches vainilla (paper-faithful) — **Resultado principal** (3 seeds)

| Método | Acc | F1 | AUC |
|---|---|---|---|
| **STFT** | **0.764 ± 0.009** | **0.762 ± 0.011** | **0.856 ± 0.022** |
| CWT | 0.677 ± 0.081 | 0.673 ± 0.094 | 0.778 ± 0.038 |

**DeLong pooled**: AUC STFT − CWT = +0.078, **p = 0.014 (SIGNIFICATIVO)**.

**Con la metodología fiel al paper, STFT supera a CWT con significancia estadística**.

### 4.4 Comparación con el paper Lopes 2023

| Métrica | Paper Lopes (T2: N vs AD, LOSO test) | Este TFM (SVM vainilla STFT, 3 seeds) |
|---|---|---|
| N sujetos | 39 (20 N + 19 AD1) | 65 (29 HC + 36 AD) |
| Accuracy | 0.71 ± 0.02 | **0.764 ± 0.009** |
| F1 | 0.61 ± 0.02 | **0.762 ± 0.011** |
| AUC | no reportado | **0.856 ± 0.022** |

**El TFM supera ligeramente al paper original**, posiblemente por:
- Más sujetos (65 vs 39).
- Población HC más sana (MMSE 30.0 ± 0.0 vs N del paper sin valor explícito).
- Anti-leakage estricto (saliency y patches por fold, no globales).

### 4.5 Análisis de saliency: correlación 2D entre métodos

Pearson r entre saliency maps `saliency_diff` (3 seeds, media ± SD):

| Comparación | r |
|---|---|
| **STFT vs CWT** con Grad-CAM | **-0.237 ± 0.119** |
| **STFT vs CWT** con vainilla | **-0.086 ± 0.076** |
| **Grad-CAM vs vainilla** con STFT | +0.114 ± 0.043 |
| **Grad-CAM vs vainilla** con CWT | -0.166 ± 0.106 |

**Interpretación**: los saliency maps STFT y CWT NO son consistentes — son aproximadamente ortogonales o ligeramente anti-correlated. Lo mismo ocurre entre Grad-CAM y vainilla. Esto demuestra que **"el biomarcador descubierto" depende fuertemente de la elección de pipeline**.

### 4.6 Análisis por banda canónica

Proporción de píxeles top-10% saliency en cada banda canónica (media entre seeds):

| Configuración | δ (0.5–4 Hz) | θ (4–8 Hz) | α (8–13 Hz) | β (13–30 Hz) | γ (30–45 Hz) |
|---|---|---|---|---|---|
| **STFT vainilla** | 5.8% | **27.9%** | **61.1%** | 5.3% | 0.0% |
| CWT vainilla | 0.8% | 2.0% | 7.6% | 54.2% | 35.5% |
| STFT Grad-CAM | 0.0% | 0.0% | 0.0% | 0.7% | 99.3% |
| CWT Grad-CAM | 0.0% | 0.0% | 0.0% | 67.5% | 30.9% |

**STFT con saliency vainilla recupera el biomarcador clásico de EA**: atenuación alfa y aumento theta concentran ~89% de la señal. Este resultado está alineado con la literatura clínica EEG-AD (Fraga 2013, Cassani 2020).

Las otras configuraciones (incluyendo CWT vainilla) se concentran en bandas beta/gamma altas, que son menos canónicas para EA. Esto explica por qué STFT vainilla tiene la mejor AUC: **aprende información clínicamente conocida**.

### 4.7 Consistencia de patches entre folds (Jaccard)

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

**Respuesta corta: NO**, al menos con la metodología fiel al paper. Tres líneas de evidencia:

1. **DeLong AUC pooled** (SVM vainilla): STFT > CWT por +0.078, p = 0.014.
2. **Consistencia entre seeds**: STFT vainilla tiene SD muy baja (acc ±0.009), CWT vainilla muy alta (±0.081), indicando que CWT es menos estable.
3. **Bandas activas**: STFT vainilla descubre alpha + theta (clásicos en EA), CWT se concentra en beta/gamma (menos clínicamente conocidos).

La hipótesis original del proyecto (CWT > STFT por mejor resolución frecuencial en bajas Hz) **no se confirma**. Posibles razones:

- El "resize bilinear a 45×45" iguala artificialmente las resoluciones de salida, eliminando parte de la ventaja teórica de CWT.
- Los modspecs de Lopes están optimizados para STFT (los hiperparámetros del CNN fueron tuned para input STFT).
- La CWT con `cmor1.5-1.0` puede ser subóptima — `cmor1-1` o n_scales mayor podrían cambiar el resultado.

### 5.2 El método de saliency cambia la conclusión

**Hallazgo metodológico relevante**: con saliency Grad-CAM, CWT tiende a ser mejor; con vainilla (paper-faithful), STFT es mejor con significancia. Las saliency maps de los dos métodos no se parecen (r ≈ 0.1 STFT, -0.17 CWT).

Esto implica que **resultados del tipo "transformada X supera transformada Y"** dependen del método de saliency, y deben reportarse con saliency vainilla para alineación con el paper original. Es un sesgo metodológico que la comunidad debería tener en cuenta.

### 5.3 Replicación del paper

Sobre el dataset público ds004504, el SVM vainilla con STFT alcanza **Acc 0.764 ± 0.009** (vs 0.71 ± 0.02 del paper en T2). La diferencia (+5 puntos) se explica por:

- Más sujetos (+27).
- Anti-leakage estricto (saliency y patches por fold, no globales como podría inferirse del paper).
- HC con MMSE perfecto (30/30), menos overlap con AD que en el N del paper.

**La replicación es exitosa y reproducible** (3 seeds dan SD ≤ 0.022 en AUC).

### 5.4 Limitaciones

1. **Sesgo de género** en el dataset (χ² p = 0.039): 67% mujeres en AD vs 38% en HC. Podría sesgar el modelo si hay diferencias EEG por género. Mitigación: análisis estratificado por género (no realizado en este TFM, queda para trabajo futuro).
2. **Patches inestables entre folds** (Jaccard ≈ 0.05): el "biomarcador descubierto" varía mucho fold-to-fold. Esto es típico de redes pequeñas con poco data — la saliency es ruidosa.
3. **Anti-leakage parcial**: aunque hicimos LOSO + saliency-por-fold, el grid search de threshold/K se hace por fold sobre train (no sobre val real). Mejor sería un nested CV.
4. **CNN sub-entrenada**: dropout 0.85 es muy agresivo. Con menos dropout (0.5) la CNN sola subiría su accuracy, pero quizás la saliency sería diferente. No exploramos este trade-off.
5. **Resize bilinear** del modspec a 45×45 puede igualar artificialmente STFT y CWT. Una comparación más justa sería en su resolución nativa, con dos CNNs distintas.
6. **Sin validación externa**: solo usamos un dataset público; replicar en otro (e.g., el privado de Cassani) reforzaría las conclusiones.
7. **Tareas binarias solamente**: el paper original tiene 5 tareas (T1–T5 con AD1/AD2). Este TFM solo replica T2.

### 5.5 Aporte propio

Más allá de la replicación, este TFM aporta:

1. **Evaluación multi-seed**: el paper original no reporta varianza por seed. Aquí mostramos que algunos resultados (especialmente CWT vainilla) tienen alta varianza (acc ±0.081), lo que cuestiona la robustez de algunas conclusiones.
2. **Comparación rigurosa STFT vs CWT**: documentamos que la elección del método de saliency cambia cualitativamente la conclusión cuantitativa.
3. **Conexión con literatura clínica**: STFT + vainilla descubre alpha-theta (61% + 28%), reproduciendo el biomarcador clásico de EA (atenuación alfa). Esta interpretación neurofisiológica falta en el paper original.
4. **Auditoría DSP + ML completa**: documentada en `docs/AUDIT.md`.
5. **Reproducibilidad total**: pipeline en GitHub público con `requirements-lock.txt`, seeds fijas, configs YAML y notebooks ejecutables.

---

## 6. Conclusiones

1. **La replicación independiente del paper de Lopes et al. 2023 es exitosa**: SVM con saliency vainilla y STFT sobre ds004504 alcanza Acc = 0.764 ± 0.009, AUC = 0.856 ± 0.022, superando ligeramente al paper original (0.71 ± 0.02).

2. **La hipótesis "CWT > STFT" no se confirma**: con la metodología fiel al paper (saliency vainilla), STFT es estadísticamente superior a CWT (DeLong p = 0.014). Solo con Grad-CAM se ve tendencia a favor de CWT, sugiriendo que esa diferencia es artefacto del método saliency.

3. **El biomarcador clásico de EA (alpha attenuation + theta increase) emerge solo con STFT + vainilla**: explica por qué esa configuración tiene la mejor AUC y por qué es la favorecida por la metodología original del paper.

4. **Hallazgos metodológicos para la comunidad**:
   - El método de saliency afecta sustantivamente las conclusiones cuantitativas.
   - Los patches descubiertos por saliency-guided ML son altamente inestables entre folds (Jaccard ≈ 0.05).
   - Reportar varianza por seed es esencial: configuraciones con SD alta deben tomarse con cautela.

5. **Limitaciones a futuras direcciones**:
   - Análisis estratificado por género (confound detectado).
   - Test externo en otro dataset EEG-AD.
   - Banco de filtros con bandas no uniformes (Condición C de la propuesta original).
   - Nested CV para grid search de patches.

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

1. **Lopes, M., Cassani, R., Falk, T.H.** (2023). Using CNN saliency maps and EEG modulation spectra for improved and more interpretable machine learning-based Alzheimer's disease diagnosis. *IEEE Trans. Neural Syst. Rehabil. Eng.* **31**, 1310–1319.
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
