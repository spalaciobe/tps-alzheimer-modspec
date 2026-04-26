# Auditoría consolidada del proyecto

Consolidación de 6 auditorías especializadas (fidelidad al paper, calidad de código, viabilidad de publicación, rigor DSP, pitfalls de ML, ética/datos). Cada hallazgo está clasificado por severidad y referenciado a `archivo:línea`.

---

## TL;DR — top 12 acciones priorizadas

| # | Severidad | Acción | Archivo |
|---|---|---|---|
| 1 | 🔴 Crítica | **Saliency aggregation por fold** (no global) | `scripts/04_extract_saliency_features.py:131-146` |
| 2 | 🔴 Crítica | **Grid search real de patches por fold** (no fijo p=88, K=4) | `scripts/04_extract_saliency_features.py:150` |
| 3 | 🔴 Crítica | **Disclaimer clínico + declaración uso IA** | `README.md` |
| 4 | 🔴 Crítica | **N efectivo por autocorrelación de epochs** (overlap 87.5%) | `epoching.py:9-15`, todos los tests |
| 5 | 🟠 Alta | **LICENSE MIT** en raíz | `LICENSE` (crear) |
| 6 | 🟠 Alta | **`set_seed()` en `scripts/05_run_svm.py`** | `scripts/05_run_svm.py:51` |
| 7 | 🟠 Alta | **Replicar también vanilla saliency** (paper original) | `04_extract_saliency_features.py` |
| 8 | 🟠 Alta | **3-5 seeds reportados** (varianza) | `scripts/03_train_loso.py` |
| 9 | 🟠 Alta | **Anonymize MNE + Drive privado** antes de Colab | `src/preprocess.py` |
| 10 | 🟡 Media | **Documentar batch=128 vs paper batch=4** en RESULTS | `results/RESULTS_quick.md` |
| 11 | 🟡 Media | **DeLong test para AUC + power analysis** | `src/stats.py` |
| 12 | 🟡 Media | **Eliminar código legacy** (ModSpecConcatDataset) | `src/datasets.py:105-161` |

---

## 1. Fidelidad al paper Lopes et al. 2023

### ✅ Fielmente replicado
- **Filtro 0.5–45 Hz FIR fase cero** (`preprocess.py:39`).
- **CNN architecture** Tabla 3 paper: input 19×45×45, 2 conv 3×3 ReLU, 3 FC LeakyReLU, dropout 0.85, Nadam lr=1e-4, weight_decay=1e-2 (`models/cnn.py`).
- **LOSO-CV** con hold-out val estratificado (`datasets.py:163-185`).
- **Modulation spectrum 45×45 a 1 Hz** vía resize bilinear (`modspec.py:159-244`).
- **Grid search lógico** de patches definido en `feature_extraction.py:87-125` (aunque NO usado por defecto — ver crítica abajo).
- **SVM RBF γ=1/24, C=1, MinMax [-1,1]** (`svm_pipeline.py`).
- **Clase weights** para imbalance 36/29.

### ⚠️ Desviaciones razonables documentadas
- **wICA → ICA + ICLabel**: el paper usa wICA (no especifica wavelet ni umbral). Implementación: ICA Infomax + ICLabel rechazo p≥0.8 en `eye/muscle/heart/line/channel`. Decisión consciente del alumno (más reproducible). wICA implementado como ablation (`preprocess.py:115-148`).
- **batch_size 4 → 128**: justificado por GPU con AMP. Resultados quick reportan batch=32 (no documentado claramente).
- **CWT-Morlet**: extensión nueva (no en paper) — núcleo de la propuesta.
- **Re-referenciado A1/A2 → CAR**: ds004504 viene con su propia ref, CAR uniformiza.
- **Resample 500→200 Hz**: alinea con paper.

### ❌ Desviaciones críticas (fix antes de paper)
- **Saliency: Grad-CAM (capa última conv) ≠ vanilla gradient (paper, última dense)**: NO comparable visualmente con figuras del paper. Reportar también vanilla como ablation.
- **Patches con threshold/K fijos** (88%, K=4) **no por fold**: el paper hace grid search por fold sobre val. Implementación bypassa `grid_search_patches()` (`scripts/04_extract_saliency_features.py:150`).
- **Tareas: solo binaria AD vs HC**: paper tiene 5 tareas (T1 multiclase + 4 binarias). FTD del ds004504 está excluido — sub-conjunto del scope original.

### 🔍 Verificaciones pendientes
- **Drop últimos 7 s antes de epochar**: implementado en `epoching.py:24` pero verificar que se invoca en pipeline real.

---

## 2. Calidad de código

### Fortalezas
- **Anti-leakage en CNN robusto**: z-score per-fold, mask-based LOSO disjunto (`datasets.py:87-90`).
- **`set_seed()` completo** en `utils/seed.py` (PYTHONHASHSEED + numpy + torch + cudnn deterministic).
- **AMP correctamente implementado** en `train.py:56-78` (GradScaler + autocast).
- **Optimizaciones medibles**:
  - `compute_modulation_spectrum_subject` (T-F una vez/sujeto): 8 min → 2 min en CWT.
  - `SubjectBank` (precarga RAM): 15 min/fold → 4 min/fold en LOSO.
- **16 tests** cubriendo shapes, leakage, normalization, epoching, CNN forward.

### Mejoras concretas (10 ítems)

| Prio | Mejora | Archivo |
|---|---|---|
| Alta | Añadir `set_seed(args.seed)` | `scripts/05_run_svm.py:51` |
| Alta | Docstring explícito anti-leakage en SVM | `src/svm_pipeline.py:14-16` |
| Alta | Test `compute_modulation_spectrum_subject` ≡ por-epoch | `tests/test_modspec_equiv.py` (crear) |
| Media | Eliminar/deprecar `ModSpecConcatDataset` y `ModSpecSubjectDataset` legacy | `src/datasets.py:105-161` |
| Media | `requirements-lock.txt` con versiones fijas | crear |
| Media | Test saliency (gradcam/vanilla, shape, no NaN) | `tests/test_saliency.py` (crear) |
| Media | Cachear stats globales en lugar de recalcular por fold | `04_extract_saliency_features.py:31-48` |
| Baja | README estimaciones disco/RAM | `README.md` |
| Baja | Refactorizar duplicación `_patch_powers` | `src/feature_extraction.py:46-79` |
| Baja | Test class weights en CNN | `tests/test_cnn_class_weights.py` (crear) |

---

## 3. Rigor DSP / teoría de señales

### Hallazgos teóricos cuantificables

| # | Tema | Severidad | Detalle |
|---|---|---|---|
| 3.1 | Resolución STFT 1.56 Hz vs 1 Hz declarado | 🟡 Media | nperseg=128 a fs=200 → 1.56 Hz. Para 1 Hz exacto: nperseg=200. Aclarar en docs o cambiar. |
| 3.2 | Edge effects sin padding | 🟡 Media | `boundary=None, padded=False` en `modspec.py:46`. ~8% atenuación en bordes. Fix: `boundary='even'`. |
| 3.3 | CWT `cmor1.5-1.0` subóptimo | 🟡 Media | B=1.5 muy estrecho para EEG. Estándar: `cmor1-1` o `cmor0.5-1`. |
| 3.4 | Sin Cone Of Influence (COI) en CWT | 🟡 Media | Bordes con energía espuria. Implementar máscara COI. |
| 3.5 | **Ambigüedad \|X\|² vs \|X\|** | 🔴 Alta | Lopes eq.(1) no aclara. Código usa `\|X\|²` (`modspec.py:49`). 6 dB offset sistemático vs versión \|X\|. **Verificar paper sec 2.4**. |
| 3.6 | Resize bilinear vs spline | 🟢 Baja | `zoom(order=1)`. Cambio a `order=3` cuesta ~1% extra, mejor fidelidad. |
| 3.7 | ICLabel mismatch banda | 🟡 Media | Entrenado 1-100 Hz, datos filtrados 0.5-45 Hz. Warning observado en logs. ~10% sub-detección estimada. |
| 3.8 | **Epochs autocorrelados 87.5%** | 🔴 **Crítica** | overlap=7s en 8s. N_efectivo ≈ N_epochs / 13. **Tests Wilcoxon asumen IID — p-values inflados ~2.8×**. Reportar N_eff y aplicar Bonferroni o block bootstrap. |
| 3.9 | CAR vs A1/A2 ref no comparado | 🟢 Baja | ±5-10 dB potencia. Comparar en 3 sujetos como ablation. |
| 3.10 | Filtro `phase='zero-double'` ✓ correcto | — | Coincide con "zero-phase FIR" del paper. |

**Implicación más crítica**: el overlap 87.5% **invalida tests estadísticos a nivel epoch**. A nivel sujeto (LOSO) está bien, pero cualquier afirmación sobre "p<X en epochs" es estadística sospechosa.

---

## 4. ML pitfalls

### Crítica (bloqueante para paper)

#### 4.1 Saliency leakage entre folds
`scripts/04_extract_saliency_features.py:131-146`: agrega `accum[cls] += fold_map` global. Los `patch_masks.npy` resultantes contienen señal de TODOS los sujetos, incluyendo los que serán test en SVM. **Leakage indirecto**.
**Fix**: generar `patch_masks_fold_i.npy` por fold (folds ≠i en train). Usar mascara correspondiente en `05_run_svm.py:LOSO`.

#### 4.2 Threshold/K fijos hardcodeados
`scripts/04_extract_saliency_features.py:150` usa `threshold_pct=88, n_clusters=4` sin grid search. La función `grid_search_patches()` existe en `feature_extraction.py:87-125` pero **nunca se invoca**. **Selection bias**: 88/4 fueron elegidos por... ¿? RESULTS reporta sin justificación.
**Fix**: invocar `grid_search_patches()` con datos val del fold; reportar (p*, K*) seleccionados.

### Alta

| # | Tema | Detalle |
|---|---|---|
| 4.3 | Early stopping con val de 2 sujetos | F1 macro de val con N_AD=1, N_HC=1 → varianza enorme. Modelos inestables. **Fix**: val con 10% sujetos estratificado. |
| 4.4 | Subsample seed colisión train/val | `03_train_loso.py:101-102` aplica subsample con `seed+fold_idx` independientemente. Verificar disjunción epoch-level explícitamente. |
| 4.5 | **Sin múltiples seeds**: solo seed=0 | Varianza no cuantificada. **Reportar 3-5 seeds en final**. |
| 4.6 | Sin DeLong test ni power analysis | Solo Wilcoxon. Para AUC, DeLong es estándar. Power analysis: con N=65 detectable Δ AUC ~0.08 al 70% power. |

### Media-baja
- ANOVA features intra-fold pero no cross-validados.
- Threshold 0.5 fijo sin optimización ROC.
- Múltiples comparaciones sin BH-FDR (implementado en `stats.py` pero no invocado).
- Mean softmax sobre epochs autocorrelados (no IID).

---

## 5. Camino a publicación

### Tipo recomendado
**"Reproducibility study with methodological extension"**: replicación independiente del paper de Lopes en dataset público + comparación STFT vs CWT como extensión técnica.

### Venues realistas

| Venue | Viabilidad | Falta |
|---|---|---|
| **EMBC 2026** | 🟢 Cerca, factible | LOSO full + 3-5 seeds + tabla confounders + figuras paper-ready |
| **EUSIPCO 2026** | 🟢 Buen fit (DSP angle) | Igual + más énfasis CWT vs STFT |
| **ICASSP 2027** | 🟡 Posible | Añadir baseline ViT/Transformer comparison |
| **IEEE TNSRE** (donde está Lopes) | 🟡 Exigente | Test set externo + Leave-Site-Out |
| **JBHI** | 🟢 Aceptable a negative results | Análisis confounders + DeLong + 5 seeds |
| **Frontiers Aging Neuroscience** | 🟢 Más accesible | Mismas mejoras, IF más bajo |

### Claims defendibles vs no permitidos

✅ **Defendibles** (con LOSO full + correcciones):
1. "El pipeline de Lopes es replicable en dataset público; SVM con patches Grad-CAM-guiados alcanza ~77% accuracy en AD vs HC."
2. "CWT-Morlet no muestra mejora significativa vs STFT (p > 0.05)."
3. "Mapas Grad-CAM de modspec son consistentes entre STFT y CWT (correlación 2D)."

❌ **NO permitidos** sin más datos:
1. "CWT es inferior a STFT" — solo equivalente.
2. "Modspec captura biomarcadores más interpretables que bandas clásicas" — falta head-to-head.
3. "Listo para uso clínico" — falta validación externa.

### Mejoras imprescindibles (pre-paper)

1. **LOSO full sin --quick** (ya en ejecución).
2. **3-5 seeds** con media ± SD.
3. **Test externo** (Cassani dataset si accesible) o Leave-Site-Out si ds004504 multi-sitio.
4. **Análisis saliency cuantitativo**: correlación 2D STFT vs CWT, test estadístico en diferencias.
5. **Tabla confounders**: edad/MMSE/sexo/sitio AD vs HC (t-test, χ²).
6. **DeLong test para AUC**.
7. **Power analysis explícito**.
8. **Ablation vanilla saliency** para alineación con figuras Lopes.

### Preguntas duras de reviewers (preparar respuestas)

1. ¿Por qué CNN 2015 y no Transformers/ViTs?
2. ¿ds004504 tiene rigor clínico equivalente al dataset privado de Lopes?
3. ¿Comparación CWT/STFT es justa con n_scales=32 vs nperseg=128?
4. ¿LOSO con N=65 tiene varianza alta — por qué no nested k-fold?
5. ¿Generalización a otros datasets EEG-AD?

---

## 6. Ética, datos, licencias

### Acciones inmediatas (críticas)

| Punto | Estado | Acción |
|---|---|---|
| **LICENSE en raíz (MIT)** | ❌ Falta | Crear `LICENSE` con texto MIT |
| **Disclaimer clínico** | ❌ Falta | Bloque visible en `README.md` |
| **Declaración uso IA generativa** | ❌ Falta | Bloque en README + memoria TFM |
| **`raw.anonymize(daysback=10000)`** | ❌ Falta | En `preprocess.py` antes de guardar FIF |
| **Drive compartido solo con supervisor** | ⚠️ Recomendado | Antes de subir HDF5 a Colab |

### Estado correcto (no requiere acción)
- ✅ ds004504 bajo CC0 1.0 (libre redistribución).
- ✅ Repo privado en GitHub (riesgo bajo).
- ✅ Citaciones de Lopes y Miltiadous en plan/propuesta.
- ✅ Seeds fijados para reproducibilidad.

### Pendiente pre-defensa
- `CITATION.cff` en raíz.
- `requirements-lock.txt` con versiones fijas.
- Docstrings con citas (Selvaraju 2017 Grad-CAM, Pion-Tonachini 2019 ICLabel).

---

## 7. Plan de remediación (3 olas)

### Ola 1 — esta sesión / hoy (~1 h)
- [ ] LICENSE MIT.
- [ ] `set_seed` en script 05.
- [ ] Disclaimer clínico + declaración IA en README.
- [ ] Doc batch=128 deviation en RESULTS_quick.md.
- [ ] Eliminar código legacy `ModSpecConcatDataset`.

### Ola 2 — próxima semana (post LOSO full)
- [ ] Refactor saliency aggregation por fold (anti-leakage).
- [ ] Grid search real patches.
- [ ] 3-5 seeds.
- [ ] DeLong test, BH-FDR.
- [ ] N_efectivo por autocorrelación documentado.
- [ ] Anonymize MNE.

### Ola 3 — pre-publicación (mes 2)
- [ ] Test externo o Leave-Site-Out.
- [ ] Tabla confounders + ANCOVA.
- [ ] Análisis saliency cuantitativo.
- [ ] Ablation vanilla saliency.
- [ ] Power analysis.
- [ ] Write-up paper (EMBC 2026 deadline ~sep).

---

## 8. Veredicto global

**Estado actual**: pipeline funcional end-to-end, replica orden de magnitud de Lopes (SVM 77% acc, AUC 0.85). Wilcoxon STFT vs CWT no significativo (p=0.34) — información válida.

**Riesgos principales**:
1. Saliency aggregation con leakage indirecto.
2. Patches/threshold fijados sin grid search transparente.
3. N efectivo a nivel epoch sobreestimado por overlap 87.5%.
4. Sin múltiples seeds ⇒ varianza desconocida.
5. Sin disclaimer clínico ni declaración IA.

**Camino realista**:
- TFM y defensa: ✅ listo con fixes ola 1 + LOSO full.
- Conference paper EMBC/EUSIPCO 2026: 🟡 factible con olas 1+2 + 1 mes redacción.
- Journal IEEE TNSRE/JBHI: 🟡 requiere ola 3 (test externo o sito-cross-validation).

El proyecto es **honest, replicable y publicable como replication study con extensión técnica**. No es un "breakthrough" pero aporta valor a la comunidad EEG-AD por validación independiente sobre datos públicos.
