# Resultados FULL — Replicación Lopes et al. 2023 sobre ds004504 (multi-seed, 3 seeds)

## Configuración común
- Dataset: OpenNeuro ds004504 — 65 sujetos AD vs HC (36 AD + 29 HC).
- Preproceso: FIR 0.5–45 Hz fase cero + ICA Infomax + ICLabel + resample 200 Hz.
- Epochs: 8 s con paso 1 s, ~470 epochs/sujeto.
- Modspec: **45×45 tras resize bilinear** (Δf real STFT = 1.5625 Hz a fs=200 Hz, no 1 Hz nominal).
- CNN: réplica funcional del paper (2 conv + 3 FC, dropout 0.85, Nadam lr=1e-4); batch_size=128 vs 4 del paper.
- Validación: LOSO-CV 65 folds × 3 seeds (s0, s1, s2).
- Anti-leakage: saliency y patches POR FOLD; z-score, scaler, ANOVA fit solo en train.

## ⚠️ Desviaciones respecto al paper

- **Saliency principal**: Grad-CAM (alineado con propuesta del alumno) + vainilla como ablation (paper-faithful, configuración usada para conclusiones cuantitativas).
- **batch_size**: 128 (GPU permite mayor que el 4 del paper; AMP + class weights mantienen estabilidad).
- **wICA → ICA + ICLabel**: alternativa reproducible y estándar en MNE (Pion-Tonachini 2019).
- **Re-referenciado**: CAR en lugar de A1/A2 (ds004504 trae su propia ref).
- **CWT-Morlet**: extensión nueva no presente en el paper.
- **Grid search de patches**: heurístico (max separabilidad AD vs HC en saliency), no nested CV con SVM en val set; ver limitación en INFORME_TFM §5.5.2.
- **Resolución STFT real**: 1.5625 Hz (Δf=fs/nperseg con nperseg=128, fs=200), no 1 Hz exacto; el "45×45" sale del resize bilinear posterior, que es interpolación.

---

## Tabla maestra de resultados multi-seed (media ± SD, n=3 seeds)

Tres representaciones T-F: **STFT**, **CWT nativa** y **CWT-fair** (CWT con el eje de modulación decimado a 3.125 Hz para igualar el Nyquist de la STFT — misma portadora, mismo eje de modulación). CWT-fair es la condición de control que aísla la transformada del artefacto de DSP (ver `docs/INFORME_TFM.md §3.3`).

| Clasificador | Saliency | Método T-F | Accuracy | F1 macro | AUC |
|---|---|---|---|---|---|
| CNN end-to-end | — | STFT | 0.656 ± 0.024 | 0.647 ± 0.024 | 0.695 ± 0.022 |
| CNN end-to-end | — | CWT nativa | 0.626 ± 0.071 | 0.625 ± 0.071 | 0.590 ± 0.069 |
| CNN end-to-end | — | **CWT-fair** | 0.656 ± 0.062 | 0.655 ± 0.061 | 0.667 ± 0.067 |
| SVM patches | Grad-CAM | STFT | 0.662 ± 0.031 | 0.643 ± 0.033 | 0.713 ± 0.020 |
| SVM patches | Grad-CAM | CWT nativa | **0.703 ± 0.009** | **0.697 ± 0.012** | **0.800 ± 0.010** |
| SVM patches | Grad-CAM | **CWT-fair** | 0.682 ± 0.054 | 0.673 ± 0.050 | 0.769 ± 0.045 |
| SVM patches | **Vainilla** (paper) | **STFT** | **0.764 ± 0.009** | **0.760 ± 0.008** | **0.856 ± 0.022** ⭐ |
| SVM patches | Vainilla | CWT nativa | 0.677 ± 0.081 | 0.670 ± 0.085 | 0.778 ± 0.038 |
| SVM patches | Vainilla | **CWT-fair** | 0.754 ± 0.046 | 0.750 ± 0.046 | 0.828 ± 0.031 |

⭐ Mejor AUC individual (STFT vainilla). **Pero** con el eje de modulación igualado (CWT-fair), STFT y CWT son estadísticamente indistinguibles: DeLong STFT vs CWT-fair p=0.318 (vainilla), 0.587 (Grad-CAM), ninguna sobrevive BH-FDR. La brecha aparente STFT↔CWT era mayoritariamente el artefacto de DSP del eje de modulación (ver sección de tests abajo).

---

## Detalle por seed

### CNN end-to-end

| Método | seed=0 | seed=1 | seed=2 |
|---|---|---|---|
| STFT | acc 0.631 / AUC 0.680 | 0.677 / 0.686 | 0.662 / 0.720 |
| CWT  | acc 0.708 / AUC 0.670 | 0.585 / 0.555 | 0.585 / 0.545 |

### SVM Grad-CAM

| Método | seed=0 | seed=1 | seed=2 |
|---|---|---|---|
| STFT | 0.692 / 0.690 | 0.631 / 0.719 | 0.662 / 0.729 |
| CWT  | 0.708 / 0.792 | 0.692 / 0.811 | 0.708 / 0.796 |

### SVM Vainilla (paper-faithful)

| Método | seed=0 | seed=1 | seed=2 |
|---|---|---|---|
| STFT | 0.769 / 0.843 | 0.754 / 0.844 | 0.769 / 0.881 |
| CWT nativa | 0.769 / 0.815 | 0.615 / 0.739 | 0.646 / 0.781 |

---

## El confound DSP del eje de modulación — y cómo se controla

- **STFT**: `nperseg=128, noverlap=64` ⇒ `dt=0.32 s` ⇒ **Nyquist modulación = 1.5625 Hz**, ~13 bins reales interpolados a 45.
- **CWT nativa**: `dt=1/200=0.005 s` ⇒ Nyquist modulación = 100 Hz ⇒ **~180 bins** subsampleados a 45.
- **CWT-fair**: decima la envolvente de la CWT a 3.125 Hz (anti-alias) ⇒ mismo Nyquist e igual nº de bins reales que STFT. Misma portadora que la CWT nativa; solo cambia el eje de modulación.

Comparar STFT vs **CWT-fair** aísla la transformada a igualdad de eje (comparación justa); CWT nativa vs CWT-fair mide el efecto del eje (el artefacto).

## Tests estadísticos — DeLong AUC por seed + combinación (Stouffer)

DeLong pareado dentro de cada seed (n=65) combinado entre seeds. La columna BH-FDR q corrige la **familia de m=3 comparaciones justas** (STFT vs CWT-fair en CNN, Grad-CAM, vainilla); las filas *confundidas* y *efecto-eje* se muestran solo como contexto.

| Saliency | Comparación | Δ AUC por seed | Stouffer p | BH-FDR q |
|---|---|---|---|---|
| Vainilla | STFT vs CWT **nativa** (confundida) | +0.028, +0.105, +0.101 | 0.061 | — (contexto) |
| Vainilla | **STFT vs CWT-fair** (justa) | −0.021, +0.036, +0.069 | **0.318** | 0.679 |
| Vainilla | CWT nativa vs CWT-fair (efecto eje) | −0.049, −0.069, −0.032 | 0.283 | — |
| Grad-CAM | STFT vs CWT nativa (confundida) | (CWT +0.087) | 0.068 | — (contexto) |
| Grad-CAM | **STFT vs CWT-fair** (justa) | −0.124, −0.004, −0.040 | **0.587** | 0.679 |
| Grad-CAM | CWT nativa vs CWT-fair (efecto eje) | −0.021, +0.088, +0.027 | 0.527 | — |
| CNN | **STFT vs CWT-fair** (justa) | −0.034, +0.096, +0.025 | **0.679** | 0.679 |

**Lectura final**:

- La ventaja marginal de STFT sobre la CWT **nativa** (vainilla, p=0.061) **se disuelve a p=0.318 cuando se iguala el eje** (STFT vs CWT-fair). Lo mismo en Grad-CAM: la ventaja aparente de la CWT nativa (p=0.068) cae a p=0.587 con CWT-fair.
- En **ambos** métodos de saliency, igualar el eje mueve la CWT hacia la STFT: |Δ AUC| baja de 0.078→0.028 (vainilla, ~64%) y 0.087→0.056 (Grad-CAM, ~36%).
- **Ninguna comparación justa (STFT vs CWT-fair) se acerca a la significancia** (q ≈ 0.68 en las tres, m=3). **A igualdad de eje de modulación, STFT y CWT-Morlet son estadísticamente indistinguibles**: las diferencias aparentes eran en gran parte el artefacto de DSP.
- **Caveat**: CWT-fair "iguala hacia abajo" (descarta modulaciones >1.56 Hz). Fisiológicamente razonable (AM diagnóstica lenta en EA), pero la definición alternativa "hacia arriba" (STFT con hop fino) queda como trabajo futuro. Ver `docs/INFORME_TFM.md §5.1, §5.5`.

## Análisis post-hoc de bajo coste (sin re-entrenar) — `results/fusion_tost_analysis.json`

- **Late-fusion STFT+CWT-fair** (promedio de scores, vainilla): AUC **0.867 ± 0.014** vs STFT 0.856 ± 0.022 y CWT-fair 0.828 ± 0.031. Mejora leve + menor varianza, pero **inconsistente** (por seed +0.033/+0.008/−0.007) y no se sostiene en el ensemble mediana (STFT 0.900 vs fusión 0.887) → indicio de complementariedad parcial, no concluyente.
- **TOST de equivalencia** (δ=±0.05) sobre ΔAUC STFT−CWT-fair: TOST seed-level p=0.246; bootstrap ensemble ΔAUC=+0.051, IC90 [−0.000, +0.107] (no cabe en ±0.05; δ mínimo ≈0.11). **La conclusión correcta es "no se detecta diferencia", NO "equivalencia demostrada"** (poca potencia con 3 seeds).

## Posicionamiento vs estado del arte

Posicionamiento vs literatura (con búsqueda web) en `docs/EVALUACION_SOTA.md`. Resumen: bajo LOSO honesto en ds004504 la literatura reporta ~71–83% acc; este trabajo (0.764/0.856) está por encima de la media LOSO y ~7 pts bajo el mejor comparable (DICE-net 83.28% LOSO), lejos de foundation models (LEAD ~91% F1). El valor del trabajo es la replicación leakage-free + el aporte metodológico (CWT-fair), no la accuracy. La novedad del control CWT-fair frente al linaje Falk/Cassani/Fraga se detalla en `docs/INFORME_TFM.md §5.7`.

---

## Comparación con el paper original

| Métrica | Paper Lopes T2 (LOSO test) | Este TFM (SVM vainilla STFT, 3 seeds) |
|---|---|---|
| N sujetos | 39 (20 N + 19 AD1) | 65 (29 HC + 36 AD) |
| Population HC | "Normal" (MMSE no especificado) | MMSE 30.0 ± 0.0 (perfecto) |
| Anti-leakage en patches | No documentado explícitamente | Por fold (estricto) |
| Accuracy | 0.71 ± 0.02 | 0.764 ± 0.009 |
| F1 | 0.61 ± 0.02 | 0.760 ± 0.008 |
| AUC | no reportado | 0.856 ± 0.022 |

**Lectura**: el pipeline replicado produce métricas **en el mismo rango** sobre datos públicos independientes (~+5 puntos). La comparación numérica directa NO es estricta por las diferencias de población, dataset y procedimientos. Lo defendible es que **el método de Lopes funciona sobre ds004504** — objetivo de una replicación.

---

## Hallazgos adicionales (post_experiments.json)

### Correlación 2D entre saliency maps (Pearson r, n=3 seeds)

| Comparación | r ± SD |
|---|---|
| STFT vs CWT (Grad-CAM) | −0.237 ± 0.119 |
| STFT vs CWT (vainilla) | −0.086 ± 0.076 |
| Grad-CAM vs vainilla (STFT) | +0.114 ± 0.043 |
| Grad-CAM vs vainilla (CWT)  | −0.166 ± 0.106 |

Saliency maps son sustancialmente diferentes entre métodos. El "biomarcador descubierto" depende del pipeline.

### Bandas canónicas (top-10% píxeles por banda) — **solo STFT**

La CWT usa eje portador log-espaciado (geomspace); la asignación píxel→banda requeriría otro mapeo, así que no se reporta (ver `docs/INFORME_TFM.md §5.5`).

| Configuración | δ | θ | α | β | γ |
|---|---|---|---|---|---|
| **STFT vainilla** | 5.8% | **27.9%** | **61.1%** | 5.3% | 0.0% |
| STFT Grad-CAM | 0.0% | 0.0% | 0.0% | 0.7% | 99.3% |

**STFT vainilla concentra su saliency en el biomarcador clásico de EA (alpha attenuation + theta increase)**, coherente con su mayor AUC media (0.856). Es una **observación correlacional post-hoc**, no prueba de superioridad: a igualdad de eje de modulación STFT y CWT-fair son indistinguibles (DeLong p=0.318, NS). Además, la atribución por banda solo aplica a STFT (la CWT usa eje portador geomspace, mapeo distinto).

### Consistencia de patches (Jaccard entre folds, 200 pares aleatorios)

| Configuración | Jaccard medio ± SD |
|---|---|
| STFT Grad-CAM | 0.050–0.063 |
| STFT vainilla | 0.036–0.051 |
| CWT Grad-CAM | 0.042–0.045 |
| CWT vainilla | 0.026–0.030 |

Patches MUY inestables entre folds. La saliency es ruidosa fold-a-fold, aunque las métricas globales sean buenas.

### Confounders AD vs HC

| Variable | AD | HC | Test | p | Conclusión |
|---|---|---|---|---|---|
| Edad | 66.4 ± 7.9 | 67.9 ± 5.4 | t-test | 0.38 | OK |
| MMSE | 17.8 ± 4.5 | 30.0 ± 0.0 | t-test | <0.001 | esperado |
| Género (F/M) | 24/12 | 11/18 | χ² | **0.039** | sesgo |

Sesgo de género detectado: discutir como limitación.

---

## Archivos generados

- `results/multiseed_analysis.json`: tablas y stats multi-seed.
- `results/post_experiments/post_experiments.json`: experimentos abiertos.
- `results/figures/`: 7 figuras estándar (Grad-CAM).
- `results/figures_vanilla/`: 7 figuras con saliency vainilla.
- `results/figures_multiseed/`: 13 figuras con error bars y boxplots.
- `results/figures_multiseed/auc_master_summary.png`: figura maestra (recomendada para portada).
- `docs/INFORME_TFM.md`: informe completo del TFM.

---

## Lectura final (con matices estadísticos)

1. **Replicación funcional** del pipeline de Lopes 2023 sobre dataset público ds004504: **Acc SVM vainilla STFT = 0.764 ± 0.009 vs 0.71 ± 0.02** del paper (comparación Acc-con-Acc; el paper no reporta AUC). El AUC 0.856 ± 0.022 se reporta como poder discriminativo. Comparación numérica orientativa por diferencias de dataset/población.
2. **A igualdad de eje de modulación, STFT y CWT-Morlet son indistinguibles; las diferencias aparentes eran DSP**. La condición de control CWT-fair (mismo eje de modulación que STFT) muestra: vainilla, la CWT pasa de −0.078 AUC vs STFT (nativa, p=0.061) a −0.028 (fair, p=0.318); Grad-CAM, de +0.087 (nativa) a +0.056 (fair, p=0.587). En ambas direcciones igualar el eje mueve la CWT hacia la STFT (magnitud distinta: ~64% vainilla, ~36% Grad-CAM). Ninguna comparación justa sobrevive BH-FDR (q≈0.68, m=3). **No encontramos evidencia de que la elección STFT vs CWT cambie el desempeño del pipeline una vez controlado el eje** (potencia limitada, 3 seeds).
3. **El confound DSP fue identificado y controlado**: STFT (Nyquist mod. 1.56 Hz) vs CWT nativa (100 Hz) no codifican el mismo eje vertical. En vez de dejarlo como limitación, se añade CWT-fair (decima la envolvente a 3.125 Hz, igualando el eje) para separar transformada de artefacto. Caveat: iguala "hacia abajo" (descarta modulaciones >1.56 Hz); la variante "hacia arriba" queda como trabajo futuro (§5.1, §5.5).
4. **El ranking según método de saliency también era, en parte, el confound**: con CWT nativa el ranking se invierte (Grad-CAM: CWT>STFT; vainilla: STFT>CWT), pero la inversión se atenúa con CWT-fair. Las saliency maps están **débilmente correlacionadas, cerca de cero** (r ≈ −0.09 vainilla, −0.24 Grad-CAM; con n=3 sin afirmar independencia); descubren regiones distintas, pero eso no se traduce en diferencia de desempeño con el eje igualado.
5. **Observaciones exploratorias (post-hoc)**: STFT + vainilla concentra saliency en bandas alpha (61%) + theta (28%) y canales occipito-temporales (O1, O2, T5, T6), coherente con literatura EEG-AD. Indicio prometedor, no biomarcador validado. **Atribución por banda solo para STFT** (eje lineal); CWT usa geomspace (ver §5.5).
6. **Limitaciones honestas**: solo 3 seeds; grid search de patches heurístico (no nested CV); CWT-fair iguala hacia abajo; CWT subexplorada (sin tuning específico); sesgo de género con poder limitado (**ausencia de evidencia, no evidencia de ausencia**); sin validación externa.
7. **Reproducibilidad**: pipeline en GitHub público, requirements-lock, 3 seeds, anti-leakage estricto en LOSO + saliency-por-fold, tests unitarios pasando.
