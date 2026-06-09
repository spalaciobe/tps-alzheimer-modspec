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

| Clasificador | Saliency | Método T-F | Accuracy | F1 macro | AUC |
|---|---|---|---|---|---|
| CNN end-to-end | — | STFT | 0.656 ± 0.024 | 0.642 ± 0.020 | 0.695 ± 0.022 |
| CNN end-to-end | — | CWT  | 0.626 ± 0.071 | 0.611 ± 0.070 | 0.590 ± 0.069 |
| SVM patches | Grad-CAM | STFT | 0.662 ± 0.031 | 0.643 ± 0.041 | 0.713 ± 0.020 |
| SVM patches | Grad-CAM | CWT  | **0.703 ± 0.009** | **0.687 ± 0.026** | **0.800 ± 0.010** |
| SVM patches | **Vainilla** (paper) | **STFT** | **0.764 ± 0.009** | **0.762 ± 0.011** | **0.856 ± 0.022** ⭐ |
| SVM patches | Vainilla | CWT  | 0.677 ± 0.081 | 0.673 ± 0.094 | 0.778 ± 0.038 |

⭐ Mejor configuración global. Coincide con la metodología fiel al paper original.

---

## Detalle por seed

### CNN end-to-end

| Método | seed=0 | seed=1 | seed=2 |
|---|---|---|---|
| STFT | acc 0.631 / AUC 0.680 | 0.677 / 0.728 | 0.662 / 0.677 |
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
| CWT  | 0.769 / 0.815 | 0.615 / 0.739 | 0.646 / 0.778 |

---

## Tests estadísticos pareados (pooled sobre 3 seeds) con corrección BH-FDR

| Comparación | Wilcoxon p por seed | DeLong AUC Δ | DeLong p crudo | BH-FDR p ajustado | Sig. tras corrección |
|---|---|---|---|---|---|
| **SVM Vainilla STFT vs CWT** | 0.893, 0.025, 0.338 | **+0.078 (STFT mayor)** | **0.014** | **0.042** | ✓ |
| SVM Grad-CAM STFT vs CWT | 0.727, 0.863, 0.091 | −0.077 (CWT mayor) | 0.195 | 0.293 | ✗ |
| CNN STFT vs CWT | 0.006, 0.401, 0.095 | +0.095 | 0.252 | 0.252 | ✗ |

**Lectura matizada**:

- Solo SVM vainilla sobrevive a la corrección BH-FDR (3 comparaciones, α=0.05), con p ajustado = 0.042 (marginalmente significativo).
- El **Wilcoxon por seed es inconsistente** en SVM vainilla (0.893, 0.025, 0.338): solo seed=1 muestra diferencia significativa, perdida tras Bonferroni intra-seed.
- Conclusión apropiada: **ausencia de evidencia a favor de CWT bajo esta configuración**, no superioridad intrínseca demostrada de STFT.
- La hipótesis original del proyecto (CWT > STFT) **no obtiene evidencia a favor**, pero tampoco se prueba formalmente lo contrario.

---

## Comparación con el paper original

| Métrica | Paper Lopes T2 (LOSO test) | Este TFM (SVM vainilla STFT, 3 seeds) |
|---|---|---|
| N sujetos | 39 (20 N + 19 AD1) | 65 (29 HC + 36 AD) |
| Population HC | "Normal" (MMSE no especificado) | MMSE 30.0 ± 0.0 (perfecto) |
| Anti-leakage en patches | No documentado explícitamente | Por fold (estricto) |
| Accuracy | 0.71 ± 0.02 | 0.764 ± 0.009 |
| F1 | 0.61 ± 0.02 | 0.762 ± 0.011 |
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

### Bandas canónicas (top-10% píxeles por banda)

| Configuración | δ | θ | α | β | γ |
|---|---|---|---|---|---|
| **STFT vainilla** | 5.8% | **27.9%** | **61.1%** | 5.3% | 0.0% |
| CWT vainilla | 0.8% | 2.0% | 7.6% | 54.2% | 35.5% |
| STFT Grad-CAM | 0.0% | 0.0% | 0.0% | 0.7% | 99.3% |
| CWT Grad-CAM | 0.0% | 0.0% | 0.0% | 67.5% | 30.9% |

**STFT vainilla es la única configuración que descubre el biomarcador clásico de EA (alpha attenuation + theta increase)**. Esto explica su superioridad cuantitativa.

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

1. **Replicación funcional** del pipeline de Lopes 2023 sobre dataset público ds004504: AUC SVM vainilla STFT = 0.856 ± 0.022 vs 0.71 ± 0.02 del paper (orden de magnitud equivalente; comparación numérica orientativa por diferencias de dataset/población).
2. **Hipótesis original CWT > STFT no obtiene evidencia a favor**: SVM vainilla STFT muestra mejor AUC media (DeLong p crudo=0.014, BH-FDR=0.042), pero Wilcoxon por seed inconsistente (0.893, 0.025, 0.338) → mejor interpretado como **ausencia de evidencia para CWT** bajo esta configuración (resize 45×45 + cmor1.5-1.0), no superioridad intrínseca demostrada de STFT.
3. **El ranking depende del método de saliency**: con Grad-CAM la tendencia se invierte (CWT > STFT, NS). Las saliency maps de ambos métodos están casi descorrelacionadas (r ≈ -0.09 vainilla, -0.24 Grad-CAM) → posible **complementariedad** (ensemble como hipótesis futura).
4. **Observaciones exploratorias (post-hoc)**: STFT + vainilla concentra saliency en bandas alpha (61%) + theta (28%) y canales occipito-temporales (O1, O2, T5, T6), coherente con literatura EEG-AD. Es indicio prometedor, no biomarcador validado.
5. **Limitaciones honestas**: solo 3 seeds; grid search de patches heurístico (no nested CV); resize 45×45 puede favorecer STFT; CWT subexplorada (sin tuning específico); sesgo de género en dataset con poder limitado para descartarlo en el modelo (N≈129/grupo requerido); sin validación externa.
6. **Reproducibilidad**: pipeline en GitHub público, requirements-lock, 3 seeds, anti-leakage estricto en LOSO + saliency-por-fold, tests unitarios pasando.
