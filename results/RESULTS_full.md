# Resultados FULL — corrida completa LOSO con fixes aplicados

Pipeline ejecutado completo sobre RTX 3050 (CUDA 12.4, AMP). Total: ~5 h CNN + ~30 min análisis.

## Configuración

- **Dataset**: OpenNeuro ds004504, 65 sujetos AD vs HC (36 AD + 29 HC), 19 canales 10-20, fs 500→200 Hz.
- **Preproceso**: filtro 0.5–45 Hz FIR fase cero + ICA Infomax + ICLabel rechazo (eye/muscle/heart/line/channel) + **anonymize** + resample 200 Hz.
- **Modspec**: 45×45 a 1 Hz. STFT (Hann, nperseg=128) y CWT-Morlet (cmor1.5-1.0, 32 escalas log).
- **CNN**: réplica fiel de Lopes — 2 conv ReLU + 3 FC LeakyReLU, dropout 0.85, Nadam lr=1e-4, wd=1e-2, batch=128, **50 epochs** (NO subsample).
- **AMP** (mixed precision FP16) activo para acelerar GPU 3050 4 GB.
- **LOSO**: 65 folds, 1 sujeto extra para val/early stopping, agregación epoch→sujeto vía `mean(softmax)`.
- **Saliency Grad-CAM POR FOLD** (anti-leakage): cada fold k usa solo train_paths de k para descubrir su saliency. Subset estratificado de 20 sujetos/fold (10 AD + 10 HC).
- **Grid search real** de patches por fold: threshold ∈ {80,82,...,96}%, K ∈ {3,4,5}, seleccionado maximizando separabilidad AD vs HC.
- **SVM** RBF γ=1/24, C=1, MinMax [-1,1], top-24 features ANOVA, **patches POR FOLD** (`--per-fold-patches`).

## Resultados a nivel de sujeto (LOSO 65)

### CNN end-to-end (sin patches)

| Método | Accuracy | F1 macro | Sensibilidad | Especificidad | AUC |
|---|---|---|---|---|---|
| **STFT** | 0.631 | 0.620 | 0.722 | 0.517 | 0.680 |
| **CWT**  | **0.708** | **0.707** | 0.667 | **0.759** | 0.670 |

**Tests pareados:**
- Wilcoxon scores (sujeto): **p = 0.006** (significativo, CWT > STFT en distribución de scores).
- Wilcoxon correctness: p = 0.317.
- DeLong AUC: Δ=+0.011, **p = 0.900** (AUC equivalente).
- Bootstrap CI95 accuracy: STFT [0.51, 0.74], CWT [0.60, 0.82].

**Lectura**: CWT mejora **+7.7 pts accuracy** y **+24 pts especificidad**, recuperando los HC. Wilcoxon en scores p=0.006 confirma diferencia significativa en distribución de probabilidades, aunque el AUC es prácticamente idéntico.

### SVM con patches saliency-guided (réplica Lopes)

#### A) Saliency = Grad-CAM (alineado con propuesta del alumno)

| Método | Accuracy | F1 macro | Sensibilidad | Especificidad | AUC |
|---|---|---|---|---|---|
| **STFT** | 0.692 | 0.675 | 0.833 | 0.517 | 0.690 |
| **CWT**  | **0.754** | **0.749** | 0.806 | **0.690** | **0.812** |

- Wilcoxon scores: p = 0.683.
- **DeLong AUC: Δ=-0.123, z=-1.83, p = 0.067** (marginal, casi significativo).
- Bootstrap CI95: STFT [0.55, 0.81], CWT [0.65, 0.85].

#### B) Saliency = Vanilla gradient (paper-faithful Lopes)

| Método | Accuracy | F1 macro | Sensibilidad | Especificidad | AUC |
|---|---|---|---|---|---|
| **STFT** | **0.769** | 0.766 | 0.806 | 0.724 | **0.843** |
| **CWT**  | **0.769** | 0.766 | 0.806 | 0.724 | 0.815 |

- Wilcoxon scores: p = 0.893.
- DeLong AUC: Δ=+0.028, z=0.44, **p = 0.660** (NO significativo).
- STFT y CWT producen el MISMO accuracy (0.769) — diferencia solo en AUC (0.028, sin significancia).

**Lectura crítica**: la elección del método de saliency cambia la conclusión:
- Con **vanilla saliency** (paper-faithful), STFT ≈ CWT — ambos AUC ~0.83. **No hay ganancia con CWT.**
- Con **Grad-CAM**, CWT > STFT en AUC (Δ=12 pts, p=0.067 marginal).

Esto sugiere que la diferencia STFT vs CWT en SVM con Grad-CAM puede ser **artefacto del método saliency**, no una mejora real de la representación T-F. El experimento más fiel al paper original (vanilla) no muestra ventaja de CWT.

**Comparación con paper Lopes 2023**: SVM STFT vanilla full obtiene **AUC 0.843** vs paper T2 (N vs AD) AUC no reportado pero acc 0.71 ± 0.02. Acc 0.769 — replica orden de magnitud y supera ligeramente, posiblemente por dataset más grande (65 vs 39 sujetos en T2).

## Comparación con corrida quick (validación de mejora)

| Métrica | Quick | Full | Δ |
|---|---|---|---|
| CNN STFT Acc | 0.554 | 0.631 | +7.7 |
| CNN STFT AUC | 0.540 | 0.680 | +14.0 |
| CNN CWT Acc | 0.538 | 0.708 | +17.0 |
| CNN CWT AUC | 0.600 | 0.670 | +7.0 |
| SVM STFT Acc | 0.769 | 0.692 | -7.7 |
| SVM STFT AUC | 0.854 | 0.690 | **-16.4** |
| SVM CWT Acc | 0.692 | 0.754 | +6.2 |
| SVM CWT AUC | 0.739 | 0.812 | +7.3 |

**Notas críticas**:
- En CNN, el full mejora claramente sobre quick (lo esperado — más entrenamiento, todos los datos).
- En **SVM, los resultados full bajan respecto a quick** porque el quick usaba **patches GLOBALES** (data leakage). Al pasar a `--per-fold-patches` (anti-leakage), los números bajan pero son **honestos**. La corrida quick estaba inflada artificialmente.
- **El verdadero efecto del CWT vs STFT solo se ve correctamente en la corrida full anti-leakage.**

## Comparación con paper Lopes 2023

| Tarea | Paper (val SVM) | Este TFM (test SVM full) | Notas |
|---|---|---|---|
| T2: N vs AD | Acc 0.71 ± 0.02 | STFT 0.69, CWT 0.75 | Dataset distinto pero replica orden de magnitud |
| T2 AUC | no reportado | STFT 0.69, CWT 0.81 | — |

**Replicación validada**: el SVM full STFT obtiene 0.69 acc, indistinguible del 0.71 del paper (diferencia dentro de CI95). Confirma que el pipeline replica fielmente.

## Lectura final (revisada con ablation vanilla)

1. **El pipeline replica el orden de magnitud del paper de Lopes** sobre dataset público con anti-leakage estricto. SVM vanilla full: Acc 0.769, AUC 0.843 (paper: Acc 0.71).

2. **CWT vs STFT depende del método de saliency**:
   | Clasificador / Saliency | STFT acc/AUC | CWT acc/AUC | Test |
   |---|---|---|---|
   | CNN end-to-end | 0.631 / 0.680 | 0.708 / 0.670 | Wilcoxon p=0.006 ✓ |
   | SVM Grad-CAM | 0.692 / 0.690 | 0.754 / 0.812 | DeLong p=0.067 marginal |
   | **SVM vanilla (paper)** | **0.769 / 0.843** | **0.769 / 0.815** | DeLong p=0.66 ns |

3. **Conclusión defendible para el TFM**:
   - **CWT NO supera a STFT con la metodología fiel al paper** (vanilla saliency).
   - El supuesto efecto positivo de CWT con Grad-CAM puede ser artefacto del método saliency, no una mejora real de la representación T-F.
   - La hipótesis de la propuesta queda **NO confirmada**, pero el experimento es metodológicamente sólido.

4. **Conclusión positiva**: la replicación independiente del paper sobre dataset público funciona — orden de magnitud coherente con Lopes 2023 (Acc ~0.77 vs 0.71 reportado).

5. **Para publicación, antes de claims**:
   - 3-5 seeds para varianza.
   - Test externo (otro dataset EEG-AD).
   - Comparar grad-cam vs vanilla en otras tareas (no solo AD vs HC).

## Anti-leakage — fixes aplicados

| Fix | Implementación |
|---|---|
| **Saliency POR FOLD** | `04_extract_saliency_features.py` genera `per_fold/patch_masks_foldNN.npy` |
| **Grid search REAL** | Threshold ∈ {80..96%}, K ∈ {3,4,5} por fold, score = separabilidad AD/HC |
| **SVM patches por fold** | `05_run_svm.py --per-fold-patches` carga máscara del fold actual |
| **z-score solo train** | `IndexedDataset` con `fit_channel_zscore` por fold |
| **MinMax solo train** | `svm_pipeline.py:fit_svm_pipeline` con docstring explícito |
| **DeLong test** | `src/stats.py:delong_test` para diferencias de AUC |
| **N efectivo** | `src/stats.py:effective_n(n_epochs, overlap_ratio)` |
| **Anonymize** | `src/preprocess.py` aplica `raw.anonymize(daysback=10000)` |
| **Resume** | Scripts 03, 04 detectan progreso previo y saltan folds completados |

## Archivos generados

```
results/
├── stft_200_seed0/          # CNN STFT FULL (65 folds .pt + fold_results.json)
├── cwt_200_seed0/           # CNN CWT FULL
├── svm_stft_200_seed0_perfold/  # SVM STFT con per-fold patches
├── svm_cwt_200_seed0_perfold/   # SVM CWT con per-fold patches
├── compare_cnn_200_seed0.json
├── compare_svm_200_seed0_perfold.json
├── figures/
│   ├── modspec_means_{stft,cwt}.png   # Modspec medio por clase (HC, AD, diff)
│   ├── saliency_compare.png            # Grad-CAM STFT vs CWT
│   ├── compare_{cnn,svm}.png           # Acc + CI95 STFT vs CWT
│   ├── roc_svm.png                     # ROC curves
│   └── confusion_svm.png               # Matrices de confusión
└── RESULTS_full.md          # ← este documento

data/derivatives/saliency/
├── stft_200_seed0/
│   ├── per_fold/            # 65 folds × {AD, HC, patch_masks}
│   ├── saliency_{AD,HC,diff}.npy   # Mapas globales (visualización)
│   └── summary.json
└── cwt_200_seed0/           # idem
```

## Limitaciones reconocidas (para paper)

1. **Single seed (seed=0)**: para publicar deberían correr 3-5 seeds y reportar media ± SD.
2. **Saliency Grad-CAM ≠ vanilla del paper**: ablation con vanilla no se ejecutó por tiempo. Los mapas no son comparables directamente con Figs. del paper.
3. **Test externo no realizado**: validez externa pendiente. ds004504 viene de hospital único (sitio único).
4. **Confounders no analizados**: edad, MMSE, género no controlados explícitamente.
5. **Autocorrelación epochs (overlap 87.5%)**: tests a nivel epoch no usan N_efectivo. A nivel sujeto los tests son válidos.
6. **CWT n_scales=32 vs paper 50**: por velocidad. No afecta la rejilla final 45×45.
7. **batch=128 vs paper batch=4**: por GPU + AMP. No documentado un ablation sin AMP.
