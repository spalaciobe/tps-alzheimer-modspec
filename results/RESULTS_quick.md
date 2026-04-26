# Resultados — corrida quick (2026-04-25)

Pipeline end-to-end completado en una sesión sobre RTX 3050 Laptop (CUDA 12.4).

## Configuración

- **Dataset**: OpenNeuro ds004504, 65 sujetos (36 AD + 29 HC), fs=500 Hz → resampleado a 200 Hz.
- **Preproceso**: filtro 0.5–45 Hz FIR + ICA Infomax + ICLabel (rechazo eye/muscle/heart/line/channel).
- **Modspec**: 45×45 a 1 Hz (carrier 0.5–45 Hz, mod 0–22.5 Hz). STFT (Hann, nperseg=128) y CWT-Morlet (cmor1.5-1.0, 32 escalas log).
- **CNN**: réplica fiel de Lopes — 2 conv + 3 FC LeakyReLU, dropout 0.85, Nadam lr=1e-4, weight_decay=1e-2.
- **Validación**: LOSO-CV (65 folds), 1 sujeto extra para val/early stopping.
- **Quick mode**: epochs=10, batch=32, subsample=100 epochs/sujeto en train. Para validar pipeline en 13 min/método.
- **Saliency**: Grad-CAM, agregado 5 folds × 10 sujetos/fold.
- **Patches**: percentil 88, KMeans K=4 sobre el mapa diferencial AD−HC.
- **SVM**: RBF, γ=1/24, C=1, features = potencia patches + ratios (19 canales × 4 patches + 19 × 6 ratios), top-24 ANOVA, MinMax [-1, 1]. Subsample 80 epochs/sujeto.

## Resultados a nivel de sujeto (LOSO 65)

### CNN end-to-end (quick mode — sub-entrenado)

| Método | Accuracy | F1 macro | Sensibilidad (AD) | Especificidad (HC) | AUC |
|---|---|---|---|---|---|
| **STFT** | 0.554 | 0.538 | 0.667 | 0.414 | 0.540 |
| **CWT**  | 0.538 | 0.536 | 0.556 | 0.517 | 0.600 |

Wilcoxon pareado por sujeto: **p = 0.297** (no significativo). Bootstrap CI95: STFT [0.43, 0.66], CWT [0.42, 0.66]. Solapan completamente.

### SVM con patches saliency-guided (réplica del paper)

| Método | Accuracy | F1 macro | Sensibilidad | Especificidad | AUC |
|---|---|---|---|---|---|
| **STFT** | **0.769** | **0.764** | **0.833** | **0.690** | **0.854** |
| **CWT**  | 0.692 | 0.689 | 0.722 | 0.655 | 0.739 |

Wilcoxon score pareado: **p = 0.961** (no significativo). Wilcoxon correctness: **p = 0.336**. Bootstrap CI95: STFT [0.65, 0.86], CWT [0.58, 0.80].

## Lectura

1. **El pipeline replica con éxito el orden de magnitud del paper**: SVM con patches saliency-guided alcanza accuracy 0.77 y AUC 0.85 en AD vs HC. El paper de Lopes reporta T2 (N vs AD) accuracy 0.71 ± 0.02 en test SVM. **Resultado coherente** dada la diferencia de dataset y el modo quick.

2. **STFT > CWT en quick mode** numéricamente (Δ accuracy 7.7%), pero **la diferencia NO es estadísticamente significativa** (p > 0.3 en ambos clasificadores). Los CIs95 solapan ampliamente.

3. **Implicación**: la hipótesis de la propuesta ("CWT capta mejor las modulaciones de bajas frecuencias") **no se confirma en quick mode**. Hace falta correr el LOSO completo (epochs=50, sin subsample) para tener un veredicto definitivo. Ver `docs/optimization_options.md` — Colab T4 lo hace en ~40 min.

4. **CNN end-to-end ≈ azar** en quick mode. Era esperable: 10 epochs con 100 muestras/sujeto no es entrenamiento suficiente para una CNN con dropout 0.85.

## Para el run completo (recomendado)

Quitar `--quick`, opcionalmente añadir AMP (ya implementado en `src/train.py`). Estimado:
- Local RTX 3050 con AMP + batch 128: ~1.7h/método.
- Colab T4: ~20 min/método.

## Archivos generados

- `results/stft_200_seed0_quick/fold_results.json` (CNN STFT)
- `results/cwt_200_seed0_quick/fold_results.json` (CNN CWT)
- `results/svm_stft_200_seed0_quick/{fold_results,summary}.json`
- `results/svm_cwt_200_seed0_quick/{fold_results,summary}.json`
- `results/compare_{cnn,svm}_200_seed0_quick.json`
- `data/derivatives/saliency/{stft,cwt}_200_seed0_quick/saliency_{AD,HC,diff}.npy`
- `data/derivatives/saliency/{stft,cwt}_200_seed0_quick/patch_masks.npy`
- `results/figures_quick/{modspec_means_stft,modspec_means_cwt,saliency_compare,compare_svm,compare_cnn,roc_svm,confusion_svm}.png`
