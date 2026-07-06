#!/usr/bin/env bash
# Piloto del fix DSP (cwt_fair) — seed 0. Encadena modspec -> CNN -> saliency -> SVM.
# Universo de paths paralelo (method=cwt_fair): NO toca resultados de cwt/stft.
# Uso: bash scripts/run_pilot_cwt_fair.sh [SEED]
set -e
SEED="${1:-0}"
PY="D:/Universidad/Maestria/TPS/Proyecto/.venv/Scripts/python.exe"
cd "D:/Universidad/Maestria/TPS/Proyecto"

echo "=== [1/6] modspec cwt_fair (una vez para todas las seeds) ==="
"$PY" scripts/02_compute_modspec.py --method cwt_fair --fs 200

echo "=== [2/6] CNN LOSO cwt_fair seed=$SEED ==="
"$PY" scripts/03_train_loso.py --method cwt_fair --fs 200 --seed "$SEED"

echo "=== [3/6] saliency VAINILLA cwt_fair seed=$SEED (headline) ==="
"$PY" scripts/04_extract_saliency_features.py --method cwt_fair --fs 200 --seed "$SEED" \
    --grid-search --max-subjects-per-fold 20 --saliency-method vanilla

echo "=== [4/6] SVM VAINILLA cwt_fair seed=$SEED (headline) ==="
"$PY" scripts/05_run_svm.py --method cwt_fair --fs 200 --seed "$SEED" \
    --per-fold-patches --epochs-per-subject 80 --saliency-method vanilla

echo "=== [5/6] saliency GRAD-CAM cwt_fair seed=$SEED (completitud) ==="
"$PY" scripts/04_extract_saliency_features.py --method cwt_fair --fs 200 --seed "$SEED" \
    --grid-search --max-subjects-per-fold 20

echo "=== [6/6] SVM GRAD-CAM cwt_fair seed=$SEED (completitud) ==="
"$PY" scripts/05_run_svm.py --method cwt_fair --fs 200 --seed "$SEED" \
    --per-fold-patches --epochs-per-subject 80

echo "=== PILOTO seed=$SEED COMPLETO ==="
