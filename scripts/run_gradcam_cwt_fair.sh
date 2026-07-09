#!/usr/bin/env bash
# Gradcam cwt_fair x 3 seeds. La CNN ya está entrenada (65/65 por seed), así que
# solo faltan saliency gradcam + SVM gradcam. Resume-safe. --no-bank no aplica
# (04/05 no usan SubjectBank). Todo en universo cwt_fair (no toca cwt/stft).
set -e
PY="D:/Universidad/Maestria/TPS/Proyecto/.venv/Scripts/python.exe"
cd "D:/Universidad/Maestria/TPS/Proyecto"

for SEED in 0 1 2; do
  echo "===== GRADCAM SEED $SEED ====="
  echo "=== saliency GRAD-CAM seed=$SEED ==="
  "$PY" scripts/04_extract_saliency_features.py --method cwt_fair --fs 200 --seed "$SEED" \
      --grid-search --max-subjects-per-fold 20
  echo "=== SVM GRAD-CAM seed=$SEED ==="
  "$PY" scripts/05_run_svm.py --method cwt_fair --fs 200 --seed "$SEED" \
      --per-fold-patches --epochs-per-subject 80
  echo "===== GRADCAM SEED $SEED COMPLETO ====="
done
echo "=== GRADCAM cwt_fair 3 SEEDS COMPLETO ==="
