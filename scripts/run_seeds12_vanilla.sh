#!/usr/bin/env bash
# Seeds 1 y 2 de cwt_fair, SOLO cadena vainilla (headline): CNN -> saliency vainilla -> SVM vainilla.
# modspec cwt_fair ya existe (compartido entre seeds) y se salta por resume.
# Gradcam se deja para después (secundario). Todo con --no-bank (RAM baja) y resume.
set -e
PY="D:/Universidad/Maestria/TPS/Proyecto/.venv/Scripts/python.exe"
cd "D:/Universidad/Maestria/TPS/Proyecto"

for SEED in 1 2; do
  echo "===== SEED $SEED ====="
  echo "=== modspec (skip si existe) ==="
  "$PY" scripts/02_compute_modspec.py --method cwt_fair --fs 200
  echo "=== CNN LOSO seed=$SEED (--no-bank) ==="
  "$PY" scripts/03_train_loso.py --method cwt_fair --fs 200 --seed "$SEED" --no-bank
  echo "=== saliency VAINILLA seed=$SEED ==="
  "$PY" scripts/04_extract_saliency_features.py --method cwt_fair --fs 200 --seed "$SEED" \
      --grid-search --max-subjects-per-fold 20 --saliency-method vanilla
  echo "=== SVM VAINILLA seed=$SEED (headline) ==="
  "$PY" scripts/05_run_svm.py --method cwt_fair --fs 200 --seed "$SEED" \
      --per-fold-patches --epochs-per-subject 80 --saliency-method vanilla
  echo "===== SEED $SEED VAINILLA COMPLETO ====="
done
echo "=== SEEDS 1 y 2 VAINILLA COMPLETO — listo para análisis 3-seed ==="
