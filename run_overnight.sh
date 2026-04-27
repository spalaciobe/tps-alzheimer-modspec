#!/usr/bin/env bash
# Pipeline encadenado para correr durante la noche.
# Espera SVM CWT, luego comparación, figuras y RESULTS_full.md.
set -e
cd "D:/Universidad/Maestria/TPS/Proyecto"
source .venv/Scripts/activate

echo "[$(date)] Esperando SVM CWT..."
until [[ -f results/svm_cwt_200_seed0_perfold/summary.json ]]; do sleep 60; done
echo "[$(date)] SVM CWT listo"

echo "[$(date)] Comparación STFT vs CWT (CNN y SVM)"
python scripts/06_compare_stft_cwt.py --classifier cnn 2>&1 | tail -3
python scripts/06_compare_stft_cwt.py --classifier svm 2>&1 | tail -3

echo "[$(date)] Generando figuras"
python scripts/07_generate_figures.py 2>&1 | tail -3

echo "[$(date)] Pipeline overnight DONE"
