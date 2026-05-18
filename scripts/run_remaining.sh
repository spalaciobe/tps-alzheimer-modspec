#!/usr/bin/env bash
# Orquestador: espera a que CWT seed=1 termine y luego corre todo lo pendiente.
# Pensado para correr en background; logs a results/run_remaining.log.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="$(pwd)/.venv/Scripts/python.exe"
LOG="$(pwd)/results/run_remaining.log"
mkdir -p "$(dirname "$LOG")"

step() {
  echo ""; echo "=== [$(date +%H:%M:%S)] $* ==="
}

wait_for_n() {
  local file="$1"; local need="$2"
  step "wait $file >= $need folds"
  until [[ -f "$file" ]] && [[ "$($PY -c "import json; print(len(json.load(open(r'$file'))))" 2>/dev/null || echo 0)" -ge "$need" ]]; do
    sleep 60
  done
  step "OK $file = $need folds"
}

run() {
  step "RUN $*"
  "$@" || step "WARN exit=$? for $*"
}

# === 1. Esperar a que CWT seed=1 termine (ya está en marcha en otra terminal) ===
wait_for_n results/cwt_200_seed1/fold_results.json 65

# === 2. STFT y CWT seed=2 (full LOSO) ===
run "$PY" scripts/03_train_loso.py --method stft --fs 200 --seed 2
run "$PY" scripts/03_train_loso.py --method cwt  --fs 200 --seed 2

# === 3. Saliency multi-seed (Grad-CAM y vanilla) ===
for S in 1 2; do
  for M in stft cwt; do
    run "$PY" scripts/04_extract_saliency_features.py --method "$M" --fs 200 --seed "$S" --grid-search --max-subjects-per-fold 20
    run "$PY" scripts/04_extract_saliency_features.py --method "$M" --fs 200 --seed "$S" --grid-search --max-subjects-per-fold 20 --saliency-method vanilla
  done
done

# === 4. SVM multi-seed × ambas saliency ===
for S in 1 2; do
  for M in stft cwt; do
    run "$PY" scripts/05_run_svm.py --method "$M" --fs 200 --seed "$S" --per-fold-patches --epochs-per-subject 80
    run "$PY" scripts/05_run_svm.py --method "$M" --fs 200 --seed "$S" --per-fold-patches --epochs-per-subject 80 --saliency-method vanilla
  done
done

# === 5. Comparativas finales (re-corren sobre seed=0 actualizado si hace falta) ===
run "$PY" scripts/06_compare_stft_cwt.py --classifier cnn
run "$PY" scripts/06_compare_stft_cwt.py --classifier svm --per-fold
run "$PY" scripts/06_compare_stft_cwt.py --classifier svm --per-fold --saliency-method vanilla

# === 6. Figuras ===
run "$PY" scripts/07_generate_figures.py --per-fold
run "$PY" scripts/07_generate_figures.py --per-fold --saliency-method vanilla

step "ALL DONE"
