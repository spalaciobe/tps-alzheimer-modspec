#!/usr/bin/env bash
# Orquestador v2: completar STFT/CWT seed=2 (con resume hasta 65/65) y
# luego saliency + SVM + comparativas + figuras para seeds 1 y 2.
#
# Diferencia con v1: verifica que cada LOSO llegue a 65/65 antes de pasar al
# siguiente paso, relanzando si exit != 0 hasta lograrlo (resume automático).
set -uo pipefail
cd "$(dirname "$0")/.."

PY="$(pwd)/.venv/Scripts/python.exe"
LOG="$(pwd)/results/run_remaining_v2.log"
mkdir -p "$(dirname "$LOG")"

step() { echo ""; echo "=== [$(date +%H:%M:%S)] $* ==="; }

count_folds() {
  local file="$1"
  [[ -f "$file" ]] && "$PY" -c "import json; print(len(json.load(open(r'$file'))))" 2>/dev/null || echo 0
}

run_loso_until_done() {
  # Reintenta el LOSO hasta que fold_results.json tenga 65 folds.
  local method="$1"; local seed="$2"
  local file="results/${method}_200_seed${seed}/fold_results.json"
  local max_retries=5
  for i in $(seq 1 $max_retries); do
    local n
    n=$(count_folds "$file")
    if [[ "$n" -ge 65 ]]; then
      step "OK ${method} s${seed} = $n/65 folds"
      return 0
    fi
    step "RUN [${i}/${max_retries}] LOSO ${method} s${seed} (actual: $n/65)"
    "$PY" scripts/03_train_loso.py --method "$method" --fs 200 --seed "$seed" || step "WARN exit=$?"
  done
  step "GIVE UP ${method} s${seed} tras $max_retries intentos"
  return 1
}

run() {
  step "RUN $*"
  "$@" || step "WARN exit=$? for $*"
}

# === 1. Completar STFT s2 ===
run_loso_until_done stft 2

# === 2. Completar CWT s2 ===
run_loso_until_done cwt 2

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

# === 5. Comparativas finales ===
run "$PY" scripts/06_compare_stft_cwt.py --classifier cnn
run "$PY" scripts/06_compare_stft_cwt.py --classifier svm --per-fold
run "$PY" scripts/06_compare_stft_cwt.py --classifier svm --per-fold --saliency-method vanilla

# === 6. Figuras ===
run "$PY" scripts/07_generate_figures.py --per-fold
run "$PY" scripts/07_generate_figures.py --per-fold --saliency-method vanilla

step "ALL DONE"
