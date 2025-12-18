#!/usr/bin/env bash
# run_experiments.sh (CPU)
# 執行三組 CPU OpenMP Monte Carlo 實驗，結果統一寫到 ../experiments/results/cpu_*.csv
# 不產生 verbose log，只顯示進度。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 校正參數
python ../tools/calibrate_from_data.py \
  --single-csv "../../data/single/AAPL.csv" \
  --multi-dir "../../data/multi" \
  --out "params.sh" > /dev/null 2>&1

source params.sh

# 統一輸出到 src/experiments/results/
OUT_DIR="../experiments/results"
mkdir -p "$OUT_DIR"

CSV_PATHS="${OUT_DIR}/cpu_paths_scaling.csv"
CSV_STEPS="${OUT_DIR}/cpu_steps_scaling.csv"
CSV_THREADS="${OUT_DIR}/cpu_threads_scaling.csv"

# 清空舊檔（如果存在）
rm -f "$CSV_PATHS" "$CSV_STEPS" "$CSV_THREADS"

echo "=== CPU Experiments ==="
echo "[1/3] Paths scaling (European / Asian)..."

PATHS_LIST=(1000000 5000000 10000000 50000000)
THREADS=16

for PATHS in "${PATHS_LIST[@]}"; do
  ./mc_pricer --type european \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --paths "$PATHS" --seed 42 --threads "$THREADS" --csv "$CSV_PATHS" > /dev/null

  ./mc_pricer --type asian \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --steps 252 --paths "$PATHS" --seed 43 --threads "$THREADS" --csv "$CSV_PATHS" > /dev/null
done

echo "[2/3] Steps scaling (Asian / Basket)..."

STEPS_LIST=(64 128 252 512)
PATHS_SINGLE=1000000
PATHS_BASKET=500000

for STEPS in "${STEPS_LIST[@]}"; do
  ./mc_pricer --type asian \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --steps "$STEPS" --paths "$PATHS_SINGLE" --seed 44 --threads "$THREADS" --csv "$CSV_STEPS" > /dev/null

  ./mc_pricer --type basket \
    --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
    --steps "$STEPS" --assets 8 --rho "$RHO_MULTI" \
    --paths "$PATHS_BASKET" --seed 45 --threads "$THREADS" --csv "$CSV_STEPS" > /dev/null
done

echo "[3/3] OpenMP threads scaling (Basket)..."

THREADS_LIST=(1 2 4 8 16 32)
PATHS_THREADS=2000000

for THR in "${THREADS_LIST[@]}"; do
  ./mc_pricer --type basket \
    --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
    --steps 252 --assets 8 --rho "$RHO_MULTI" \
    --paths "$PATHS_THREADS" --seed 46 --threads "$THR" --csv "$CSV_THREADS" > /dev/null
done

echo "Done. Results in: $OUT_DIR"
echo "  - cpu_paths_scaling.csv"
echo "  - cpu_steps_scaling.csv"
echo "  - cpu_threads_scaling.csv"
