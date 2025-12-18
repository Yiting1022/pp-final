#!/usr/bin/env bash
# run_experiments.sh (GPU)
# 執行三組 GPU Monte Carlo 實驗，結果統一寫到 ../experiments/results/gpu_*.csv
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

CSV_PATHS="${OUT_DIR}/gpu_paths_scaling.csv"
CSV_STEPS="${OUT_DIR}/gpu_steps_scaling.csv"
CSV_GPUS="${OUT_DIR}/gpu_multi_gpu_scaling.csv"

# 清空舊檔（如果存在）
rm -f "$CSV_PATHS" "$CSV_STEPS" "$CSV_GPUS"

echo "=== GPU Experiments ==="
echo "[1/3] Paths scaling (European / Asian)..."

PATHS_LIST=(10000000 50000000 100000000 500000000)
for PATHS in "${PATHS_LIST[@]}"; do
  ./gpu_pricer --type european \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --paths "$PATHS" --seed 42 --csv "$CSV_PATHS" > /dev/null

  ./gpu_pricer --type asian \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --steps 252 --paths "$PATHS" --seed 43 --csv "$CSV_PATHS" > /dev/null
done

echo "[2/3] Steps scaling (Asian / Basket)..."

STEPS_LIST=(64 128 252 512)
PATHS_SINGLE=1000000
PATHS_BASKET=500000

for STEPS in "${STEPS_LIST[@]}"; do
  ./gpu_pricer --type asian \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --steps "$STEPS" --paths "$PATHS_SINGLE" --seed 44 --csv "$CSV_STEPS" > /dev/null

  ./gpu_pricer --type basket \
    --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
    --steps "$STEPS" --assets 8 --rho "$RHO_MULTI" \
    --paths "$PATHS_BASKET" --seed 45 --gpus 1 --csv "$CSV_STEPS" > /dev/null
done

echo "[3/3] Multi-GPU scaling (Basket)..."

GPUS_LIST=(1 2 4)
PATHS_MULTI=2000000

for GPUS in "${GPUS_LIST[@]}"; do
  ./gpu_pricer --type basket \
    --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
    --steps 252 --assets 8 --rho "$RHO_MULTI" \
    --paths "$PATHS_MULTI" --gpus "$GPUS" --seed 46 --csv "$CSV_GPUS" > /dev/null
done

echo "Done. Results in: $OUT_DIR"
echo "  - gpu_paths_scaling.csv"
echo "  - gpu_steps_scaling.csv"
echo "  - gpu_multi_gpu_scaling.csv"
