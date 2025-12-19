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
CSV_ASSETS="${OUT_DIR}/gpu_assets_scaling.csv"
CSV_BLOCKS="${OUT_DIR}/gpu_block_size_tuning.csv"
CSV_COMPARE="${OUT_DIR}/gpu_vs_cpu_comparison.csv"
CSV_OPTION_TYPES="${OUT_DIR}/option_types_comparison.csv"

# 清空舊檔（如果存在）
rm -f "$CSV_PATHS" "$CSV_STEPS" "$CSV_GPUS" "$CSV_ASSETS" "$CSV_BLOCKS" "$CSV_COMPARE" "$CSV_OPTION_TYPES"

echo "=== GPU Experiments ==="
echo "[1/7] Paths scaling (European / Asian)..."

PATHS_LIST=(1000000 5000000 10000000 50000000 100000000 500000000 1000000000)
for PATHS in "${PATHS_LIST[@]}"; do
  ./gpu_pricer --type european \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --paths "$PATHS" --seed 42 --csv "$CSV_PATHS" > /dev/null

  ./gpu_pricer --type asian \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --steps 252 --paths "$PATHS" --seed 43 --csv "$CSV_PATHS" > /dev/null
done

echo "[2/7] Steps scaling (Asian / European)..."

STEPS_LIST=(64 128 252 512)
PATHS_SINGLE=1000000

for STEPS in "${STEPS_LIST[@]}"; do
  ./gpu_pricer --type asian \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --steps "$STEPS" --paths "$PATHS_SINGLE" --seed 44 --csv "$CSV_STEPS" > /dev/null

  ./gpu_pricer --type european \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --steps "$STEPS" --paths "$PATHS_SINGLE" --seed 45 --csv "$CSV_STEPS" > /dev/null
done

echo "[3/7] Multi-GPU scaling (Basket)..."

GPUS_LIST=(1 2 4)
PATHS_MULTI=2000000

for GPUS in "${GPUS_LIST[@]}"; do
  ./gpu_pricer --type basket \
    --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
    --steps 252 --assets 8 --rho "$RHO_MULTI" \
    --paths "$PATHS_MULTI" --gpus "$GPUS" --seed 46 --csv "$CSV_GPUS" > /dev/null
done

echo "[4/7] Assets scaling (Basket - curse of dimensionality)..."

ASSETS_LIST=(2 4 8 16 32)
PATHS_ASSETS=500000

for ASSETS in "${ASSETS_LIST[@]}"; do
  ./gpu_pricer --type basket \
    --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
    --steps 252 --assets "$ASSETS" --rho "$RHO_MULTI" \
    --paths "$PATHS_ASSETS" --seed 47 --gpus 1 --csv "$CSV_ASSETS" > /dev/null
done

echo "[5/7] Block size tuning (European)..."

BLOCK_SIZES=(128 256 512)
BLOCKS_PER_SM_LIST=(2 4 8)
PATHS_TUNE=10000000

for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
  for BLOCKS_SM in "${BLOCKS_PER_SM_LIST[@]}"; do
    ./gpu_pricer --type european \
      --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
      --paths "$PATHS_TUNE" --seed 48 \
      --block_size "$BLOCK_SIZE" --blocks_per_sm "$BLOCKS_SM" \
      --csv "$CSV_BLOCKS" > /dev/null
  done
done

echo "[6/7] GPU vs CPU comparison (European / Asian / Basket)..."

COMPARE_PATHS_LIST=(100000 500000 1000000 5000000 10000000)

for PATHS in "${COMPARE_PATHS_LIST[@]}"; do
  # European
  ./gpu_pricer --type european \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --paths "$PATHS" --seed 50 --csv "$CSV_COMPARE" > /dev/null

  # Asian
  ./gpu_pricer --type asian \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --steps 252 --paths "$PATHS" --seed 51 --csv "$CSV_COMPARE" > /dev/null

  # Basket (8 assets)
  ./gpu_pricer --type basket \
    --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
    --steps 252 --assets 8 --rho "$RHO_MULTI" \
    --paths "$PATHS" --gpus 1 --seed 52 --csv "$CSV_COMPARE" > /dev/null
done

echo "[7/7] Option types comparison (European / Asian / Basket - same path count)..."

OPTION_TYPES_PATHS_LIST=(1000000 5000000 10000000)

for PATHS in "${OPTION_TYPES_PATHS_LIST[@]}"; do
  # European
  ./gpu_pricer --type european \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --paths "$PATHS" --seed 60 --csv "$CSV_OPTION_TYPES" > /dev/null

  # Asian
  ./gpu_pricer --type asian \
    --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
    --steps 252 --paths "$PATHS" --seed 61 --csv "$CSV_OPTION_TYPES" > /dev/null

  # Basket (8 assets)
  ./gpu_pricer --type basket \
    --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
    --steps 252 --assets 8 --rho "$RHO_MULTI" \
    --paths "$PATHS" --gpus 1 --seed 62 --csv "$CSV_OPTION_TYPES" > /dev/null
done

echo "Done. Results in: $OUT_DIR"
echo "  - gpu_paths_scaling.csv"
echo "  - gpu_steps_scaling.csv"
echo "  - gpu_multi_gpu_scaling.csv"
echo "  - gpu_assets_scaling.csv"
echo "  - gpu_block_size_tuning.csv"
echo "  - gpu_vs_cpu_comparison.csv"
echo "  - option_types_comparison.csv"
