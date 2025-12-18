#!/usr/bin/env bash
# run_all.sh
# Build all CUDA binaries and run European / Asian / Basket
# Monte Carlo pricers using parameters calibrated from real market data
# under ../data/single and ../data/multi.

set -euo pipefail

# Go to script directory (finalproject)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional CLI parameters:
#   $1 = PATHS_SINGLE  (for single-asset products, default 1000000)
#   $2 = PATHS_BASKET  (for basket/multi-asset products, default 500000)
PATHS_SINGLE="${1:-1000000}"
PATHS_BASKET="${2:-500000}"

echo "[1/3] Calibrating GBM parameters from market data..."
python ../tools/calibrate_from_data.py \
  --single-csv "../../data/single/AAPL.csv" \
  --multi-dir "../../data/multi" \
  --out "params.sh"

source params.sh

echo
echo "[2/3] Building CUDA pricer targets (shared mc_pricer core)..."
make -j4

echo
echo "[3/3] Running pricing examples with calibrated parameters..."

echo
echo "-- gpu_pricer --type european (single-asset European call) --"
./gpu_pricer \
  --type european \
  --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
  --paths "$PATHS_SINGLE" --seed 42

echo
echo "-- gpu_pricer --type asian (single-asset Asian arithmetic call) --"
./gpu_pricer \
  --type asian \
  --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
  --steps 252 --paths "$PATHS_SINGLE" --seed 43

echo
echo "-- gpu_pricer --type basket (multi-asset basket European call, 8 assets, single GPU) --"
./gpu_pricer \
  --type basket \
  --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
  --steps 252 --assets 8 --rho "$RHO_MULTI" \
  --paths "$PATHS_BASKET" --seed 44 --gpus 1

echo
echo "-- gpu_pricer --type basket (fixed 8-asset basket European call, single GPU) --"
./gpu_pricer \
  --type basket \
  --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
  --steps 252 --assets 8 --rho "$RHO_MULTI" \
  --paths "$PATHS_BASKET" --seed 45 --gpus 1

echo
echo "-- gpu_pricer --type basket (multi-GPU basket European call) --"
./gpu_pricer \
  --type basket \
  --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
  --steps 252 --assets 8 --rho "$RHO_MULTI" \
  --paths "$PATHS_BASKET" --gpus 2 --seed 46

echo
echo "All runs finished."
