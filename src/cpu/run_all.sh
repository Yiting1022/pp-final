#!/usr/bin/env bash
# run_all.sh (CPU / OpenMP)
# Run OpenMP Monte Carlo baseline for the same four products as GPU:
#   - european_pricer       => --type european
#   - asian_pricer          => --type asian
#   - basket_pricer         => --type basket (assets configurable)
#   - multiasset_pricer     => --type basket, fixed 8 assets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional CLI parameters:
#   $1 = PATHS_SINGLE  (for single-asset products, default 1000000)
#   $2 = PATHS_BASKET  (for basket/multi-asset products, default 1000000)
PATHS_SINGLE="${1:-1000000}"
PATHS_BASKET="${2:-1000000}"

echo "[1/3] Calibrating GBM parameters from market data..."
python ../tools/calibrate_from_data.py \
    --single-csv "../../data/single/AAPL.csv" \
    --multi-dir "../../data/multi" \
    --out "params.sh"

source params.sh

echo
echo "[2/3] Building CPU OpenMP mc_pricer..."
make -j4

echo
echo "[3/3] Running CPU OpenMP baselines with calibrated parameters..."

echo
echo "-- European (CPU OpenMP) --"
./mc_pricer --type european --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
            --paths "$PATHS_SINGLE" --seed 42 --threads 16

echo
echo "-- Asian (CPU OpenMP) --"
./mc_pricer --type asian --S0 "$S0_SINGLE" --K "$S0_SINGLE" --r 0.05 --sigma "$SIGMA_SINGLE" --T 1.0 \
            --steps 252 --paths "$PATHS_SINGLE" --seed 42 --threads 16

echo
echo "-- Basket (CPU OpenMP, configurable assets) --"
./mc_pricer --type basket --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
            --steps 252 --assets 4 --rho "$RHO_MULTI" \
            --paths "$PATHS_BASKET" --seed 42 --threads 16

echo
echo "-- Multi-asset (CPU OpenMP, fixed 8 assets) --"
./mc_pricer --type basket --S0 "$S0_MULTI" --K "$S0_MULTI" --r 0.05 --sigma "$SIGMA_MULTI" --T 1.0 \
            --steps 252 --assets 8 --rho "$RHO_MULTI" \
            --paths "$PATHS_BASKET" --seed 42 --threads 16

echo
echo "All CPU runs finished."
