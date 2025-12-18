#!/usr/bin/env bash
# Top-level run_all.sh under src/
# Integrates GPU and CPU pipelines:
# - Both use tools/calibrate_from_data.py on real AAPL + 8-stock basket
# - GPU: src/gpu/run_all.sh
# - CPU: src/cpu/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional: ensure sub-scripts are executable
chmod +x gpu/run_all.sh cpu/run_all.sh || true

# Optional top-level parameters:
#   $1 = PATHS_SINGLE  (for single-asset products, default 1000000)
#   $2 = PATHS_BASKET  (for basket/multi-asset products, default 500000)
PATHS_SINGLE="${1:-1000000}"
PATHS_BASKET="${2:-500000}"

echo "[1/2] Running GPU pipeline (build + calibrate + price)..."
(
  cd gpu
  ./run_all.sh "$PATHS_SINGLE" "$PATHS_BASKET"
)

echo
echo "[2/2] Running CPU pipeline (build + calibrate + price)..."
(
  cd cpu
  ./run_all.sh "$PATHS_SINGLE" "$PATHS_BASKET"
)

echo
echo "All CPU/GPU runs finished."
