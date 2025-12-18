#!/usr/bin/env bash
# run_all_experiments.sh
# 一鍵執行 GPU + CPU 實驗並產生圖表

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  Monte Carlo Experiments: GPU + CPU"
echo "========================================="
echo

# 清空舊結果
echo "Cleaning old results..."
rm -rf experiments/results/* experiments/plots/*

# GPU 實驗（快）
echo
echo "[1/3] Running GPU experiments..."
(cd gpu && ./run_experiments.sh)

# CPU 實驗（慢）
echo
echo "[2/3] Running CPU experiments..."
(cd cpu && ./run_experiments.sh)

# 畫圖
echo
echo "[3/3] Generating plots..."
python plot_experiments.py

echo
echo "========================================="
echo "  All experiments completed!"
echo "========================================="
echo "Results: experiments/results/*.csv"
echo "Plots:   experiments/plots/*.png"
