#!/usr/bin/env bash
# run_all_experiment.sh
# 設定多組 Monte Carlo path 數目，批次跑 GPU+CPU pipeline。
# 會呼叫同目錄下的 run_all.sh，並把每一次實驗輸出存成 log。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------
# 可自行調整的實驗組合
#   PATHS_SINGLE: 單資產產品 (European / Asian)
#   PATHS_BASKET: 多資產 / Basket 產品
# 以下是三組範例，從小到大：
#   1) 1e5  / 5e4
#   2) 5e5  / 2.5e5
#   3) 1e6  / 5e5
# 如需更多組合，直接修改陣列即可。
# ------------------------------
PATHS_SINGLE_LIST=(100000 500000 1000000)
PATHS_BASKET_LIST=(50000 250000 500000)

if [[ ${#PATHS_SINGLE_LIST[@]} -ne ${#PATHS_BASKET_LIST[@]} ]]; then
  echo "[ERROR] PATHS_SINGLE_LIST 與 PATHS_BASKET_LIST 長度需相同" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="experiments/${RUN_ID}"
mkdir -p "$OUT_DIR"

echo "實驗輸出將存到: $OUT_DIR"

for idx in "${!PATHS_SINGLE_LIST[@]}"; do
  PATHS_SINGLE="${PATHS_SINGLE_LIST[$idx]}"
  PATHS_BASKET="${PATHS_BASKET_LIST[$idx]}"

  echo
  echo "========== 實驗 $((idx+1)) / ${#PATHS_SINGLE_LIST[@]} =========="
  echo "PATHS_SINGLE = $PATHS_SINGLE, PATHS_BASKET = $PATHS_BASKET"

  LOG_PREFIX="${OUT_DIR}/paths_single_${PATHS_SINGLE}_basket_${PATHS_BASKET}"

  # 呼叫原本的 top-level run_all.sh (GPU + CPU 一起跑)
  ./run_all.sh "$PATHS_SINGLE" "$PATHS_BASKET" \
    2>&1 | tee "${LOG_PREFIX}.log"

done

echo
echo "所有實驗完成。log 位置：$OUT_DIR"
