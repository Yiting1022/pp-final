# Monte Carlo Option Pricing: CPU vs GPU Experiments

這個專案提供 CPU (OpenMP) 與 GPU (CUDA) 的 Monte Carlo 選擇權定價實驗框架。

## 目錄結構

```
src/
├── cpu/
│   ├── mc_pricer           # CPU 版本執行檔 (OpenMP)
│   ├── mc_pricer.cpp
│   └── run_experiments.sh  # CPU 實驗腳本
├── gpu/
│   ├── gpu_pricer          # GPU 版本執行檔 (CUDA)
│   ├── gpu_pricer.cu
│   └── run_experiments.sh  # GPU 實驗腳本
├── experiments/
│   ├── results/            # 所有實驗 CSV 統一輸出位置
│   └── plots/              # 所有實驗圖表統一輸出位置
├── tools/
│   └── calibrate_from_data.py  # 從真實市場數據校正參數
└── plot_experiments.py     # 統一畫圖腳本
```

## 快速開始

### 1. 編譯程式

```bash
cd src
make -C cpu    # 編譯 CPU 版本
make -C gpu    # 編譯 GPU 版本
```

### 2. 執行實驗

#### GPU 實驗（推薦先跑，速度快）
```bash
cd src/gpu
./run_experiments.sh
```

執行三組實驗：
- **Paths scaling**: 不同 Monte Carlo paths 數量 (1e7, 5e7, 1e8, 5e8)
- **Steps scaling**: 不同時間步數 (64, 128, 252, 512)
- **Multi-GPU scaling**: 使用不同 GPU 數量 (1, 2, 4)

輸出：
- `experiments/results/gpu_paths_scaling.csv`
- `experiments/results/gpu_steps_scaling.csv`
- `experiments/results/gpu_multi_gpu_scaling.csv`

#### CPU 實驗（需時較長）
```bash
cd src/cpu
./run_experiments.sh
```

執行三組實驗：
- **Paths scaling**: 不同 Monte Carlo paths 數量 (1e6, 5e6, 1e7, 5e7)
- **Steps scaling**: 不同時間步數 (64, 128, 252, 512)
- **Multi-thread scaling**: 不同 OpenMP threads (1, 2, 4, 8, 16, 32)

輸出：
- `experiments/results/cpu_paths_scaling.csv`
- `experiments/results/cpu_steps_scaling.csv`
- `experiments/results/cpu_threads_scaling.csv`

### 3. 產生圖表

```bash
cd src
python plot_experiments.py
```

這會從 `experiments/results/` 讀取所有 CSV，產生對比圖表到 `experiments/plots/`：

- `paths_*_time.png`: CPU vs GPU 時間比較
- `paths_*_speedup.png`: GPU 加速比
- `paths_*_stderr.png`: 統計誤差比較
- `steps_*_time.png`: 不同步數的時間
- `multi_gpu_speedup.png`: Multi-GPU 加速效率
- `multi_thread_speedup.png`: OpenMP 加速效率
- `time_histogram.png`: 執行時間分布直方圖

## 選擇權類型

支援三種選擇權定價：

1. **European Call**: 單資產歐式買權
   - 最簡單，有 Black-Scholes 解析解可對照
   
2. **Asian Arithmetic Call**: 單資產亞式買權（算術平均）
   - 需要時間步數離散化，計算量較大
   
3. **Basket European Call**: 多資產籃子買權
   - 8 個資產的籃子，需要 Cholesky 分解處理相關性

## CSV 格式

所有實驗結果統一格式：

```
engine,type,workers,paths,steps,assets,rho,S0,K,r,sigma,T,price,std_error,time_ms
GPU,EuropeanCall,1,10000000,1,1,0.0,271.84,271.84,0.05,0.318,1.0,40.569255,0.020677,0.993432
```

欄位說明：
- `engine`: CPU 或 GPU
- `type`: 選擇權類型
- `workers`: GPU 數量或 CPU threads 數
- `paths`: Monte Carlo 路徑數
- `steps`: 時間離散化步數
- `price`: 計算出的選擇權價格
- `std_error`: 標準誤差
- `time_ms`: 執行時間（毫秒）

## 手動執行單次定價

如果需要測試特定參數組合：

```bash
# GPU 版本
cd src/gpu
./gpu_pricer --type european --paths 1000000 --csv result.csv

# CPU 版本
cd src/cpu
./mc_pricer --type asian --steps 252 --paths 1000000 --threads 16 --csv result.csv
```

查看完整參數：
```bash
./gpu_pricer --help
./mc_pricer --help
```

## 依賴項

- **編譯**: GCC (C++17), NVCC (CUDA)
- **畫圖**: Python 3, pandas, matplotlib

安裝 Python 依賴：
```bash
pip install pandas matplotlib
```

## 注意事項

1. GPU 實驗需要 CUDA 環境，arch 設定為 sm_80（可在 `gpu/Makefile` 調整）
2. Multi-GPU 實驗會自動偵測可用 GPU 數量，超過實際數量會失敗
3. 實驗腳本會自動校正市場參數（需要 `data/single/AAPL.csv` 和 `data/multi/` 資料）
4. 所有實驗腳本設計為無 verbose 輸出，只顯示進度，結果全部寫入 CSV

## 清理舊結果

```bash
rm -rf experiments/results/* experiments/plots/*
```
