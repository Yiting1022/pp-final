#!/usr/bin/env python3
"""
plot_experiments.py
統一畫圖腳本：讀取 experiments/results/ 底下的 CSV，產生對比圖表到 experiments/plots/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "experiments"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# 讀取 CSV
def load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

gpu_paths = load_csv("gpu_paths_scaling.csv")
gpu_steps = load_csv("gpu_steps_scaling.csv")
gpu_gpus = load_csv("gpu_multi_gpu_scaling.csv")

cpu_paths = load_csv("cpu_paths_scaling.csv")
cpu_steps = load_csv("cpu_steps_scaling.csv")
cpu_threads = load_csv("cpu_threads_scaling.csv")


# ==================== 1. Paths Scaling: CPU vs GPU ====================
def plot_paths_scaling():
    if gpu_paths is None or cpu_paths is None:
        print("Skipping paths_scaling: missing CSV")
        return

    for opt_type in ["EuropeanCall", "AsianArithmeticCall"]:
        gpu_sub = gpu_paths[gpu_paths["type"] == opt_type].sort_values("paths")
        cpu_sub = cpu_paths[cpu_paths["type"] == opt_type].sort_values("paths")

        if len(gpu_sub) == 0 or len(cpu_sub) == 0:
            continue

        # Time comparison - bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(gpu_sub))
        width = 0.35
        labels = [f"{p/1e6:.0f}M" for p in gpu_sub["paths"]]
        
        ax.bar(x - width/2, gpu_sub["time_ms"], width, label="GPU", alpha=0.8)
        ax.bar(x + width/2, cpu_sub["time_ms"], width, label="CPU", alpha=0.8)
        
        ax.set_xlabel("Paths")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{opt_type}: Time vs Paths")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"paths_{opt_type}_time.png"), dpi=150)
        plt.close()

        # Std error comparison - bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, gpu_sub["std_error"], width, label="GPU", alpha=0.8)
        ax.bar(x + width/2, cpu_sub["std_error"], width, label="CPU", alpha=0.8)
        
        ax.set_xlabel("Paths")
        ax.set_ylabel("Standard Error")
        ax.set_title(f"{opt_type}: Standard Error vs Paths")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"paths_{opt_type}_stderr.png"), dpi=150)
        plt.close()

        # Speedup - bar chart
        merged = pd.merge(gpu_sub, cpu_sub, on=["paths"], suffixes=("_gpu", "_cpu"))
        if len(merged) > 0:
            merged["speedup"] = merged["time_ms_cpu"] / merged["time_ms_gpu"]
            fig, ax = plt.subplots(figsize=(10, 6))
            x_merged = np.arange(len(merged))
            labels_merged = [f"{p/1e6:.0f}M" for p in merged["paths"]]
            
            ax.bar(x_merged, merged["speedup"], width*1.5, alpha=0.8, color='green')
            ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
            
            ax.set_xlabel("Paths")
            ax.set_ylabel("Speedup (CPU time / GPU time)")
            ax.set_title(f"{opt_type}: GPU Speedup vs Paths")
            ax.set_xticks(x_merged)
            ax.set_xticklabels(labels_merged)
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"paths_{opt_type}_speedup.png"), dpi=150)
            plt.close()

    print("✓ Paths scaling plots saved")


# ==================== 2. Steps Scaling: CPU vs GPU ====================
def plot_steps_scaling():
    if gpu_steps is None or cpu_steps is None:
        print("Skipping steps_scaling: missing CSV")
        return

    for opt_type in ["AsianArithmeticCall", "BasketEuropeanCall"]:
        gpu_sub = gpu_steps[gpu_steps["type"] == opt_type].sort_values("steps")
        cpu_sub = cpu_steps[cpu_steps["type"] == opt_type].sort_values("steps")

        if len(gpu_sub) == 0 or len(cpu_sub) == 0:
            continue

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(gpu_sub))
        width = 0.35
        labels = [str(s) for s in gpu_sub["steps"]]
        
        ax.bar(x - width/2, gpu_sub["time_ms"], width, label="GPU", alpha=0.8)
        ax.bar(x + width/2, cpu_sub["time_ms"], width, label="CPU", alpha=0.8)
        
        ax.set_xlabel("Steps")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{opt_type}: Time vs Steps")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"steps_{opt_type}_time.png"), dpi=150)
        plt.close()

    print("✓ Steps scaling plots saved")


# ==================== 3. Multi-GPU / Multi-Thread Scaling ====================
def plot_workers_scaling():
    # GPU multi-GPU - bar chart
    if gpu_gpus is not None:
        basket_gpu = gpu_gpus[gpu_gpus["type"] == "BasketEuropeanCall"].sort_values("workers")
        if len(basket_gpu) > 0:
            t1_gpu = basket_gpu.iloc[0]["time_ms"]
            basket_gpu["speedup"] = t1_gpu / basket_gpu["time_ms"]

            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(basket_gpu))
            width = 0.6
            
            ax.bar(x, basket_gpu["speedup"], width, alpha=0.8, label="GPU Actual", color='steelblue')
            ax.plot(x, basket_gpu["workers"], marker='o', linestyle="--", 
                   color='red', linewidth=2, label="Ideal", markersize=8)
            
            ax.set_xlabel("GPUs")
            ax.set_ylabel("Speedup")
            ax.set_title("Basket: Multi-GPU Speedup")
            ax.set_xticks(x)
            ax.set_xticklabels(basket_gpu["workers"].astype(int))
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "multi_gpu_speedup.png"), dpi=150)
            plt.close()

    # CPU multi-thread - bar chart
    if cpu_threads is not None:
        basket_cpu = cpu_threads[cpu_threads["type"] == "BasketEuropeanCall"].sort_values("workers")
        if len(basket_cpu) > 0:
            t1_cpu = basket_cpu.iloc[0]["time_ms"]
            basket_cpu["speedup"] = t1_cpu / basket_cpu["time_ms"]

            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(basket_cpu))
            width = 0.6
            
            ax.bar(x, basket_cpu["speedup"], width, alpha=0.8, label="CPU (OpenMP) Actual", color='coral')
            ax.plot(x, basket_cpu["workers"], marker='s', linestyle="--", 
                   color='red', linewidth=2, label="Ideal", markersize=8)
            
            ax.set_xlabel("Threads")
            ax.set_ylabel("Speedup")
            ax.set_title("Basket: OpenMP Threads Speedup")
            ax.set_xticks(x)
            ax.set_xticklabels(basket_cpu["workers"].astype(int))
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "multi_thread_speedup.png"), dpi=150)
            plt.close()

    print("✓ Workers scaling plots saved")


# ==================== 4. Histogram: Time Distribution ====================
def plot_time_histogram():
    all_data = []
    if gpu_paths is not None:
        all_data.append(("GPU", gpu_paths))
    if cpu_paths is not None:
        all_data.append(("CPU", cpu_paths))

    if len(all_data) == 0:
        print("Skipping histogram: no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for opt_idx, opt_type in enumerate(["EuropeanCall", "AsianArithmeticCall"]):
        ax = axes[opt_idx]
        for name, df in all_data:
            sub = df[df["type"] == opt_type]
            if len(sub) > 0:
                ax.hist(sub["time_ms"], bins=10, alpha=0.6, label=name, edgecolor="black")

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{opt_type}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "time_histogram.png"), dpi=150)
    plt.close()

    print("✓ Histogram plot saved")


# ==================== Main ====================
if __name__ == "__main__":
    print("Generating plots from experiments/results/...")
    plot_paths_scaling()
    plot_steps_scaling()
    plot_workers_scaling()
    plot_time_histogram()
    print(f"\nAll plots saved to: {PLOTS_DIR}/")
