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
gpu_assets = load_csv("gpu_assets_scaling.csv")
gpu_blocks = load_csv("gpu_block_size_tuning.csv")
gpu_compare = load_csv("gpu_vs_cpu_comparison.csv")

cpu_paths = load_csv("cpu_paths_scaling.csv")
cpu_steps = load_csv("cpu_steps_scaling.csv")
cpu_threads = load_csv("cpu_threads_scaling.csv")
cpu_compare = load_csv("cpu_vs_gpu_comparison.csv")

option_types_compare = load_csv("option_types_comparison.csv")


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

        # Merge on common paths for comparison
        merged = pd.merge(gpu_sub, cpu_sub, on=["paths"], suffixes=("_gpu", "_cpu"))
        
        if len(merged) == 0:
            continue

        # Time comparison - bar chart (only common paths)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(merged))
        width = 0.35
        labels = [f"{p/1e6:.0f}M" for p in merged["paths"]]
        
        ax.bar(x - width/2, merged["time_ms_gpu"], width, label="GPU", alpha=0.8)
        ax.bar(x + width/2, merged["time_ms_cpu"], width, label="CPU", alpha=0.8)
        
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
        ax.bar(x - width/2, merged["std_error_gpu"], width, label="GPU", alpha=0.8)
        ax.bar(x + width/2, merged["std_error_cpu"], width, label="CPU", alpha=0.8)
        
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

    for opt_type in ["AsianArithmeticCall", "EuropeanCall"]:
        gpu_sub = gpu_steps[gpu_steps["type"] == opt_type].sort_values("steps")
        cpu_sub = cpu_steps[cpu_steps["type"] == opt_type].sort_values("steps")

        if len(gpu_sub) == 0 or len(cpu_sub) == 0:
            continue

        # Merge on common steps for comparison
        merged = pd.merge(gpu_sub, cpu_sub, on=["steps"], suffixes=("_gpu", "_cpu"))
        
        if len(merged) == 0:
            continue

        # Time comparison - bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(merged))
        width = 0.35
        labels = [str(s) for s in merged["steps"]]
        
        ax.bar(x - width/2, merged["time_ms_gpu"], width, label="GPU", alpha=0.8)
        ax.bar(x + width/2, merged["time_ms_cpu"], width, label="CPU", alpha=0.8)
        
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

        # Speedup - bar chart
        merged["speedup"] = merged["time_ms_cpu"] / merged["time_ms_gpu"]
        fig, ax = plt.subplots(figsize=(10, 6))
        x_merged = np.arange(len(merged))
        labels_merged = [str(s) for s in merged["steps"]]
        
        ax.bar(x_merged, merged["speedup"], width*1.5, alpha=0.8, color='green')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
        
        ax.set_xlabel("Steps")
        ax.set_ylabel("Speedup (CPU time / GPU time)")
        ax.set_title(f"{opt_type}: GPU Speedup vs Steps")
        ax.set_xticks(x_merged)
        ax.set_xticklabels(labels_merged)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"steps_{opt_type}_speedup.png"), dpi=150)
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


# ==================== 5. Assets Scaling (Curse of Dimensionality) ====================
def plot_assets_scaling():
    if gpu_assets is None:
        print("Skipping assets_scaling: missing CSV")
        return

    basket = gpu_assets[gpu_assets["type"] == "BasketEuropeanCall"].sort_values("assets")
    if len(basket) == 0:
        return

    # Time vs Assets - bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(basket))
    width = 0.6
    labels = [str(a) for a in basket["assets"]]
    
    ax.bar(x, basket["time_ms"], width, alpha=0.8, color='steelblue')
    ax.set_xlabel("Number of Assets")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Basket Option: Time vs Assets (Curse of Dimensionality)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "assets_scaling_time.png"), dpi=150)
    plt.close()

    # Time complexity - log-log to check O(n^2) behavior
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(basket["assets"], basket["time_ms"], marker='o', markersize=8, linewidth=2)
    
    # Fit O(n^2) reference line
    n_ref = np.array([2, 32])
    t_ref = basket.iloc[0]["time_ms"] * (n_ref / basket.iloc[0]["assets"])**2
    ax.loglog(n_ref, t_ref, 'r--', linewidth=2, label='O(n²) reference')
    
    ax.set_xlabel("Number of Assets")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Basket Option: Computational Complexity")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "assets_scaling_complexity.png"), dpi=150)
    plt.close()

    print("✓ Assets scaling plots saved")


# ==================== 6. Block Size Tuning (GPU Configuration) ====================
def plot_block_size_tuning():
    if gpu_blocks is None:
        print("Skipping block_size_tuning: missing CSV")
        return

    df = gpu_blocks.copy()
    if len(df) == 0 or 'block_size' not in df.columns:
        print("Skipping block_size_tuning: invalid data")
        return

    # Get unique block sizes and blocks_per_sm
    block_sizes = sorted(df['block_size'].unique())
    blocks_per_sm_vals = sorted(df['blocks_per_sm'].unique())

    # Create heatmap-style visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    width = 0.15
    x = np.arange(len(block_sizes))
    
    for i, bps in enumerate(blocks_per_sm_vals):
        subset = df[df['blocks_per_sm'] == bps].sort_values('block_size')
        if len(subset) > 0:
            offset = width * (i - len(blocks_per_sm_vals)/2 + 0.5)
            ax.bar(x + offset, subset['time_ms'], width, 
                   label=f'blocks_per_sm={bps}', alpha=0.8)
    
    ax.set_xlabel('Block Size (threads per block)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('GPU Configuration Tuning: Block Size vs Blocks-per-SM')
    ax.set_xticks(x)
    ax.set_xticklabels(block_sizes)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "block_size_tuning.png"), dpi=150)
    plt.close()

    # Also create a heatmap
    try:
        pivot_table = df.pivot_table(values='time_ms', 
                                     index='blocks_per_sm', 
                                     columns='block_size')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticklabels(pivot_table.index)
        
        ax.set_xlabel('Block Size')
        ax.set_ylabel('Blocks per SM')
        ax.set_title('GPU Configuration Heatmap (Time in ms)')
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                text = ax.text(j, i, f'{pivot_table.values[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        fig.colorbar(im, ax=ax, label='Time (ms)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "block_size_heatmap.png"), dpi=150)
        plt.close()
    except:
        pass  # Skip heatmap if pivot fails

    # Find and report best configuration
    best_idx = df['time_ms'].idxmin()
    best_config = df.loc[best_idx]
    print(f"   Best config: block_size={best_config['block_size']}, "
          f"blocks_per_sm={best_config['blocks_per_sm']}, "
          f"time={best_config['time_ms']:.3f} ms")

    print("✓ Block size tuning plots saved")


# ==================== 7. Option Types Comparison (Asian vs European vs Multi) ====================
def plot_option_types_comparison():
    if option_types_compare is None:
        print("Skipping option_types_comparison: missing CSV")
        return

    df = option_types_compare.copy()
    if len(df) == 0:
        print("Skipping option_types_comparison: empty data")
        return

    # Group by path count and compare option types
    path_counts = sorted(df['paths'].unique())
    
    for paths in path_counts:
        subset = df[df['paths'] == paths].sort_values('type')
        
        if len(subset) == 0:
            continue

        # Time comparison bar chart with LOG SCALE
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(subset))
        width = 0.6
        
        colors = ['steelblue', 'coral', 'mediumseagreen']
        bars = ax.bar(x, subset['time_ms'], width, alpha=0.8, color=colors[:len(subset)])
        
        ax.set_xlabel('Option Type')
        ax.set_ylabel('Time (ms, log scale)')
        ax.set_yscale('log')  # Use logarithmic scale
        path_label = f"{paths/1e6:.0f}M" if paths >= 1e6 else f"{paths/1e3:.0f}K"
        ax.set_title(f'Option Types Performance Comparison ({path_label} paths)')
        ax.set_xticks(x)
        ax.set_xticklabels(subset['type'], rotation=15, ha='right')
        ax.grid(True, axis='y', alpha=0.3, which='both')
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(subset.iterrows()):
            ax.text(i, row['time_ms'] * 1.15, 
                   f"{row['time_ms']:.2f} ms", 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"option_types_compare_{path_label}.png"), dpi=150)
        plt.close()

    # Create an overall comparison chart with all path counts (LOG SCALE)
    option_types = sorted(df['type'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    x = np.arange(len(path_counts))
    
    for i, opt_type in enumerate(option_types):
        opt_data = df[df['type'] == opt_type].sort_values('paths')
        if len(opt_data) > 0:
            offset = width * (i - len(option_types)/2 + 0.5)
            bars = ax.bar(x + offset, opt_data['time_ms'], width, 
                   label=opt_type, alpha=0.8)
            
            # Add value labels on bars
            for j, (idx, row) in enumerate(opt_data.iterrows()):
                ax.text(x[j] + offset, row['time_ms'] * 1.15, 
                       f"{row['time_ms']:.1f}", 
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Number of Paths')
    ax.set_ylabel('Time (ms, log scale)')
    ax.set_yscale('log')  # Use logarithmic scale
    ax.set_title('Option Types Performance Comparison (All Path Counts)')
    ax.set_xticks(x)
    labels = [f"{p/1e6:.0f}M" if p >= 1e6 else f"{p/1e3:.0f}K" for p in path_counts]
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "option_types_comparison_all.png"), dpi=150)
    plt.close()
    
    # Also create a normalized comparison (relative to European)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for paths in path_counts:
        subset = df[df['paths'] == paths].sort_values('type')
        if len(subset) == 0:
            continue
        
        # Find European time as baseline
        european_time = subset[subset['type'].str.contains('European', case=False)]['time_ms'].values
        if len(european_time) == 0:
            continue
        european_time = european_time[0]
        
        # Calculate relative slowdown
        subset_copy = subset.copy()
        subset_copy['relative'] = subset_copy['time_ms'] / european_time
        
        path_label = f"{paths/1e6:.0f}M" if paths >= 1e6 else f"{paths/1e3:.0f}K"
        x_pos = np.arange(len(subset_copy))
        
        ax.plot(x_pos, subset_copy['relative'], marker='o', label=path_label, linewidth=2, markersize=8)
    
    ax.set_xlabel('Option Type')
    ax.set_ylabel('Relative Time (vs European)')
    ax.set_title('Option Types Complexity: Relative Performance (European = 1.0x)')
    ax.set_xticks(np.arange(len(option_types)))
    ax.set_xticklabels(option_types, rotation=15, ha='right')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='European baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "option_types_relative_complexity.png"), dpi=150)
    plt.close()

    print("✓ Option types comparison plots saved")


# ==================== 8. GPU vs CPU Direct Comparison ====================
def plot_gpu_vs_cpu_comparison():
    if gpu_compare is None or cpu_compare is None:
        print("Skipping gpu_vs_cpu_comparison: missing CSV")
        return

    gpu_df = gpu_compare.copy()
    cpu_df = cpu_compare.copy()

    if len(gpu_df) == 0 or len(cpu_df) == 0:
        print("Skipping gpu_vs_cpu_comparison: empty data")
        return

    # For each option type, create comparison plots
    for opt_type in ["EuropeanCall", "AsianArithmeticCall", "BasketCall"]:
        gpu_sub = gpu_df[gpu_df["type"] == opt_type].sort_values("paths")
        cpu_sub = cpu_df[cpu_df["type"] == opt_type].sort_values("paths")

        if len(gpu_sub) == 0 or len(cpu_sub) == 0:
            continue

        # Time comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(gpu_sub))
        width = 0.35
        labels = [f"{p/1e6:.1f}M" if p >= 1e6 else f"{p/1e3:.0f}K" for p in gpu_sub["paths"]]
        
        ax.bar(x - width/2, gpu_sub["time_ms"], width, label="GPU", alpha=0.8, color='steelblue')
        ax.bar(x + width/2, cpu_sub["time_ms"], width, label="CPU (16 threads)", alpha=0.8, color='coral')
        
        ax.set_xlabel("Number of Paths")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"GPU vs CPU Performance: {opt_type}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"compare_{opt_type}_time.png"), dpi=150)
        plt.close()

        # Speedup calculation
        speedup = cpu_sub["time_ms"].values / gpu_sub["time_ms"].values
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['green' if s > 1 else 'red' for s in speedup]
        ax.bar(x, speedup, alpha=0.8, color=colors)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
        
        ax.set_xlabel("Number of Paths")
        ax.set_ylabel("Speedup (CPU time / GPU time)")
        ax.set_title(f"GPU Speedup over CPU (16 threads): {opt_type}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(speedup):
            ax.text(i, v + max(speedup)*0.02, f'{v:.2f}x', 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"compare_{opt_type}_speedup.png"), dpi=150)
        plt.close()

        # Print statistics
        print(f"   {opt_type}: avg speedup = {speedup.mean():.2f}x, "
              f"max = {speedup.max():.2f}x, min = {speedup.min():.2f}x")

    print("✓ GPU vs CPU comparison plots saved")


# ==================== Main ====================
if __name__ == "__main__":
    print("Generating plots from experiments/results/...")
    plot_paths_scaling()
    plot_steps_scaling()
    plot_workers_scaling()
    plot_time_histogram()
    plot_assets_scaling()
    plot_block_size_tuning()
    plot_option_types_comparison()
    plot_gpu_vs_cpu_comparison()
    print(f"\nAll plots saved to: {PLOTS_DIR}/")
