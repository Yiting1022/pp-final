# 檔名範例：plot_hist_time.py
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

base = "experiments_gpu"
latest_dir = sorted(glob.glob(os.path.join(base, "*")))[-1]
csv_path = os.path.join(latest_dir, "exp1_paths_scaling.csv")
print("using:", csv_path)

df = pd.read_csv(csv_path)

# 只看 EuropeanCall 的 time_ms
eu = df[df["type"] == "EuropeanCall"]

plt.figure()
plt.hist(eu["time_ms"], bins=10)  # bins 可以自己調
plt.xlabel("time_ms")
plt.ylabel("count")
plt.title("EuropeanCall GPU MC time_ms histogram")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(latest_dir, "exp1_european_time_hist.png"))
# 或 plt.show()