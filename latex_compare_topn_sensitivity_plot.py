import pandas as pd
import matplotlib.pyplot as plt

# === 替换为你的文件名 ===
summary_file = "topN_strategy_summary.csv"

# === 读取数据 ===
df = pd.read_csv(summary_file)

# === 提取各列 ===
N = df["Top_N"]
sharpe = df["Sharpe Ratio"]
drawdown = df["Max Drawdown"]
cum_return = df["Cumulative Return"]

# === 创建图表 ===
plt.figure(figsize=(10, 6))
plt.plot(N, sharpe, marker="o", label="Sharpe Ratio")
plt.plot(N, drawdown, marker="s", label="Max Drawdown")
plt.plot(N, cum_return, marker="^", label="Cumulative Return")

plt.xlabel("Top N")
plt.ylabel("Metric Value")
plt.title("Strategy Sensitivity to Top-N Selection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("topN_sensitivity_plot.png", dpi=300)
plt.show()
