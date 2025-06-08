
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取策略结果
df = pd.read_csv("day4_result.csv")
df["DATE"] = pd.to_datetime(df["DATE"])

# ==== 1. Summary metrics ====
summary = {
    "Total Days": len(df),
    "Mean Daily Return": df["LS"].mean(),
    "Std Dev of Return": df["LS"].std(),
    "Sharpe Ratio": df["LS"].mean() / df["LS"].std() * (252**0.5),
    "Cumulative Return": df["Cumulative_LS"].iloc[-1] - 1,
    "Max Drawdown": (df["Cumulative_LS"] / df["Cumulative_LS"].cummax() - 1).min()
}
summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
summary_df.to_csv("summary_metrics.csv")

# ==== 2. Quantile return plot (robust to duplicate bin edges) ====
try:
    quantile_bins = pd.qcut(df["LS"], 5, duplicates="drop")
    num_bins = quantile_bins.cat.categories.size
    labels = [f"Q{i+1}" for i in range(num_bins)]
    df["quantile"] = pd.qcut(df["LS"], num_bins, labels=labels, duplicates="drop")

    quantile_ret = df.groupby("quantile")["LS"].mean()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=quantile_ret.index, y=quantile_ret.values)
    plt.title(f"Average LS Return by Signal Quantile (Actual {num_bins} Quantiles)")
    plt.xlabel("Quantile")
    plt.ylabel("Avg LS Return")
    plt.tight_layout()
    plt.savefig("04_quantile_returns.png")
    plt.close()
except ValueError as e:
    print(f"Quantile plot skipped due to error: {e}")

# ==== 3. Drawdown plot ====
cum = df["Cumulative_LS"]
dd = cum / cum.cummax() - 1

plt.figure(figsize=(10, 4))
plt.plot(df["DATE"], dd, label="Drawdown", color="red")
plt.fill_between(df["DATE"], dd, 0, color="red", alpha=0.3)
plt.title("Strategy Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid()
plt.tight_layout()
plt.savefig("drawdown_plot.png")
plt.close()
