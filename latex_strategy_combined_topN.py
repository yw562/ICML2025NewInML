import pandas as pd
import matplotlib.pyplot as plt

# === 读取数据 ===
signal_df = pd.read_csv("tweet_level_preds.csv")
price_df = pd.read_csv("filter_2017_cleaned_aligned_data.csv")
summary_file = "topN_strategy_summary.csv"

# === 合并推文和价格数据 ===
merged = pd.merge(signal_df, price_df, on=["DATE", "STOCK_CODE"], how="inner")
merged["DATE"] = pd.to_datetime(merged["DATE"])

# === 设置 Top-N 分组 ===
N_values = [5, 10, 30, 50]
results = []

for N in N_values:
    daily_returns = []
    for date, group in merged.groupby("DATE"):
        top_long = group.nlargest(N, "net_tone")
        top_short = group.nsmallest(N, "net_tone")
        long_ret = top_long["1_DAY_RETURN"].mean()
        short_ret = top_short["1_DAY_RETURN"].mean()
        ls_ret = long_ret - short_ret
        daily_returns.append({"DATE": date, "LS": ls_ret})
    df = pd.DataFrame(daily_returns).sort_values("DATE")
    df["Cumulative_LS"] = (1 + df["LS"]).cumprod()
    results.append((N, df))

# === 读取 summary 数据 ===
summary_df = pd.read_csv(summary_file)

# === 创建合图 ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

# --- 子图1：累计收益 ---
for N, df in results:
    ax1.plot(df["DATE"], df["Cumulative_LS"], label=f"Top {N}")
ax1.set_title("Cumulative Return — Strategy Comparison by Top-N")
ax1.set_ylabel("Cumulative Return")
ax1.legend()
ax1.grid(True)

# --- 子图2：Sharpe / Drawdown / CumReturn 随 N ---
N = summary_df["Top_N"]
ax2.plot(N, summary_df["Sharpe Ratio"], marker="o", label="Sharpe Ratio")
ax2.plot(N, summary_df["Max Drawdown"], marker="s", label="Max Drawdown")
ax2.plot(N, summary_df["Cumulative Return"], marker="^", label="Cumulative Return")
ax2.set_title("Top-N Sensitivity — Risk/Return Metrics")
ax2.set_xlabel("Top N")
ax2.set_ylabel("Metric Value")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("strategy_combined_topN.pdf", dpi=300)
plt.show()
