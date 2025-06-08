
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
signal_df = pd.read_csv("tweet_level_preds.csv")
price_df = pd.read_csv("filter_2017_cleaned_aligned_data.csv")

# 合并两个表格：以日期和股票代码为键
merged = pd.merge(signal_df, price_df, on=["DATE", "STOCK_CODE"], how="inner")
merged["DATE"] = pd.to_datetime(merged["DATE"])

# ✅ 截尾处理：只保留 2018-09-01 之前的数据
cutoff_date = "2017-12-31"
merged = merged[merged["DATE"] < cutoff_date]

# 设置要比较的 N 值
N_values = [5, 10, 30, 50]

# 存储结果
results = {}

for N in N_values:
    daily_returns = []

    for date, group in merged.groupby("DATE"):
        top_long = group.nlargest(N, "net_tone")
        top_short = group.nsmallest(N, "net_tone")

        long_ret = top_long["1_DAY_RETURN"].mean()
        short_ret = top_short["1_DAY_RETURN"].mean()
        ls_ret = long_ret - short_ret

        daily_returns.append({"DATE": date, "Long": long_ret, "Short": short_ret, "LS": ls_ret})

    df = pd.DataFrame(daily_returns).sort_values("DATE")
    df["Cumulative_LS"] = (1 + df["LS"]).cumprod()
    results[N] = df

# 绘制比较图（截尾版）
plt.figure(figsize=(12, 6))
for N, df in results.items():
    plt.plot(df["DATE"], df["Cumulative_LS"], label=f"Top {N}")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Strategy Comparison for Different N Values (Truncated)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("compare_topN_returns_truncated20171231.png")
plt.show()

# 生成 summary 表格（截尾版）
summary = []
for N, df in results.items():
    sharpe = df["LS"].mean() / df["LS"].std() * (252 ** 0.5)
    max_dd = (df["Cumulative_LS"] / df["Cumulative_LS"].cummax() - 1).min()
    summary.append({
        "Top_N": N,
        "Mean Daily Return": df["LS"].mean(),
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Cumulative Return": df["Cumulative_LS"].iloc[-1] - 1
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("topN_strategy_summary_truncated20171231.csv", index=False)
