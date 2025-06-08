
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
signal_df = pd.read_csv("tweet_level_preds.csv")
price_df = pd.read_csv("cleaned_aligned_data.csv")

# 日期处理
signal_df["DATE"] = pd.to_datetime(signal_df["DATE"])
price_df["DATE"] = pd.to_datetime(price_df["DATE"])

# 合并 signal 和 price
merged = pd.merge(signal_df, price_df, on=["DATE", "STOCK_CODE"], how="inner")

# 去除异常 signal 天（极端股票数）
signal_counts = merged.groupby("DATE")["STOCK_CODE"].count()
valid_dates = signal_counts[(signal_counts > 20) & (signal_counts < 100000)].index
merged = merged[merged["DATE"].isin(valid_dates)]

# 截尾处理：只保留 2017-12-31 之前的数据
merged = merged[merged["DATE"] <= "2017-12-31"]

# 设置 Top-N 组
N_values = [5, 10, 30, 50]
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

# 绘制累计收益图
plt.figure(figsize=(12, 6))
for N, df in results.items():
    plt.plot(df["DATE"], df["Cumulative_LS"], label=f"Top {N}")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Strategy Comparison (Filtered Days, Truncated Before 2018)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("compare_topN_returns_filtered_truncated_day9.png")
plt.close()

# 生成 summary 表
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
summary_df.to_csv("topN_strategy_summary_filtered_truncated_day9.csv", index=False)
