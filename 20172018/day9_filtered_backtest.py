
import pandas as pd
import matplotlib.pyplot as plt

# 读取原始数据
signal_df = pd.read_csv("tweet_level_preds.csv")
price_df = pd.read_csv("cleaned_aligned_data.csv")

# 合并并处理日期
signal_df["DATE"] = pd.to_datetime(signal_df["DATE"])
price_df["DATE"] = pd.to_datetime(price_df["DATE"])
merged = pd.merge(signal_df, price_df, on=["DATE", "STOCK_CODE"], how="inner")
merged["DATE"] = pd.to_datetime(merged["DATE"])

# ==== 1. 过滤掉 signal 数量过多或过少的日子 ====
stock_count = merged.groupby("DATE")["STOCK_CODE"].count()
valid_dates = stock_count[(stock_count > 20) & (stock_count < 100000)].index
filtered = merged[merged["DATE"].isin(valid_dates)]

# ==== 2. 回测 Top-N 策略 ====
N_values = [5, 10, 30, 50]
results = {}

for N in N_values:
    daily_returns = []
    for date, group in filtered.groupby("DATE"):
        top_long = group.nlargest(N, "net_tone")
        top_short = group.nsmallest(N, "net_tone")
        long_ret = top_long["1_DAY_RETURN"].mean()
        short_ret = top_short["1_DAY_RETURN"].mean()
        ls_ret = long_ret - short_ret
        daily_returns.append({"DATE": date, "Long": long_ret, "Short": short_ret, "LS": ls_ret})

    df = pd.DataFrame(daily_returns).sort_values("DATE")
    df["Cumulative_LS"] = (1 + df["LS"]).cumprod()
    results[N] = df

# ==== 3. 绘图 ====
plt.figure(figsize=(12, 6))
for N, df in results.items():
    plt.plot(df["DATE"], df["Cumulative_LS"], label=f"Top {N}")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Strategy Performance with Filtered Signal Days")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("compare_topN_filtered_returns.png")
plt.show()

# ==== 4. Summary 输出 ====
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
summary_df.to_csv("topN_strategy_summary_filtered.csv", index=False)
