
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 加载数据 ===
signal_df = pd.read_csv("tweet_level_preds.csv")
price_df = pd.read_csv("filter_2017_cleaned_aligned_data.csv")

signal_df["DATE"] = pd.to_datetime(signal_df["DATE"])
price_df["DATE"] = pd.to_datetime(price_df["DATE"])

merged = pd.merge(signal_df, price_df, on=["DATE", "STOCK_CODE"], how="inner")
merged = merged[(merged["DATE"] <= "2017-12-31")]

# === 信号分布随时间变化 ===
plt.figure(figsize=(10, 4))
merged.groupby("DATE")["net_tone"].mean().plot()
plt.title("Daily Average Signal Strength (net_tone)")
plt.ylabel("Mean net_tone")
plt.tight_layout()
plt.savefig("signal_strength_over_time.png")
plt.close()

# === 分析 Top-N 股票出现频率 ===
topN = 10
top_stock_counter = {}

for date, group in merged.groupby("DATE"):
    top_stocks = group.nlargest(topN, "net_tone")["STOCK_CODE"]
    for stock in top_stocks:
        top_stock_counter[stock] = top_stock_counter.get(stock, 0) + 1

top_stock_freq = pd.Series(top_stock_counter).sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_stock_freq.values, y=top_stock_freq.index, color="dodgerblue")
plt.title(f"Most Frequently Selected Stocks in Top {topN}")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("topN_stock_frequency.png")
plt.close()

# === 月度收益和 Sharpe ===
daily_returns = []
N = 10  # 可调
for date, group in merged.groupby("DATE"):
    top_long = group.nlargest(N, "net_tone")
    top_short = group.nsmallest(N, "net_tone")
    long_ret = top_long["1_DAY_RETURN"].mean()
    short_ret = top_short["1_DAY_RETURN"].mean()
    ls_ret = long_ret - short_ret
    daily_returns.append({"DATE": date, "LS": ls_ret})

ret_df = pd.DataFrame(daily_returns).sort_values("DATE")
ret_df["Month"] = ret_df["DATE"].dt.to_period("M")
monthly_perf = ret_df.groupby("Month")["LS"].agg(["mean", "std", "count"])
monthly_perf["Sharpe"] = monthly_perf["mean"] / monthly_perf["std"] * (252 ** 0.5)

monthly_perf.reset_index(inplace=True)
monthly_perf["Month"] = monthly_perf["Month"].astype(str)

plt.figure(figsize=(10, 5))
sns.barplot(data=monthly_perf, x="Month", y="Sharpe", color="teal")
plt.title(f"Monthly Sharpe Ratio (Top {N} Strategy)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sharpe.png")
plt.close()

monthly_perf.to_csv("monthly_sharpe_table.csv", index=False)
