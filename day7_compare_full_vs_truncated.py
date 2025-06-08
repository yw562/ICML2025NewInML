
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 1. 加载 summary 文件 ====
full = pd.read_csv("topN_strategy_summary.csv")
trunc = pd.read_csv("topN_strategy_summary_truncated20171231.csv")

full["Source"] = "Full Period"
trunc["Source"] = "Truncated"

combined = pd.concat([full, trunc])

# ==== 2. 横向对比图 ====
metrics = ["Sharpe Ratio", "Max Drawdown", "Cumulative Return"]
colors = {"Full Period": "skyblue", "Truncated": "orange"}

for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=combined, x=metric, y="Top_N", hue="Source", palette=colors)
    plt.title(f"{metric} Comparison: Full vs Truncated Period")
    plt.tight_layout()
    plt.savefig(f"compare_{metric.lower().replace(' ', '_')}_full_vs_truncated.png")
    plt.close()

# ==== 3. 尾部下跌归因分析 ====

# 加载原始合并数据
signal_df = pd.read_csv("tweet_level_preds.csv")
price_df = pd.read_csv("filter_2017_cleaned_aligned_data.csv")

merged = pd.merge(signal_df, price_df, on=["DATE", "STOCK_CODE"], how="inner")
merged["DATE"] = pd.to_datetime(merged["DATE"])

# 聚焦在策略急剧下跌的时间窗口（例如2018-09-01之后）
tail = merged[merged["DATE"] >= "2017-12-31"]

# 按股票统计这段期间的平均日收益
stock_dd = tail.groupby("STOCK_CODE")["1_DAY_RETURN"].mean().sort_values()

# 可视化跌幅最猛的前20个股票
worst_20 = stock_dd.head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=worst_20.values, y=worst_20.index, color="crimson")
plt.title("Stocks with Most Negative Avg Daily Return (Post 31 Dec 2017)")
plt.xlabel("Avg 1-Day Return")
plt.tight_layout()
plt.savefig("worst_stocks_post20171231.png")
plt.close()
