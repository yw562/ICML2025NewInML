
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
signal_df = pd.read_csv("tweet_level_preds.csv")
price_df = pd.read_csv("cleaned_aligned_data.csv")

# 日期转换
signal_df["DATE"] = pd.to_datetime(signal_df["DATE"])
price_df["DATE"] = pd.to_datetime(price_df["DATE"])

# 合并数据
merged = pd.merge(signal_df, price_df, on=["DATE", "STOCK_CODE"], how="inner")
merged = merged.sort_values(["STOCK_CODE", "DATE"])

# ==== 1. 有效 signal 天数统计 ====
signal_days = merged["DATE"].nunique()
stock_days = merged.groupby("DATE")["STOCK_CODE"].count()

with open("diagnostics_log.txt", "w") as f:
    f.write(f"✅ 有效策略交易日总数: {signal_days}\n")
    f.write(f"✅ 每日平均 signal 股票数: {stock_days.mean():.2f}\n")
    f.write(f"✅ 最多 signal 股票数: {stock_days.max()}\n")
    f.write(f"✅ 最少 signal 股票数: {stock_days.min()}\n\n")

# ==== 2. 分析 signal 分布 ====
plt.figure(figsize=(8, 5))
sns.histplot(merged["net_tone"], bins=50, kde=True)
plt.title("Distribution of Signal (net_tone)")
plt.xlabel("net_tone")
plt.tight_layout()
plt.savefig("signal_distribution.png")
plt.close()

# ==== 3. 检查未来泄漏风险 ====
# 判断是否有未来收益被用于计算 signal
# 检查同一股票是否在 t 日的 signal 对应的是 t+1 的 return
merged["next_date"] = merged.groupby("STOCK_CODE")["DATE"].shift(-1)
merged["next_return"] = merged.groupby("STOCK_CODE")["1_DAY_RETURN"].shift(-1)

check_df = merged.dropna(subset=["next_return"])

# 如果有很多 signal 来自未来的 return，这里相关性会很高（是错误的）
correlation = check_df["net_tone"].corr(check_df["next_return"])

with open("diagnostics_log.txt", "a") as f:
    f.write(f"🧪 Signal (t) 与 next day return (t+1) 的相关系数: {correlation:.4f}\n")
    if abs(correlation) > 0.6:
        f.write("⚠️ 警告：可能存在未来信息泄漏，请检查 signal 是否来自当日数据。\n")
    else:
        f.write("✅ 未发现明显的未来泄漏风险。\n")

# ==== 4. 输出 signal 数量随时间变化 ====
count_by_day = merged.groupby("DATE")["STOCK_CODE"].count()

plt.figure(figsize=(10, 4))
count_by_day.plot()
plt.title("Number of Signal Stocks per Day")
plt.xlabel("Date")
plt.ylabel("Stock Count")
plt.tight_layout()
plt.savefig("signal_count_timeseries.png")
plt.close()
