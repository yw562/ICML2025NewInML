
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
signal_df = pd.read_csv("tweet_level_preds.csv")
price_df = pd.read_csv("filter_2017_cleaned_aligned_data.csv")

# 合并两个表格：以日期和股票代码为键
merged = pd.merge(signal_df, price_df, on=["DATE", "STOCK_CODE"], how="inner")

# 确保日期格式正确
merged["DATE"] = pd.to_datetime(merged["DATE"])

# 设定每天选前N个long（看涨）+ N个short（看跌）
N = 30
daily_returns = []

# 按天循环构建组合
for date, group in merged.groupby("DATE"):
    top_long = group.nlargest(N, "net_tone")  # 得分最高的做多
    top_short = group.nsmallest(N, "net_tone")  # 得分最低的做空

    long_ret = top_long["1_DAY_RETURN"].mean()
    short_ret = top_short["1_DAY_RETURN"].mean()
    ls_ret = long_ret - short_ret

    daily_returns.append({
        "DATE": date,
        "Long": long_ret,
        "Short": short_ret,
        "LS": ls_ret
    })

# 构建DataFrame并排序
result_df = pd.DataFrame(daily_returns).sort_values("DATE")
result_df["Cumulative_LS"] = (1 + result_df["LS"]).cumprod()

# 保存结果到 CSV
result_df.to_csv("day4_result.csv", index=False)

# 画图
plt.figure(figsize=(10, 5))
plt.plot(result_df["DATE"], result_df["Cumulative_LS"], label="Long-Short Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Daily Long-Short Strategy (Top 30 by Net Tone)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("day4_strategy_performance.png")
