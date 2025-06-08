
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# === 加载数据 ===
signal_df = pd.read_csv("tweet_level_preds.csv")
price_df = pd.read_csv("cleaned_aligned_data.csv")

signal_df["DATE"] = pd.to_datetime(signal_df["DATE"])
price_df["DATE"] = pd.to_datetime(price_df["DATE"])

# 合并数据
merged = pd.merge(signal_df, price_df, on=["DATE", "STOCK_CODE"], how="inner")
merged = merged[(merged["DATE"] <= "2017-12-31")]

# 设置预测信号
merged["pred_up"] = (merged["net_tone"] > 0).astype(int)

return_columns = ["1_DAY_RETURN", "2_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]
results = []

for col in return_columns:
    horizon = col.replace("_DAY_RETURN", "D")

    # 计算每日多空收益
    daily_returns = []
    turnover_list = []

    for date, group in merged.groupby("DATE"):
        group = group.dropna(subset=[col])
        top_long = group.nlargest(10, "net_tone")
        top_short = group.nsmallest(10, "net_tone")

        long_ret = top_long[col].mean()
        short_ret = top_short[col].mean()
        ls_ret = long_ret - short_ret
        turnover = len(set(top_long["STOCK_CODE"]).union(set(top_short["STOCK_CODE"])))

        daily_returns.append({"DATE": date, "LS": ls_ret})
        turnover_list.append(turnover)

    df = pd.DataFrame(daily_returns).sort_values("DATE")
    df["cum_return"] = (1 + df["LS"]).cumprod()
    df["Month"] = df["DATE"].dt.to_period("M")

    # Sharpe Ratio
    sharpe = df["LS"].mean() / df["LS"].std() * (252 ** 0.5)

    # IC
    ic = merged["net_tone"].corr(merged[col], method="spearman")

    # Win Rate
    merged[f"actual_up_{col}"] = (merged[col] > 0).astype(int)
    merged[f"correct_{col}"] = (merged["pred_up"] == merged[f"actual_up_{col}"]).astype(int)
    win_rate = merged[f"correct_{col}"].mean()

    # Turnover
    avg_turnover = np.mean(turnover_list)

    # Alpha / R2 via CAPM-style regression: LS vs Market
    # 用市场平均作为 proxy
    merged_daily = merged.groupby("DATE")[col].mean().rename("MKT").reset_index()
    df = df.merge(merged_daily, on="DATE", how="left")
    df = df.dropna()
    X = sm.add_constant(df["MKT"])
    model = sm.OLS(df["LS"], X).fit()
    alpha = model.params["const"]
    r2 = model.rsquared

    # Append summary
    results.append({
        "Horizon": horizon,
        "Sharpe Ratio": sharpe,
        "IC": ic,
        "Win Rate": win_rate,
        "Avg Turnover": avg_turnover,
        "Alpha": alpha,
        "R2": r2,
        "Cumulative Return": df["cum_return"].iloc[-1] - 1
    })

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(df["DATE"], df["cum_return"])
    plt.title(f"Cumulative LS Return ({horizon})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"cumulative_return_{horizon}.png")
    plt.close()

# 保存汇总表
result_df = pd.DataFrame(results)
result_df.to_csv("t0_strategy_summary_extended.csv", index=False)

# 可视化：关键指标图
plt.figure(figsize=(10, 5))
sns.barplot(data=result_df, x="Horizon", y="Sharpe Ratio", color="skyblue")
plt.title("Sharpe Ratio across Horizons")
plt.tight_layout()
plt.savefig("sharpe_across_horizon.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.barplot(data=result_df, x="Horizon", y="IC", color="salmon")
plt.title("Spearman IC across Horizons")
plt.tight_layout()
plt.savefig("ic_across_horizon.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.barplot(data=result_df, x="Horizon", y="Win Rate", color="lightgreen")
plt.title("Win Rate across Horizons")
plt.tight_layout()
plt.savefig("winrate_across_horizon.png")
plt.close()
