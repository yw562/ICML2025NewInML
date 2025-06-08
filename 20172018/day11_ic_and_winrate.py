
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# 初始化结果容器
ic_results = []
win_results = []

return_columns = ["1_DAY_RETURN", "2_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]

for col in return_columns:
    # === IC（Spearman） ===
    daily_ic = merged.groupby("DATE").apply(lambda x: x["net_tone"].corr(x[col], method="spearman"))
    avg_ic = daily_ic.mean()
    ic_results.append({"Horizon": col, "Avg_IC": avg_ic})

    # === Win Rate ===
    merged[f"actual_up_{col}"] = (merged[col] > 0).astype(int)
    merged[f"correct_{col}"] = (merged["pred_up"] == merged[f"actual_up_{col}"]).astype(int)
    win_rate = merged[f"correct_{col}"].mean()
    win_results.append({"Horizon": col, "WinRate": win_rate})

    # === 画每日 IC 图 ===
    plt.figure(figsize=(10, 4))
    daily_ic.plot()
    plt.title(f"Daily Spearman IC for {col}")
    plt.ylabel("Spearman IC")
    plt.xlabel("Date")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"daily_ic_{col}.png")
    plt.close()

    # === 画每日 Win Rate 图 ===
    daily_win = merged.groupby("DATE")[f"correct_{col}"].mean()
    plt.figure(figsize=(10, 4))
    daily_win.plot()
    plt.title(f"Daily Win Rate for {col}")
    plt.ylabel("Win Rate")
    plt.xlabel("Date")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"daily_winrate_{col}.png")
    plt.close()

# 保存汇总表
ic_df = pd.DataFrame(ic_results)
ic_df.to_csv("ic_summary.csv", index=False)

win_df = pd.DataFrame(win_results)
win_df.to_csv("winrate_summary.csv", index=False)
