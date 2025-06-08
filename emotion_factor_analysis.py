import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ========== CONFIG ==========
FILE_PATH = "llm_emotion_type_labeling_samples_labeled_gemini.csv"  # 修改为你本地路径
RET_COLUMNS = ["1_DAY_RETURN", "2_DAY_RETURN", "7_DAY_RETURN"]
MIN_SAMPLES = 15
TOP_K = 10
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# ========== LOAD DATA ==========
df = pd.read_csv(FILE_PATH, parse_dates=["DATE"])
df["DATE"] = pd.to_datetime(df["DATE"])

# ========== MAIN LOOP ==========
for ret_col in RET_COLUMNS:
    print(f"\n📊 Processing return column: {ret_col}")

    df_use = df.dropna(subset=["label", "net_tone", ret_col]).copy()
    df_use = df_use.rename(columns={ret_col: "fwd_ret"})

    # Label explode
    rows = []
    for _, row in df_use.iterrows():
        labels = [l.strip() for l in str(row["label"]).split(",") if l.strip()]
        for label in labels:
            rows.append({
                "DATE": row["DATE"],
                "label": label,
                "net_tone": row["net_tone"],
                "fwd_ret": row["fwd_ret"]
            })
    df_exp = pd.DataFrame(rows)

    # ---------- SHARPE RATIO + T-TEST ----------
    sharpe_data = []
    for label, group in df_exp.groupby("label"):
        if len(group) >= MIN_SAMPLES:
            mu, sigma = group["fwd_ret"].mean(), group["fwd_ret"].std()
            sr = mu / sigma
            t_stat, p_val = ttest_1samp(group["fwd_ret"], 0.0)
            stars = (
                "***" if p_val < 0.01 else
                "**" if p_val < 0.05 else
                "*" if p_val < 0.1 else ""
            )
            sharpe_data.append({
                "label": label,
                "sharpe_ratio": sr,
                "t_stat": t_stat,
                "p_val": p_val,
                "significance": stars,
                "n_obs": len(group)
            })
    sharpe_df = pd.DataFrame(sharpe_data).sort_values("sharpe_ratio", ascending=False)

    # ---------- IC STABILITY ----------
    df_exp["month"] = df_exp["DATE"].dt.to_period("M")
    ic_by_month = df_exp.groupby(["label", "month"]).apply(
        lambda x: spearmanr(x["net_tone"], x["fwd_ret"])[0] if len(x) >= MIN_SAMPLES else np.nan
    ).unstack("label")

    # ---------- SUMMARY TABLE ----------
    summary_table = sharpe_df.set_index("label")[[
        "sharpe_ratio", "t_stat", "p_val", "significance", "n_obs"
    ]].join(
        ic_by_month.mean().rename("avg_ic")
    ).sort_values("avg_ic", ascending=False)

    # ---------- SAVE CSV ----------
    summary_table.to_csv(OUT_DIR / f"factor_summary_{ret_col}.csv")

    # ---------- RETURN DISTRIBUTION PLOTS ----------
    for label in summary_table.index:
        label_data = df_exp[df_exp["label"] == label]["fwd_ret"]
        if len(label_data) >= MIN_SAMPLES:
            plt.figure(figsize=(6, 4))
            sns.histplot(label_data, kde=True, bins=30)
            plt.title(f"Return Distribution: {label} — {ret_col}")
            plt.xlabel("Forward Return")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"ret_dist_{ret_col}_{label.replace('/', '_').replace(' ', '_')}.png")
            plt.close()

    # ---------- IC STABILITY PLOT ----------
    top_k = summary_table.dropna(subset=["avg_ic"]).head(TOP_K)
    if not top_k.empty:
        top_k_labels = top_k.index.tolist()
        ic_by_month[top_k_labels].plot(figsize=(12, 6), title=f"IC Stability by Month — {ret_col}")
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Month")
        plt.ylabel("IC")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"ic_by_month_topk_{ret_col}.png")
        plt.close()

    # ---------- SHARPE RATIO BARPLOT ----------
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary_table.reset_index(),
        x="sharpe_ratio",
        y="label",
        order=summary_table.index
    )
    plt.title(f"Sharpe Ratios by Label — {ret_col}")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Label")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"sharpe_ratio_bar_{ret_col}.png")
    plt.close()

print("\n✅ All return columns processed. Results saved in:", OUT_DIR)
