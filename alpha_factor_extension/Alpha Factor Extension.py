import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# ========= CONFIG ==========
LABEL_FILE = Path("llm_emotion_type_labeling_samples_labeled_gemini.csv")
RET_COLUMNS = ["1_DAY_RETURN", "2_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]
MIN_SAMPLES = 15
ROLLING_WINDOW = 90
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# ========= LOAD DATA ==========
df = pd.read_csv(LABEL_FILE, parse_dates=["DATE"])

for ret_col in RET_COLUMNS:
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

    # ========= STEP 1: ROLLING SHARPE ==========
    print(f"\n📊 Rolling Sharpe for {ret_col}")
    rolling_results = []

    for label in df_exp["label"].unique():
        temp = df_exp[df_exp["label"] == label].sort_values("DATE")
        if len(temp) < ROLLING_WINDOW:
            continue

        temp.set_index("DATE", inplace=True)

        # 修复：只对数值列进行重采样
        numeric_cols = temp.select_dtypes(include='number').columns
        non_numeric_cols = temp.select_dtypes(exclude='number').columns
        temp_num = temp[numeric_cols].resample("D").mean().ffill()
        temp_cat = temp[non_numeric_cols].resample("D").ffill()
        temp = pd.concat([temp_num, temp_cat], axis=1)

        temp["rolling_mean"] = temp["fwd_ret"].rolling(ROLLING_WINDOW).mean()
        temp["rolling_std"] = temp["fwd_ret"].rolling(ROLLING_WINDOW).std()
        temp["rolling_sharpe"] = temp["rolling_mean"] / temp["rolling_std"]
        temp = temp.dropna(subset=["rolling_sharpe"])
        temp["label"] = label
        rolling_results.append(temp[["rolling_sharpe", "label"]])

    if rolling_results:
        df_rolling = pd.concat(rolling_results).reset_index()
        plt.figure(figsize=(12, 6))
        for label in df_rolling["label"].unique():
            subset = df_rolling[df_rolling["label"] == label]
            plt.plot(subset["DATE"], subset["rolling_sharpe"], label=label)
        plt.axhline(0, color="black", linestyle="--")
        plt.title(f"Rolling Sharpe Ratio (window={ROLLING_WINDOW}) — {ret_col}")
        plt.xlabel("Date")
        plt.ylabel("Sharpe Ratio")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"rolling_sharpe_{ret_col}.png")
        plt.close()

    # ========= STEP 2: ORTHOGONALITY ==========
    print(f"\n🔬 Orthogonality test for {ret_col}")
    if "MARKET_RET" in df.columns:
        df_orth = df_exp.merge(df[["DATE", "MARKET_RET"]], on="DATE", how="left")
        orth_stats = []
        for label, sub in df_orth.groupby("label"):
            if len(sub) < MIN_SAMPLES:
                continue
            X = add_constant(sub["MARKET_RET"])
            y = sub["fwd_ret"]
            model = OLS(y, X).fit()
            orth_stats.append({
                "label": label,
                "beta_to_market": model.params.get("MARKET_RET", np.nan),
                "r2": model.rsquared
            })
        df_orth_out = pd.DataFrame(orth_stats)
        df_orth_out.to_csv(OUT_DIR / f"orthogonality_{ret_col}.csv", index=False)

    # ========= STEP 3: IC STABILITY ==========
    print(f"\n📐 IC Stability for {ret_col}")
    df_exp["month"] = df_exp["DATE"].dt.to_period("M")
    ic_by_month = df_exp.groupby(["label", "month"]).apply(
        lambda x: spearmanr(x["net_tone"], x["fwd_ret"])[0] if len(x) >= MIN_SAMPLES else np.nan
    ).unstack("label")

    ic_by_month.to_csv(OUT_DIR / f"ic_by_month_{ret_col}.csv")
    ic_by_month.plot(figsize=(12, 6), title=f"IC by Month — {ret_col}")
    plt.axhline(0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ic_by_month_{ret_col}.png")
    plt.close()
    