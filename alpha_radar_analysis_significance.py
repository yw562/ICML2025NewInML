import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_1samp

###############################
# 配置
###############################
LABEL_FILE = Path("llm_emotion_type_labeling_samples_labeled_gemini.csv")
RET_COLUMNS = ["1_DAY_RETURN", "2_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]
TOP_K_PLOT = 12
MIN_SAMPLES = 15  # ⬅️ 更宽容的样本数阈值
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

###############################
# 核心函数
###############################

def calc_ic(subdf):
    try:
        return np.corrcoef(subdf["net_tone"], subdf["fwd_ret"])[0, 1]
    except:
        return np.nan

def explode_labels(df, ret_col):
    df = df.dropna(subset=["net_tone", "label", ret_col])
    df = df[df["label"].astype(str).str.strip() != ""]
    df = df.rename(columns={ret_col: "fwd_ret"})
    records = []
    for _, row in df.iterrows():
        labels = [l.strip() for l in str(row["label"]).split(",") if l.strip()]
        for l in labels:
            records.append({
                "label": l,
                "net_tone": row["net_tone"],
                "fwd_ret": row["fwd_ret"]
            })
    return pd.DataFrame(records)

def calc_significance(df_exp):
    p_values = {}
    for label, group in df_exp.groupby("label"):
        if len(group) >= MIN_SAMPLES:
            _, p = ttest_1samp(group["fwd_ret"], popmean=0)
            p_values[label] = p
        else:
            p_values[label] = np.nan
    return pd.Series(p_values)

def add_stars(pval):
    if pd.isna(pval):
        return ""
    elif pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return ""

def analyze_ret_window(df, ret_col):
    print(f"\n📈 Analyzing: {ret_col}")
    df_exp = explode_labels(df.copy(), ret_col)
    
    # 导出 label 出现频率
    label_counts = df_exp["label"].value_counts()
    label_counts.to_csv(OUT_DIR / f"label_counts_{ret_col}.csv")
    print(f"✅ Saved label count: label_counts_{ret_col}.csv")

    # 情绪强度分组
    df_exp["tone_group"] = pd.qcut(df_exp["net_tone"], q=3, labels=["Low", "Med", "High"])

    # IC / Sharpe 指标
    agg = df_exp.groupby("label").agg(
        samples=("fwd_ret", "count"),
        meanret=("fwd_ret", "mean"),
        stdret=("fwd_ret", "std")
    )
    agg["ic"] = df_exp.groupby("label").apply(calc_ic)
    agg["sharpe"] = agg["meanret"] / agg["stdret"]

    # 显著性检验
    agg["p_value"] = calc_significance(df_exp)
    agg["sig"] = agg["p_value"].apply(add_stars)

    # ➕ 调试输出：过滤前
    print(f"🧮 Total unique labels: {agg.shape[0]}")
    
    # 检查样本不足
    dropped_small = agg[agg["samples"] < MIN_SAMPLES]
    print(f"⚠️ Dropped {len(dropped_small)} labels due to low samples:\n{dropped_small[['samples']].T.to_string()}")

    # 检查 NaN
    dropped_nan = agg[agg.isna().any(axis=1)]
    print(f"⚠️ Dropped {len(dropped_nan)} labels due to NaN:\n{dropped_nan.index.tolist()}")

    # 最终保留
    agg = agg[agg["samples"] >= MIN_SAMPLES]
    agg = agg.dropna(subset=["meanret", "stdret", "sharpe", "ic"])
    print(f"✅ Remaining labels for output: {agg.shape[0]}")

    # 保存指标表
    agg = agg.sort_values("sharpe", ascending=False)
    out_csv = OUT_DIR / f"metrics_by_label_{ret_col}.csv"
    agg.to_csv(out_csv, float_format="%.6f")
    print(f"📄 Saved metrics: {out_csv}")

    # 雷达图 + 柱状图
    sel = agg.head(TOP_K_PLOT)
    plot_radar(sel["sharpe"], sel.index.tolist(), ret_col)
    plot_bar(agg["sharpe"], agg["sig"], f"Sharpe Ratio ({ret_col})", ret_col)

    # 分组导出
    tone_agg = df_exp.groupby(["label", "tone_group"])["fwd_ret"].mean().unstack()
    out_tone = OUT_DIR / f"tone_quantiles_{ret_col}.csv"
    tone_agg.to_csv(out_tone, float_format="%.6f")
    print(f"📄 Saved tone quantile returns: {out_tone}")

def plot_radar(values, labels, ret_tag):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=9)
    ax.set_title(f"Alpha Radar — Sharpe by Event ({ret_tag})", va="bottom")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"radar_{ret_tag}.png", dpi=300)
    plt.close()

def plot_bar(series, sigs, title, ret_tag):
    fig, ax = plt.subplots(figsize=(10, min(12, len(series) * 0.4)))
    sns.barplot(x=series.values, y=series.index, ax=ax, palette="coolwarm")
    for i, (v, s) in enumerate(zip(series.values, sigs)):
        ax.text(v + 0.001*np.sign(v), i, s, color="black", va="center", fontweight="bold")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Sharpe Ratio (with significance)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"bar_sharpe_{ret_tag}.png", dpi=300)
    plt.close()

###############################
# 主流程
###############################

print("📥 Loading data …")
df_raw = pd.read_csv(LABEL_FILE, parse_dates=["DATE"])

for ret_col in RET_COLUMNS:
    analyze_ret_window(df_raw, ret_col)

print("\n🎉 All analysis complete. See 'output/' folder.")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from scipy.stats import ttest_1samp

# ###############################
# # 配置
# ###############################
# LABEL_FILE = Path("llm_emotion_type_labeling_samples_labeled_gemini.csv")
# RET_COLUMNS = ["1_DAY_RETURN", "2_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]
# TOP_K_PLOT = 12
# MIN_SAMPLES = 30
# OUT_DIR = Path("output")
# OUT_DIR.mkdir(exist_ok=True)

# ###############################
# # 函数定义
# ###############################

# def calc_ic(subdf):
#     try:
#         return np.corrcoef(subdf["net_tone"], subdf["fwd_ret"])[0, 1]
#     except:
#         return np.nan

# def explode_labels(df, ret_col):
#     df = df.dropna(subset=["net_tone", "label", ret_col])
#     df = df[df["label"].astype(str).str.strip() != ""]
#     df = df.rename(columns={ret_col: "fwd_ret"})
#     records = []
#     for _, row in df.iterrows():
#         labels = [l.strip() for l in str(row["label"]).split(",") if l.strip()]
#         for l in labels:
#             records.append({
#                 "label": l,
#                 "net_tone": row["net_tone"],
#                 "fwd_ret": row["fwd_ret"]
#             })
#     return pd.DataFrame(records)

# def calc_significance(df_exp):
#     p_values = {}
#     for label, group in df_exp.groupby("label"):
#         if len(group) >= MIN_SAMPLES:
#             _, p = ttest_1samp(group["fwd_ret"], popmean=0)
#             p_values[label] = p
#         else:
#             p_values[label] = np.nan
#     return pd.Series(p_values)

# def add_stars(pval):
#     if pd.isna(pval):
#         return ""
#     elif pval < 0.001:
#         return "***"
#     elif pval < 0.01:
#         return "**"
#     elif pval < 0.05:
#         return "*"
#     else:
#         return ""

# def analyze_ret_window(df, ret_col):
#     print(f"\n📈 Analyzing: {ret_col}")
#     df_exp = explode_labels(df.copy(), ret_col)
    
#     # 情绪强度分组
#     df_exp["tone_group"] = pd.qcut(df_exp["net_tone"], q=3, labels=["Low", "Med", "High"])

#     # IC / Sharpe 指标
#     agg = df_exp.groupby("label").agg(
#         samples=("fwd_ret", "count"),
#         meanret=("fwd_ret", "mean"),
#         stdret=("fwd_ret", "std")
#     )
#     agg["ic"] = df_exp.groupby("label").apply(calc_ic)
#     agg["sharpe"] = agg["meanret"] / agg["stdret"]
    
#     # 显著性检验
#     agg["p_value"] = calc_significance(df_exp)
#     agg["sig"] = agg["p_value"].apply(add_stars)
    
#     agg = agg.dropna()
#     agg = agg[agg["samples"] >= MIN_SAMPLES]
#     agg = agg.sort_values("sharpe", ascending=False)

#     out_csv = OUT_DIR / f"metrics_by_label_{ret_col}.csv"
#     agg.to_csv(out_csv, float_format="%.6f")
#     print(f"✅ Saved: {out_csv}")

#     # 保存雷达图
#     sel = agg.head(TOP_K_PLOT)
#     plot_radar(sel["sharpe"], sel.index.tolist(), ret_col)

#     # 保存柱状图（含显著性）
#     plot_bar(agg["sharpe"], agg["sig"], f"Sharpe Ratio ({ret_col})", ret_col)

#     # 情绪强度分位组收益输出
#     tone_agg = df_exp.groupby(["label", "tone_group"])["fwd_ret"].mean().unstack()
#     out_tone = OUT_DIR / f"tone_quantiles_{ret_col}.csv"
#     tone_agg.to_csv(out_tone, float_format="%.6f")
#     print(f"✅ Saved tone group mean returns to {out_tone}")


# def plot_radar(values, labels, ret_tag):
#     print(f"📡 Plotting radar for {ret_tag} …")
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
#     values = np.concatenate((values, [values[0]]))
#     angles = np.concatenate((angles, [angles[0]]))
#     fig = plt.figure(figsize=(8, 8))
#     ax = plt.subplot(111, polar=True)
#     ax.plot(angles, values, "o-", linewidth=2)
#     ax.fill(angles, values, alpha=0.25)
#     ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=9)
#     ax.set_title(f"Alpha Radar — Sharpe by Event ({ret_tag})", va="bottom")
#     ax.grid(True)
#     plt.tight_layout()
#     fig.savefig(OUT_DIR / f"radar_{ret_tag}.png", dpi=300)
#     plt.close()


# def plot_bar(series, sigs, title, ret_tag):
#     print(f"📊 Plotting bar chart for {ret_tag} …")
#     fig, ax = plt.subplots(figsize=(10, min(12, len(series) * 0.4)))
#     sns.barplot(x=series.values, y=series.index, ax=ax, palette="coolwarm")
    
#     for i, (v, s) in enumerate(zip(series.values, sigs)):
#         ax.text(v + 0.001*np.sign(v), i, s, color="black", va="center", fontweight="bold")
    
#     ax.axvline(0, color="black", linewidth=1)
#     ax.set_title(title)
#     ax.set_xlabel("Sharpe Ratio (with significance)")
#     plt.tight_layout()
#     fig.savefig(OUT_DIR / f"bar_sharpe_{ret_tag}.png", dpi=300)
#     plt.close()

# ###############################
# # 主流程
# ###############################

# print("📥 Loading data …")
# df_raw = pd.read_csv(LABEL_FILE, parse_dates=["DATE"])

# for ret_col in RET_COLUMNS:
#     analyze_ret_window(df_raw, ret_col)

# print("\n🎉 All analysis done. Check output folder.")
