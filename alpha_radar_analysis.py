import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

###############################
# 配置
###############################
LABEL_FILE = Path("llm_emotion_type_labeling_samples_labeled_gemini.csv")

RET_COLUMN = "1_DAY_RETURN"  # 可替换为 '2_DAY_RETURN', '3_DAY_RETURN', 等
TOP_K_PLOT = 12
MIN_SAMPLES = 50
OUT_METRICS = "metrics_by_label.csv"
OUT_RADAR = "alpha_radar.png"

###############################
# 1. 加载数据
###############################
print("📥 Loading label data...")
df = pd.read_csv(LABEL_FILE, parse_dates=["DATE"])
df = df.dropna(subset=["net_tone", "label", RET_COLUMN])
df = df[df["label"].astype(str).str.strip() != ""]

# 重命名收益列为 fwd_ret，统一后续处理
df = df.rename(columns={RET_COLUMN: "fwd_ret"})

###############################
# 2. 拆分多标签
###############################
print("🧹 Exploding multi-labels …")
records = []
for _, row in df.iterrows():
    labels = [l.strip() for l in str(row["label"]).split(",") if l.strip()]
    for l in labels:
        records.append({
            "label": l,
            "net_tone": row["net_tone"],
            "fwd_ret": row["fwd_ret"]
        })
df_exp = pd.DataFrame(records)
print(f"🎯 Unique labels: {df_exp['label'].nunique()}")

###############################
# 3. 计算每类标签的 IC / Sharpe
###############################
print("📊 Calculating metrics …")
def calc_ic(subdf):
    try:
        return np.corrcoef(subdf["net_tone"], subdf["fwd_ret"])[0, 1]
    except:
        return np.nan

agg = df_exp.groupby("label").agg(
    samples=("fwd_ret", "count"),
    meanret=("fwd_ret", "mean"),
    stdret=("fwd_ret", "std")
)
agg["ic"] = df_exp.groupby("label").apply(calc_ic)
agg["sharpe"] = agg["meanret"] / agg["stdret"]
agg = agg.dropna()
agg = agg[agg["samples"] >= MIN_SAMPLES]
agg = agg.sort_values("sharpe", ascending=False)
agg.to_csv(OUT_METRICS, float_format="%.6f")
print(f"📄 Saved metrics to {OUT_METRICS}")

###############################
# 4. 雷达图
###############################
print("📈 Plotting radar chart …")
sel = agg.head(TOP_K_PLOT)
labels = sel.index.tolist()
values = sel["sharpe"].values

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
values = np.concatenate((values, [values[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, "o-", linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=9)
ax.set_title("Alpha Radar — Sharpe Ratio by Event Type", va="bottom")
ax.grid(True)
plt.tight_layout()
fig.savefig(OUT_RADAR, dpi=300)
print(f"✅ Radar chart saved to {OUT_RADAR}")

###############################
# 5. 打印 top 结果
###############################
print("\n🏆 Top-10 by Sharpe:")
print(sel[["samples", "ic", "sharpe"]].round(4))
