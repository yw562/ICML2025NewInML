import pandas as pd
from pathlib import Path

# ==== 配置 ====
metrics_dir = Path("output")
ret_tags = ["1_DAY_RETURN", "2_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]
top_k = 5  # 每个 horizon 显示前 K 高 Sharpe 的标签
out_tex = "merged_latex_table2.tex"

# ==== 收集数据 ====
dfs = []
for tag in ret_tags:
    file = metrics_dir / f"metrics_by_label_{tag}.csv"
    if file.exists():
        df = pd.read_csv(file)
        df["Horizon"] = tag.replace("_DAY_RETURN", "D")
        dfs.append(df)
    else:
        print(f"⚠️ Missing file: {file}")

# ==== 合并并整理 ====
all_df = pd.concat(dfs, ignore_index=True)
all_df = all_df.dropna(subset=["label", "sharpe", "ic", "p_value", "sig"])
all_df = all_df.sort_values(["Horizon", "sharpe"], ascending=[True, False])

# ==== 筛选 top-K per horizon ====
top_df = all_df.groupby("Horizon").head(top_k).copy()

# ==== 格式化 ====
def fmt(x, digits=3):
    return f"{x:.{digits}f}" if pd.notnull(x) else ""

top_df["Sharpe"] = top_df.apply(lambda x: f"{fmt(x['sharpe'])}{x['sig']}", axis=1)
top_df["IC"] = top_df["ic"].apply(fmt)
top_df["N"] = top_df["samples"].astype(int)

final_df = top_df[["Horizon", "label", "N", "Sharpe", "IC"]].rename(columns={
    "label": "Event Label"
})

# ==== 转 Latex ====
latex_table = final_df.to_latex(
    index=False,
    escape=False,
    caption="Top predictive event labels by Sharpe ratio across horizons. *, **, *** denote p-value < 0.05, 0.01, 0.001 respectively.",
    label="tab:alpha_by_event",
    longtable=True,
    column_format="llccc"
)

with open(out_tex, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(f"✅ Latex table saved to {out_tex}")
