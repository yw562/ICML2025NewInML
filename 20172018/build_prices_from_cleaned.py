#!/usr/bin/env python
# build_prices_from_cleaned.py
import pandas as pd
import numpy as np

CSV_FILE  = "cleaned_aligned_data.csv"   # ← 你的原始文件
DATE_COL  = "DATE"                       # 日期列
TICKER_COL= "STOCK_CODE"                 # 股票代码列
PRICE_COL = "LAST_PRICE"                 # 收盘价列
OUT_PQ    = "prices.parquet"             # 输出路径

# 1) 读取并选取需要的列
df = pd.read_csv(CSV_FILE, usecols=[DATE_COL, TICKER_COL, PRICE_COL],
                 parse_dates=[DATE_COL])

# 2) 透视成 行=日期×列=ticker 的矩阵
price_mat = (
    df.pivot_table(index=DATE_COL,
                   columns=TICKER_COL,
                   values=PRICE_COL,
                   aggfunc="last")        # 同日多行取最后价；也可用 mean
    .sort_index()
    .astype(float)
)

# 3) 可选：剔除全空列 / 填充极少量缺口
price_mat.dropna(axis=1, how="all", inplace=True)
price_mat.to_parquet(OUT_PQ)

print(f"✅ 生成 prices.parquet  → {OUT_PQ}  shape={price_mat.shape}")
