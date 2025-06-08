# day2_export_signals.py
from pathlib import Path
import pandas as pd, numpy as np, pickle

DATA_CSV      = "filter_2017_cleaned_aligned_data.csv"
TEXT_COL      = "cleaned_text"
DATE_COL      = "DATE"
TICKER_COL    = "STOCK_CODE"
LAST_PRICE_COL = "LAST_PRICE"
ONE_DAY_RETURN_COL = "1_DAY_RETURN"
TWO_DAY_RETURN_COL = "2_DAY_RETURN"
THREE_DAY_RETURN_COL = "3_DAY_RETURN"
SEVEN_DAY_RETURN_COL = "7_DAY_RETURN"



MODEL_DIR     = Path("models")
OUT_TWEET_CSV = "tweet_level_preds.csv"
OUT_SIGNAL_PQ = "signals.parquet"

# ---------- 载入模型 ----------
vec = pickle.load(open(MODEL_DIR / "vectorizer.pkl", "rb"))
lda = pickle.load(open(MODEL_DIR / "lda_model.pkl", "rb"))
clf = pickle.load(open(MODEL_DIR / "logreg.pkl",   "rb"))

# ---------- 推文级预测 ----------
df = pd.read_csv(DATA_CSV, parse_dates=[DATE_COL])
X  = vec.transform(df[TEXT_COL])
doc_topic = lda.transform(X)

p_pos = clf.predict_proba(doc_topic)[:, 1]
p_neg = 1 - p_pos
net_tone = doc_topic[:, 0] - doc_topic[:, 1]
text = df[TEXT_COL].astype(str)  # 确保文本列为字符串类型
tweet = df["TWEET"].astype(str)

tweet_preds = df[[DATE_COL, TICKER_COL]].copy()
tweet_preds["p_pos"]    = p_pos
tweet_preds["p_neg"]    = p_neg
tweet_preds["net_tone"] = net_tone
tweet_preds["TEXT_COL"] = text
tweet_preds["LAST_PRICE_COL"] = df[LAST_PRICE_COL]
tweet_preds[ONE_DAY_RETURN_COL] = df[ONE_DAY_RETURN_COL]
tweet_preds[TWO_DAY_RETURN_COL] = df[TWO_DAY_RETURN_COL]
tweet_preds[THREE_DAY_RETURN_COL] = df[THREE_DAY_RETURN_COL]
tweet_preds[SEVEN_DAY_RETURN_COL] = df[SEVEN_DAY_RETURN_COL]
tweet_preds["TWEET"] = tweet.astype(str)  # 确保推文列为字符串类型

tweet_preds.to_csv(OUT_TWEET_CSV, index=False)

# ---------- 日度信号 ----------
daily = (
    tweet_preds
    .groupby([DATE_COL, TICKER_COL])["net_tone"]
    .mean()
    .unstack(fill_value=np.nan)
)
daily.sort_index(inplace=True)
daily.to_parquet(OUT_SIGNAL_PQ)

print("✅ 导出完成：", OUT_SIGNAL_PQ, daily.shape)
