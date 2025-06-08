
import pandas as pd

# 加载 signal 数据
df = pd.read_csv("tweet_level_preds.csv")
df["DATE"] = pd.to_datetime(df["DATE"])

# 只保留必要列
df = df[["DATE", "STOCK_CODE", "TEXT_COL", "net_tone", "TWEET","LAST_PRICE_COL", "1_DAY_RETURN", "2_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]]
df = df.dropna(subset=["TWEET", "net_tone"])

# 选择 top-K 正向信号和 bottom-K 负向信号
top_k = 100
top_positive = df.sort_values("net_tone", ascending=False).head(top_k).copy()
top_negative = df.sort_values("net_tone", ascending=True).head(top_k).copy()

# 添加空列供手动/LLM 填写情绪类型标签
top_positive["label"] = ""
top_negative["label"] = ""

# 合并导出
combined = pd.concat([top_positive, top_negative], ignore_index=True)
combined.to_csv("llm_emotion_type_labeling_samples.csv", index=False)
print("✅ 已导出待标注样本文件：llm_emotion_type_labeling_samples.csv")
