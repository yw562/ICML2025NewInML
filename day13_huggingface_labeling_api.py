import pandas as pd
import requests
import time

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
TOKEN = ""
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Load data
df = pd.read_csv("llm_emotion_type_labeling_samples.csv")
df = df[df["label"].isnull() | (df["label"] == "")].copy()
df = df.head(5)  # limit rows for demo

labels = []

for i, row in df.iterrows():
    prompt = f"""You are a financial NLP expert. Given the following news or text, identify its emotional category (choose only one label):

Text:
"{row['TWEET']}"

Possible categories:
- Policy Support
- Company Earnings
- Industry Crisis
- Market Panic
- Market Rebound
- Investor Expectations


Final answer
"""

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 20}}
        )
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            generated = result[0]["generated_text"]
            label = generated.split("Final answer")[-1].strip().split("\n")[0].strip(" <>:")
        else:
            label = "ERROR"
            print("Unexpected response:", result)
    except Exception as e:
        print(f"Error at index {i}: {e}")
        label = "ERROR"

    print(f"[{i}] Label: {label}")
    labels.append(label)
    time.sleep(2)

df["label"] = labels
df.to_csv("llm_emotion_type_labeling_samples_labeled_hfapi.csv", index=False)
print("✅ Saved: llm_emotion_type_labeling_samples_labeled_hfapi.csv")
