import pandas as pd
import google.generativeai as genai
import time

# Authenticate with your Gemini API key
genai.configure(api_key="")

model = genai.GenerativeModel("gemini-pro")

# Load your sample file
df = pd.read_csv("llm_emotion_type_labeling_samples_reconstructed.csv")
df = df[df["label"].isnull() | (df["label"] == "")].copy()
df = df.head(50)  # adjust based on your quota

labels = []

for i, row in df.iterrows():
    prompt = f"""You are a financial NLP expert. Based on the following text, classify it into one or more of these market-relevant explanation tags (separated by commas if multiple):

Tags (select only from this list):
- Monetary Policy Easing
- Monetary Policy Tightening
- Fiscal Stimulus
- Regulatory Crackdown
- Regulatory Relaxation
- Trade Policy Change
- Central Bank Decision
- Interest Rate Hike
- Interest Rate Cut
- Currency Intervention
- Geopolitical Tension
- Election Result Impact
- Stock Price Surge
- Stock Price Drop
- Unusual Volume Spike
- Market-wide Rally
- Market-wide Selloff
- Volatility Spike
- Short Squeeze
- Panic Selling
- Liquidity Crunch
- Earnings Beat
- Earnings Miss
- Revenue Growth
- Profit Warning
- Dividend Announcement
- Stock Buyback
- Secondary Offering
- Bankruptcy Filing
- Restructuring Plan
- Insider Buying
- Insider Selling
- Accounting Irregularity
- Merger Announcement
- Acquisition Target
- Strategic Partnership
- Joint Venture
- Spin-off Announcement
- Leadership Change
- CEO Appointment
- CEO Resignation
- New Product Launch
- Product Recall
- Patent Grant
- Patent Lawsuit
- R&D Breakthrough
- Clinical Trial Result
- FDA Approval
- Regulatory Rejection
- Viral Marketing Campaign
- Influencer Endorsement
- Social Media Backlash
- Customer Lawsuit
- Brand Boycott
- Positive Media Coverage
- Negative Press
- Industry Regulation Change
- Commodity Price Shock
- Supply Chain Disruption
- Technological Shift
- ESG Scandal
- Environmental Impact
- GDP Growth Report
- Inflation Report
- Employment Data
- Consumer Sentiment Drop
- Retail Sales Report
- Manufacturing PMI
- Meme Stock Activity
- Retail Investor Buzz
- Speculation/Rumor
- Data Breach Incident
- Cybersecurity Risk
- Whistleblower Report
- Litigation Risk
- Activist Investor Action
- SPAC Announcement
- Delisting Risk
- Short Report Released

Text:
\"\"\"{row['reconstructed_text']}\"\"\"

Output format:
Comma-separated list of labels. If unclear, output: Other
"""

    try:
        response = model.generate_content(prompt)
        label = response.text.strip().split("\n")[0].strip(" <>:-")
    except Exception as e:
        print(f"Error at row {i}: {e}")
        label = "ERROR"

    print(f"[{i}] Label: {label}")
    labels.append(label)
    time.sleep(1.5)  # avoid rate limit

df["label"] = labels
df.to_csv("llm_emotion_type_labeled_refined.csv", index=False)
print("✅ Saved: llm_emotion_type_labeled_refined.csv")
