import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
file_paths = {
    '1D': '/Users/yueyi/Downloads/Styles/predicting-returns-with-text-data/predicting-returns-with-text-data/output/tone_quantiles_1_DAY_RETURN.csv',
    '2D': '/Users/yueyi/Downloads/Styles/predicting-returns-with-text-data/predicting-returns-with-text-data/output/tone_quantiles_2_DAY_RETURN.csv',
    '3D': '/Users/yueyi/Downloads/Styles/predicting-returns-with-text-data/predicting-returns-with-text-data/output/tone_quantiles_3_DAY_RETURN.csv',
    '7D': '/Users/yueyi/Downloads/Styles/predicting-returns-with-text-data/predicting-returns-with-text-data/output/tone_quantiles_7_DAY_RETURN.csv',
}

# 合并数据并标记 horizon
dfs = []
for horizon, path in file_paths.items():
    df = pd.read_csv(path)
    df['horizon'] = horizon
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

# 清洗：reshape 为 long format
long_df = all_data.melt(
    id_vars=['label', 'horizon'],
    value_vars=['Low', 'Med', 'High'],
    var_name='quantile',
    value_name='return'
).dropna()

# 可视化：每个时间 horizon 下不同 quantile 的 return 分布
plt.figure(figsize=(14, 6))
sns.boxplot(data=long_df, x='horizon', y='return', hue='quantile')
plt.title("Return Distribution by Tone Quantiles Across Time Horizons")
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel("Return")
plt.xlabel("Horizon")
plt.legend(title='Tone Quantile')
plt.tight_layout()
plt.show()
