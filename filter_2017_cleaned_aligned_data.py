import pandas as pd

# 读取原始文件（假设文件名为 data.csv）
df = pd.read_csv('cleaned_aligned_data.csv')

# 将 DATE 列转换为 datetime 格式
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# 筛选出 2017 年的数据
df_2017 = df[df['DATE'].dt.year == 2017]

# 保存筛选结果为新的文件
df_2017.to_csv('data_2017.csv', index=False)

print(f"筛选完成，共 {len(df_2017)} 条 2017 年的数据已保存为 data_2017.csv")
