import pandas as pd

def stock_data_summary(file_path, threshold=1000):
    df = pd.read_csv(file_path)
    if 'STOCK_CODE' not in df.columns or 'DATE' not in df.columns:
        print("数据里缺少 'STOCK_CODE' 或 'DATE' 列，请检查数据")
        return
    
    # 确保日期列是datetime格式
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    
    stock_counts = df['STOCK_CODE'].value_counts()
    
    print(f"总共股票数：{stock_counts.shape[0]}\n")
    print("每只股票数据量：")
    for stock_code, count in stock_counts.items():
        print(f"{stock_code}: {count}")
    
    print("\n数据量超过{0}条的股票，按数量降序排列：".format(threshold))
    filtered = stock_counts[stock_counts > threshold].sort_values(ascending=False)
    for stock_code, count in filtered.items():
        print(f"{stock_code}: {count}")
    
    print("\n超过{0}条数据的股票时间区间及频率估计：".format(threshold))
    for stock_code in filtered.index:
        stock_df = df[df['STOCK_CODE'] == stock_code]
        start_date = stock_df['DATE'].min()
        end_date = stock_df['DATE'].max()
        total_days = (end_date - start_date).days
        
        # 避免除0错误
        if filtered[stock_code] > 1 and total_days > 0:
            avg_freq = total_days / filtered[stock_code]
        else:
            avg_freq = None
        
        # 粗略判断频率
        if avg_freq is None:
            freq_desc = "数据量或时间跨度不足"
        elif avg_freq < 1:
            freq_desc = "高频（可能为分钟或小时级别）"
        elif 0.9 <= avg_freq <= 1.1:
            freq_desc = "日频"
        else:
            freq_desc = f"低频，平均每{avg_freq:.2f}天一条"
        
        print(f"{stock_code}:")
        print(f"  时间范围: {start_date.date()} 到 {end_date.date()} ({total_days} 天)")
        print(f"  估计时间频率: {freq_desc}\n")

if __name__ == '__main__':
    data_file = 'cleaned_aligned_data.csv'  # 替换成你的数据文件路径
    stock_data_summary(data_file)

# import pandas as pd

# def stock_data_summary(file_path, threshold=1000):
#     df = pd.read_csv(file_path)
#     if 'STOCK_CODE' not in df.columns:
#         print("数据里没有 'STOCK_CODE' 列，请检查数据")
#         return
    
#     stock_counts = df['STOCK_CODE'].value_counts()
    
#     print(f"总共股票数：{stock_counts.shape[0]}\n")
#     print("每只股票数据量：")
#     for stock_code, count in stock_counts.items():
#         print(f"{stock_code}: {count}")
    
#     print("\n数据量超过{0}条的股票，按数量降序排列：".format(threshold))
#     filtered = stock_counts[stock_counts > threshold].sort_values(ascending=False)
#     for stock_code, count in filtered.items():
#         print(f"{stock_code}: {count}")
    
#     return stock_counts, filtered

# if __name__ == '__main__':
#     data_file = 'cleaned_aligned_data.csv'  # 替换成你的数据文件路径
#     stock_data_summary(data_file)
