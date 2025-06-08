import pandas as pd
from align import align_stock_data
from preprocessing import preprocess_aligned_data

def main():
    file_path = r"C:\Users\flab\Downloads\predicting returns with text data\data\reduced_dataset-release.csv"

    # 读取原始数据
    df = pd.read_csv(file_path)
    print(f"原始样本数: {len(df)}")

    # 调用对齐函数
    df_aligned = align_stock_data(df)

    # 调用清洗函数
    df_cleaned = preprocess_aligned_data(df_aligned)

    # 显示清洗后的信息
    print(f"清洗后样本数: {len(df_cleaned)}")
    print("清洗后前五行数据：")
    print(df_cleaned.head())

if __name__ == "__main__":
    main()
