# import pandas as pd

# # Windows 文件路径，使用原始字符串防止 \ 被误解释
# file_path = r"C:\Users\flab\Downloads\predicting returns with text data\data\reduced_dataset-release.csv"

# # 读取 CSV 文件
# df = pd.read_csv(file_path)

# # 显示前五行数据
# print("前五行数据：")
# print(df.head())

# # 显示数据的行列数
# print(f"\n总行数: {df.shape[0]}, 总列数: {df.shape[1]}")

# # 显示列名
# print("\n列名：")
# print(df.columns.tolist())

# # 检查缺失值情况
# print("\n每列缺失值数量：")
# print(df.isnull().sum())
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("前五行数据：")
    print(df.head())
    print(f"\n总行数: {df.shape[0]}, 总列数: {df.shape[1]}")
    print("\n列名：")
    print(df.columns.tolist())
    print("\n每列缺失值数量：")
    print(df.isnull().sum())
    return df


if __name__ == "__main__":
    file_path = r"C:\Users\flab\Downloads\predicting returns with text data\data\reduced_dataset-release.csv"
    df = load_data(file_path)
