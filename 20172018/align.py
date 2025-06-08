import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # 去除链接
    text = re.sub(r"@\w+", "", text)     # 去除 @用户名
    text = re.sub(r"#", "", text)        # 去除 #
    text = re.sub(r"[^a-z\s]", "", text) # 去除非字母字符（保留空格）
    tokens = word_tokenize(text)
    # 保留 no, not，过滤其他停用词
    cleaned = [word for word in tokens if word not in stop_words or word in ['no', 'not']]
    return cleaned

def align_stock_data(df):
    # 日期格式转换，指定日/月/年格式
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')

    # 过滤关键字段缺失的行
    required_cols = ['TWEET', 'STOCK', 'DATE', '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN']
    df = df.dropna(subset=required_cols)

    # 去重：去除重复的推文（基于TWEET列）
    df = df.drop_duplicates(subset=['TWEET'])

    # 如果有股票代码映射字典，可以这样映射：
    # 假设stock_code_map = {'PayPal': 'PYPL', 'Amazon': 'AMZN', ...}
    # 这里暂时直接用股票名称代替代码
    df['STOCK_CODE'] = df['STOCK']

    # 按股票代码和日期排序（升序）
    df = df.sort_values(['STOCK_CODE', 'DATE']).reset_index(drop=True)

    print("对齐后样本数:", len(df))
    print("排序后前五行数据：")
    print(df[['STOCK', 'STOCK_CODE', 'DATE']].head())

    return df

def preprocess_aligned_data(df_aligned):
    print(f"清洗前样本数: {len(df_aligned)}")

    df = df_aligned.dropna(subset=['TWEET', '1_DAY_RETURN']).copy()

    print("正在预处理文本...")

    df['cleaned_text'] = df['TWEET'].apply(clean_text)

    # 过滤掉清洗后空文本的行
    df = df[df['cleaned_text'].map(len) > 0].reset_index(drop=True)

    print(f"预处理完成，有效样本数: {len(df)}")
    print("清洗后前五行数据：")
    pd.set_option('display.max_columns', None)
    print(df.head())
    pd.reset_option('display.max_columns')

    print("当前DataFrame列名：")
    print(df.columns.tolist())

    return df

if __name__ == "__main__":
    file_path = r"C:\Users\flab\Downloads\predicting returns with text data\data\reduced_dataset-release.csv"
    df = pd.read_csv(file_path, low_memory=False)

    print(f"原始样本数: {len(df)}")

    df_aligned = align_stock_data(df)
    df_cleaned = preprocess_aligned_data(df_aligned)

# import pandas as pd

# def align_stock_data(df):
#     # 转换日期格式，错误转换为 NaT
#     df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')

#     # 过滤掉任何关键字段缺失的行
#     required_cols = ['TWEET', 'STOCK', 'DATE', '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN']
#     df = df.dropna(subset=required_cols)

#     df = df.reset_index(drop=True)
#     print("对齐完成，有效样本数:", len(df))
#     print("对齐后前五行数据：")
#     print(df.head())
#     return df


# if __name__ == "__main__":
#     # 示例用法：读取文件，执行对齐
#     file_path = r"C:\Users\flab\Downloads\predicting returns with text data\data\reduced_dataset-release.csv"
#     df = pd.read_csv(file_path)
#     print(f"原始样本数: {len(df)}")
    
#     df_aligned = align_stock_data(df)


# import pandas as pd

# def align_stock_data(df):
#     # 转换日期格式，错误转换为 NaT
#     df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')

#     # 过滤掉任何关键字段缺失的行
#     required_cols = ['TWEET', 'STOCK', 'DATE', '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN']
#     df = df.dropna(subset=required_cols)

#     df = df.reset_index(drop=True)
#     print("对齐完成，有效样本数:", len(df))
#     return df


# if __name__ == "__main__":
#     # 示例用法：读取文件，执行对齐
#     file_path = r"C:\Users\flab\Downloads\predicting returns with text data\data\reduced_dataset-release.csv"
#     df = pd.read_csv(file_path)
#     print(f"原始样本数: {len(df)}")
    
#     df_aligned = align_stock_data(df)
