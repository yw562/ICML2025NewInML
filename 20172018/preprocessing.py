import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

nltk.download('punkt')
nltk.download('stopwords')

# 载入停用词集合，后续可根据需求调整
stop_words = set(stopwords.words('english'))

# 保留金融常用词示例（你可以根据需求扩充）
finance_keep_words = {'no', 'not', 'buy', 'sell', 'hold', 'gain', 'loss'}

def clean_text(text):
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"http\S+", "", text)    # 去除链接
    text = re.sub(r"@\w+", "", text)       # 去除 @用户名
    text = re.sub(r"#", "", text)          # 去除 #
    text = re.sub(r"[^a-z\s]", "", text)   # 去除非字母字符（保留空格）

    tokens = word_tokenize(text)

    # 过滤停用词，但保留金融关键动词
    cleaned = [word for word in tokens if (word not in stop_words or word in finance_keep_words)]

    return cleaned

def preprocess_aligned_data(df_aligned):
    print(f"清洗前样本数: {len(df_aligned)}")

    # 保留所有列，过滤缺失 TWEET 和 1_DAY_RETURN 的行
    df = df_aligned.dropna(subset=['TWEET', '1_DAY_RETURN']).copy()

    print("正在预处理文本...")

    df['cleaned_text'] = df['TWEET'].apply(clean_text)

    # 过滤掉清洗后空文本的行
    df = df[df['cleaned_text'].map(len) > 0].reset_index(drop=True)

    print(f"预处理完成，有效样本数: {len(df)}")
    print("清洗后前五行数据：")

    # 显示所有列，方便调试
    pd.set_option('display.max_columns', None)
    print(df.head())
    pd.reset_option('display.max_columns')

    print("当前DataFrame列名：")
    print(df.columns.tolist())

    # 打印当前工作目录，方便确认文件保存位置
    print("当前工作目录是:", os.getcwd())

    # 保存清洗后的数据
    df.to_csv('cleaned_aligned_data.csv', index=False)
    print("已保存清洗后的数据到 cleaned_aligned_data.csv")

    return df

def load_and_preprocess(filepath="data/cleaned_aligned_data.csv"):
    import os

    if os.path.exists(filepath):
        print(f"📂 直接加载已存在的清洗数据：{filepath}")
        return pd.read_csv(filepath, parse_dates=['DATE'])  # 注意列名 DATE 是大写
    else:
        print(f"⚙️ 找不到清洗数据，尝试重新清洗：{filepath}")
        df_aligned = pd.read_csv(filepath, parse_dates=['DATE'])  # 原始数据路径
        return preprocess_aligned_data(df_aligned)



# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')

# stop_words = set(stopwords.words('english'))

# def clean_text(text):
#     if not isinstance(text, str):
#         return []
#     text = text.lower()
#     text = re.sub(r"http\S+", "", text)  # 去除链接
#     text = re.sub(r"@\w+", "", text)     # 去除 @用户名
#     text = re.sub(r"#", "", text)        # 去除 #
#     text = re.sub(r"[^a-z\s]", "", text) # 去除非字母字符（保留空格）
#     tokens = word_tokenize(text)
#     # 不过滤所有 stopwords，保留 'no', 'not'
#     cleaned = [word for word in tokens if word not in stop_words or word in ['no', 'not']]
#     return cleaned

# def preprocess_aligned_data(df_aligned):
#     print(f"清洗前样本数: {len(df_aligned)}")

#     # 保留所有列，过滤掉 TWEET 和 1_DAY_RETURN 缺失的行
#     df = df_aligned.dropna(subset=['TWEET', '1_DAY_RETURN']).copy()

#     print("正在预处理文本...")

#     df['cleaned_text'] = df['TWEET'].apply(clean_text)

#     # 过滤掉清洗后空文本的行
#     df = df[df['cleaned_text'].map(len) > 0].reset_index(drop=True)

#     print(f"预处理完成，有效样本数: {len(df)}")
#     print("清洗后前五行数据：")

#     # 设置显示所有列，打印前5行
#     pd.set_option('display.max_columns', None)
#     print(df.head())
#     pd.reset_option('display.max_columns')  # 恢复默认

#     print("当前DataFrame列名：")
#     print(df.columns.tolist())

#     return df



# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')

# stop_words = set(stopwords.words('english'))

# def clean_text(text):
#     if not isinstance(text, str):
#         return []
#     text = text.lower()
#     text = re.sub(r"http\S+", "", text)  # 去除链接
#     text = re.sub(r"@\w+", "", text)     # 去除 @用户名
#     text = re.sub(r"#", "", text)        # 去除 #
#     text = re.sub(r"[^a-z\s]", "", text) # 去除非字母字符（保留空格）
#     tokens = word_tokenize(text)
#     # 不要过滤所有 stopwords，保留一部分常见词（例如 not, no 可能对金融情感很关键）
#     cleaned = [word for word in tokens if word not in stop_words or word in ['no', 'not']]
#     return cleaned

# def load_and_preprocess(file_path):
#     df = pd.read_csv(file_path)
#     df = df[['TWEET', '1_DAY_RETURN']].dropna()
#     print("原始样本数:", len(df))

#     print("正在预处理文本...")
#     df['cleaned_text'] = df['TWEET'].apply(clean_text)

#     # 过滤掉空文本（预处理后长度为0）
#     df = df[df['cleaned_text'].map(len) > 0]

#     print("预处理完成，有效样本数:", len(df))
#     return df
