import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import load_and_preprocess

def join_words(word_list):
    """将词列表转为一句字符串供Tfidf处理"""
    return ' '.join(word_list)

def build_features(df, max_features=5000):
    """构建TF-IDF向量"""
    df['text_joined'] = df['cleaned_text'].apply(join_words)
    df = df[df['text_joined'].str.strip() != '']
    df = df[df['1_DAY_RETURN'].notna()]
    print("剩余有效样本数:", len(df))
    print("\n示例文本：\n", df['text_joined'].iloc[0])
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['text_joined'])
    y = df['1_DAY_RETURN'].values
    return X, y, vectorizer

def tune_lasso(X, y):
    """使用GridSearchCV寻找最优的alpha参数"""
    print("开始调参...")
    alphas = np.logspace(-5, -1, 10)
    model = Lasso(max_iter=10000)
    grid = GridSearchCV(model, {'alpha': alphas}, cv=5, scoring='r2')
    grid.fit(X, y)
    print(f"最佳 alpha: {grid.best_params_['alpha']:.1e}")
    print(f"最佳交叉验证 R²: {grid.best_score_:.4f}")
    return grid.best_estimator_

def report(model, X_test, y_test, vectorizer):
    """评估模型表现并输出重要特征"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 模型测试集均方误差(MSE): {mse:.6f}")
    print(f"📈 模型测试集R2得分: {r2:.4f}")

    # 输出重要特征
    coef = model.coef_
    features = vectorizer.get_feature_names_out()
    important_features = [(f, c) for f, c in zip(features, coef) if abs(c) > 1e-5]
    important_features = sorted(important_features, key=lambda x: abs(x[1]), reverse=True)

    if important_features:
        print("\n🧠 最重要的词和对应系数:")
        for word, weight in important_features[:20]:
            print(f"{word}: {weight:.6f}")
    else:
        print("\n⚠️ 没有非零重要特征（模型可能没学到什么）")

def plot_sentiment_vs_return(model, X_test, y_test):
    """按预测得分分组，绘制平均回报趋势图"""
    predicted_scores = model.predict(X_test)
    df = pd.DataFrame({'pred_score': predicted_scores, 'actual_return': y_test})
    # 解决重复边界的问题，duplicates='drop'
    df['decile'] = pd.qcut(df['pred_score'], 10, labels=False, duplicates='drop')

    avg_returns = df.groupby('decile')['actual_return'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_returns.index, avg_returns.values, marker='o', label='Avg. Next-Day Return')
    plt.title("📊 Predicted Sentiment Score vs. Avg. Next-Day Return", fontsize=16)
    plt.xlabel("Sentiment Score Group (Negative → Positive)", fontsize=14)
    plt.ylabel("Average Actual Next-Day Return", fontsize=14)
    plt.grid(True)
    plt.xticks(range(len(avg_returns)), [f"Q{i+1}" for i in range(len(avg_returns))], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\n各分组平均次日回报：")
    for i, r in enumerate(avg_returns):
        print(f"Q{i+1}: {r:.6f}")

def main():
    print("开始加载并预处理数据...")
    df = load_and_preprocess("data/reduced_dataset-release.csv")

    print("转换文本格式，准备特征提取...")
    X, y, vectorizer = build_features(df)

    print("拆分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("训练并调参 Lasso 模型...")
    model = tune_lasso(X_train, y_train)

    print("评估模型...")
    report(model, X_test, y_test, vectorizer)

    print("分析情感得分与回报的关系...")
    plot_sentiment_vs_return(model, X_test, y_test)

if __name__ == "__main__":
    main()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, r2_score
# from preprocessing import load_and_preprocess

# def join_words(word_list):
#     """将词列表转为一句字符串供Tfidf处理"""
#     return ' '.join(word_list)

# def build_features(df, max_features=5000):
#     """构建TF-IDF向量"""
#     df['text_joined'] = df['cleaned_text'].apply(join_words)
#     df = df[df['text_joined'].str.strip() != '']
#     df = df[df['1_DAY_RETURN'].notna()]
#     print("剩余有效样本数:", len(df))
#     print("\n示例文本：\n", df['text_joined'].iloc[0])
    
#     vectorizer = TfidfVectorizer(max_features=max_features)
#     X = vectorizer.fit_transform(df['text_joined'])
#     y = df['1_DAY_RETURN'].values
#     return X, y, vectorizer

# def tune_lasso(X, y):
#     """使用GridSearchCV寻找最优的alpha参数"""
#     print("开始调参...")
#     alphas = np.logspace(-5, -1, 10)
#     model = Lasso(max_iter=10000)
#     grid = GridSearchCV(model, {'alpha': alphas}, cv=5, scoring='r2')
#     grid.fit(X, y)
#     print(f"最佳 alpha: {grid.best_params_['alpha']:.1e}")
#     print(f"最佳交叉验证 R²: {grid.best_score_:.4f}")
#     return grid.best_estimator_

# def report(model, X_test, y_test, vectorizer):
#     """评估模型表现并输出重要特征"""
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print(f"\n📊 模型测试集均方误差(MSE): {mse:.6f}")
#     print(f"📈 模型测试集R2得分: {r2:.4f}")

#     # 输出重要特征
#     coef = model.coef_
#     features = vectorizer.get_feature_names_out()
#     important_features = [(f, c) for f, c in zip(features, coef) if abs(c) > 1e-5]
#     important_features = sorted(important_features, key=lambda x: abs(x[1]), reverse=True)

#     if important_features:
#         print("\n🧠 最重要的词和对应系数:")
#         for word, weight in important_features[:20]:
#             print(f"{word}: {weight:.6f}")
#     else:
#         print("\n⚠️ 没有非零重要特征（模型可能没学到什么）")

# def plot_sentiment_vs_return(model, X_test, y_test):
#     """按预测得分分组，绘制平均回报趋势图"""
#     predicted_scores = model.predict(X_test)
#     df = pd.DataFrame({'pred_score': predicted_scores, 'actual_return': y_test})
#     # 解决重复边界的问题，duplicates='drop'
#     df['decile'] = pd.qcut(df['pred_score'], 10, labels=False, duplicates='drop')

#     avg_returns = df.groupby('decile')['actual_return'].mean()

#     plt.figure(figsize=(10, 6))
#     plt.plot(avg_returns.index, avg_returns.values, marker='o', label='平均次日回报')
#     plt.title("📊 模型预测情感分数 vs 平均次日回报", fontsize=16)
#     plt.xlabel("预测情感得分分组（从负面到正面）", fontsize=14)
#     plt.ylabel("平均实际次日回报", fontsize=14)
#     plt.grid(True)
#     plt.xticks(range(len(avg_returns)), [f"Q{i+1}" for i in range(len(avg_returns))], fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.show()

#     print("\n各分组平均次日回报：")
#     for i, r in enumerate(avg_returns):
#         print(f"Q{i+1}: {r:.6f}")

# def main():
#     print("开始加载并预处理数据...")
#     df = load_and_preprocess("data/reduced_dataset-release.csv")

#     print("转换文本格式，准备特征提取...")
#     X, y, vectorizer = build_features(df)

#     print("拆分训练集和测试集...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     print("训练并调参 Lasso 模型...")
#     model = tune_lasso(X_train, y_train)

#     print("评估模型...")
#     report(model, X_test, y_test, vectorizer)

#     print("分析情感得分与回报的关系...")
#     plot_sentiment_vs_return(model, X_test, y_test)

# if __name__ == "__main__":
#     main()




# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, r2_score
# from preprocessing import load_and_preprocess

# def join_words(word_list):
#     """将词列表转为一句字符串供Tfidf处理"""
#     return ' '.join(word_list)

# def build_features(df, max_features=5000):
#     """构建TF-IDF向量"""
#     df['text_joined'] = df['cleaned_text'].apply(join_words)
#     df = df[df['text_joined'].str.strip() != '']
#     df = df[df['1_DAY_RETURN'].notna()]
#     print("剩余有效样本数:", len(df))
#     print("\n示例文本：\n", df['text_joined'].iloc[0])
    
#     vectorizer = TfidfVectorizer(max_features=max_features)
#     X = vectorizer.fit_transform(df['text_joined'])
#     y = df['1_DAY_RETURN'].values
#     return X, y, vectorizer

# def tune_lasso(X, y):
#     """使用GridSearchCV寻找最优的alpha参数"""
#     print("开始调参...")
#     alphas = np.logspace(-5, -1, 10)
#     model = Lasso(max_iter=10000)
#     grid = GridSearchCV(model, {'alpha': alphas}, cv=5, scoring='r2')
#     grid.fit(X, y)
#     print(f"最佳 alpha: {grid.best_params_['alpha']:.1e}")
#     print(f"最佳交叉验证 R²: {grid.best_score_:.4f}")
#     return grid.best_estimator_

# def report(model, X_test, y_test, vectorizer):
#     """评估模型表现并输出重要特征"""
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print(f"\n📊 模型测试集均方误差(MSE): {mse:.6f}")
#     print(f"📈 模型测试集R2得分: {r2:.4f}")

#     # 输出重要特征
#     coef = model.coef_
#     features = vectorizer.get_feature_names_out()
#     important_features = [(f, c) for f, c in zip(features, coef) if abs(c) > 1e-5]
#     important_features = sorted(important_features, key=lambda x: abs(x[1]), reverse=True)

#     if important_features:
#         print("\n🧠 最重要的词和对应系数:")
#         for word, weight in important_features[:20]:
#             print(f"{word}: {weight:.6f}")
#     else:
#         print("\n⚠️ 没有非零重要特征（模型可能没学到什么）")

# def main():
#     print("开始加载并预处理数据...")
#     df = load_and_preprocess("data/reduced_dataset-release.csv")

#     print("转换文本格式，准备特征提取...")
#     X, y, vectorizer = build_features(df)

#     print("拆分训练集和测试集...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     print("训练并调参 Lasso 模型...")
#     model = tune_lasso(X_train, y_train)

#     print("评估模型...")
#     report(model, X_test, y_test, vectorizer)

# if __name__ == "__main__":
#     main()

