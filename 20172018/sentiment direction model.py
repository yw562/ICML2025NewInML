import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from scipy.stats import spearmanr
from preprocessing import load_and_preprocess

def join_words(word_list):
    return ' '.join(word_list)

def tune_logistic(X, y):
    Cs = np.logspace(-3, 2, 6)
    model = LogisticRegression(max_iter=10000, solver='lbfgs', multi_class='multinomial', class_weight='balanced')
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(model, {'C': Cs}, cv=5, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_estimator_

def evaluate_ic(y_true_cont, scores):
    ic = spearmanr(scores, y_true_cont).correlation
    print(f"📈 IC（Information Coefficient）: {ic:.4f}")

def report(model, X_test, y_test, y_test_continuous):
    scores = model.predict_proba(X_test) @ np.array([-1, 0, 1])
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    evaluate_ic(y_test_continuous, scores)
    r2 = r2_score(y_test_continuous, scores)
    print(f"📉 R² vs true return: {r2:.4f}")

    return scores

def plot_score_vs_return(scores, y_true_cont):
    df = pd.DataFrame({'score': scores, 'ret': y_true_cont})
    df['quantile'] = pd.qcut(df['score'], 10, labels=False, duplicates='drop')
    avg_ret = df.groupby('quantile')['ret'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_ret.index, avg_ret.values, marker='o')
    plt.title("📊 Score Quantile vs Avg. Return")
    plt.xlabel("Quantile (Q0 = lowest score)")
    plt.ylabel("Avg. 1-Day Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n各分组平均收益：")
    for i, r in enumerate(avg_ret):
        print(f"Q{i}: {r:.6f}")

def build_features(df, vectorizer=None, max_features=5000):
    df['text_joined'] = df['cleaned_text']#.apply(join_words)
    df = df[df['text_joined'].str.strip() != '']
    df = df[df['1_DAY_RETURN'].notna()]
    
    df['label'] = df['1_DAY_RETURN'].apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))

    # If no vectorizer provided, create and fit one (training phase)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(df['text_joined'])
    else:
        # Use existing vectorizer (testing phase)
        X = vectorizer.transform(df['text_joined'])
    
    y = df['label'].values
    y_continuous = df['1_DAY_RETURN'].values

    return X, y, y_continuous, vectorizer, df

# ...existing code...

def main():
    print("📦 读取并预处理数据...")
    df = load_and_preprocess("data/cleaned_aligned_data.csv")
    stock_list = df['STOCK_CODE'].unique()

    for stock in stock_list:
        print(f"\n🔍 正在处理股票：{stock}")
        df_stock = df[df['STOCK_CODE'] == stock].copy()
        if len(df_stock) < 100:
            print(f"🚫 样本数不足，跳过 {stock}")
            continue

        df_stock = df_stock.sort_values('DATE')
        split_date = df_stock['DATE'].quantile(0.8)
        train_df = df_stock[df_stock['DATE'] <= split_date]
        test_df = df_stock[df_stock['DATE'] > split_date]

        if len(train_df) < 50 or len(test_df) < 20:
            print(f"🚫 训练或测试数据太少，跳过 {stock}")
            continue

        # Fit vectorizer on training data only
        X_train, y_train, _, vectorizer, _ = build_features(train_df)
        # Apply fitted vectorizer to test data
        X_test, y_test, y_test_cont, _, _ = build_features(test_df, vectorizer=vectorizer)

        print(f"✅ 训练样本数: {X_train.shape[0]}，测试样本数: {X_test.shape[0]}")
        model = tune_logistic(X_train, y_train)
        scores = report(model, X_test, y_test, y_test_cont)
        plot_score_vs_return(scores, y_test_cont)

if __name__ == "__main__":
    main()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix, r2_score
# from scipy.stats import spearmanr
# from imblearn.over_sampling import RandomOverSampler  # ✅ 新增
# from preprocessing import load_and_preprocess

# def join_words(word_list):
#     return ' '.join(word_list)

# def build_features(df, max_features=5000):
#     df['text_joined'] = df['cleaned_text'].apply(join_words)
#     df = df[df['text_joined'].str.strip() != '']
#     df = df[df['1_DAY_RETURN'].notna()]
    
#     print("Remaining valid samples:", len(df))
#     print("\nSample text:\n", df['text_joined'].iloc[0])

#     # ✅ 重新定义标签：±1%为中性
#     df['label'] = df['1_DAY_RETURN'].apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))

#     vectorizer = TfidfVectorizer(max_features=max_features)
#     X = vectorizer.fit_transform(df['text_joined'])
#     y = df['label'].values
#     y_continuous = df['1_DAY_RETURN'].values

#     return X, y, y_continuous, vectorizer, df

# def tune_logistic(X, y):
#     print("Starting hyperparameter tuning...")
#     Cs = np.logspace(-3, 2, 6)
#     model = LogisticRegression(max_iter=10000, solver='lbfgs', multi_class='multinomial')
#     grid = GridSearchCV(model, {'C': Cs}, cv=5, scoring='accuracy')
#     grid.fit(X, y)
#     print(f"Best C: {grid.best_params_['C']}")
#     print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")
#     return grid.best_estimator_

# def evaluate_ic(y_true_cont, scores):
#     ic = spearmanr(scores, y_true_cont).correlation
#     print(f"📈 Information Coefficient (IC): {ic:.4f}")

# def report(model, X_test, y_test, y_test_continuous):
#     scores = model.predict_proba(X_test) @ np.array([-1, 0, 1])
#     y_pred = model.predict(X_test)

#     print("\n🧪 Classification Report:")
#     print(classification_report(y_test, y_pred))

#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))

#     evaluate_ic(y_test_continuous, scores)
#     r2 = r2_score(y_test_continuous, scores)
#     print(f"📊 R² score (vs true continuous returns): {r2:.4f}")

# def plot_score_vs_return(scores, y_true_cont):
#     df = pd.DataFrame({'pred_score': scores, 'actual_return': y_true_cont})
#     df['decile'] = pd.qcut(df['pred_score'], 10, labels=False, duplicates='drop')
#     avg_returns = df.groupby('decile')['actual_return'].mean()

#     plt.figure(figsize=(10, 6))
#     plt.plot(avg_returns.index, avg_returns.values, marker='o')
#     plt.title("📊 Predicted Score vs Avg. Next-Day Return", fontsize=16)
#     plt.xlabel("Predicted Score Quantile (Q1 = most negative)", fontsize=14)
#     plt.ylabel("Avg. Next-Day Return", fontsize=14)
#     plt.grid(True)
#     plt.xticks(range(len(avg_returns)), [f"Q{i+1}" for i in range(len(avg_returns))], fontsize=12)
#     plt.tight_layout()
#     plt.show()

#     print("\nAverage next-day returns by group:")
#     for i, r in enumerate(avg_returns):
#         print(f"Q{i+1}: {r:.6f}")

# def main():
#     print("📦 Loading and preprocessing data...")
#     df = load_and_preprocess("data/reduced_dataset-release.csv")

#     print("🔧 Building features and labels...")
#     X, y, y_continuous, vectorizer, df = build_features(df)

#     print("📊 Label 分布（-1=下跌，0=平盘，1=上涨）:")
#     print(df['label'].value_counts())

#     print("🔀 Splitting data...")
#     X_train, X_test, y_train, y_test, y_train_continuous, y_test_continuous = train_test_split(
#         X, y, y_continuous, test_size=0.2, random_state=42)

#     print("🧪 Balancing data with RandomOverSampler...")
#     ros = RandomOverSampler(random_state=42)
#     X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

#     print("🎯 Training model...")
#     model = tune_logistic(X_train_resampled, y_train_resampled)

#     print("📈 Evaluating model...")
#     report(model, X_test, y_test, y_test_continuous)

#     print("📉 Plotting score vs return...")
#     scores = model.predict_proba(X_test) @ np.array([-1, 0, 1])
#     plot_score_vs_return(scores, y_test_continuous)

# if __name__ == "__main__":
#     main()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix, r2_score
# from scipy.stats import spearmanr
# from preprocessing import load_and_preprocess

# LABEL_THRESHOLD = 0.002  # ✅ 你可以改成 0.001、0.003 来平衡标签

# def join_words(word_list):
#     """Join list of words into a single string for Tfidf processing"""
#     return ' '.join(word_list)

# def build_features(df, max_features=5000):
#     """Build TF-IDF features and generate labels"""
#     df['text_joined'] = df['cleaned_text'].apply(join_words)
#     df = df[df['text_joined'].str.strip() != '']
#     df = df[df['1_DAY_RETURN'].notna()]
#     print("Remaining valid samples:", len(df))
#     print("\nSample text:\n", df['text_joined'].iloc[0])

#     # Label generation: positive=1, negative=-1, neutral=0
#     df['label'] = df['1_DAY_RETURN'].apply(
#         lambda x: 1 if x > LABEL_THRESHOLD else (-1 if x < -LABEL_THRESHOLD else 0)
#     )

#     # ✅ 标签分布统计和可视化
#     label_counts = df['label'].value_counts().sort_index()
#     print("\n Label 分布（-1=下跌，0=平盘，1=上涨）:")
#     print(label_counts)

#     sns.barplot(x=label_counts.index.astype(str), y=label_counts.values)
#     plt.title("Label 分布情况")
#     plt.xlabel("Label")
#     plt.ylabel("样本数量")
#     plt.show()

#     vectorizer = TfidfVectorizer(max_features=max_features)
#     X = vectorizer.fit_transform(df['text_joined'])
#     y = df['label'].values
#     y_continuous = df['1_DAY_RETURN'].values

#     return X, y, y_continuous, vectorizer

# def tune_logistic(X, y):
#     """Use GridSearchCV to find the best C parameter"""
#     print("Starting hyperparameter tuning...")
#     Cs = np.logspace(-3, 2, 6)
#     model = LogisticRegression(
#         max_iter=10000, 
#         solver='lbfgs', 
#         multi_class='multinomial',
#         class_weight='balanced'  # ✅ 自动处理标签不平衡
#     )
#     grid = GridSearchCV(model, {'C': Cs}, cv=5, scoring='accuracy')
#     grid.fit(X, y)
#     print(f"Best C: {grid.best_params_['C']}")
#     print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")
#     return grid.best_estimator_

# def evaluate_ic(y_true_cont, scores):
#     """Calculate Information Coefficient (Spearman rank correlation)"""
#     y_true_cont = np.ravel(y_true_cont)
#     scores = np.ravel(scores)
#     ic = spearmanr(scores, y_true_cont).correlation
#     print(f"Information Coefficient (IC): {ic:.4f}")

# def report(model, X_test, y_test, y_test_continuous):
#     """Evaluate model performance: classification report, confusion matrix, and IC"""
#     scores = model.predict_proba(X_test) @ np.array([-1, 0, 1])  # expected direction score
#     y_pred = model.predict(X_test)

#     print("\n Classification Report:")
#     print(classification_report(y_test, y_pred))

#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))

#     evaluate_ic(y_test_continuous, scores)
#     r2 = r2_score(y_test_continuous, scores)
#     print(f" R² score (vs true continuous returns): {r2:.4f}")

# def plot_score_vs_return(scores, y_true_cont):
#     """Plot predicted direction scores vs average actual next-day returns"""
#     df = pd.DataFrame({'pred_score': scores, 'actual_return': y_true_cont})
#     df['decile'] = pd.qcut(df['pred_score'], 10, labels=False, duplicates='drop')
#     avg_returns = df.groupby('decile')['actual_return'].mean()

#     plt.figure(figsize=(10, 6))
#     plt.plot(avg_returns.index, avg_returns.values, marker='o')
#     plt.title("Model Predicted Direction Score vs Average Next-Day Return", fontsize=16)
#     plt.xlabel("Predicted Direction Score Group (Negative to Positive)", fontsize=14)
#     plt.ylabel("Average Actual Next-Day Return", fontsize=14)
#     plt.grid(True)
#     plt.xticks(range(len(avg_returns)), [f"Q{i+1}" for i in range(len(avg_returns))], fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.show()

#     print("\nAverage next-day returns by group:")
#     for i, r in enumerate(avg_returns):
#         print(f"Q{i+1}: {r:.6f}")

# def main():
#     print(" Loading and preprocessing data...")
#     df = load_and_preprocess("data/reduced_dataset-release.csv")

#     print(" Converting text and building features...")
#     X, y, y_continuous, vectorizer = build_features(df)

#     print(" Splitting training and testing sets...")
#     X_train, X_test, y_train, y_test, y_train_continuous, y_test_continuous = train_test_split(
#         X, y, y_continuous, test_size=0.2, random_state=42)

#     print(" Training and tuning logistic regression model...")
#     model = tune_logistic(X_train, y_train)

#     print(" Evaluating model performance...")
#     report(model, X_test, y_test, y_test_continuous)

#     print(" Analyzing predicted scores vs returns...")
#     scores = model.predict_proba(X_test) @ np.array([-1, 0, 1])
#     plot_score_vs_return(scores, y_test_continuous)

# if __name__ == "__main__":
#     main()
