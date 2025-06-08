from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from scipy.stats import spearmanr

def train_binary_model(X, y_binary, pos_label):
    neg_weight = 1
    pos_weight = (y_binary == 0).sum() / (y_binary == 1).sum()
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=pos_weight
    )
    sample_weights = np.where(y_binary == 1, pos_weight, neg_weight)
    model.fit(X, y_binary, sample_weight=sample_weights)
    return model

def train_dual_binary_models(X, y):
    y_up = (y == 2).astype(int)
    y_down = (y == 0).astype(int)

    model_up = train_binary_model(X, y_up, pos_label=2)
    model_down = train_binary_model(X, y_down, pos_label=0)

    return model_up, model_down

def predict_with_dual_models(model_up, model_down, X):
    pred_up = model_up.predict(X)
    pred_down = model_down.predict(X)

    y_pred = np.full_like(pred_up, fill_value=1)  # default to neutral
    y_pred[pred_up == 1] = 2
    y_pred[pred_down == 1] = 0
    return y_pred

def predict_proba_with_dual_models(model_up, model_down, X):
    proba_up = model_up.predict_proba(X)[:, 1]
    proba_down = model_down.predict_proba(X)[:, 1]
    proba_neutral = 1 - np.maximum(proba_up, proba_down)

    # Normalize to make probabilities sum to 1
    total = proba_up + proba_down + proba_neutral
    return np.vstack([
        proba_down / total,
        proba_neutral / total,
        proba_up / total
    ]).T

def report_dual(model_up, model_down, X_test, y_test, y_test_continuous):
    y_pred = predict_with_dual_models(model_up, model_down, X_test)
    y_proba = predict_proba_with_dual_models(model_up, model_down, X_test)
    scores = y_proba @ np.array([-1, 0, 1])

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    evaluate_ic(y_test_continuous, scores)
    r2 = r2_score(y_test_continuous, scores)
    print(f"📉 R² vs true return: {r2:.4f}")

    correct_positive = ((y_pred == 2) & (y_test == 2)).sum()
    correct_negative = ((y_pred == 0) & (y_test == 0)).sum()
    total_predicted_trades = ((y_pred == 2) | (y_pred == 0)).sum()
    if total_predicted_trades > 0:
        win_rate = (correct_positive + correct_negative) / total_predicted_trades
        print(f"🏆 Win Rate（胜率）: {win_rate:.2%}")
    else:
        print("⚠️ 没有预测出正向或负向交易，无法计算胜率")

    return scores


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix, r2_score
# from xgboost import XGBClassifier
# from scipy.stats import spearmanr
# from collections import Counter
# from preprocessing import load_and_preprocess

# def join_words(word_list):
#     return ' '.join(word_list)

# def get_sample_weights(y, weights_dict=None):
#     counter = Counter(y)
#     if weights_dict is None:
#         majority = max(counter.values())
#         return np.array([majority / counter[label] for label in y])
#     else:
#         return np.array([weights_dict.get(label, 1) for label in y])

# def tune_xgboost(X, y):
#     weights_dict = {0: 3, 1: 1, 2: 3}  # 给少数类更大权重
#     sample_weights = get_sample_weights(y, weights_dict)
#     model = XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', use_label_encoder=False)
#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [3, 5],
#         'learning_rate': [0.01, 0.1],
#         'subsample': [0.8, 1.0]
#     }
#     grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
#     grid.fit(X, y, sample_weight=sample_weights)
#     return grid.best_estimator_

# def evaluate_ic(y_true_cont, scores):
#     ic = spearmanr(scores, y_true_cont).correlation
#     print(f"📈 IC（Information Coefficient）: {ic:.4f}")

# def report(model, X_test, y_test, y_test_continuous):
#     # 预测时给三个类别赋值 -1,0,1 得分，用于连续相关分析
#     scores = model.predict_proba(X_test) @ np.array([-1, 0, 1])
#     y_pred = model.predict(X_test)

#     print("\n📊 Classification Report:")
#     print(classification_report(y_test, y_pred))
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))

#     evaluate_ic(y_test_continuous, scores)
#     r2 = r2_score(y_test_continuous, scores)
#     print(f"📉 R² vs true return: {r2:.4f}")

#     # 计算 Win Rate（胜率）
#     correct_positive = ((y_pred == 2) & (y_test == 2)).sum()
#     correct_negative = ((y_pred == 0) & (y_test == 0)).sum()
#     total_predicted_trades = ((y_pred == 2) | (y_pred == 0)).sum()
#     if total_predicted_trades > 0:
#         win_rate = (correct_positive + correct_negative) / total_predicted_trades
#         print(f"🏆 Win Rate（胜率）: {win_rate:.2%}")
#     else:
#         print("⚠️ 没有预测出正向或负向交易，无法计算胜率")

#     return scores

# def plot_score_vs_return(scores, y_true_cont):
#     df = pd.DataFrame({'score': scores, 'ret': y_true_cont})
#     df['quantile'] = pd.qcut(df['score'], 10, labels=False, duplicates='drop')
#     avg_ret = df.groupby('quantile')['ret'].mean()

#     plt.figure(figsize=(10, 6))
#     plt.plot(avg_ret.index, avg_ret.values, marker='o')
#     plt.title("📊 Score Quantile vs Avg. Return")
#     plt.xlabel("Quantile (Q0 = lowest score)")
#     plt.ylabel("Avg. 1-Day Return")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     print("\n各分组平均收益：")
#     for i, r in enumerate(avg_ret):
#         print(f"Q{i}: {r:.6f}")

# def build_features(df, vectorizer=None, max_features=5000):
#     df['text_joined'] = df['cleaned_text']  # 你数据本身已经是预处理好的文本列表或字符串
#     df = df[df['text_joined'].str.strip() != '']
#     df = df[df['1_DAY_RETURN'].notna()]
    
#     # 分类标签：2代表上涨（>1%），0代表下跌（<-1%），1代表中性
#     df['label'] = df['1_DAY_RETURN'].apply(lambda x: 2 if x > 0.01 else (0 if x < -0.01 else 1))

#     if vectorizer is None:
#         vectorizer = TfidfVectorizer(max_features=max_features)
#         X = vectorizer.fit_transform(df['text_joined'])
#     else:
#         X = vectorizer.transform(df['text_joined'])
    
#     y = df['label'].values
#     y_continuous = df['1_DAY_RETURN'].values

#     return X, y, y_continuous, vectorizer, df

# def main():
#     print("📦 读取并预处理数据...")
#     df = load_and_preprocess(r"c:\\Users\\yw562\\Downloads\\predicting-returns-with-text-data-master\\predicting-returns-with-text-data-master\\data\\cleaned_aligned_data.csv")
#     stock_list = ["Nike", "eBay", "Reuters", "Netflix", "Amazon"]
#     print("📈 股票列表：", stock_list)

#     for stock in stock_list:
#         print(f"\n🔍 正在处理股票：{stock}")
#         df_stock = df[df['STOCK_CODE'] == stock].copy()
#         if len(df_stock) < 100:
#             print(f"🚫 样本数不足，跳过 {stock}")
#             continue

#         df_stock = df_stock.sort_values('DATE')
#         split_date = df_stock['DATE'].quantile(0.8)
#         train_df = df_stock[df_stock['DATE'] <= split_date]
#         test_df = df_stock[df_stock['DATE'] > split_date]

#         if len(train_df) < 50 or len(test_df) < 20:
#             print(f"🚫 训练或测试数据太少，跳过 {stock}")
#             continue

#         X_train, y_train, _, vectorizer, _ = build_features(train_df)
#         X_test, y_test, y_test_cont, _, _ = build_features(test_df, vectorizer=vectorizer)

#         print(f"✅ 训练样本数: {X_train.shape[0]}，测试样本数: {X_test.shape[0]}")
#         model = tune_xgboost(X_train, y_train)
#         scores = report(model, X_test, y_test, y_test_cont)
#         plot_score_vs_return(scores, y_test_cont)

# if __name__ == "__main__":
#     main()
