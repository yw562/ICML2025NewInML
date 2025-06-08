

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 2 – SESTM 最小复现流水线  v4
================================
* χ² / MI 词筛选 → 2‑Topic LDA → Logistic 分类
* **进度监控** : logging + tqdm 进度条
* **断点续跑** : 如已检测到 models/*.pkl 且类型正确，跳过训练
* **健壮性**  : 自动检测坏模型（例如旧版本存下来的 ndarray），若不合法自动重训
* **可视化**  : 词云 & 情感分布，--visualize / --no_visualize 开关

运行示例
---------
```bash
pip install pandas scikit-learn wordcloud matplotlib tqdm nltk
python day2_sestm_pipeline.py \
    --data cleaned_aligned_data.csv \
    --text_col cleaned_text \
    --return_col 1_DAY_RETURN
```
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import seaborn as sns
# 顶部 import 里加一行
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS   # 新增


# 可选可视化依赖
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError:
    WordCloud = None  # type: ignore

# ---------------------------- 工具函数 ---------------------------- #

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset(path: str,
                 text_col: str,
                 label_col: str = None,
                 return_col: str = None) -> pd.DataFrame:
    """读取 CSV 并生成二值标签列 `label`"""
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise KeyError(f"文本列 {text_col} 不存在！实际列: {df.columns.tolist()[:10]} …")

    if label_col and label_col in df.columns:
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    else:
        # 需根据收益列生成 label
        candidate = return_col or next(
            (c for c in df.columns if str(c).lower() in {"1_day_return", "return_nextday", "ret", "return"}),
            None,
        )
        if candidate is None:
            raise ValueError("未找到可用收益列，请通过 --return_col 指定！")
        df = df[[text_col, candidate]].rename(columns={text_col: "text", candidate: "ret"})
        df["label"] = (df["ret"] > 0).astype(int)
    df.dropna(subset=["text"], inplace=True)
    return df


def select_top_k_terms(X, y, vectorizer: CountVectorizer, k: int = 5000, method: str = "chi2") -> Tuple[np.ndarray, CountVectorizer]:
    """根据 χ² 或互信息打分，截取 top‑k 词。
    返回稀疏矩阵及新 vectorizer（仅包含 top‑k 词）"""
    if method == "chi2":
        scores, _ = chi2(X, y)
    else:
        scores = mutual_info_classif(X, y, discrete_features=True)

    top_idx = np.argsort(scores)[-k:]
    # 获取所有特征词
    all_features = vectorizer.get_feature_names_out()
    # 选择top-k词汇构建新的词汇表
    selected_features = all_features[top_idx]
    vocab = {term: i for i, term in enumerate(selected_features)}
    
    # 创建新的vectorizer，只包含选中的词汇
    vec_sel = CountVectorizer(vocabulary=vocab)
    
    # 从原始X矩阵中提取对应的列
    X_sel = X[:, top_idx]
    
    return X_sel, vec_sel

def train_lda(X, n_topics: int = 2) -> Tuple[LatentDirichletAllocation, np.ndarray]:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=50,
        random_state=42,
    )
    logging.info("训练 LDA …")
    doc_topic = lda.fit_transform(X)
    return lda, doc_topic


def train_classifier(features: np.ndarray, labels: np.ndarray) -> Tuple[LogisticRegression, float, float]:
    X_tr, X_val, y_tr, y_val = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    acc = accuracy_score(y_val, (proba > 0.5).astype(int))
    return clf, auc, acc


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------- 可视化 ---------------------------- #

def save_wordcloud_dict(lda: LatentDirichletAllocation, vec: CountVectorizer, outdir: Path, topn: int = 50):
    """保存每个主题的词云数据为词典格式"""
    ensure_dir(outdir)
    terms = np.array(vec.get_feature_names_out())
    
    all_topics_dict = {}
    for k in range(lda.n_components):
        top_idx = np.argsort(lda.components_[k])[-topn:]
        topic_dict = {terms[idx]: float(lda.components_[k, idx]) for idx in top_idx}
        # 按权重降序排列
        topic_dict = dict(sorted(topic_dict.items(), key=lambda x: x[1], reverse=True))
        all_topics_dict[f"topic_{k}"] = topic_dict
        
        # 保存单个主题词典为JSON
        topic_file = outdir / f"topic{k}_worddict.json"
        with open(topic_file, "w", encoding="utf-8") as f:
            json.dump(topic_dict, f, ensure_ascii=False, indent=2)
        logging.info(f"主题{k}词典已保存 → {topic_file}")
    
    # 保存所有主题的综合词典
    all_file = outdir / "all_topics_worddict.json"
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(all_topics_dict, f, ensure_ascii=False, indent=2)
    logging.info(f"所有主题词典已保存 → {all_file}")
    
    return all_topics_dict

def generate_wordclouds(lda: LatentDirichletAllocation, vec: CountVectorizer, outdir: Path, topn: int = 50):
    if WordCloud is None:
        logging.warning("wordcloud 库未安装，跳过词云绘制。")
        return
    ensure_dir(outdir)
    terms = np.array(vec.get_feature_names_out())
    for k in range(lda.n_components):
        top_idx = np.argsort(lda.components_[k])[-topn:]
        freqs = {t: lda.components_[k, idx] for t, idx in zip(terms[top_idx], top_idx)}
        wc = WordCloud(width=800, height=500, background_color="white").generate_from_frequencies(freqs)
        plt.figure(figsize=(8, 5))
        plt.imshow(wc)
        plt.axis("off")
        out_path = outdir / f"topic{k}_wordcloud.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"词云已保存 → {out_path}")


def plot_tone_distribution(net_tone: np.ndarray, labels: np.ndarray, outdir: Path):
    if WordCloud is None:
        return  # 缺 matplotlib 也会在导入阶段被捕获
    ensure_dir(outdir)
    plt.figure(figsize=(6, 4))
    for lab, name in zip([1, 0], ["Up", "Down"]):
        sns.kdeplot(net_tone[labels == lab], label=name, fill=True)
    plt.title("Net‑Tone Density")
    plt.legend()
    out_path = outdir / "tone_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"情感分布图已保存 → {out_path}")

# ---------------------------- 主流程 ---------------------------- #

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cleaned_aligned_data.csv", help="CSV 数据路径")
    ap.add_argument("--text_col", default="cleaned_text", help="文本列名")
    ap.add_argument("--label_col", default=None, help="已有标签列名 (0/1)")
    ap.add_argument("--return_col", default="1_DAY_RETURN", help="收益列名，用于派生标签")
    ap.add_argument("--top_k", type=int, default=5000, help="词筛选 top‑k 大小")
    ap.add_argument("--method", choices=["chi2", "mi"], default="chi2", help="筛选统计量")
    ap.add_argument("--model_dir", default="models", help="模型保存目录")
    ap.add_argument("--fig_dir", default="figs", help="可视化输出目录")
    ap.add_argument("--no_visualize", dest="visualize", action="store_false", default=True, help="禁用可视化")
    ap.add_argument("--force_retrain", action="store_true", help="忽略现有模型并重新训练")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    vec_pkl = model_dir / "vectorizer.pkl"
    lda_pkl = model_dir / "lda_model.pkl"
    clf_pkl = model_dir / "logreg.pkl"

    # ------------- 检测已有模型 ------------- #
    if (not args.force_retrain and vec_pkl.exists() and lda_pkl.exists() and clf_pkl.exists()):
        vectorizer = load_pickle(vec_pkl)
        lda = load_pickle(lda_pkl)
        clf = load_pickle(clf_pkl)
        # 简单类型检查
        if not hasattr(vectorizer, "transform"):
            logging.warning("检测到旧版 vectorizer 非 CountVectorizer，将重新训练……")
        else:
            logging.info("检测到现有合法模型，直接加载并跳过训练。")
            # 评估 & 可视化  ---------------------
            df = load_dataset(args.data, args.text_col, args.label_col, args.return_col)
            X_transformed = vectorizer.transform(df["text"])
            doc_topic = lda.transform(X_transformed)
            net_tone = doc_topic[:, 0] - doc_topic[:, 1]
            auc = roc_auc_score(df["label"], clf.predict_proba(doc_topic)[:, 1])
            logging.info(f"整体 AUC = {auc:.3f}")
            if args.visualize:
                generate_wordclouds(lda, vectorizer, Path(args.fig_dir))
                save_wordcloud_dict(lda, vectorizer, Path(args.fig_dir))
                plot_tone_distribution(net_tone, df["label"].values, Path(args.fig_dir))
            return
    # ---------------- 重训 ---------------- #
    logging.info("开始重新训练模型…")
    ensure_dir(model_dir)

    # 数据加载
    df = load_dataset(args.data, args.text_col, args.label_col, args.return_col)
    texts = df["text"].tolist()
    y = df["label"].values

    # # 向量化
    # logging.info("→ 文本向量化 …")
    # # ↓↓↓ 在 main() 里创建 base_vec 的那一行改成这样 ↓↓↓
    # custom_stop = ENGLISH_STOP_WORDS.union({"rt", "amp"})            # 把 rt/amp 合并进英文停用词表
    # base_vec = CountVectorizer(min_df=3, stop_words=custom_stop, lowercase=True)
    # # base_vec = CountVectorizer(min_df=3, stop_words="english", lowercase=True)
    # X_full = base_vec.fit_transform(texts)

    # 向量化
    logging.info("→ 文本向量化 …")
    # 把 rt/amp 合并进英文停用词表，并转换为列表
    custom_stop = list(ENGLISH_STOP_WORDS.union({"rt", "amp"}))
    base_vec = CountVectorizer(min_df=3, stop_words=custom_stop, lowercase=True)
    X_full = base_vec.fit_transform(texts)

    # 词筛选
    logging.info("→ 词项筛选 (top‑%d) …", args.top_k)
    X_sel, vec_sel = select_top_k_terms(X_full, y, base_vec, k=args.top_k, method=args.method)

    # LDA 训练
    lda, doc_topic = train_lda(X_sel)

    # 分类器
    logging.info("→ 训练 Logistic 分类器 …")
    clf, auc, acc = train_classifier(doc_topic, y)
    logging.info(f"验证 AUC = {auc:.3f} | Accuracy = {acc:.3f}")

    # 保存
    save_pickle(vec_sel, vec_pkl)
    save_pickle(lda, lda_pkl)
    save_pickle(clf, clf_pkl)
    logging.info(f"✔ 模型已保存到 {model_dir}")

    # 可视化
    if args.visualize:
        generate_wordclouds(lda, vec_sel, Path(args.fig_dir))
        save_wordcloud_dict(lda, vec_sel, Path(args.fig_dir))
        net_tone = doc_topic[:, 0] - doc_topic[:, 1]
        plot_tone_distribution(net_tone, y, Path(args.fig_dir))


if __name__ == "__main__":
    main()
