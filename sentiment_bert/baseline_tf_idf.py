"""
baseline_tf_idf.py
使用 TF-IDF + Logistic Regression 在 IMDB 数据集上做情感分析 baseline
"""

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_imdb_subset(n_samples: int = 10000):

    print("🔄 Loading IMDB dataset from HuggingFace...")
    dataset = load_dataset("imdb")  # 会自动下载到本地缓存

    # 取一部分 train + 一部分 test，合在一起再切分
    train_data = dataset["train"].shuffle(seed=42).select(range(min(n_samples, len(dataset["train"]))))
    test_data = dataset["test"].shuffle(seed=42).select(range(min(n_samples // 2, len(dataset["test"]))))

    texts = list(train_data["text"]) + list(test_data["text"])
    labels = list(train_data["label"]) + list(test_data["label"])

    df = pd.DataFrame({"text": texts, "label": labels})
    print(f"✅ Loaded {len(df)} samples.")
    return df


def build_tfidf_features(train_texts, val_texts, max_features: int = 20000):
    """
    使用 TF-IDF 把文本转换成稀疏向量
    """
    print("🔧 Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # 使用 1-gram + 2-gram
        stop_words="english"
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    print(f"✅ TF-IDF features shape: {X_train.shape}")
    return X_train, X_val, vectorizer


def train_logistic_regression(X_train, y_train, C: float = 2.0):
    """
    训练一个逻辑回归分类器
    """
    print("🚀 Training Logistic Regression baseline...")
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    print("✅ Training finished.")
    return clf


def main():
    # 1. 加载数据
    df = load_imdb_subset(n_samples=12000)

    # 2. 切分训练 / 验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # 3. TF-IDF 特征
    X_train, X_val, vectorizer = build_tfidf_features(train_texts, val_texts, max_features=20000)

    # 4. 训练逻辑回归xi
    clf = train_logistic_regression(X_train, train_labels, C=2.0)

    # 5. 在验证集上评估
    print("📈 Evaluating on validation set...")
    val_preds = clf.predict(X_val)
    acc = accuracy_score(val_labels, val_preds)
    print(f"\n⭐ Baseline Accuracy: {acc * 100:.2f}%\n")

    print("Classification report:")
    print(classification_report(val_labels, val_preds, target_names=["negative", "positive"]))

    print("Confusion matrix:")
    print(confusion_matrix(val_labels, val_preds))


if __name__ == "__main__":
    main()
