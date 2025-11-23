"""
eval_bert.py
加载训练好的 DistilBERT 情感分析模型并进行评估 / 单句预测
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "models/distilbert_imdb_best"


def load_model():
    print(f"🔄 Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()

    label = "positive" if pred == 1 else "negative"
    return label


def evaluate_full_testset(tokenizer, model, max_samples=2000):
    print("🔍 Evaluating model on IMDB test set...")

    dataset = load_dataset("imdb")["test"].shuffle(seed=42).select(range(max_samples))

    correct = 0
    total = len(dataset)

    for item in dataset:
        text = item["text"]
        true_label = item["label"]  # 0 or 1
        pred_label = 1 if predict(text, tokenizer, model) == "positive" else 0

        if true_label == pred_label:
            correct += 1

    acc = correct / total
    print(f"📈 Eval Accuracy on {total} samples: {acc * 100:.2f}%")
    return acc


def main():
    tokenizer, model = load_model()

    # 单句测试
    print("\n=== Demo Prediction ===")
    test_sentence = "I really loved this movie. It was fantastic!"
    print("Sentence:", test_sentence)
    print("Prediction:", predict(test_sentence, tokenizer, model))

    # 完整测试集评估
    print("\n=== Full Test Set Evaluation ===")
    evaluate_full_testset(tokenizer, model)


if __name__ == "__main__":
    main()
