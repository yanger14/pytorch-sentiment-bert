"""
train_bert.py
使用 DistilBERT 在 IMDB 数据集上做情感分析微调
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)


def main():
    # ========= 0. 设备 =========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ========= 1. 超参数 =========
    model_name = "distilbert-base-uncased"
    num_epochs = 3
    train_batch_size = 16
    eval_batch_size = 32
    learning_rate = 2e-5
    max_train_samples = 8000     # 为了跑得快，先用部分数据，后面你可以调大
    max_eval_samples = 2000

    # ========= 2. 加载 IMDB 数据 =========
    print("🔄 Loading IMDB dataset...")
    raw_datasets = load_dataset("imdb")

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets["test"]

    # 抽样一部分，方便快速实验
    if max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=42).select(range(max_train_samples))
    if max_eval_samples is not None:
        test_dataset = test_dataset.shuffle(seed=42).select(range(max_eval_samples))

    print("Train samples:", len(train_dataset))
    print("Eval samples:", len(test_dataset))

    # ========= 3. 加载 tokenizer =========
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=256
        )

    print("🔧 Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 设置为 PyTorch 格式
    tokenized_train.set_format(type="torch")
    tokenized_eval.set_format(type="torch")

    # 动态 padding 的 collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_train,
        shuffle=True,
        batch_size=train_batch_size,
        collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_eval,
        shuffle=False,
        batch_size=eval_batch_size,
        collate_fn=data_collator
    )

    # ========= 4. 加载预训练 BERT 模型 =========
    print(f"📦 Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    model.to(device)

    # ========= 5. 优化器 & 学习率调度器 =========
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # ========= 6. 训练循环 =========
    global_step = 0
    best_eval_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if (step + 1) % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Step [{step + 1}/{len(train_dataloader)}] "
                      f"Loss: {avg_loss:.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # ========= 7. 每个 epoch 结束后在验证集上评估 =========
        eval_acc = evaluate(model, eval_dataloader, device)
        print(f"Epoch {epoch} Eval Accuracy: {eval_acc * 100:.2f}%")

        # 保存最好的一版模型
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            save_dir = "models/distilbert_imdb_best"
            print(f"💾 New best model, saving to {save_dir}")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    print(f"\n🎉 Training finished. Best Eval Accuracy: {best_eval_acc * 100:.2f}%")


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]  # IMDB 的标签字段名是 labels
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc


if __name__ == "__main__":
    main()
