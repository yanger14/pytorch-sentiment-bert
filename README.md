# pytorch-sentiment-bert

# 🚀 BERT Sentiment Analysis with PyTorch + FastAPI + Gradio

A complete end-to-end **Sentiment Analysis system** built with **PyTorch**, **HuggingFace Transformers**, **FastAPI**, and **Gradio**.  
This project fine-tunes **DistilBERT** on the IMDB movie review dataset and provides:

✔ A trained DistilBERT model (~90% accuracy)  
✔ A REST API for real-time inference  
✔ A web demo UI built using Gradio  
✔ Baseline model using TF-IDF + Logistic Regression  
✔ Full training, evaluation, and deployment pipeline  

---

## ✨ Features

### 🔥 1. Fine-tuned DistilBERT model
- Achieved **~90% test accuracy**  
- GPU-accelerated training  
- Tokenization + attention masks  
- Softmax probability scoring  

### ⚡ 2. FastAPI REST API
- Lightweight inference endpoint  
- Accepts raw text, returns JSON  
- Suitable for production deployment (Docker / Render / Railway)

### 🎨 3. Web Demo with Gradio
- Clean and interactive interface  
- Allows anyone to try the model  
- Locally hosted or cloud-deployed  

### 📊 4. Classical ML Baseline
- TF-IDF + Logistic Regression  
- Helps compare BERT vs traditional NLP  

---

## 🧠 Tech Stack

| Category | Technology |
|---------|------------|
| Language | Python |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Classical ML | scikit-learn (TF-IDF + Logistic Regression) |
| Deployment | FastAPI, Uvicorn |
| UI | Gradio |
| Environment | Conda, CUDA (RTX 4060) |
| Version Control | Git + GitHub |

---

## 📁 Project Structure

```
pytorch-sentiment-bert/
├── train_bert.py            # Training DistilBERT model
├── eval_bert.py             # Evaluation on IMDB
├── baseline_tf_idf.py       # TF-IDF baseline model
│
├── sentiment_bert/
│   └── api.py               # FastAPI inference server
│
├── gradio_app.py            # Gradio web demo
│
├── models/
│   └── distilbert_imdb_best/   # Saved fine-tuned model (ignored in git)
│
├── test_torch.py            # CUDA + PyTorch test
├── requirements.txt
└── README.md
```
