from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles   # 如果以后要放单独的 css/js 文件


# 注意：模型路径来自 sentiment_bert 的上一级目录
MODEL_PATH = "../models/distilbert_imdb_best"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading model from: {MODEL_PATH}, device={device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

app = FastAPI(title="IMDB Sentiment API", version="1.0")
@app.get("/demo", response_class=HTMLResponse)
def sentiment_demo():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>IMDB Sentiment Demo</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #0f172a;
                color: #e5e7eb;
                display: flex;
                justify-content: center;
                padding: 40px;
            }
            .card {
                background: #111827;
                padding: 24px 28px;
                border-radius: 16px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.6);
                width: 600px;
            }
            h1 {
                font-size: 24px;
                margin-bottom: 4px;
            }
            p.subtitle {
                font-size: 13px;
                color: #9ca3af;
                margin-bottom: 16px;
            }
            textarea {
                width: 100%;
                min-height: 120px;
                padding: 10px 12px;
                border-radius: 10px;
                border: 1px solid #374151;
                background: #020617;
                color: #e5e7eb;
                resize: vertical;
                font-size: 14px;
            }
            button {
                margin-top: 14px;
                padding: 8px 16px;
                border-radius: 999px;
                border: none;
                background: #3b82f6;
                color: white;
                font-weight: 600;
                cursor: pointer;
            }
            button:hover {
                background: #2563eb;
            }
            .result {
                margin-top: 18px;
                padding: 10px 12px;
                border-radius: 12px;
                background: #020617;
                border: 1px solid #1f2937;
                font-size: 14px;
            }
            .label-positive { color: #22c55e; font-weight: 600; }
            .label-negative { color: #f97316; font-weight: 600; }
            .score { color: #9ca3af; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>IMDB Sentiment Demo</h1>
            <p class="subtitle">输入一段英文影评，点击按钮，模型会返回情感和置信度。</p>
            <textarea id="inputText" placeholder="Type an English review here..."></textarea>
            <button onclick="sendRequest()">Analyze Sentiment</button>
            <div id="result" class="result" style="display:none;"></div>
        </div>

        <script>
        async function sendRequest() {
            const text = document.getElementById("inputText").value;
            const resultDiv = document.getElementById("result");

            if (!text.trim()) {
                resultDiv.style.display = "block";
                resultDiv.innerHTML = "请输入一段文本再分析。";
                return;
            }

            resultDiv.style.display = "block";
            resultDiv.innerHTML = "Analyzing...";

            try {
                const resp = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text })
                });
                if (!resp.ok) {
                    resultDiv.innerHTML = "Request failed: " + resp.status;
                    return;
                }
                const data = await resp.json();
                const labelClass = data.label === "positive" ? "label-positive" : "label-negative";
                resultDiv.innerHTML = `
                    <span class="${labelClass}">${data.label.toUpperCase()}</span>
                    <div class="score">Confidence: ${(data.score * 100).toFixed(2)}%</div>
                `;
            } catch (e) {
                resultDiv.innerHTML = "Error: " + e;
            }
        }
        </script>
    </body>
    </html>
    """



class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


@app.get("/")
def home():
    return {"message": "IMDB Sentiment API is running. Visit /docs to try it."}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        score = probs[pred_id].item()

    label = "positive" if pred_id == 1 else "negative"
    return PredictResponse(label=label, score=score)
