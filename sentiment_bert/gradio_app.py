import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/distilbert_imdb_best"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        labels = ["negative", "positive"]
        return {labels[i]: float(probs[i]) for i in range(2)}

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, label="Review"),
    outputs=gr.Label(num_top_classes=2, label="Sentiment"),
    title="IMDB Sentiment Demo",
    description="Type a movie review and see the sentiment scores."
)

demo.launch(share=True)
