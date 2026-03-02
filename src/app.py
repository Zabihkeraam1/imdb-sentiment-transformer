from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="A simple API for sentiment analysis using a fine-tuned DistilBERT model on the IMDB dataset.",
    version="1.0.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

model.to(device)
model.eval()


@app.post("/predict")
def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    return {
        "sentiment": "Positive" if prediction == 1 else "Negative",
        "confidence": round(confidence, 4)
    }


@app.get("/health")
def health():
    return {"status": "ok"}