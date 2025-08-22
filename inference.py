import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

MODEL_DIR = "best_fake_news_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

def predict(texts):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return [{"text": t, "prediction": "FAKE" if p==1 else "REAL", "prob_fake": float(prob[1]), "prob_real": float(prob[0])}
            for t, p, prob in zip(texts, preds, probs)]

if __name__ == "__main__":
    samples = [
        "NASA announces successful Artemis I mission.",
        "Aliens secretly built the pyramids, scientists confirm."
    ]
    results = predict(samples)
    df = pd.DataFrame(results)
    print(df)
