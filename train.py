import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm

# =============================
# Training Script for Fake News Detector
# =============================

class NewsDataset(Dataset):
    def __init__(self, dataframe, text_col="clean_text", label_col="label_id", max_len=256, tokenizer_name="distilbert-base-uncased"):
        self.texts = dataframe[text_col].astype(str).tolist()
        self.labels = dataframe[label_col].tolist()
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, preds, labels = 0, [], []
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=target)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        labels.extend(target.cpu().numpy())

    acc = accuracy_score(labels, preds)
    return total_loss / len(loader), acc

def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss, preds, labels = 0, [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=target)
            total_loss += outputs.loss.item()
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(target.cpu().numpy())

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, zero_division=0)
    return total_loss / len(loader), acc, report

def train_model(train_df, val_df, model_name="distilbert-base-uncased", epochs=2, lr=2e-5, batch_size=16, save_dir="best_fake_news_model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NewsDataset(train_df, tokenizer_name=model_name)
    val_dataset   = NewsDataset(val_df, tokenizer_name=model_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        val_loss, val_acc, report = eval_one_epoch(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(report)

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(save_dir)
    print(f"✅ Model saved at {save_dir}")
    return model

