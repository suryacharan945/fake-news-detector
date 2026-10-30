# 📰 AI-Powered Fake News Detection System

[![Fake News Detector CI](https://github.com/suryacharan945/fake-news-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/suryacharan945/fake-news-detector/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/suryacharan945/fake-news-detector)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-orange)](https://gradio.app/)

An end-to-end Natural Language Processing (NLP) system designed to identify misinformation and classify news articles as **REAL** or **FAKE**. Built using fine-tuned **DistilBERT** (`distilbert-base-uncased`) sequence classification, PyTorch, and an interactive Gradio web application with batch CSV evaluation.

* **Hugging Face Model Hub:** [`suryacharan945/fake-news-detector`](https://huggingface.co/suryacharan945/fake-news-detector)

---

## 🏛️ Architecture & Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                   NLP MISINFORMATION DETECTION PIPELINE                │
└────────────────────────────────────────────────────────────────────────┘
          │
          ├── [Input Layer]: Raw Text / News Headlines / Batch CSV
          │
          ├── [Tokenizer]: DistilBERT Tokenizer (`distilbert-base-uncased`)
          │     ├── Padding & Truncation (`max_length=256`)
          │     └── PyTorch Tensor Encoding (`input_ids`, `attention_mask`)
          │
          ├── [Transformer Backbone]: DistilBERT (6-layer, 768-dim, 12 heads, 66M params)
          │     ├── Multi-Head Self-Attention Blocks
          │     └── Pooled [CLS] Representation
          │
          ├── [Classification Head]: Linear Layer (`768 -> 2 classes`) + Softmax
          │     ├── Index 0: FAKE
          │     └── Index 1: REAL
          │
          └── [Application Layer]: Gradio Web UI (`app.py`) & Python API (`inference.py`)
                ├── Single Article Scoring with Label Confidence Breakdown
                └── Batch CSV Ingestion with Downloadable Predictions
```

---

## 🚀 Key Features

- **Fine-Tuned DistilBERT Backbone:** Leverages 97% of BERT's language understanding while cutting parameter count by 40% and providing 2x faster inference.
- **Hugging Face Hub Integration:** Model checkpoints are loaded directly from the Hugging Face Model Hub (`suryacharan945/fake-news-detector`).
- **Interactive Gradio Demonstration:** Web-based UI supporting single-text prediction with probability confidences and batch CSV processing.
- **Reproducible Training Pipeline:** Modular PyTorch dataset loader, AdamW optimization with linear warmup scheduler, and evaluation reporting (`train.py`).

---

## ⚙️ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/suryacharan945/fake-news-detector.git
cd fake-news-detector

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Tests

```bash
python -m unittest discover tests
```

### 3. Launch Gradio Web Application

```bash
python app.py
```
Open `http://localhost:7860` in your browser to interact with the model.

---

## 📂 Repository Layout

```text
├── app.py                  # Gradio interactive web application
├── inference.py            # Programmatic inference pipeline
├── train.py                # PyTorch / Transformers model training script
├── requirements.txt        # Python dependency manifest
├── tests/                  # Automated unit test suite
│   └── test_pipeline.py    # Pipeline syntax, constants, and math validation
├── .github/workflows/      # Automated CI workflow
│   └── ci.yml
├── .gitignore              # Version control ignore rules
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
