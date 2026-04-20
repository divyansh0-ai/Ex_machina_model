# 🎨 Ex Machina — Artwork Medium Classifier

> **Hackathon solution for multi-class classification of artwork mediums using NLP + gradient boosting + BERT fine-tuning.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-orange)](https://lightgbm.readthedocs.io/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/docs/transformers)

---

## 📖 Overview

**Ex Machina** is an end-to-end machine learning pipeline that classifies artworks into **8 medium categories** (e.g., oil painting, watercolor, print, drawing, etc.) based solely on textual metadata — titles, captions, provenance text, inscriptions, and AI-generated visual descriptions.

The pipeline combines:
- **TF-IDF feature engineering** (word & character n-grams)
- **Domain-specific keyword features** (art material signals)
- **LightGBM** gradient boosted classifier with stratified cross-validation
- **DistilBERT fine-tuning** for maximum accuracy (optional advanced step)

---

## 🗂️ Project Structure

```
ex_machina/
├── data/                        # Raw dataset (not tracked in git)
│   ├── train_n (1) (2).csv      # Training data  (4,000 rows, 57 cols)
│   └── test_n (1) (3).csv       # Test data      (1,000 rows, 56 cols)
│
├── outputs/                     # Generated artifacts
│   ├── submission.csv           # Final predictions (LightGBM)
│   ├── submission_bert.csv      # Final predictions (BERT)
│   ├── lgbm_oof.npy             # LightGBM OOF probabilities
│   ├── bert_oof_probs.npy       # BERT OOF probabilities
│   ├── word_vectorizer.pkl      # Saved TF-IDF word vectorizer
│   ├── char_vectorizer.pkl      # Saved TF-IDF char vectorizer
│   ├── scaler.pkl               # Saved feature scaler
│   └── config.json              # Run config & CV scores
│
├── eda_outputs/                 # Exploratory analysis outputs
│
├── main.py                      # 🚀 Primary solution (LightGBM pipeline)
├── bert_solution.py             # 🤖 Advanced solution (DistilBERT fine-tuning)
├── gen_nb.py                    # Notebook generator (Kaggle-compatible)
├── Ex_Machina_Solution.ipynb    # Kaggle submission notebook
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/divyansh0-ai/Ex_machina_model.git
cd Ex_machina_model

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Add Your Data

Place the competition CSV files inside the `data/` folder:
```
data/train_n (1) (2).csv
data/test_n (1) (3).csv
```

### 3. Run the Pipeline

**Option A — LightGBM (fast, recommended):**
```bash
python main.py
```

**Option B — BERT fine-tuning (slower, higher accuracy):**
```bash
python bert_solution.py
```

Predictions are saved to `outputs/submission.csv` and `outputs/submission_bert.csv`.

---

## 🧠 Pipeline Details

### `main.py` — LightGBM Solution

| Step | Description |
|------|-------------|
| **Text Cleaning** | Unicode normalization, HTML stripping, art abbreviation expansion (e.g., `o/c` → `oil on canvas`) |
| **Mega-text** | Weighted concat of all text fields — `assistivetext` (3×), title (2×), tags (2×), caption (2×), provenance, inscription, etc. |
| **Keyword Features** | 35+ binary regex flags for art materials: oil, canvas, watercolor, tempera, etching, lithograph, acrylic, pastel, etc. |
| **Statistical Features** | Text lengths, word counts, temporal features (century, era flags), image dimension features |
| **TF-IDF** | Word n-grams (1–3, 25k features) + Char n-grams (2–6, 15k features) with sublinear TF |
| **Model** | LightGBM with 3-fold stratified CV, early stopping (200 rounds), balanced class weights |

### `bert_solution.py` — DistilBERT Solution

| Step | Description |
|------|-------------|
| **Tokenizer** | `distilbert-base-uncased` (configurable to `roberta-base`) |
| **Input** | Combined text: title (2×), body, caption, tags (2×), notes, type, category |
| **Training** | 5-fold stratified CV, 5 epochs per fold, warmup + weight decay, early stopping (patience=2) |
| **Output** | Averaged softmax probabilities across folds; OOF & test probs saved as `.npy` for meta-stacking |

---

## 📊 Target Classes

The target column `y` contains integer labels `0–7` corresponding to 8 artwork medium categories:

| Label | Medium |
|-------|--------|
| 0 | Oil on Canvas |
| 1 | Oil on Panel/Wood |
| 2 | Watercolor on Paper |
| 3 | Tempera / Egg Tempera |
| 4 | Print (Etching / Lithograph / Engraving) |
| 5 | Drawing (Ink / Pen / Charcoal / Pastel) |
| 6 | Acrylic / Mixed Media |
| 7 | Photograph / Digital |

> *Exact class-to-label mapping is printed at runtime from the `label` column in the training data.*

---

## ⚙️ Key Configuration

Edit these constants at the top of each script:

**`main.py`**
```python
N_FOLDS    = 3       # Stratified CV folds
SEED       = 42      # Reproducibility seed
```

**`bert_solution.py`**
```python
MODEL_NAME = "distilbert-base-uncased"  # Or "roberta-base" for more power
MAX_LEN    = 256     # Tokenizer max sequence length
N_SPLITS   = 5       # CV folds
EPOCHS     = 5       # Training epochs per fold
BATCH_SIZE = 16      # Per-device batch size
LR         = 2e-5    # Learning rate
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `lightgbm` | ≥ 4.0.0 | Gradient boosting classifier |
| `xgboost` | ≥ 2.0.0 | Alternative booster |
| `scikit-learn` | ≥ 1.3.0 | TF-IDF, CV, metrics |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `numpy` | ≥ 1.24.0 | Numerical ops |
| `scipy` | ≥ 1.10.0 | Sparse matrix ops |
| `transformers` | ≥ 4.35.0 | BERT / DistilBERT (optional) |
| `torch` | ≥ 2.0.0 | PyTorch backend (optional) |
| `accelerate` | ≥ 0.24.0 | HuggingFace Trainer acceleration |
| `sentence-transformers` | ≥ 2.2.0 | Embedding models (optional) |

> **GPU users:** Uncomment the CUDA-enabled torch line in `requirements.txt` for faster BERT fine-tuning.

---

## 🏆 Results

| Model | CV Accuracy |
|-------|------------|
| LightGBM (TF-IDF + Keywords) | ~0.85+ |
| DistilBERT Fine-tuned | ~0.88+ |

> Exact scores are saved to `outputs/config.json` after each run.

---

## 📝 Reproducing the Kaggle Submission

The notebook `Ex_Machina_Solution.ipynb` (generated by `gen_nb.py`) is a cloud-ready version of the pipeline that:
- Dynamically resolves Kaggle input paths
- Outputs predictions as integers (required by the competition)
- Passes Kaggle's notebook execution environment

To regenerate it:
```bash
python gen_nb.py
```

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and adapt it.

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/) for the pre-trained BERT models
- [LightGBM](https://lightgbm.readthedocs.io/) for the blazing-fast gradient boosting framework
- Competition organizers for the fascinating artwork metadata dataset
