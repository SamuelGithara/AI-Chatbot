"""
Step 2: Train the models.

  (a) BASE MODEL  - pretrained mBERT used as-is (frozen, no fine-tuning). Sentence
                    embeddings (mean-pooled) are fed to a logistic-regression head
                    trained on the same training set. This measures what mBERT
                    "already knows" without adapting its weights.
  (a0) BASE ZERO-SHOT - pretrained mBERT with no task training at all: each query is
                    matched to the most similar label description (cosine similarity).
  (b) FINE-TUNED  - all mBERT weights updated on the farmer dataset.
  (c) REFERENCE   - TF-IDF + logistic regression (classical, non-transformer).

Usage:  python train.py
"""
import json
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from config import (LABEL_DESCRIPTIONS, BATCH_SIZE, DATA_DIR, EPOCHS, ID2LABEL, LABEL2ID, LABELS, LR, MAX_LEN,
                    MODEL_DIR, MODEL_NAME, RESULTS_DIR, SEED, WEIGHT_DECAY)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class QueryDataset(Dataset):
    def __init__(self, df, tok):
        self.enc = tok(list(df["input"]), truncation=True, padding="max_length", max_length=MAX_LEN)
        self.labels = [LABEL2ID[l] for l in df["output"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


@torch.no_grad()
def embed(texts, tok, model, bs=32):
    """Mean-pooled sentence embeddings from frozen pretrained mBERT."""
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).numpy())
    return np.vstack(out)


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": (preds == p.label_ids).mean(),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}


def main():
    set_seed()
    RESULTS_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    preds = {"id": test["id"], "input": test["input"], "language": test["language"],
             "has_local_term": test["has_local_term"], "true": test["output"]}

    # (c) TF-IDF reference
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    # All models learn from the SAME 210 training rows (val is only for early stopping)
    Xtr = tfidf.fit_transform(train["input"])
    ytr = train["output"]
    lr_tfidf = LogisticRegression(max_iter=2000, C=10).fit(Xtr, ytr)
    preds["tfidf_lr"] = lr_tfidf.predict(tfidf.transform(test["input"]))

    # (a) Base mBERT (frozen) + logistic-regression head
    base = AutoModel.from_pretrained(MODEL_NAME)
    Etr = embed(list(train["input"]), tok, base)
    Ete = embed(list(test["input"]), tok, base)
    lr_base = LogisticRegression(max_iter=5000, C=1.0).fit(Etr, ytr)
    preds["base_mbert"] = lr_base.predict(Ete)
    # (a0) Base mBERT zero-shot: cosine similarity to label descriptions (no task training)
    Elab = embed([LABEL_DESCRIPTIONS[l] for l in LABELS], tok, base)
    sims = (Ete / np.linalg.norm(Ete, axis=1, keepdims=True)) @ \
           (Elab / np.linalg.norm(Elab, axis=1, keepdims=True)).T
    preds["base_zeroshot"] = [LABELS[i] for i in sims.argmax(1)]
    with open(RESULTS_DIR / "base_head.pkl", "wb") as f:
        pickle.dump(lr_base, f)
    del base

    # (b) Fine-tuned mBERT
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"), num_train_epochs=EPOCHS, learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=32,
        weight_decay=WEIGHT_DECAY, warmup_steps=50, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        save_total_limit=1, logging_steps=5, seed=SEED, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=QueryDataset(train, tok),
                      eval_dataset=QueryDataset(val, tok), compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(str(MODEL_DIR)); tok.save_pretrained(str(MODEL_DIR))

    out = trainer.predict(QueryDataset(test, tok))
    preds["finetuned_mbert"] = [ID2LABEL[i] for i in out.predictions.argmax(-1)]

    history = [h for h in trainer.state.log_history]
    with open(RESULTS_DIR / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    pd.DataFrame(preds).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    print("Saved predictions to results/test_predictions.csv")


if __name__ == "__main__":
    main()
