"""
Step 4: Compare base vs fine-tuned predictions on new queries (demo for the report).

Usage:
  python predict.py "Mahindi yangu ina viwavi jeshi, nifanye nini?"
  python predict.py            # runs the built-in demo queries
"""
import pickle
import sys

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from config import MODEL_DIR, MODEL_NAME, RESULTS_DIR
from train import embed

DEMO = [
    "Ugali yetu itakosekana, mahindi yameliwa na viwavi jeshi.",
    "Kiwele cha ng'ombe kimevimba na maziwa yana damu.",
    "Mvua za vuli zimechelewa, nipande lini?",
    "Madalali wanataka kununua gunia la maharagwe kwa bei ya kutupa.",
    "Nyanya zangu zimeshika blight baada ya mvua.",
    "Nitumie samadi au DAP kwa sukuma wiki?",
]


def main():
    queries = sys.argv[1:] or DEMO
    tok_b = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModel.from_pretrained(MODEL_NAME)
    head = pickle.load(open(RESULTS_DIR / "base_head.pkl", "rb"))
    base_preds = head.predict(embed(queries, tok_b, base))

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ft = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()
    with torch.no_grad():
        enc = tok(queries, padding=True, truncation=True, return_tensors="pt")
        probs = ft(**enc).logits.softmax(-1)
    for q, bp, p in zip(queries, base_preds, probs):
        i = int(p.argmax())
        print(f"\nQuery: {q}\n  Base mBERT      -> {bp}\n"
              f"  Fine-tuned mBERT -> {ft.config.id2label[i]} ({p[i]:.2f})")


if __name__ == "__main__":
    main()
