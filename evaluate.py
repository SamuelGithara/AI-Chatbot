"""
Step 3: Evaluate base vs fine-tuned mBERT (and the TF-IDF reference).

Outputs (results/):
  metrics_summary.csv        accuracy, precision, recall, macro-F1 per model
  per_class_report.csv       precision/recall/F1 for each label and model
  per_language_accuracy.csv  accuracy by language (en / sw / sheng / mix)
  mcnemar_test.txt           significance of base vs fine-tuned difference
  local_adaptation_examples.csv  cases where base fails and fine-tuned succeeds
  confusion_*.png, model_comparison.png, training_curve.png

Usage:  python evaluate.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from statsmodels.stats.contingency_tables import mcnemar

from config import LABELS, RESULTS_DIR

MODELS = {"base_zeroshot": "Base mBERT (zero-shot)",
          "tfidf_lr": "TF-IDF + LR (reference)",
          "base_mbert": "Base mBERT (frozen + LR head)",
          "finetuned_mbert": "Fine-tuned mBERT"}


def main():
    df = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    y = df["true"]

    # Overall metrics
    rows, per_class = [], []
    for col, name in MODELS.items():
        p, r, f, _ = precision_recall_fscore_support(y, df[col], average="macro", zero_division=0)
        rows.append({"model": name, "accuracy": accuracy_score(y, df[col]),
                     "macro_precision": p, "macro_recall": r, "macro_f1": f,
                     "n_correct": int((df[col] == y).sum()), "n_test": len(y)})
        rep = classification_report(y, df[col], labels=LABELS, output_dict=True, zero_division=0)
        for lab in LABELS:
            per_class.append({"model": name, "label": lab, **{k: rep[lab][k] for k in
                              ["precision", "recall", "f1-score", "support"]}})
    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(per_class).round(4).to_csv(RESULTS_DIR / "per_class_report.csv", index=False)
    print(summary.to_string(index=False))

    # Accuracy by language and by local-term presence
    lang = df.groupby("language").apply(
        lambda g: pd.Series({MODELS[c]: (g[c] == g["true"]).mean() for c in MODELS} | {"n": len(g)}),
        include_groups=False).round(3)
    lang.to_csv(RESULTS_DIR / "per_language_accuracy.csv")
    print("\nAccuracy by language:\n", lang)

    # McNemar tests: each base model vs fine-tuned, on the same test items
    f_ok = df["finetuned_mbert"] == y
    b_ok = df["base_mbert"] == y
    txt = ""
    for col in ["base_zeroshot", "base_mbert"]:
        ok = df[col] == y
        table = [[int((ok & f_ok).sum()), int((ok & ~f_ok).sum())],
                 [int((~ok & f_ok).sum()), int((~ok & ~f_ok).sum())]]
        res = mcnemar(table, exact=True)
        txt += (f"=== {MODELS[col]} vs Fine-tuned mBERT ===\n"
                f"Contingency table (rows: base correct/wrong, cols: fine-tuned correct/wrong)\n"
                f"{np.array(table)}\n"
                f"b = base right, fine-tuned wrong = {table[0][1]}\n"
                f"c = base wrong, fine-tuned right = {table[1][0]}\n"
                f"Exact McNemar statistic = {res.statistic}, p-value = {res.pvalue:.4g}\n\n")
    (RESULTS_DIR / "mcnemar_test.txt").write_text(txt)
    print("\n" + txt)

    # Local-adaptation examples
    ex = df[~b_ok & f_ok][["input", "language", "true", "base_mbert", "finetuned_mbert"]]
    ex.to_csv(RESULTS_DIR / "local_adaptation_examples.csv", index=False)
    print(f"{len(ex)} examples where base fails and fine-tuned succeeds")

    # Confusion matrices
    for col, name in MODELS.items():
        cm = confusion_matrix(y, df[col], labels=LABELS)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right")
        ax.set_yticks(range(len(LABELS)), LABELS)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix: {name}")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"confusion_{col}.png", dpi=150); plt.close(fig)

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(summary)); w = 0.35
    ax.bar(x - w / 2, summary["accuracy"], w, label="Accuracy", color="#2e7d32")
    ax.bar(x + w / 2, summary["macro_f1"], w, label="Macro-F1", color="#a5d6a7")
    for i, (a, f) in enumerate(zip(summary["accuracy"], summary["macro_f1"])):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [m.replace(" (", "\n(") for m in summary["model"]], fontsize=8); ax.set_ylim(0, 1.1); ax.legend()
    ax.set_title("Test-set performance"); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150); plt.close(fig)

    # Training curve
    log = json.loads((RESULTS_DIR / "training_log.json").read_text())
    ev = [h for h in log if "eval_macro_f1" in h]
    tr = [h for h in log if "loss" in h and "eval_loss" not in h]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot([h["epoch"] for h in tr], [h["loss"] for h in tr], label="Train loss", color="#795548")
    ax1.plot([h["epoch"] for h in ev], [h["eval_loss"] for h in ev], "o-", label="Val loss", color="#ff9800")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot([h["epoch"] for h in ev], [h["eval_macro_f1"] for h in ev], "s--", label="Val macro-F1", color="#2e7d32")
    ax2.set_ylabel("Macro-F1"); ax2.set_ylim(0, 1.05)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="center right", bbox_to_anchor=(0.98, 0.55)); ax1.set_title("Fine-tuning progress"); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
