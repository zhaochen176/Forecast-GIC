"""Evaluate and plot one completed CNN-BiLSTM run.

This is the single post-training entry point. It writes final metrics, ROC/PR
curves, training history, probability time series, and threshold trade-offs
when the corresponding CSV files exist in ``--run-dir``.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

THRESHOLDS = (3, 5, 10, 20)

def metrics(frame: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        y = frame[f"target_exceeds_{threshold}A_30min"].astype(int).to_numpy()
        score = frame[f"probability_exceeds_{threshold}A_30min"].to_numpy()
        pred = (score >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({"threshold_A": threshold, "decision_threshold": cutoff,
            "roc_auc": roc_auc_score(y, score) if np.unique(y).size > 1 else np.nan,
            "pr_auc": average_precision_score(y, score), "POD": tp/(tp+fn) if tp+fn else np.nan,
            "POFD": fp/(fp+tn) if fp+tn else np.nan, "FAR": fp/(tp+fp) if tp+fp else np.nan,
            "CSI": tp/(tp+fp+fn) if tp+fp+fn else np.nan, "F1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
            "Bias": (tp+fp)/(tp+fn) if tp+fn else np.nan, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    return pd.DataFrame(rows)

def plot_curves(frame: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for threshold in THRESHOLDS:
        y = frame[f"target_exceeds_{threshold}A_30min"]; score = frame[f"probability_exceeds_{threshold}A_30min"]
        if y.nunique() < 2: continue
        fpr, tpr, _ = roc_curve(y, score); precision, recall, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, label=f"{threshold} A"); axes[1].plot(recall, precision, label=f"{threshold} A")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate"); axes[1].set(xlabel="Recall", ylabel="Precision")
    for axis in axes: axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(out / "roc_pr_curves.png", dpi=180); plt.close(fig)

def plot_history(run: Path, out: Path) -> None:
    path = run / "training_history.csv"
    if not path.exists(): return
    data = pd.read_csv(path); numeric = [c for c in data.columns if c.lower() in {"epoch", "train_loss", "validation_loss", "val_loss", "learning_rate"}]
    if len(numeric) < 2: return
    x = data["epoch"] if "epoch" in data else np.arange(len(data)); fig, ax = plt.subplots(figsize=(7, 4))
    for col in numeric:
        if col != "epoch": ax.plot(x, data[col], label=col)
    ax.set(xlabel="Epoch", ylabel="Value"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "training_history.png", dpi=180); plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed 04 output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(); run = args.run_dir.resolve(); pred_path = run / "test_probabilities.csv"
    if not pred_path.exists(): raise FileNotFoundError(f"Missing {pred_path}; run 04_train_cnn_bilstm.py first.")
    frame = pd.read_csv(pred_path); out = run / "final_evaluation"; out.mkdir(exist_ok=True)
    metrics(frame, args.threshold).to_csv(out / "metrics.csv", index=False); plot_curves(frame, out); plot_history(run, out)
    probability_columns = [c for c in frame.columns if c.startswith("probability_exceeds_")]
    id_columns = [c for c in ("time", "event_id", "split") if c in frame.columns]
    frame[id_columns + probability_columns].to_csv(out / "probability_series.csv", index=False)
    print(f"Evaluation written to {out}")

if __name__ == "__main__": main()
