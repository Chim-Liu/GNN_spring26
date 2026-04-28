"""
evaluate_test.py — Evaluate BaselineCNN and SquareActCNN on the official
10,000-image MNIST test partition.

Saves to results/:
    confusion_baseline_test.png
    confusion_square_test.png
    reliability_baseline_test.png
    reliability_square_test.png
    test_evaluation_report.txt

Usage
-----
    python evaluate_test.py
"""

from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data  import get_datasets
from model import BaselineCNN, SquareActCNN

RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 10
DIGIT_NAMES  = [str(i) for i in range(10)]
BATCH_SIZE   = 512


# ---------------------------------------------------------------------------
# Metrics (identical to evaluate_full.py so Part 4 / Part 5 are comparable)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs  = F.softmax(logits, dim=1)
        all_labels.append(y.numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def accuracy(labels, probs):
    return (probs.argmax(axis=1) == labels).mean()


def per_class_accuracy(labels, probs):
    preds = probs.argmax(axis=1)
    return np.array([
        (preds[labels == c] == c).mean() if (labels == c).sum() > 0 else 0.0
        for c in range(NUM_CLASSES)
    ])


def per_class_counts(labels):
    return np.array([(labels == c).sum() for c in range(NUM_CLASSES)])


def confusion_matrix(labels, probs):
    preds = probs.argmax(axis=1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    return cm


def precision_recall_f1(labels, probs):
    preds = probs.argmax(axis=1)
    p_list, r_list, f_list = [], [], []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f  = 2 * p * r / (p + r + 1e-9)
        p_list.append(p); r_list.append(r); f_list.append(f)
    return np.mean(p_list), np.mean(r_list), np.mean(f_list)


def ece(labels, probs, n_bins=15):
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == labels).astype(float)
    bins        = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val     = 0.0
    n           = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        ece_val += (mask.sum() / n) * abs(correct[mask].mean() - confidences[mask].mean())
    return ece_val


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=DIGIT_NAMES, yticklabels=DIGIT_NAMES,
           xlabel="Predicted label", ylabel="True label", title=title)
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_reliability_diagram(labels, probs, title, save_path, n_bins=15):
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == labels).astype(float)
    bins        = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc, bin_conf = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            bin_acc.append(0); bin_conf.append((lo + hi) / 2)
        else:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidences[mask].mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bins[:-1], bin_acc, width=1/n_bins, align="edge",
           alpha=0.6, label="Accuracy")
    ax.bar(bins[:-1], bins[:-1], width=1/n_bins, align="edge",
           alpha=0.3, color="red", label="Gap (over-confidence)")
    ax.set(xlabel="Confidence", ylabel="Accuracy", title=title,
           xlim=[0, 1], ylim=[0, 1])
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    _, _, test_ds = get_datasets()
    test_loader   = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Print per-class counts in test set
    test_labels_all = np.array([test_ds[i][1].item() for i in range(len(test_ds))])
    counts = per_class_counts(test_labels_all)
    print("\nTest-set class distribution:")
    for c, n in enumerate(counts):
        print(f"  Digit {c}: {n} samples")
    print(f"  Total : {counts.sum()}")

    configs = [
        ("BaselineCNN (ReLU)", BaselineCNN,  RESULTS_DIR / "baseline_cnn.pt",  "baseline"),
        ("SquareActCNN (x^2)", SquareActCNN, RESULTS_DIR / "square_act_cnn.pt", "square"),
    ]

    report_lines = []

    for name, cls, ckpt, tag in configs:
        print(f"\nEvaluating {name} on test set ...")
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

        model = cls().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

        labels, probs = collect_predictions(model, test_loader)

        acc     = accuracy(labels, probs)
        pc_acc  = per_class_accuracy(labels, probs)
        cm      = confusion_matrix(labels, probs)
        mp, mr, mf = precision_recall_f1(labels, probs)
        ece_val = ece(labels, probs)
        n_correct = int(round(acc * len(labels)))
        n_errors  = len(labels) - n_correct

        plot_confusion_matrix(
            cm, f"Confusion Matrix — {name} (Test Set)",
            RESULTS_DIR / f"confusion_{tag}_test.png"
        )
        plot_reliability_diagram(
            labels, probs,
            f"Reliability Diagram — {name} (Test Set)",
            RESULTS_DIR / f"reliability_{tag}_test.png"
        )

        block = (
            f"\n{'='*60}\n"
            f"Model: {name}\n"
            f"{'='*60}\n"
            f"  Test accuracy : {acc*100:.2f}%  ({n_correct:,} / {len(labels):,} correct, {n_errors} errors)\n\n"
            f"  Per-class accuracy on test set:\n"
        )
        for c in range(NUM_CLASSES):
            block += f"    Digit {c}: {pc_acc[c]*100:.2f}%  ({counts[c]} samples)\n"
        block += (
            f"\n"
            f"  Macro Precision : {mp*100:.2f}%\n"
            f"  Macro Recall    : {mr*100:.2f}%\n"
            f"  Macro F1        : {mf*100:.2f}%\n\n"
            f"  ECE (15 bins)   : {ece_val*100:.3f}%\n"
        )
        print(block)
        report_lines.append(block)

    report_path = RESULTS_DIR / "test_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MNIST TEST-SET EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")
        for block in report_lines:
            f.write(block)

    print(f"\nReport saved to {report_path}")
    print("Plots saved to results/confusion_*_test.png and reliability_*_test.png")


if __name__ == "__main__":
    main()
