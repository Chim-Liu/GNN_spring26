"""
evaluate_full.py — Comprehensive evaluation of BaselineCNN vs SquareActCNN.

Computes and reports:
  - Overall accuracy on training and validation sets
  - Per-class accuracy
  - Confusion matrix (saved as PNG)
  - Precision / Recall / F1 (macro-average)
  - Expected Calibration Error (ECE)

Usage
-----
    python evaluate_full.py

Outputs saved to results/:
    confusion_baseline.png
    confusion_square.png
    evaluation_report.txt
"""

from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data  import get_datasets, get_loaders
from model import BaselineCNN, SquareActCNN

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
DIGIT_NAMES = [str(i) for i in range(10)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader):
    """Return (all_labels, all_probs) arrays over the full loader."""
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
    preds = probs.argmax(axis=1)
    return (preds == labels).mean()


def per_class_accuracy(labels, probs):
    preds = probs.argmax(axis=1)
    accs = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        accs.append((preds[mask] == c).mean() if mask.sum() > 0 else 0.0)
    return np.array(accs)


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
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    macro_p = np.mean(p_list)
    macro_r = np.mean(r_list)
    macro_f = np.mean(f_list)
    return macro_p, macro_r, macro_f, np.array(p_list), np.array(r_list), np.array(f_list)


def ece(labels, probs, n_bins=15):
    """Expected Calibration Error with equal-width confidence bins."""
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == labels).astype(float)

    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = correct[mask].mean()
        ece_val += (mask.sum() / n) * abs(avg_acc - avg_conf)
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
    """Reliability diagram: mean confidence vs accuracy per bin."""
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == labels).astype(float)

    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc, bin_conf, bin_sizes = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            bin_acc.append(0); bin_conf.append((lo + hi) / 2); bin_sizes.append(0)
        else:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_sizes.append(mask.sum())

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

def evaluate_one(name, model_cls, ckpt_path, train_loader, val_loader):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = model_cls().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    tr_labels, tr_probs = collect_predictions(model, train_loader)
    vl_labels, vl_probs = collect_predictions(model, val_loader)

    tr_acc = accuracy(tr_labels, tr_probs)
    vl_acc = accuracy(vl_labels, vl_probs)

    pc_acc = per_class_accuracy(vl_labels, vl_probs)
    cm     = confusion_matrix(vl_labels, vl_probs)
    mp, mr, mf, pp, pr, pf = precision_recall_f1(vl_labels, vl_probs)
    ece_v  = ece(vl_labels, vl_probs)

    return {
        "name": name,
        "tr_acc": tr_acc, "vl_acc": vl_acc,
        "per_class_acc": pc_acc,
        "cm": cm,
        "macro_p": mp, "macro_r": mr, "macro_f1": mf,
        "per_class_p": pp, "per_class_r": pr, "per_class_f1": pf,
        "ece": ece_v,
        "vl_labels": vl_labels, "vl_probs": vl_probs,
    }


def print_report(results, file=None):
    def pr(*args, **kwargs):
        print(*args, **kwargs)
        if file:
            print(*args, **kwargs, file=file)

    for r in results:
        pr(f"\n{'='*60}")
        pr(f"Model: {r['name']}")
        pr(f"{'='*60}")
        pr(f"  Training   accuracy : {r['tr_acc']*100:.2f}%  "
           f"({int(r['tr_acc']*45000):,} / 45,000 correct)")
        pr(f"  Validation accuracy : {r['vl_acc']*100:.2f}%  "
           f"({int(r['vl_acc']*15000):,} / 15,000 correct)")
        pr()
        pr("  Per-class accuracy on validation set:")
        for c in range(NUM_CLASSES):
            pr(f"    Digit {c}: {r['per_class_acc'][c]*100:.2f}%")
        pr()
        pr(f"  Macro Precision : {r['macro_p']*100:.2f}%")
        pr(f"  Macro Recall    : {r['macro_r']*100:.2f}%")
        pr(f"  Macro F1        : {r['macro_f1']*100:.2f}%")
        pr()
        pr(f"  ECE (15 bins)   : {r['ece']*100:.3f}%")


def main():
    print(f"Device: {DEVICE}")

    train_loader, val_loader, _ = get_loaders(batch_size=256)

    results = []
    for name, cls, ckpt in [
        ("BaselineCNN (ReLU)",   BaselineCNN,  RESULTS_DIR / "baseline_cnn.pt"),
        ("SquareActCNN (x^2)",   SquareActCNN, RESULTS_DIR / "square_act_cnn.pt"),
    ]:
        print(f"\nEvaluating {name} ...")
        r = evaluate_one(name, cls, ckpt, train_loader, val_loader)
        results.append(r)

        tag = "baseline" if "Baseline" in name else "square"
        plot_confusion_matrix(
            r["cm"], f"Confusion Matrix — {name}",
            RESULTS_DIR / f"confusion_{tag}.png"
        )
        plot_reliability_diagram(
            r["vl_labels"], r["vl_probs"],
            f"Reliability Diagram — {name}",
            RESULTS_DIR / f"reliability_{tag}.png"
        )

    with open(RESULTS_DIR / "evaluation_report.txt", "w") as f:
        print_report(results, file=f)

    print_report(results)

    # Side-by-side comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    b, s = results[0], results[1]
    print(f"{'Metric':<30} {'Baseline':>12} {'SquareAct':>12} {'Delta':>10}")
    print("-"*65)
    metrics = [
        ("Train Accuracy (%)",    b['tr_acc']*100,   s['tr_acc']*100),
        ("Val Accuracy (%)",      b['vl_acc']*100,   s['vl_acc']*100),
        ("Macro Precision (%)",   b['macro_p']*100,  s['macro_p']*100),
        ("Macro Recall (%)",      b['macro_r']*100,  s['macro_r']*100),
        ("Macro F1 (%)",          b['macro_f1']*100, s['macro_f1']*100),
        ("ECE (%)",               b['ece']*100,      s['ece']*100),
    ]
    for label, bv, sv in metrics:
        delta = sv - bv
        sign = "+" if delta >= 0 else ""
        print(f"{label:<30} {bv:>12.3f} {sv:>12.3f} {sign}{delta:>9.3f}")

    print("\nPlots saved to results/")
    print("Report saved to results/evaluation_report.txt")


if __name__ == "__main__":
    main()
