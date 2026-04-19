"""
evaluate.py — Evaluate saved model checkpoints on the held-out test set.

Computes accuracy and per-class breakdown.  Designed to be run after
training scripts have produced their .pt artefacts.

Usage
-----
    # Evaluate baseline
    python evaluate.py --model baseline

    # Evaluate square-activation model
    python evaluate.py --model square

    # Evaluate a custom checkpoint
    python evaluate.py --checkpoint results/my_model.pt --arch square
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data  import get_datasets
from model import BaselineCNN, SquareActCNN

RESULTS_DIR = Path("results")

CHECKPOINT_MAP = {
    "baseline" : (RESULTS_DIR / "baseline_cnn.pt",   BaselineCNN),
    "square"   : (RESULTS_DIR / "square_act_cnn.pt", SquareActCNN),
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, loader, device, num_classes=10):
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds  = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += len(y)
        for c in range(num_classes):
            mask = (y == c)
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c]   += mask.sum().item()

    overall_acc = correct / total
    per_class   = [class_correct[c] / max(class_total[c], 1)
                   for c in range(num_classes)]
    return overall_acc, per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine checkpoint path and architecture
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        arch_cls  = BaselineCNN if args.arch == "baseline" else SquareActCNN
    else:
        ckpt_path, arch_cls = CHECKPOINT_MAP[args.model]

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run the corresponding training script first."
        )

    model = arch_cls().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded: {ckpt_path}")

    _, _, test_ds = get_datasets(seed=42)
    test_loader   = DataLoader(test_ds, batch_size=256, shuffle=False)

    overall_acc, per_class = evaluate_model(model, test_loader, device)

    print(f"\nTest accuracy (overall): {overall_acc:.4f} "
          f"({overall_acc * 100:.2f}%)")
    print("\nPer-class accuracy:")
    for c, acc in enumerate(per_class):
        print(f"  Digit {c}: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", choices=["baseline", "square"],
                       default="baseline",
                       help="Named preset (loads from results/)")
    group.add_argument("--checkpoint", type=str,
                       help="Path to a custom .pt checkpoint")
    parser.add_argument("--arch", choices=["baseline", "square"],
                        default="square",
                        help="Architecture to use with --checkpoint")
    main(parser.parse_args())
