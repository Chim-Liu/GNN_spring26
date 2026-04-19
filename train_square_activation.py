"""
train_square_activation.py — Train the FHE-adapted CNN (square activations).

Key differences from train_baseline.py
---------------------------------------
  * Uses SquareActCNN (f(x) = x^2 instead of ReLU).
  * Lower default learning rate (1e-4) to mitigate gradient explosion.
  * Gradient clipping is applied as an additional safeguard.
  * Weight initialisation uses scaled Xavier uniform to keep initial
    activations in a stable range before BatchNorm has converged.

Saved artefacts
---------------
  results/square_act_cnn.pt   — best model weights
  results/square_act_log.csv  — per-epoch metrics for analysis

Usage
-----
    python train_square_activation.py [--epochs N] [--lr LR]
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data  import get_loaders
from model import SquareActCNN

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module):
    """
    Conservative Xavier initialisation.

    Square activations with large initial weights produce very large
    gradients in the first few steps.  Scaling down the fan-out by a
    factor of 2 empirically reduces early divergence.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight, gain=0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    clip_norm: float = 1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        # Gradient clipping — essential for square activations
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_loaders(
        batch_size=args.batch_size, seed=args.seed
    )

    model = SquareActCNN().to(device)
    model.apply(_init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_acc = 0.0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            clip_norm=args.clip_norm
        )
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )
        log_rows.append([epoch, tr_loss, tr_acc, vl_loss, vl_acc])

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(),
                       RESULTS_DIR / "square_act_cnn.pt")
            print(f"  → Saved best model (val_acc={best_val_acc:.4f})")

    # Save CSV log
    log_path = RESULTS_DIR / "square_act_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc"])
        writer.writerows(log_rows)
    print(f"\nLog saved to {log_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--clip-norm",  type=float, default=1.0)
    parser.add_argument("--seed",       type=int,   default=42)
    main(parser.parse_args())
