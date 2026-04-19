"""
predict_single.py — Run both models on one MNIST validation sample.

Usage
-----
    # Default: run on the included sample
    python predict_single.py

    # Custom image (28x28 grayscale PNG)
    python predict_single.py --image path/to/image.png

    # Custom .pt tensor (shape [1,28,28] or [28,28])
    python predict_single.py --image path/to/sample.pt

Output
------
    Predicted class and per-class confidence table for each model.
    No edits needed — runs out of the box from the repo root.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

RESULTS_DIR  = Path("results")
SAMPLE_PATH  = Path("samples") / "val_sample.png"  # bundled sample (grayscale PNG)

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from model import BaselineCNN, SquareActCNN

DEVICE = torch.device("cpu")   # single-sample inference never needs GPU


def load_sample(path: Path) -> torch.Tensor:
    """Load a single MNIST sample from .pt tensor or grayscale PNG."""
    if path.suffix == ".pt":
        t = torch.load(path, map_location="cpu")
        if t.dim() == 2:
            t = t.unsqueeze(0)   # (28,28) -> (1,28,28)
        if t.dim() == 3:
            t = t.unsqueeze(0)   # (1,28,28) -> (1,1,28,28)
        return t.float()
    else:
        from PIL import Image
        img = Image.open(path).convert("L").resize((28, 28))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)


def load_model(model_cls, ckpt_path: Path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train_baseline.py / train_square_activation.py first."
        )
    model = model_cls().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor):
    logits = model(image_tensor.to(DEVICE))
    probs  = F.softmax(logits, dim=1).squeeze(0).numpy()
    pred   = probs.argmax()
    return int(pred), probs


def print_result(model_name: str, pred: int, probs: np.ndarray, true_label=None):
    print(f"\n--- {model_name} ---")
    if true_label is not None:
        correct = "CORRECT" if pred == true_label else f"WRONG (true={true_label})"
        print(f"  Prediction  : {pred}  [{correct}]")
    else:
        print(f"  Prediction  : {pred}")
    print(f"  Confidence  : {probs[pred]*100:.2f}%")
    print("  All class confidences:")
    for c, p in enumerate(probs):
        bar = "#" * int(p * 40)
        print(f"    Digit {c}: {p*100:6.2f}%  {bar}")


def main(args):
    sample_path = Path(args.image)
    if not sample_path.exists():
        print(f"Error: sample not found at '{sample_path}'")
        sys.exit(1)

    image = load_sample(sample_path)
    print(f"Sample loaded from: {sample_path}")
    print(f"Tensor shape: {image.shape}  (min={image.min():.3f}, max={image.max():.3f})")

    true_label = None
    label_path = sample_path.with_suffix(".label.txt")
    if label_path.exists():
        true_label = int(label_path.read_text().strip())
        print(f"True label: {true_label}")

    models = [
        ("BaselineCNN (ReLU)", BaselineCNN,  RESULTS_DIR / "baseline_cnn.pt"),
        ("SquareActCNN (x^2)", SquareActCNN, RESULTS_DIR / "square_act_cnn.pt"),
    ]

    for name, cls, ckpt in models:
        try:
            model = load_model(cls, ckpt)
            pred, probs = predict(model, image)
            print_result(name, pred, probs, true_label)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a single MNIST sample."
    )
    parser.add_argument(
        "--image", type=str, default=str(SAMPLE_PATH),
        help="Path to .pt tensor or grayscale PNG (default: samples/val_sample.pt)"
    )
    main(parser.parse_args())
