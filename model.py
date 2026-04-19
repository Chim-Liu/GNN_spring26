"""
model.py — CNN architectures for plaintext and FHE-compatible inference.

Two model variants are defined:

  BaselineCNN      — standard 2-layer CNN using ReLU; establishes a
                     performance ceiling on plaintext MNIST.

  SquareActCNN     — the same topology but with ReLU replaced by the
                     square activation f(x) = x^2.  This is the first
                     step toward FHE compatibility because squaring
                     requires only a single multiplication, keeping the
                     multiplicative depth minimal.

Both models share the same __init__ signature so they can be swapped
transparently in the training scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Custom activation
# ---------------------------------------------------------------------------

class SquareActivation(nn.Module):
    """f(x) = x^2 — the simplest FHE-compatible nonlinearity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x   # element-wise square


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """
    2-layer CNN with ReLU activations — standard plaintext baseline.

    Architecture
    ------------
    Conv(1→16, 3×3, pad=1) → ReLU → AvgPool(2×2)
    Conv(16→32, 3×3, pad=1) → ReLU → AvgPool(2×2)
    Flatten → FC(1568→128) → ReLU → FC(128→10)

    Notes
    -----
    * Average pooling is used instead of max pooling because max pooling
      is not directly expressible as a low-degree polynomial, making it
      incompatible with FHE.
    * All bias terms are retained; they map to plaintext additions in FHE
      and are therefore free operations noise-budget-wise.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),                              # 28 → 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),                              # 14 → 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SquareActCNN(nn.Module):
    """
    FHE-adapted CNN: ReLU replaced by square activations.

    The architecture mirrors BaselineCNN exactly, but every ReLU is
    substituted with SquareActivation.  This choice has several
    consequences explored in detail in the project report:

      1. Gradient explosion risk — x^2 has unbounded derivative 2x,
         so aggressive learning rates and/or batch normalisation are
         required.
      2. Symmetry — unlike ReLU, x^2 maps both +x and -x to the same
         positive output.  The network must therefore encode sign
         information in the magnitude of its weights rather than through
         dead-neuron sparsity.
      3. Multiplicative depth — each SquareActivation layer consumes
         one level of the FHE noise budget.  With two convolutional
         activations plus one fully-connected activation, the total
         multiplicative depth of this model is 3.

    Notes on BatchNorm
    ------------------
    BatchNorm is included during training (track_running_stats=True) to
    stabilise the square activation.  At FHE inference time, running
    statistics are folded into the preceding linear layer's weights and
    biases (a standard technique), so BatchNorm introduces no additional
    multiplicative depth in the encrypted circuit.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            SquareActivation(),
            nn.AvgPool2d(2),                              # 28 → 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            SquareActivation(),
            nn.AvgPool2d(2),                              # 14 → 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            SquareActivation(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy = torch.randn(4, 1, 28, 28)
    for cls in (BaselineCNN, SquareActCNN):
        m = cls()
        out = m(dummy)
        print(f"{cls.__name__:20s}  output={out.shape}  "
              f"params={count_parameters(m):,}")
