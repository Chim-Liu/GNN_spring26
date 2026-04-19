"""
data.py — MNIST data loading and splitting utilities.

Provides a consistent interface for loading the raw IDX-format MNIST files
from the local data/mnist/ directory and splitting them into train / val / test
subsets according to the 60/20/20 strategy described in the project README.
"""

import struct
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data" / "mnist"

TRAIN_IMAGES = DATA_DIR / "train-images.idx3-ubyte"
TRAIN_LABELS = DATA_DIR / "train-labels.idx1-ubyte"
TEST_IMAGES  = DATA_DIR / "t10k-images.idx3-ubyte"
TEST_LABELS  = DATA_DIR / "t10k-labels.idx1-ubyte"

# ---------------------------------------------------------------------------
# Low-level IDX readers
# ---------------------------------------------------------------------------

def _read_images(path: Path) -> np.ndarray:
    """Return float32 array of shape (N, 1, 28, 28) in [0, 1]."""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    """Return int64 array of shape (N,)."""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MNISTDataset(Dataset):
    """Thin wrapper around numpy arrays so we can use DataLoader."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 transform=None):
        self.images = torch.from_numpy(images)   # (N, 1, 28, 28) float32
        self.labels = torch.from_numpy(labels)   # (N,) int64
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_datasets(val_fraction: float = 0.25, seed: int = 42):
    """
    Load and split MNIST into train / val / test datasets.

    The official MNIST 60k training split is further partitioned into
    train (75 %) and val (25 %), giving the 60/20/20 split described in
    the project proposal.

    Returns
    -------
    train_ds, val_ds, test_ds : MNISTDataset
    """
    train_images = _read_images(TRAIN_IMAGES)
    train_labels = _read_labels(TRAIN_LABELS)
    test_images  = _read_images(TEST_IMAGES)
    test_labels  = _read_labels(TEST_LABELS)

    full_train = MNISTDataset(train_images, train_labels)
    test_ds    = MNISTDataset(test_images,  test_labels)

    n_val   = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                    generator=generator)
    return train_ds, val_ds, test_ds


def get_loaders(batch_size: int = 128, val_fraction: float = 0.25,
                num_workers: int = 0, seed: int = 42):
    """
    Convenience wrapper that returns DataLoaders for train / val / test.

    Parameters
    ----------
    batch_size    : samples per mini-batch
    val_fraction  : fraction of the 60k training split held out for validation
    num_workers   : DataLoader worker processes (0 = main process only)
    seed          : random seed for the train/val split

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    train_ds, val_ds, test_ds = get_datasets(val_fraction=val_fraction,
                                              seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")
    x, y = next(iter(train_loader))
    print(f"Batch shape   : {x.shape}, dtype={x.dtype}")
    print(f"Label range   : {y.min().item()} – {y.max().item()}")
