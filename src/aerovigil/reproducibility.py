"""Utilities for reproducible AEROVIGIL experiments."""

import os
import random

import numpy as np


DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for Python, NumPy and (if installed) PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    set_global_seed()

    print(f"Random seed configured: {DEFAULT_SEED}")
    print("AEROVIGIL reproducibility configuration successful.")
