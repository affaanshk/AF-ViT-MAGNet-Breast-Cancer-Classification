# src/utils.py
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Ensures multi-run reproducibility across seeds[cite: 3]."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)