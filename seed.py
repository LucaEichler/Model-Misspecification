import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    # Python built-in random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For single-GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: control Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
