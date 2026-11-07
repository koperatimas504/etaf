from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def derive(seed: int, *tags: int) -> int:
    acc = seed & 0xFFFFFFFF
    for t in tags:
        acc = (acc * 1000003 + (t & 0xFFFFFFFF)) & 0xFFFFFFFF
    return acc
