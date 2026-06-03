import random
import numpy as np
import torch
from typing import Tuple


def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    """
    Simple moving average for a 1D array.
    Returns an array with the same length as the input.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x

    out = np.zeros_like(x, dtype=np.float32)
    csum = 0.0
    for i in range(len(x)):
        csum += float(x[i])
        if i >= window:
            csum -= float(x[i - window])
            out[i] = csum / float(window)
        else:
            out[i] = csum / float(i + 1)
    return out


def block_avg_1d(x: np.ndarray, block: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: [T]
    returns:
      xs: [num_blocks]   -> end step index of each block
      ys: [num_blocks]   -> block mean
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    T = x.shape[0]

    if T == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)

    xs, ys = [], []
    for start in range(0, T, block):
        chunk = x[start:start + block]
        if chunk.size == 0:
            continue
        xs.append(start + chunk.size)
        ys.append(float(chunk.mean()))

    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float32)