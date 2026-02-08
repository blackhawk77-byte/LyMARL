# norms.py
import torch
import torch.nn as nn
from typing import Optional


class ValueNorm(nn.Module):
    """Running mean/std for scalar targets."""
    def __init__(self, eps: float = 1e-5, device: Optional[torch.device] = None):
        super().__init__()
        self.eps = eps
        self.device = device if device is not None else torch.device("cpu")
        self.register_buffer("count", torch.tensor(0.0, device=self.device))
        self.register_buffer("mean", torch.tensor(0.0, device=self.device))
        self.register_buffer("m2", torch.tensor(1.0, device=self.device))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().view(-1).to(self.device)
        if x.numel() == 0:
            return
        for v in x:
            self.count += 1.0
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.m2 += delta * delta2

    def variance(self):
        denom = torch.clamp(self.count - 1.0, min=1.0)
        return self.m2 / denom

    def std(self):
        return torch.sqrt(self.variance() + self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std() + self.mean


# ============================================================
# Value Normalization (vector, per-dimension) (BS critic)
# ============================================================
class ValueNormVec(nn.Module):
    """Running mean/std for vector targets: shape [..., D]. Keeps per-dim stats."""
    def __init__(self, dim: int, eps: float = 1e-5, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.device = device if device is not None else torch.device("cpu")

        self.register_buffer("count", torch.zeros(self.dim, device=self.device))
        self.register_buffer("mean", torch.zeros(self.dim, device=self.device))
        self.register_buffer("m2", torch.ones(self.dim, device=self.device))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """
        x: [..., D]
        update per dim using streaming update over samples
        """
        x = x.detach().to(self.device)
        x = x.view(-1, self.dim)
        if x.numel() == 0:
            return

        for i in range(x.shape[0]):
            v = x[i]  # [D]
            self.count += 1.0
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.m2 += delta * delta2

    def variance(self):
        denom = torch.clamp(self.count - 1.0, min=1.0)
        return self.m2 / denom

    def std(self):
        return torch.sqrt(self.variance() + self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std() + self.mean
