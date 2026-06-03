import torch
import torch.nn as nn
from typing import Optional


class ValueNorm(nn.Module):
    """
    Running mean/std for scalar targets.
    Used for UE critic target normalization.
    """
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


class ValueNormVec(nn.Module):
    """
    Running mean/std for vector targets: shape [..., D].
    Used for BS critic target normalization.
    """
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
        x = x.detach().to(self.device).view(-1, self.dim)
        if x.numel() == 0:
            return

        for i in range(x.shape[0]):
            v = x[i]
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


class UEActorNetwork(nn.Module):
    """
    Shared actor network for all UEs.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class BSActorNetwork(nn.Module):
    """
    Shared actor network for all BSs.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class CentralizedCriticUE(nn.Module):
    """
    Centralized critic for UE objective.
    Outputs a scalar value.
    """
    def __init__(self, global_obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs):
        return self.net(global_obs).squeeze(-1)


class CentralizedCriticBS(nn.Module):
    """
    Centralized critic for BS objective.
    Outputs a vector of size [B].
    """
    def __init__(self, global_obs_dim: int, n_bs: int, hidden_dim: int = 256):
        super().__init__()
        self.n_bs = int(n_bs)
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.n_bs),
        )

    def forward(self, global_obs):
        return self.net(global_obs)