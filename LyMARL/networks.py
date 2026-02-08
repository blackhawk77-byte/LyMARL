# networks.py
import torch.nn as nn

class UEActorNetwork(nn.Module):
    """Actor network (shared across UEs)"""
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
    """Actor network (shared across BSs)"""
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
    """Centralized critic for UE objective (predicts normalized scalar value)"""
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
        return self.net(global_obs).squeeze(-1)  # [batch]


class CentralizedCriticBS(nn.Module):
    """
    Centralized critic for BS objective
    - outputs per-BS value: [batch, B]
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
        return self.net(global_obs)  # [batch, B]