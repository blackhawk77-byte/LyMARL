#qplex.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Agent network
# -------------------------
class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.grucell = nn.GRUCell(hidden_dim, hidden_dim)  
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, last_action, his_in):
        # obs: (B, obs_dim)
        # last_action: (B, action_dim) one-hot
        x = torch.cat([obs, last_action], dim=-1)
        x = F.relu(self.fc1(x))
        if his_in is None:
            his_in = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            if his_in.dim() == 1:
                his_in = his_in.unsqueeze(0)  # (B, H)
            assert his_in.shape == (x.size(0), self.hidden_dim), \
                f"GRU hidden should be (B,{self.hidden_dim}), got {his_in.shape}"
        his_out = self.grucell(x, his_in)   # (B, H)
        q_all = self.q_out(his_out)             # (B, A)
        return q_all, his_out


# -------------------------
# Helpers
# -------------------------
def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x

def dueling_from_q(q_all: torch.Tensor, actions: torch.Tensor):
    """
    q_all:   (B, N, A)
    actions: (B, N)  chosen actions indices

    returns:
      v_local:     (B, N)
      a_local_sel: (B, N)  advantage for chosen action only
      q_sel:       (B, N)  (optional, sometimes useful)
    """
    v_local = q_all.max(dim=-1).values                                
    q_sel = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1) 
    a_local_sel = q_sel - v_local                                          
    return v_local, a_local_sel, q_sel

# -------------------------
# Transformation on (V, A) using Eq.(7):
#   V_i(τ) = w_i(τ) V_i(τ_i) + b_i(τ)
#   A_i(τ,a_i) = w_i(τ) A_i(τ_i,a_i)
# -------------------------
class Transformation(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.eps = eps
        
        self.hyper_w = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents) # (B, N)
        )

        self.hyper_b = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents) # (B, N)
        )

    def forward(self, state, v_local: torch.Tensor, a_local: torch.Tensor):
        """
        state:      (B,S)
        v_local:    (B,N)
        a_local_sel:(B,N)

        returns:
          v_tau: (B,N)
          a_tau: (B,N)
          w,b:   (B,N)
        """

        state = _ensure_2d(state)  
        v_local = _ensure_2d(v_local) 
        a_local_sel = _ensure_2d(a_local)
    
        B, N = v_local.shape
        assert N == self.n_agents, f"Expected v_local to have {self.n_agents} agents, got {N}"
        assert a_local_sel.shape == (B, N), f"Expected a_local_sel to have shape (B,{self.n_agents}), got {a_local_sel.shape}"

        # w > 0
        w = F.softplus(self.hyper_w(state)) + self.eps  # (B, N)
        b = self.hyper_b(state)  # (B, N)

        v_tau = w * v_local + b  # (B, N)
        a_tau = w * a_local_sel  # (B, N)
        return v_tau, a_tau, w, b

# -------------------------
# Multi-head attention-like λ module (Eq.(10))
#   λ_i(τ,a) = Σ_k  λ_{i,k}(τ,a) * φ_{i,k}(τ) * υ_k(τ)
# with:
#   λ_{i,k}, φ_{i,k} : sigmoid gates  (0,1)
#   υ_k(τ) : positive key (softplus >0)
# This matches the paper text; Figure 1a shows λ>0 from an MLP;
# here that MLP is "multi-head" internally.
# -------------------------
class MultiHeadLambda(nn.Module):
    def __init__(self, n_agents, n_actions, state_dim, hidden_dim=64, n_heads=4, eps=1e-6):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.eps = eps

        joint_action_dim = n_agents * n_actions

        # λ_{i,k}(τ,a): depends on (state, joint_action)
        self.lambda_gate = nn.Sequential(
            nn.Linear(state_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * n_heads),
        )

        # φ_{i,k}(τ): depends only on state
        self.phi_gate = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * n_heads),
        )

        # υ_k(τ) > 0: depends only on state
        self.v_key = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_heads),
        )

    def forward(self, state: torch.Tensor, actions:torch.Tensor):
        """
        state:   (B,S)
        actions: (B,N)
        returns:
          lambda_i: (B,N) strictly > 0
        """

        state = _ensure_2d(state)  
        actions = _ensure_2d(actions)  

        B, N = actions.shape
        assert N == self.n_agents, f"Expected action to have {self.n_agents} agents, got {N}"

        #joint action one-hot encoding(B, N*A)
        act_1hot = F.one_hot(actions.long(), num_classes=self.n_actions).float()  # (B, N*A)
        act_1hot = act_1hot.view(B, N * self.n_actions)  # (B, N*A)

        x = torch.cat([state, act_1hot], dim=-1)  # (B, S + N*A)
        
        lam_ik = torch.sigmoid(self.lambda_gate(x)).view(B, N, self.n_heads)  # (B, N, n_heads)
        phi_ik = torch.sigmoid(self.phi_gate(state)).view(B, N, self.n_heads)  # (B, N, n_heads)
        v_k = F.softplus(self.v_key(state)) + self.eps  # (B, n_heads)
        v_k = v_k.unsqueeze(1)  # (B, 1, n_heads)

        lambda_i = (lam_ik * phi_ik * v_k).sum(dim=-1) + self.eps # (B, N)
        return lambda_i
    

# -------------------------
# Dueling Mixing network (Figure 1a)
# Input: [V_i(τ), A_i(τ,a_i)] and outputs Q_tot(τ,a)=V_tot + A_tot
# where:
#   V_tot(τ) = Σ_i V_i(τ)           (Eq.8)
#   A_tot(τ,a) = Σ_i λ_i(τ,a) A_i   (Eq.9)  (= dot product)
# -------------------------
class DuelingMixing(nn.Module):
    def __init__(self, n_agents, n_actions, state_dim, hidden_dim=64, n_heads=4):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim

        self.lambda_net = MultiHeadLambda(
            n_agents = n_agents,
            n_actions = n_actions,
            state_dim = state_dim,
            hidden_dim = hidden_dim,
            n_heads = n_heads
        )
    def forward(self, state:torch.Tensor, actions:torch.Tensor, v_tau:torch.Tensor, a_tau:torch.Tensor):
        """
        state:   (B,S)
        actions: (B,N)
        v_tau:   (B,N) transformed values
        a_tau:   (B,N) transformed advantages for chosen action only

        returns:
          q_tot: (B,)
          info: dict
        """
        state = _ensure_2d(state)
        actions = _ensure_2d(actions)
        v_tau = _ensure_2d(v_tau)
        a_tau = _ensure_2d(a_tau)

        B, N = v_tau.shape
        assert N == self.n_agents and a_tau.shape == (B, N), f"Expected v_tau and a_tau to have shape (B,{self.n_agents}), got {v_tau.shape} and {a_tau.shape}"

        # λ_i(τ,a) > 0
        lambda_i = self.lambda_net(state, actions)               # (B,N)
        
        v_tot = v_tau.sum(dim=-1)  # (B,N)
        a_tot = (lambda_i * a_tau).sum(dim=-1)                     # (B,)
        q_tot = v_tot + a_tot      
                                      # (B,)
        return q_tot, {
            "lambda_i": lambda_i,
            "v_tau": v_tau, "a_tau": a_tau,
            "v_tot": v_tot, "a_tot": a_tot
        }

# -------------------------
# Duplex Dueling wrapper:
#   Q_local_all -> (V_local, A_local_sel) -> Transform(V,A) -> Mix -> Q_tot
# -------------------------
class QPLEXDuplexDueling(nn.Module):
    def __init__(self, n_agents, n_actions, state_dim, hidden_dim=64, n_heads=4):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.transform = Transformation(n_agents, state_dim, hidden_dim)
        self.mixer = DuelingMixing(n_agents, n_actions, state_dim, hidden_dim, n_heads)

    def forward(self, q_local_all: torch.Tensor, actions: torch.Tensor, state: torch.Tensor):
        """
        q_local_all: (B,N,A) from agent networks (local)
        actions:     (B,N)
        state:       (B,S)

        returns:
            q_tot: (B,)
            info: dict
        """
        v_local, a_local_sel, q_sel = dueling_from_q(q_local_all, actions)  # (B,N), (B,N), (B,N)
        v_tau, a_tau, w, b = self.transform(state, v_local, a_local_sel)  # (B,N), (B,N), (B,N), (B,N)
        q_tot, mix_info = self.mixer(state, actions, v_tau, a_tau)
        
        mix_info.update({
            "v_local": v_local,
            "a_local_sel": a_local_sel,
            "q_sel": q_sel,
            "w": w,
            "b": b
        })
        return q_tot, mix_info