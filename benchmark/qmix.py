'''
QMIX
Lqmix: E(Sum_i=1^bs)[(yi_tot - Qi_tot(τ, u, s; θ))^2] 
yi_tot = r +γ maxu′ Qtot(τ ′, u′, s′; θ−)
Qtot(τ, u, s; θ) = f(Q1(τ1, u1, s; θ), Q2(τ2, u2, s; θ), ..., Qn(τn, un, s; θ); s)
f: mixing network
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.grucell = nn.GRUCell(hidden_dim, hidden_dim)  
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, last_action, his_in):
        x = torch.cat([obs, last_action], dim=-1)
        x = F.relu(self.fc1(x))
        if his_in is None:
            his_in = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            if his_in.dim() == 1:
                his_in = his_in.unsqueeze(0)  # (batch_size, hidden_dim)
            assert his_in.shape == (x.size(0), self.hidden_dim), f"GRU hidden should be (B,{self.hidden_dim}), got {his_in.shape}"

        his_out = self.grucell(x, his_in) 
        q = self.q_out(his_out)
        return q, his_out


class HyperNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, state):
        return torch.abs(self.fc(state))


class MixingNetwork(nn.Module):
    """
    QMIX mixing + (optional) Q_ind head
    Inputs:
        agents_q: (B, N)
        state: (B, S)
    Outputs:
        q_total: (B, )
        q_ind: (B, N)
    """
    def __init__(self, n_agents, state_dim, hidden_dim=64):

        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.hyper_w1 = HyperNetwork(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w2 = HyperNetwork(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Q_tot
        )
        self.hyper_w_ind = HyperNetwork(state_dim, hidden_dim * n_agents)
        self.hyper_b_ind = nn.Linear(state_dim, n_agents)

    def forward(self, agents_q, state): 

        agents_q = agents_q.unsqueeze(0) if agents_q.dim() == 1 else agents_q   # agents_q: (B,N) or (N,)  → (B,N)
        state = state.unsqueeze(0) if state.dim() ==1 else state                # state: (B,S) or (S,) → (B,S)

        w1 = self.hyper_w1(state).view(-1, self.n_agents, self.hidden_dim)  # W1 = [B, n_agents, hidden_dim]
        b1 = self.hyper_b1(state).view(-1, 1, self.hidden_dim)              # b1 = [B, 1, hidden_dim]
        
        hidden = F.elu(torch.bmm(agents_q.unsqueeze(1), w1) + b1).squeeze(1)   # hidden = [B, 1, n_agents] * [B, n_agents, hidden_dim] + [B, 1, hidden_dim] -> squeeze [B, hidden_dim]
        
        w2 = self.hyper_w2(state).view(-1, self.hidden_dim, 1)              # W2 = [B, hidden_dim, 1]
        b2 = self.hyper_b2(state)                                           # b2 = [B, 1] -> [B, 1]

        w_ind = self.hyper_w_ind(state).view(-1, self.hidden_dim, self.n_agents)
        b_ind = self.hyper_b_ind(state)

        q_total = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2        # q_total = [B, 1, hidden_dim] * [B, hidden_dim, 1] + [B, 1]
        q_total = q_total.squeeze(-1) # (B,)

        q_ind_list = []
        for i in range(self.n_agents):
            aq = agents_q.clone()     # (B, N)
            mask = torch.ones(self.n_agents, dtype=torch.bool, device=aq.device)
            mask[i] = False           # i번째 agent만 False, 나머지는 True
            aq[:, mask] = aq[:, mask].detach()      # (B, N)에서 i번째 agent의 q값만 gradient 흐르게 하고 나머지는 상수 취급
            hidden_i = F.elu(torch.bmm(aq.unsqueeze(1), w1) + b1).squeeze(1)    # hidden_i = [B, 1, n_agents] * [B, n_agents, hidden_dim] + [B, 1, hidden_dim] -> squeeze [B, hidden_dim]
            qi = (hidden_i * w_ind[:, :, i]).sum(dim=-1) + b_ind[:, i]          # qi = [B, hidden_dim] * [B, hidden_dim] + [B, 1] -> (B,)
            q_ind_list.append(qi.unsqueeze(-1))     # (B, 1)
        q_ind = torch.cat(q_ind_list, dim=-1)       # (B, N)

        return q_total , q_ind 
