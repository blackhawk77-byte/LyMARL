"""
# Hetero Learner:
#   - UE: QMIX (team reward = ue_team_reward)
#   - BS: Individual TD loss (indiv reward = bs_rewards[B])
"""

from dataclasses import dataclass
from tqdm import trange
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import os, csv

from qmix import AgentNetwork, MixingNetwork
from replaybuffer import ReplayBufferRNN


# -------------------------
# Utils
# -------------------------
def hard_update(target: nn.Module, online: nn.Module):
    target.load_state_dict(online.state_dict())

@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), online.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * param.data)

def one_hot(a: torch.Tensor, num_actions: int) -> torch.Tensor:
    # a: (B, )
    return F.one_hot(a.long(), num_classes=num_actions).float()

def apply_mask_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # q: (A, ) or (B, A), mask: same
    q = q.clone()
    q[~mask] = float('-inf')  # Set invalid actions to -inf
    all_invalid = (~mask).all(dim=-1, keepdim=True)
    q = torch.where(all_invalid, torch.zeros_like(q), q)
    return q

# -------------------------
# Config
# -------------------------
@dataclass 
class HeteroQMIXcfg:
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.01
    grad_clip: float = 10.0

    batch_size: int = 64
    seq_len: int = 10
    capacity_episodes: int = 10000
    update_interval_steps: int = 128

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.9995


class HeteroQMIXAgent:
    """
    EXACT MAPPO flow:
      reset()
      UE action (masked)
      bs_obs, bs_mask, cand_lists = build_bs_decision_inputs(ue_actions)
      BS action (masked)
      step_joint(ue_actions, bs_actions, cand_lists)

    Learning:
      - UE: QMIX with team reward (ue_team_reward)
      - BS: Individual TD loss with r_indiv = bs_rewards (vector length B)
    """
    def __init__(self, env, cfg: HeteroQMIXcfg, log_dir: str = "logs/qmix_lymarl_logs", device: Optional[str] = None):
        super(HeteroQMIXAgent, self).__init__()
        
        # Environment
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.users = env.users
        self.base_stations = env.base_stations
        self.N_ue = len(self.users)
        self.N_bs = len(self.base_stations)

        self.ue_obs_dim = env.local_obs_dim
        self.ue_act_dim = env.action_dim

        self.bs_obs_dim = env.bs_obs_dim
        self.bs_act_dim = env.bs_action_dim

        self.state_dim = env.global_state_dim

        # ----------------- UE QMIX ------------------
        self.ue_net = AgentNetwork(self.ue_obs_dim, self.ue_act_dim, cfg.hidden_dim).to(self.device)
        self.ue_tgt = AgentNetwork(self.ue_obs_dim, self.ue_act_dim, cfg.hidden_dim).to(self.device)
        self.ue_mix = MixingNetwork(self.N_ue, self.state_dim, cfg.hidden_dim).to(self.device)
        self.ue_mix_tgt = MixingNetwork(self.N_ue, self.state_dim, cfg.hidden_dim).to(self.device)

        # ----------------- BS (indiv TD only) ------------------
        self.bs_net = AgentNetwork(self.bs_obs_dim, self.bs_act_dim, cfg.hidden_dim).to(self.device)
        self.bs_tgt = AgentNetwork(self.bs_obs_dim, self.bs_act_dim, cfg.hidden_dim).to(self.device)

        hard_update(self.ue_tgt, self.ue_net)
        hard_update(self.ue_mix_tgt, self.ue_mix)
        hard_update(self.bs_tgt, self.bs_net)
        params = (
            list(self.ue_net.parameters()) +
            list(self.ue_mix.parameters()) +
            list(self.bs_net.parameters())
        )
        self.optimizer = optim.Adam(params, lr=cfg.lr, amsgrad=True)

        # buffers (ReplayBufferRNN: full trajectories + fixed_length sample)
        self.buf_ue = ReplayBufferRNN(capacity=cfg.capacity_episodes, device=self.device)
        self.buf_bs = ReplayBufferRNN(capacity=cfg.capacity_episodes, device=self.device)

        self.eps = cfg.eps_start
        self.total_env_stpes = 0
    
    def _decay_eps(self):
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    # -------------------------
    # Action selection (MAPPOTrainer-compatible)
    # -------------------------
    @torch.no_grad()
    def select_actions(self, local_obs: Dict[str, np.ndarray], global_obs: np.ndarray):
        # ---- UE -----
        ue_obs_batch = np.stack([local_obs[u.ue_id] for u in self.users], axis=0).astype(np.float32)  # (N_ue, obs_dim)
        ue_mask_batch = np.stack([self.env._get_action_mask(u.ue_id) for u in self.users], axis=0).astype(np.bool_)  # (N_ue, act_dim)
        
        ue_obs_t = torch.as_tensor(ue_obs_batch, dtype=torch.float32, device=self.device)  # (N_ue, obs_dim)
        ue_mask_t = torch.as_tensor(ue_mask_batch, dtype=torch.bool, device=self.device)  # (N_ue, act_dim)

        last_a0 = torch.zeros((self.N_ue, self.ue_act_dim), device=self.device)  # (N_ue, act_dim)
        q_ue_all, _ = self.ue_net(ue_obs_t, last_a0, his_in=None)  # (N_ue, act_dim)
        q_ue_all = apply_mask_q(q_ue_all, ue_mask_t)  # (N_ue, act_dim)

        ue_actions_arr = []
        for i in range(self.N_ue):
            if random.random() < self.eps:
                valid_actions = np.where(ue_mask_batch[i])[0]
                a = int(np.random.choice(valid_actions))
            else:
                a = int(torch.argmax(q_ue_all[i]).item())
            ue_actions_arr.append(a)

        ue_actions = {u.ue_id: ue_actions_arr[i] for i, u in enumerate(self.users)}

        # ---- BS -----
        bs_obs_batch, bs_mask_batch, cand_lists = self.env.build_bs_decision_inputs(ue_actions)
        bs_obs_t = torch.as_tensor(bs_obs_batch, dtype=torch.float32, device=self.device)  # (N_bs, obs_dim)
        bs_mask_t = torch.as_tensor(bs_mask_batch, dtype=torch.bool, device=self.device)  # (N_bs, act_dim)

        last_a0_bs = torch.zeros((self.N_bs, self.bs_act_dim), device=self.device)  # (N_bs, act_dim)
        q_bs_all, _ = self.bs_net(bs_obs_t, last_a0_bs, his_in=None)  # (N_bs, act_dim)
        q_bs_all = apply_mask_q(q_bs_all, bs_mask_t)  # (N_bs, act_dim)

        bs_actions_arr = []
        for j in range(self.N_bs):
            if random.random() < self.eps:
                valid_actions = np.where(bs_mask_batch[j])[0]
                a = int(np.random.choice(valid_actions))
            else:
                a = int(torch.argmax(q_bs_all[j]).item())
            bs_actions_arr.append(a)

        bs_actions = {b.bs_id: bs_actions_arr[j] for j, b in enumerate(self.base_stations)}
        return (ue_actions, ue_actions_arr, ue_obs_batch, ue_mask_batch,
                bs_actions, bs_actions_arr, bs_obs_batch, bs_mask_batch, cand_lists)
        
    # -------------------------
    # Rollout (MAPPOTrainer-style) + store to buffers
    # -------------------------
    def rollout_episode(self, n_steps: int = 200) -> Dict[str, float]:
        local_obs, global_obs = self.env.reset()

        # UE trajectory
        ue_lo, ue_s, ue_a, ue_rtot, ue_nlo, ue_ns, ue_done = [], [], [], [], [], [], []

        # BS trajectory
        bs_lo, bs_s, bs_a, bs_rindiv, bs_rtot, bs_nlo, bs_ns, bs_done = [], [], [], [], [], [], [], []
        bs_mask, bs_next_mask = [], []

        ep_r_ue = 0.0
        ep_r_bs_mean = 0.0

        for _ in range(n_steps):
            (ue_actions, ue_actions_arr, ue_obs_batch, ue_masks_batch,
                bs_actions, bs_actions_arr, bs_obs_batch, bs_masks_batch, cand_lists) = self.select_actions(local_obs, global_obs)

            next_local_obs, next_global_obs, info, done = self.env.step_joint(
                ue_actions=ue_actions, 
                bs_actions=bs_actions, 
                cand_lists=cand_lists
            )

            rew_ue = float(info['ue_team_reward'])
            rew_bs_vec = np.asarray(info["bs_rewards"], dtype=np.float32)  # (N_bs, )
            assert rew_bs_vec.shape[0] == self.N_bs

            ep_r_ue += rew_ue
            ep_r_bs_mean += float(np.mean(rew_bs_vec))

            # ---- next obs for UE ----
            ue_next_obs_batch = np.stack([next_local_obs[u.ue_id] for u in self.users], axis=0).astype(np.float32)  # (N_ue, obs_dim)

            # ---- next obs for BS ----
            (next_ue_actions, _, _, _, _, _, _, _, _) = self.select_actions(next_local_obs, next_global_obs)
            bs_next_obs_batch, bs_next_mask_batch, _ = self.env.build_bs_decision_inputs(next_ue_actions)

            # done replicated
            ue_done_batch = np.full((self.N_ue,), bool(done), dtype=bool)
            bs_done_batch = np.full((self.N_bs,), bool(done), dtype=bool)

            # append UE
            ue_lo.append(torch.tensor(ue_obs_batch, dtype=torch.float32, device=self.device))
            ue_s.append(torch.tensor(global_obs, dtype=torch.float32, device=self.device))
            ue_a.append(torch.tensor(ue_actions_arr, dtype=torch.long, device=self.device))
            ue_rtot.append(torch.tensor(rew_ue, dtype=torch.float32, device=self.device))
            ue_nlo.append(torch.tensor(ue_next_obs_batch, dtype=torch.float32, device=self.device))
            ue_ns.append(torch.tensor(next_global_obs, dtype=torch.float32, device=self.device))
            ue_done.append(torch.tensor(ue_done_batch, dtype=torch.bool, device=self.device))

            # append BS
            bs_lo.append(torch.tensor(bs_obs_batch, dtype=torch.float32, device=self.device))
            bs_s.append(torch.tensor(global_obs, dtype=torch.float32, device=self.device))
            bs_a.append(torch.tensor(bs_actions_arr, dtype=torch.long, device=self.device))
            bs_rindiv.append(torch.tensor(rew_bs_vec, dtype=torch.float32, device=self.device))
            bs_rtot.append(torch.tensor(np.sum(rew_bs_vec), dtype=torch.float32, device=self.device))
            bs_nlo.append(torch.tensor(bs_next_obs_batch, dtype=torch.float32, device=self.device))
            bs_ns.append(torch.tensor(next_global_obs, dtype=torch.float32, device=self.device))
            bs_done.append(torch.tensor(bs_done_batch, dtype=torch.bool, device=self.device))
            bs_mask.append(torch.tensor(bs_masks_batch, dtype=torch.bool, device=self.device))
            bs_next_mask.append(torch.tensor(bs_next_mask_batch, dtype=torch.bool, device=self.device))

            local_obs, global_obs = next_local_obs, next_global_obs
            self.total_env_stpes += 1
            self._decay_eps()

            if done:
                break

        T = len(ue_lo)
        if T == 0:
            return {"ep_len": 0.0, "ep_r_ue_sum": 0.0, "ep_r_bs_mean": 0.0, "epsilon": float(self.eps)}

        # stack as (T, N, dim)
        ue_lo = torch.stack(ue_lo, dim=0)
        ue_s = torch.stack(ue_s, dim=0)
        ue_a = torch.stack(ue_a, dim=0)
        ue_rtot = torch.stack(ue_rtot, dim=0)
        ue_nlo = torch.stack(ue_nlo, dim=0)
        ue_ns = torch.stack(ue_ns, dim=0)
        ue_done = torch.stack(ue_done, dim=0)

        bs_lo = torch.stack(bs_lo, dim=0)
        bs_s = torch.stack(bs_s, dim=0)
        bs_a = torch.stack(bs_a, dim=0)
        bs_rindiv = torch.stack(bs_rindiv, dim=0)
        bs_rtot = torch.stack(bs_rtot, dim=0)
        bs_nlo = torch.stack(bs_nlo, dim=0)
        bs_ns = torch.stack(bs_ns, dim=0)
        bs_done = torch.stack(bs_done, dim=0)
        bs_mask = torch.stack(bs_mask, dim=0)
        bs_next_mask = torch.stack(bs_next_mask, dim=0)

        # store to buffers
        self.buf_ue.push(ue_lo, ue_s, ue_a, ue_rtot, ue_nlo, ue_ns, ue_done, None, None, None)
        self.buf_bs.push(bs_lo, bs_s, bs_a, bs_rtot, bs_nlo, bs_ns, bs_done, bs_rindiv, bs_mask, bs_next_mask)
        
        return {"ep_len": float(T),
                "ep_r_ue_sum": float(ep_r_ue),
                "ep_r_bs_mean": float(ep_r_bs_mean / max(1, T)),
                "epsilon": float(self.eps)
            }  
    
    # -------------------------
    # UE QMIX loss (team reward)
    # -------------------------                                                                                    
    def _loss_ue_qmix(self, batch) -> torch.Tensor:
        obs, state, action, r_tot, next_obs, next_state, done, _, _, _= batch
        B, L, N, _ = obs.shape
        assert N == self.N_ue

        agent_qs, target_qs = [], []

        for i in range(N):
            a_i = action[:, :, i]                    # (B, L)
            s_i = obs[:, :, i, :]                  # (B, L, obs)
            ns_i = next_obs[:, :, i, :]            # (B, L, obs) 

            h = torch.zeros(B, self.ue_net.hidden_dim, device=self.device)  # (B, hidden_dim)
            h_tgt = torch.zeros_like(h)  # (B, hidden_dim)

            q_seq, tq_seq = [], []

            for t in range(L):
                obs_t = s_i[:, t]  # (B, obs_dim)
                act_t = a_i[:, t]  # (B,)

                if t==0:
                    a_prev_1hot = torch.zeros(B, self.ue_act_dim, device=self.device)
                else:
                    a_prev_1hot = one_hot(a_i[:,t-1], self.ue_act_dim)  # (B, A)
                
                q_all, h = self.ue_net(obs_t, a_prev_1hot, h)  # (B, A), (B, hidden_dim)
                q_sel = q_all.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)  # (B, )
                q_seq.append(q_sel.unsqueeze(1))  # (B, 1)

                with torch.no_grad():
                    next_obs_t = ns_i[:, t]  # (B, obs_dim)
                    a_curr_1hot = one_hot(act_t, self.ue_act_dim)  # (B, A)
                    q_next_online, _ = self.ue_net(next_obs_t, a_curr_1hot, h.detach())  # (B, A), (B, hidden_dim)
                    next_a = q_next_online.argmax(dim=-1, keepdim=True)         # (B, 1)

                    q_next_tgt, h_tgt = self.ue_tgt(next_obs_t, a_curr_1hot, h_tgt)  # (B, A), (B, hidden_dim)
                    tq = q_next_tgt.gather(-1, next_a).squeeze(-1)  # (B, )
                    tq_seq.append(tq.unsqueeze(1))  # (B, 1)
                
            agent_qs.append(torch.cat(q_seq, dim =1))       # (B, L)
            target_qs.append(torch.cat(tq_seq, dim =1))     # (B, L)
        
        agent_qs = torch.stack(agent_qs, dim=-1)     # (B, L, N)
        target_qs = torch.stack(target_qs, dim=-1)   # (B, L, N)

        q_tot, tq_tot = [], []
        for t in range(L):
            q_tot.append(self.ue_mix(agent_qs[:, t], state[:, t]))          # (B, )
            tq_tot.append(self.ue_mix_tgt(target_qs[:, t], next_state[:, t])) # (B, )
        q_tot = torch.stack(q_tot, dim=1)     # (B, L)
        tq_tot = torch.stack(tq_tot, dim=1)   # (B, L)

        done_any = done[:,:,0].float()
        y = r_tot + self.cfg.gamma * (1.0 - done_any) * tq_tot
        return F.smooth_l1_loss(q_tot, y.detach())
    
    # -------------------------
    # BS individual TD loss (indiv reward)
    # -------------------------
    def _loss_bs_indiv(self, batch) -> torch.Tensor:
        obs, state, action, r_tot, next_obs, next_state, done, r_indiv, mask, next_mask = batch

        assert r_indiv is not None, "BS buffer must store individual rewards for indiv TD loss"
        B, L, Nb, _ = obs.shape
        assert Nb == self.N_bs
        assert r_indiv.shape == (B, L, Nb)
        assert next_mask is not None, "BS buffer must store next action mask for indiv TD loss"

        losses=[]
        for j in range(Nb):
            a_j = action[:, :, j]                  # (B, L)
            s_j = obs[:, :, j, :]                  # (B, L, obs)
            ns_j = next_obs[:, :, j, :]            # (B, L, obs)
            r_ind_j = r_indiv[:, :, j]             # (B, L)
            d_j = done[:, :, j].float()            # (B, L)

            h = torch.zeros(B, self.bs_net.hidden_dim, device=self.device)  # (B, hidden_dim)
            h_tgt = torch.zeros_like(h)  # (B, hidden_dim)

            q_sel_seq, tq_seq = [], []
            for t in range(L):
                obs_t = s_j[:, t]  # (B, obs_dim)
                act_t = a_j[:, t]  # (B,)

                if t==0:
                    a_prev_1hot = torch.zeros(B, self.bs_act_dim, device=self.device)
                else:
                    a_prev_1hot = one_hot(a_j[:,t-1], self.bs_act_dim)  # (B, A)
                
                q_all, h = self.bs_net(obs_t, a_prev_1hot, h)  # (B, A), (B, hidden_dim)
                q_sel = q_all.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)  # (B, )
                q_sel_seq.append(q_sel)  # (B, 1)

                with torch.no_grad():
                    next_obs_t = ns_j[:, t]  # (B, obs_dim)
                    a_curr_1hot = one_hot(act_t, self.bs_act_dim)  # (B, A)
                    next_mask_t = next_mask[:, t, j, :]  # (B, A)

                    q_next_online, _ = self.bs_net(next_obs_t, a_curr_1hot, h.detach())  # (B, A), (B, hidden_dim)
                    q_next_online = apply_mask_q(q_next_online, next_mask_t)  # (B, A)
                    next_a = q_next_online.argmax(dim=-1, keepdim=True)       # (B, 1)

                    q_next_tgt, h_tgt = self.bs_tgt(next_obs_t, a_curr_1hot, h_tgt)  # (B, A), (B, hidden_dim)
                    q_next_tgt = apply_mask_q(q_next_tgt, next_mask_t)  # (B, A)
                    tq = q_next_tgt.gather(-1, next_a).squeeze(-1)  # (B, )
                    tq_seq.append(tq)  # (B, 1)

            agent_q_seq = torch.stack(q_sel_seq,dim=1)    # (B,L)
            target_q_seq = torch.stack(tq_seq,dim=1)      # (B,L)

            y = r_ind_j + self.cfg.gamma * (1.0 - d_j) * target_q_seq
            loss = F.smooth_l1_loss(agent_q_seq,y.detach())
            losses.append(loss)

        return torch.stack(losses).mean()
    
    # -------------------------
    # Update
    # -------------------------
    def update(self) -> Dict[str, float]:
        if len(self.buf_ue) < self.cfg.batch_size or len(self.buf_bs) < self.cfg.batch_size:
            return {}
        
        batch_ue = self.buf_ue.sample(self.cfg.batch_size, self.cfg.seq_len, use_indiv=False)
        batch_bs = self.buf_bs.sample(self.cfg.batch_size, self.cfg.seq_len, use_indiv=True)
    
        loss_ue = self._loss_ue_qmix(batch_ue)
        loss_bs = self._loss_bs_indiv(batch_bs)
        loss = loss_ue + loss_bs

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.cfg.grad_clip)
        self.optimizer.step()

        soft_update(self.ue_tgt, self.ue_net, self.cfg.tau)
        soft_update(self.ue_mix_tgt, self.ue_mix, self.cfg.tau)
        soft_update(self.bs_tgt, self.bs_net, self.cfg.tau)

        return {"loss": float(loss.item()), 
                "loss_ue": float(loss_ue.item()), 
                "loss_bs": float(loss_bs.item()),
                "epsilon": float(self.eps),
            }
    
    def train(self, n_env_steps: int, rollout_horizon: int = 200) -> List[Dict[str, float]]:
        logs = []
        while self.total_env_steps < n_env_steps:
            ep = self.rollout_episode(n_steps=rollout_horizon)
            logs.append({"type": "rollout", **ep})

            if (self.total_env_steps % self.cfg.update_interval_steps) == 0:
                upd = self.update()
                if upd:
                    logs.append({"type": "update", **upd})
        return logs
    