#HeteroQPLEXAgent.py
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

from benchmark.qplex.qplex import AgentNetwork as QAgentNet
from benchmark.qplex.qplex import QPLEXDuplexDueling
from benchmark.qmix.replaybuffer import ReplayBufferRNN
from LyMARL.networks_mappo import ValueNorm

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
    return q

# -------------------------
# Config
# -------------------------
@dataclass 
class HeteroQPLEXcfg:
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.01
    grad_clip: float = 10.0

    batch_size: int = 64
    seq_len: int = 64      # L for training
    chunk_len: int = 100    # T for saving in buffer
    capacity_episodes: int = 10000
    update_interval_steps: int = 100

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.99995

    n_heads: int = 8


class HeteroQPLEXAgent:
    """
    Hetero learner:
      - UE: QPLEX Qtot with team reward
      - BS: QPLEX Qtot with team reward
    Same rollout/buffer/update skeleton as your HeteroQMIXAgent.
    """
    def __init__(self, env, cfg: HeteroQPLEXcfg, device: Optional[str] = None):
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
        self.state_dim = env.global_obs_dim

        # ----------------- UE (QPLEX) ------------------
        self.ue_net = QAgentNet(self.ue_obs_dim, self.ue_act_dim, cfg.hidden_dim).to(self.device)
        self.ue_tgt = QAgentNet(self.ue_obs_dim, self.ue_act_dim, cfg.hidden_dim).to(self.device)
        self.ue_duplex = QPLEXDuplexDueling(self.N_ue, self.ue_act_dim, self.state_dim, cfg.hidden_dim, cfg.n_heads).to(self.device)
        self.ue_duplex_tgt = QPLEXDuplexDueling(self.N_ue, self.ue_act_dim, self.state_dim, cfg.hidden_dim, cfg.n_heads).to(self.device)

        # ----------------- BS (QPLEX) ------------------
        self.bs_net = QAgentNet(self.bs_obs_dim, self.bs_act_dim, cfg.hidden_dim).to(self.device)
        self.bs_tgt = QAgentNet(self.bs_obs_dim, self.bs_act_dim, cfg.hidden_dim).to(self.device)
        self.bs_duplex = QPLEXDuplexDueling(self.N_bs, self.bs_act_dim, self.state_dim, cfg.hidden_dim, cfg.n_heads).to(self.device)
        self.bs_duplex_tgt = QPLEXDuplexDueling(self.N_bs, self.bs_act_dim, self.state_dim, cfg.hidden_dim, cfg.n_heads).to(self.device)

        hard_update(self.ue_tgt, self.ue_net)
        hard_update(self.ue_duplex_tgt, self.ue_duplex)
        hard_update(self.bs_tgt, self.bs_net)
        hard_update(self.bs_duplex_tgt, self.bs_duplex)

        self.opt_ue = optim.Adam(list(self.ue_net.parameters()) + list(self.ue_duplex.parameters()), lr=cfg.lr, amsgrad=True)
        self.opt_bs = optim.Adam(list(self.bs_net.parameters()) + list(self.bs_duplex.parameters()), lr=cfg.lr, amsgrad=True)

        # buffers (ReplayBufferRNN: full trajectories + fixed_length sample, team reward only)
        self.buf_ue = ReplayBufferRNN(capacity=cfg.capacity_episodes, device=self.device)
        self.buf_bs = ReplayBufferRNN(capacity=cfg.capacity_episodes, device=self.device)

        self.eps = cfg.eps_start
        self.total_env_steps = 0

        # rollout rnn states / last actions
        self._ue_h = None
        self._bs_h = None
        self._ue_last_a = None
        self._bs_last_a = None

        self._cur_local_obs = None
        self._cur_global_obs = None
        self._need_env_reset = True

        self.reward_history_ue = []
        self.reward_history_bs = []
        self.thr_history = []
        self.fair_history_100step = []
        self.on_ratio_history = []
        self.step_history=[]

        self.ue_vnorm = ValueNorm(eps=1e-5, device=self.device)
        self.bs_vnorm = ValueNorm(eps=1e-5, device=self.device)
    
    def _decay_eps(self):
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    def _reset_rollout_rnn_state(self):
        self._ue_h = torch.zeros((self.N_ue, self.ue_net.hidden_dim), device=self.device)
        self._bs_h = torch.zeros((self.N_bs, self.bs_net.hidden_dim), device=self.device)
        self._ue_last_a = torch.zeros((self.N_ue, self.ue_act_dim), device=self.device)
        self._bs_last_a = torch.zeros((self.N_bs, self.bs_act_dim), device=self.device)

    def jain_fairness(self, rate_history, window: int = 100, eps: float = 1e-12) -> float:
        recent = rate_history if len(rate_history) < window else rate_history[-window:]
        if len(recent) == 0:
            return 0.0
        rate_array= np.asarray(recent, dtype=np.float64) # (T, N)
        rate_array = np.maximum(rate_array, 0.0)  

        x = rate_array.mean(axis=0)  # (N,)
        num = (x.sum() ** 2)
        den = (len(x) * (np.square(x).sum() + eps)) + eps
        return float(num / den)
    
    def _maybe_reset_env(self):
        if self._need_env_reset or (self._cur_local_obs is None) or (self._cur_global_obs is None):
            self._cur_local_obs, self._cur_global_obs = self.env.reset()
            self._need_env_reset = False
            self._reset_rollout_rnn_state()
    
    # -------------------------
    # Action selection
    # -------------------------
    @torch.no_grad()
    def select_actions(self, 
                       local_obs: Dict[str, np.ndarray], 
                       global_obs: np.ndarray, 
                       *, 
                       update_rnn_state: bool = True,
                       eps_override: Optional[float] = None):
        eps = self.eps if eps_override is None else float(eps_override)
        
        # ---- UE -----
        ue_obs_batch = np.stack([local_obs[u.ue_id] for u in self.users], axis=0).astype(np.float32)  # (N_ue, obs_dim)
        ue_mask_batch = np.stack([self.env._get_action_mask(u.ue_id) for u in self.users], axis=0).astype(np.bool_)  # (N_ue, act_dim)
        
        ue_obs_t = torch.as_tensor(ue_obs_batch, dtype=torch.float32, device=self.device)  # (N_ue, obs_dim)
        ue_mask_t = torch.as_tensor(ue_mask_batch, dtype=torch.bool, device=self.device)  # (N_ue, act_dim)

        q_ue_all, ue_h_out = self.ue_net(ue_obs_t, self._ue_last_a, his_in=self._ue_h)  # (N_ue, act_dim)
        q_ue_all = apply_mask_q(q_ue_all, ue_mask_t)  # (N_ue, act_dim)

        # -------------------------------------------------
        # Load-aware / Z-aware bias for UE decisions
        # -------------------------------------------------
        running_req_counts = np.zeros(self.N_bs, dtype=np.float32)
        k_cong = 0.0   # 몰림 패널티 1.0~1.1
        k_z = 0.0     # Z 큰 BS 회피 0.0~0.005
        ideal = 1.0 / max(1, self.N_bs)

        ue_actions_arr = []
        for i in range(self.N_ue):
            q_i = q_ue_all[i].clone() # (act_dim, )
            tot_req_so_far = float(running_req_counts.sum())

            for b in range(self.N_bs):
                # 1) 현재까지 이미 몰린 BS면 bias
                if tot_req_so_far > 0:
                    cong = running_req_counts[b] / (tot_req_so_far + 1e-12)
                    excess = max(0.0, cong - ideal)
                    q_i[b] -= k_cong * excess

                # 2) Z_b가 큰 BS면 bias
                bs_id = self.base_stations[b].bs_id
                q_i[b] -= k_z * float(self.env.Z_b[bs_id])
            
            # mask 다시 안전하게 적용
            invalid_mask = ~ue_mask_t[i]
            q_i[invalid_mask] = -1e9
            
            if random.random() < eps:
                valid_actions = np.where(ue_mask_batch[i])[0]
                # exploration에서도 no_request 남발 방지
                bs_valid = [a for a in valid_actions if a < self.N_bs]
                if len(bs_valid) > 0:
                    a = int(np.random.choice(bs_valid))
                else:
                    a = int(np.random.choice(valid_actions))
            else:
                a = int(torch.argmax(q_i).item())

            ue_actions_arr.append(a)
            # UE가 실제로 고른 BS면 running count 반영
            if 0 <= a < self.N_bs:
                running_req_counts[a] += 1.0

        ue_actions = {u.ue_id: ue_actions_arr[i] for i, u in enumerate(self.users)}
        
        if update_rnn_state:
            self._ue_h = ue_h_out.detach()
            self._ue_last_a = one_hot(torch.tensor(ue_actions_arr, device=self.device), num_actions = self.ue_act_dim)

        # ---- BS -----
        bs_obs_batch, bs_mask_batch, cand_lists = self.env.build_bs_decision_inputs(ue_actions)
        bs_obs_t = torch.as_tensor(bs_obs_batch, dtype=torch.float32, device=self.device)  # (N_bs, obs_dim)
        bs_mask_t = torch.as_tensor(bs_mask_batch, dtype=torch.bool, device=self.device)  # (N_bs, act_dim)

        q_bs_all, bs_h_out = self.bs_net(bs_obs_t, self._bs_last_a, his_in=self._bs_h)  # (N_bs, act_dim)
        q_bs_all = apply_mask_q(q_bs_all, bs_mask_t)  # (N_bs, act_dim)

        bs_actions_arr = []
        for j in range(self.N_bs):
            if random.random() < eps:
                valid_actions = np.where(bs_mask_batch[j])[0]
                a = int(np.random.choice(valid_actions))
            else:
                a = int(torch.argmax(q_bs_all[j]).item())
            bs_actions_arr.append(a)

        bs_actions = {b.bs_id: bs_actions_arr[j] for j, b in enumerate(self.base_stations)}
        
        if update_rnn_state:
            self._bs_h = bs_h_out.detach()
            self._bs_last_a = one_hot(torch.tensor(bs_actions_arr, device=self.device), num_actions=self.bs_act_dim)

        return (ue_actions, ue_actions_arr, ue_obs_batch, ue_mask_batch,
                bs_actions, bs_actions_arr, bs_obs_batch, bs_mask_batch, cand_lists)
        
    # -------------------------
    # Rollout + store to buffers
    # -------------------------
    def rollout_episode(self, n_steps: int = 200) -> Dict[str, float]:
        if n_steps is None:
            n_steps = self.cfg.chunk_len
        
        self._maybe_reset_env()
        local_obs, global_obs = self._cur_local_obs, self._cur_global_obs

        thr_sum = 0.0
        thr_last = 0.0
        rate_history = []
        recent_on_hist = []

        # UE trajectory
        ue_lo, ue_s, ue_a, ue_rtot, ue_nlo, ue_ns, ue_done = [], [], [], [], [], [], []
        ue_mask, ue_next_mask = [], []

        # BS trajectory
        bs_lo, bs_s, bs_a, bs_rtot, bs_nlo, bs_ns, bs_done = [], [], [], [], [], [], []
        bs_mask, bs_next_mask = [], []

        ep_r_ue = 0.0
        ep_r_bs = 0.0 
        done_flag = False

        reward_ue_hist, reward_bs_hist = [], []

        for _ in range(n_steps):
            (ue_actions, ue_actions_arr, ue_obs_batch, ue_masks_batch,
            bs_actions, bs_actions_arr, bs_obs_batch, bs_masks_batch, cand_lists) = self.select_actions(local_obs, global_obs)

            next_local_obs, next_global_obs, info, done = self.env.step_joint(
                ue_actions=ue_actions, 
                bs_actions=bs_actions, 
                cand_lists=cand_lists
            )
            # =========================
            # Throughput / Fairness stats
            # =========================
            thr_last = float(info.get("total_throughput", 0.0))
            thr_sum += thr_last
            self.thr_history.append(thr_last)

            served_rates = info.get("served_rates", None)  # dict {ue_id: rate}
            if served_rates is not None:
                step_rates = np.zeros(self.N_ue, dtype=np.float64)
                for ue_id, r in served_rates.items():
                    idx = int(ue_id) - 1
                    if 0 <= idx < self.N_ue:
                        step_rates[idx] = float(r)
                rate_history.append(step_rates)
                fair_t = self.jain_fairness(rate_history, window=100)
                self.fair_history_100step.append(fair_t)
            
            # =========================
            # On-ratio stats
            # =========================
            power_consumed = info.get("power_consumed", None)  # dict {bs_id: power}

            if power_consumed is not None:
                on_now = np.array(
                    [1.0 if float(power_consumed[b.bs_id]) > 0.0 else 0.0 for b in self.base_stations], 
                    dtype=np.float64)
                recent_on_hist.append(float(np.mean(on_now)))
                on_ratio_100 = float(np.mean(recent_on_hist[-100:]))
                self.on_ratio_history.append(on_ratio_100)
            else:
                on_feats = info.get("on_feats", None)  # dict {bs_id: on_feat}
                if on_feats is not None:
                    if isinstance(on_feats, dict):
                        vals = np.asarray(list(on_feats.values()), dtype=np.float64)
                    else:
                        vals = np.asarray(on_feats, dtype=np.float64)
                    recent_on_hist.append(float(np.mean(vals)))
                    on_ratio_100 = float(np.mean(recent_on_hist[-100:]))
                    self.on_ratio_history.append(on_ratio_100)
                else:
                    self.on_ratio_history.append(float("nan"))
                
            # reward
            rew_ue = float(info['ue_team_reward'])
            rew_bs = float(info['bs_team_reward'])
            ep_r_ue += rew_ue
            ep_r_bs += rew_bs
            reward_ue_hist.append(rew_ue)
            reward_bs_hist.append(rew_bs)

            # ---- next obs for UE ----
            ue_next_obs_batch = np.stack([next_local_obs[u.ue_id] for u in self.users], axis=0).astype(np.float32)  # (N_ue, obs_dim)
            ue_next_mask_batch = np.stack([self.env._get_action_mask(u.ue_id) for u in self.users], axis=0).astype(np.bool_)  # (N_ue, act_dim)
            # ---- next obs for BS ----
            (next_ue_actions, _, _, _, _, _, _, _, _) = self.select_actions(next_local_obs, next_global_obs, update_rnn_state=False, eps_override=0.0)
            bs_next_obs_batch, bs_next_mask_batch, _ = self.env.build_bs_decision_inputs(next_ue_actions)

            # done replicated
            ue_done_batch = np.full((self.N_ue,), bool(done), dtype=bool)
            bs_done_batch = np.full((self.N_bs,), bool(done), dtype=bool)

            # append UE
            ue_lo.append(torch.tensor(ue_obs_batch, dtype=torch.float32, device="cpu"))
            ue_s.append(torch.tensor(global_obs, dtype=torch.float32, device="cpu"))
            ue_a.append(torch.tensor(ue_actions_arr, dtype=torch.long, device="cpu"))
            ue_rtot.append(torch.tensor(rew_ue, dtype=torch.float32, device="cpu"))
            ue_nlo.append(torch.tensor(ue_next_obs_batch, dtype=torch.float32, device="cpu"))
            ue_ns.append(torch.tensor(next_global_obs, dtype=torch.float32, device="cpu"))
            ue_done.append(torch.tensor(ue_done_batch, dtype=torch.bool, device="cpu"))
            ue_mask.append(torch.tensor(ue_masks_batch, dtype=torch.bool, device="cpu"))
            ue_next_mask.append(torch.tensor(ue_next_mask_batch, dtype=torch.bool, device="cpu"))

            # append BS
            bs_lo.append(torch.tensor(bs_obs_batch, dtype=torch.float32, device="cpu"))
            bs_s.append(torch.tensor(global_obs, dtype=torch.float32, device="cpu"))
            bs_a.append(torch.tensor(bs_actions_arr, dtype=torch.long, device="cpu"))
            bs_rtot.append(torch.tensor(rew_bs, dtype=torch.float32, device="cpu"))
            bs_nlo.append(torch.tensor(bs_next_obs_batch, dtype=torch.float32, device="cpu"))
            bs_ns.append(torch.tensor(next_global_obs, dtype=torch.float32, device="cpu"))
            bs_done.append(torch.tensor(bs_done_batch, dtype=torch.bool, device="cpu"))
            bs_mask.append(torch.tensor(bs_masks_batch, dtype=torch.bool, device="cpu"))
            bs_next_mask.append(torch.tensor(bs_next_mask_batch, dtype=torch.bool, device="cpu"))

            local_obs, global_obs = next_local_obs, next_global_obs
            self.reward_history_ue.append(rew_ue)
            self.reward_history_bs.append(rew_bs)
            self.step_history.append(self.total_env_steps)
            self.total_env_steps += 1

            Q_u = info.get("Q_u", None)
            Z_b = info.get("Z_b", None)
            if (self.total_env_steps % 50) == 0:
                no_req_idx = int(getattr(self.env, "no_request_action", self.ue_act_dim - 1))
                no_req_ratio = sum(int(a)==no_req_idx for a in ue_actions_arr) / max(1, self.N_ue)
                # ue action -> counts
                counts = np.zeros(self.N_bs+1,dtype=np.int32)
                for a in ue_actions_arr:
                    if a == no_req_idx:
                        counts[-1] += 1
                    elif 0<= a <self.N_bs:
                        counts[a] += 1
                    else:
                        pass
                bs_counts = counts[:self.N_bs]
                mean_Q = float(np.mean(list(Q_u.values()))) if isinstance(Q_u, dict) and len(Q_u) > 0 else 0.0
                mean_Z = float(np.mean(list(Z_b.values()))) if isinstance(Z_b, dict) and len(Z_b) > 0 else 0.0
                print("thr:", info.get("total_throughput", 0.0),
                      "| no_req_ratio:", round(no_req_ratio, 3),
                      "| bs_counts:", bs_counts.tolist(),
                      "| bs_mask_true_mean:", float(np.mean(bs_masks_batch.sum(axis=1))),
                      "| bs_mask_true_min:", int(bs_masks_batch.sum(axis=1).min()),
                      "| cand_mean:", float(np.mean([len(c) for c in cand_lists])),
                      "| cand_min:", int(min(len(c) for c in cand_lists)),
                     f"| mean_Q: {mean_Q:.3f}",
                     f"| mean_Z: {mean_Z:.3f}",
                    )
            self._decay_eps()

            if done:
                done_flag = True
                break

        T = len(ue_lo)
        if T == 0:
            return {"ep_len": 0.0, 
                    "ep_r_ue_sum": 0.0, 
                    "ep_r_bs_sum": 0.0, 
                    "epsilon": float(self.eps),
                    "reward_ue_hist": [],
                    "reward_bs_hist": []
            }

        # stack as (T, N, dim)
        ue_lo = torch.stack(ue_lo, dim=0)
        ue_s = torch.stack(ue_s, dim=0)
        ue_a = torch.stack(ue_a, dim=0)
        ue_rtot = torch.stack(ue_rtot, dim=0)
        ue_nlo = torch.stack(ue_nlo, dim=0)
        ue_ns = torch.stack(ue_ns, dim=0)
        ue_done = torch.stack(ue_done, dim=0)
        ue_mask = torch.stack(ue_mask, dim=0)
        ue_next_mask = torch.stack(ue_next_mask, dim=0)

        bs_lo = torch.stack(bs_lo, dim=0)
        bs_s = torch.stack(bs_s, dim=0)
        bs_a = torch.stack(bs_a, dim=0)
        bs_rtot = torch.stack(bs_rtot, dim=0)
        bs_nlo = torch.stack(bs_nlo, dim=0)
        bs_ns = torch.stack(bs_ns, dim=0)
        bs_done = torch.stack(bs_done, dim=0)
        bs_mask = torch.stack(bs_mask, dim=0)
        bs_next_mask = torch.stack(bs_next_mask, dim=0)

        # store to buffers
        if T>= self.cfg.seq_len:
            self.buf_ue.push(ue_lo, ue_s, ue_a, ue_rtot, ue_nlo, ue_ns, ue_done, None, ue_mask, ue_next_mask)
            self.buf_bs.push(bs_lo, bs_s, bs_a, bs_rtot, bs_nlo, bs_ns, bs_done, None, bs_mask, bs_next_mask)
        
        self._cur_local_obs, self._cur_global_obs = local_obs, global_obs

        if done_flag:
            self._need_env_reset = True
            self._cur_local_obs, self._cur_global_obs = None, None

        thr_mean = thr_sum / max(1, T)
        fair_mean = float(self.fair_history_100step[-1]) if len(self.fair_history_100step) > 0 else float("nan")
        on_ratio_mean = float(self.on_ratio_history[-1]) if len(self.on_ratio_history) > 0 else float("nan")

        return {"ep_len": float(T),
                "thr_sum": float(thr_sum),
                "thr_mean": float(thr_mean),
                "thr_last": float(thr_last),
                "fair_mean": float(fair_mean),              # 에피소드 누적 rate 기준 Jain
                "on_ratio_mean": float(on_ratio_mean),
                "ep_r_ue_sum": float(ep_r_ue),
                "ep_r_bs_sum": float(ep_r_bs),
                "epsilon": float(self.eps),
                "reward_ue_hist": reward_ue_hist,
                "reward_bs_hist": reward_bs_hist
            }  
    
    # -------------------------
    # Generic QPLEX team TD loss (for UE or BS)
    # -------------------------                                                                                  
    def _loss_team_qplex(self, batch, *, is_ue: bool, update_vnorm: bool=True) -> torch.Tensor:
        obs, state, action, r_tot, next_obs, next_state, done, r_indiv, mask, next_mask= batch
        B, L, N, _ = obs.shape
        assert next_mask is not None

        if is_ue:
            net, tgt = self.ue_net, self.ue_tgt
            duplex, duplex_tgt = self.ue_duplex, self.ue_duplex_tgt
            act_dim = self.ue_act_dim
            vnorm = self.ue_vnorm
        else:
            net, tgt = self.bs_net, self.bs_tgt
            duplex, duplex_tgt = self.bs_duplex, self.bs_duplex_tgt
            act_dim = self.bs_act_dim
            vnorm = self.bs_vnorm

        # build q_all_online (B,L,N,A), q_all_tgt_next (B,L,N,A), next_a_star (B,L,N)
        q_all_online = torch.zeros((B, L, N, act_dim), device=self.device)
        q_all_tgt_next = torch.zeros((B, L, N, act_dim), device=self.device)
        next_a_star = torch.zeros((B, L, N), device=self.device, dtype=torch.long)

        for i in range(N):
            a_i = action[:, :, i]                   # (B, L)
            s_i = obs[:, :, i, :]                   # (B, L, obs)
            ns_i = next_obs[:, :, i, :]             # (B, L, obs)
            next_mask_i = next_mask[:, :, i, :]     # (B, L, A)
            mask_i = mask[:, :, i, :] if mask is not None else None              # (B, L, A) 

            h = torch.zeros(B, net.hidden_dim, device=self.device)  # (B, hidden_dim)
            h_tgt = torch.zeros_like(h)  # (B, hidden_dim)

            for t in range(L):
                obs_t = s_i[:, t]  # (B, obs_dim)
                act_t = a_i[:, t]  # (B,)
                next_obs_t = ns_i[:, t]  # (B, obs_dim)
                next_mask_t = next_mask_i[:, t]  # (B, act_dim)
                cur_mask_t = mask_i[:, t] if mask_i is not None else None  # (B, act_dim)

                a_prev_1hot = torch.zeros(B, act_dim, device=self.device) if t==0 else one_hot(a_i[:,t-1], act_dim)  # (B, A)
                
                # online Q at current
                q_all_t, h = net(obs_t, a_prev_1hot, h)  # (B, A), (B, hidden_dim)
                if cur_mask_t is not None:
                    q_all_t = apply_mask_q(q_all_t, cur_mask_t)  # (B, A)
                q_all_online[:, t, i, :] = q_all_t  # save for online action

                with torch.no_grad():
                    _, h_tgt = tgt(obs_t, a_prev_1hot, h_tgt)  # (B, A), (B, hidden_dim)
                    a_curr_1hot = one_hot(act_t, act_dim)  # (B, A)
                    # next action selection -> online argmax(net)
                    q_next_online_all, _ = net(next_obs_t, a_curr_1hot, h.detach())  # (B, A), (B, hidden_dim)
                    q_next_online_all = apply_mask_q(q_next_online_all, next_mask_t)  # (B, A)
                    next_a = q_next_online_all.argmax(dim=-1)         
                    next_a_star[:, t, i] = next_a 
                    # target Q -> target eval(ue_tgt)
                    q_next_tgt_all, h_tgt_next = tgt(next_obs_t, a_curr_1hot, h_tgt)  # (B, A), (B, hidden_dim)
                    q_next_tgt_all = apply_mask_q(q_next_tgt_all, next_mask_t)  # (B, A)
                    q_all_tgt_next[:, t, i, :] = q_next_tgt_all  
                    h_tgt = h_tgt_next

        # Qtot online & target
        q_tot, tq_tot = [], []
        
        for t in range(L):
            q_t, _ = duplex(q_all_online[:, t], action[:,t], state[:, t]) # (B, )
            with torch.no_grad():
                tq_t, _ = duplex_tgt(q_all_tgt_next[:, t], next_a_star[:, t], next_state[:, t])
            q_tot.append(q_t)       # (B, )
            tq_tot.append(tq_t)     # (B, )
        q_tot = torch.stack(q_tot, dim=1)     # (B, L)
        tq_tot = torch.stack(tq_tot, dim=1)   # (B, L)

        done_any = done[:,:,0].float()
        y = r_tot + self.cfg.gamma * (1.0 - done_any) * tq_tot

        if update_vnorm:
            with torch.no_grad():
                vnorm.update(y)
        y_n = vnorm.normalize(y)
        loss_tot = F.smooth_l1_loss(q_tot, y_n.detach())

        return loss_tot 
    
    # -------------------------
    # Update
    # -------------------------
    def update(self) -> Dict[str, float]:
        if len(self.buf_ue) < self.cfg.batch_size or len(self.buf_bs) < self.cfg.batch_size:
            return {}
        
        batch_ue = self.buf_ue.sample(self.cfg.batch_size, self.cfg.seq_len, use_indiv=False)
        batch_bs = self.buf_bs.sample(self.cfg.batch_size, self.cfg.seq_len, use_indiv=False)
    
        loss_ue = self._loss_team_qplex(batch_ue, is_ue = True, update_vnorm=True)
        self.opt_ue.zero_grad()
        loss_ue.backward()
        nn.utils.clip_grad_norm_(list(self.ue_net.parameters()) + list(self.ue_duplex.parameters()), self.cfg.grad_clip)
        self.opt_ue.step()
        
        loss_bs = self._loss_team_qplex(batch_bs, is_ue = False, update_vnorm=True)
        self.opt_bs.zero_grad()
        loss_bs.backward()
        nn.utils.clip_grad_norm_(list(self.bs_net.parameters()) + list(self.bs_duplex.parameters()), self.cfg.grad_clip)
        self.opt_bs.step()        

        soft_update(self.ue_tgt, self.ue_net, self.cfg.tau)
        soft_update(self.ue_duplex_tgt, self.ue_duplex, self.cfg.tau)
        soft_update(self.bs_tgt, self.bs_net, self.cfg.tau)
        soft_update(self.bs_duplex_tgt, self.bs_duplex, self.cfg.tau)

        return {
                "loss_ue": float(loss_ue.item()), 
                "loss_bs": float(loss_bs.item()),
                "epsilon": float(self.eps),
                "ue_v_mean": float(self.ue_vnorm.mean.item()),
                "ue_v_std": float(self.ue_vnorm.std().item()),
                "bs_v_mean": float(self.bs_vnorm.mean.item()),
                "bs_v_std": float(self.bs_vnorm.std().item()),
            }
    
    def train(self, n_env_steps: int, rollout_horizon: Optional[int]=None) -> List[Dict[str, float]]:
        if rollout_horizon is None:
            rollout_horizon = self.cfg.chunk_len
        logs = []
        pbar = tqdm(total=n_env_steps, desc="Training")
        while self.total_env_steps < n_env_steps:
            prev= self.total_env_steps
            remaining = n_env_steps - self.total_env_steps
            steps_to_run= min(rollout_horizon, remaining)
            
            ep = self.rollout_episode(n_steps=steps_to_run)
            pbar.update(self.total_env_steps - prev)
            logs.append({"type": "rollout", **ep})

            if (self.total_env_steps % self.cfg.update_interval_steps) == 0:
                upd = self.update()
                if upd:
                    logs.append({"type": "update", **upd})
                    pbar.set_postfix({"loss_ue": f"{upd['loss_ue']:.4f}", 
                                      "loss_bs": f"{upd['loss_bs']:.4f}",
                                      "epsilon": f"{upd['epsilon']:.3f}"
                                    })
        pbar.close()
        return logs
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "ue_net": self.ue_net.state_dict(),
            "ue_tgt": self.ue_tgt.state_dict(),
            "ue_duplex": self.ue_duplex.state_dict(),
            "ue_duplex_tgt": self.ue_duplex_tgt.state_dict(),
            "bs_net": self.bs_net.state_dict(),
            "bs_tgt": self.bs_tgt.state_dict(),
            "bs_duplex": self.bs_duplex.state_dict(),
            "bs_duplex_tgt": self.bs_duplex_tgt.state_dict(),
            "opt_ue": self.opt_ue.state_dict(),
            "opt_bs": self.opt_bs.state_dict(),
            "ue_vnorm": self.ue_vnorm.state_dict(),
            "bs_vnorm": self.bs_vnorm.state_dict(),
            "eps": self.eps,
            "total_env_steps": self.total_env_steps,
            "cfg": self.cfg,
        }
        torch.save(checkpoint, path)
        print(f"[SAVE] Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.ue_net.load_state_dict(checkpoint["ue_net"])
        self.ue_tgt.load_state_dict(checkpoint["ue_tgt"])
        self.ue_duplex.load_state_dict(checkpoint["ue_duplex"])
        self.ue_duplex_tgt.load_state_dict(checkpoint["ue_duplex_tgt"])
        self.bs_net.load_state_dict(checkpoint["bs_net"])
        self.bs_tgt.load_state_dict(checkpoint["bs_tgt"])
        self.bs_duplex.load_state_dict(checkpoint["bs_duplex"])
        self.bs_duplex_tgt.load_state_dict(checkpoint["bs_duplex_tgt"])
        self.opt_ue.load_state_dict(checkpoint["opt_ue"])
        self.opt_bs.load_state_dict(checkpoint["opt_bs"])
        self.ue_vnorm.load_state_dict(checkpoint["ue_vnorm"])
        self.bs_vnorm.load_state_dict(checkpoint["bs_vnorm"])
        self.eps = checkpoint.get("eps", self.cfg.eps_start)
        self.total_env_steps = checkpoint.get("total_env_steps", 0)
        print(f"[LOAD] Model loaded from {path}")