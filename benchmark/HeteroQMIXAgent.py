"""
# Hetero Learner:
#   - UE: QMIX (team reward = ue_team_reward)
#   - BS: Individual TD loss (indiv reward = bs_rewards[B])
"""

from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

from benchmark.qmix import AgentNetwork, MixingNetwork
from benchmark.replaybuffer import ReplayBufferRNN
from LyMARL.norms import ValueNorm, ValueNormVec


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
    seq_len: int = 128      # L for training
    chunk_len: int = 200    # T for saving in buffer
    capacity_episodes: int = 10000
    update_interval_steps: int = 200

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.99995

    lambda_ue_ind: float = 1.0
    beta_bs_ind: float = 1.0


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

        self.state_dim = env.global_obs_dim

        # ----------------- UE QMIX ------------------
        self.ue_net = AgentNetwork(self.ue_obs_dim, self.ue_act_dim, cfg.hidden_dim).to(self.device)
        self.ue_tgt = AgentNetwork(self.ue_obs_dim, self.ue_act_dim, cfg.hidden_dim).to(self.device)
        self.ue_mix = MixingNetwork(self.N_ue, self.state_dim, cfg.hidden_dim).to(self.device)
        self.ue_mix_tgt = MixingNetwork(self.N_ue, self.state_dim, cfg.hidden_dim).to(self.device)

        # ----------------- BS (Qtot + Qindiv) ------------------
        self.bs_net = AgentNetwork(self.bs_obs_dim, self.bs_act_dim, cfg.hidden_dim).to(self.device)
        self.bs_tgt = AgentNetwork(self.bs_obs_dim, self.bs_act_dim, cfg.hidden_dim).to(self.device)
        self.bs_mix = MixingNetwork(self.N_bs, self.state_dim, cfg.hidden_dim).to(self.device)
        self.bs_mix_tgt = MixingNetwork(self.N_bs, self.state_dim, cfg.hidden_dim).to(self.device)

        hard_update(self.ue_tgt, self.ue_net)
        hard_update(self.ue_mix_tgt, self.ue_mix)
        hard_update(self.bs_tgt, self.bs_net)
        hard_update(self.bs_mix_tgt, self.bs_mix)

        self.opt_ue = optim.Adam(list(self.ue_net.parameters()) + list(self.ue_mix.parameters()), lr=cfg.lr, amsgrad=True)
        self.opt_bs = optim.Adam(list(self.bs_net.parameters()) + list(self.bs_mix.parameters()), lr=cfg.lr, amsgrad=True)

        # buffers (ReplayBufferRNN: full trajectories + fixed_length sample)
        self.buf_ue = ReplayBufferRNN(capacity=cfg.capacity_episodes, device=self.device)
        self.buf_bs = ReplayBufferRNN(capacity=cfg.capacity_episodes, device=self.device)

        self.eps = cfg.eps_start
        self.total_env_steps = 0

        self._ue_h = None
        self._bs_h = None
        self._ue_last_a = None
        self._bs_last_a = None

        self._cur_local_obs = None
        self._cur_global_obs = None
        self._need_env_reset = True

        self.reward_history_ue = []
        self.reward_history_bs = []
        self.step_history=[]

        self.ue_vnorm = ValueNorm(eps=1e-5, device=self.device)
        self.bs_vnorm = ValueNorm(eps=1e-5, device=self.device)
        self.bs_vnorm_vec = ValueNormVec(self.N_bs, eps=1e-5, device=self.device)
    
    def _decay_eps(self):
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    def _reset_rollout_rnn_state(self):
        self._ue_h = torch.zeros((self.N_ue, self.ue_net.hidden_dim), device=self.device)
        self._bs_h = torch.zeros((self.N_bs, self.bs_net.hidden_dim), device=self.device)
        self._ue_last_a = torch.zeros((self.N_ue, self.ue_act_dim), device=self.device)
        self._bs_last_a = torch.zeros((self.N_bs, self.bs_act_dim), device=self.device)

    def jain_fairness(self, x, eps: float = 1e-12) -> float:
        x = np.asarray(x, dtype=np.float64)
        x = np.maximum(x, 0.0)
        num = (x.sum() ** 2)
        den = (len(x) * (np.square(x).sum() + eps)) + eps
        return float(num / den)
    
    def _maybe_reset_env(self):
        if self._need_env_reset or (self._cur_local_obs is None) or (self._cur_global_obs is None):
            self._cur_local_obs, self._cur_global_obs = self.env.reset()
            self._need_env_reset = False
            self._reset_rollout_rnn_state()
    
    # -------------------------
    # Action selection (MAPPOTrainer-compatible)
    # -------------------------
    @torch.no_grad()
    def select_actions(self, local_obs: Dict[str, np.ndarray], 
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

        if self._ue_h is None or self._ue_last_a is None:
            self._ue_h = torch.zeros((self.N_ue, self.ue_net.hidden_dim), device=self.device)
            self._ue_last_a = torch.zeros((self.N_ue, self.ue_act_dim), device=self.device)

        q_ue_all, ue_h_out = self.ue_net(ue_obs_t, self._ue_last_a, his_in=self._ue_h)  # (N_ue, act_dim)
        if update_rnn_state:
            self._ue_h = ue_h_out.detach()
        q_ue_all = apply_mask_q(q_ue_all, ue_mask_t)  # (N_ue, act_dim)

        ue_actions_arr = []
        for i in range(self.N_ue):
            if random.random() < eps:
                valid_actions = np.where(ue_mask_batch[i])[0]
                a = int(np.random.choice(valid_actions))
            else:
                a = int(torch.argmax(q_ue_all[i]).item())
            ue_actions_arr.append(a)

        ue_actions = {u.ue_id: ue_actions_arr[i] for i, u in enumerate(self.users)}
        if update_rnn_state:
            ue_a_t = torch.tensor(ue_actions_arr, dtype=torch.long, device=self.device)  # (N_ue, )
            self._ue_last_a = F.one_hot(ue_a_t, num_classes=self.ue_act_dim).float()  # (N_ue, act_dim)

        # ---- BS -----
        bs_obs_batch, bs_mask_batch, cand_lists = self.env.build_bs_decision_inputs(ue_actions)
        bs_obs_t = torch.as_tensor(bs_obs_batch, dtype=torch.float32, device=self.device)  # (N_bs, obs_dim)
        bs_mask_t = torch.as_tensor(bs_mask_batch, dtype=torch.bool, device=self.device)  # (N_bs, act_dim)

        if self._bs_h is None or self._bs_last_a is None:
            self._bs_h = torch.zeros((self.N_bs, self.bs_net.hidden_dim), device=self.device)
            self._bs_last_a = torch.zeros((self.N_bs, self.bs_act_dim), device=self.device)

        q_bs_all, bs_h_out = self.bs_net(bs_obs_t, self._bs_last_a, his_in=self._bs_h)  # (N_bs, act_dim)
        if update_rnn_state:
            self._bs_h = bs_h_out.detach()
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
            bs_a_t = torch.as_tensor(bs_actions_arr, device=self.device, dtype=torch.long)
            self._bs_last_a = F.one_hot(bs_a_t, num_classes=self.bs_act_dim).float()
        return (ue_actions, ue_actions_arr, ue_obs_batch, ue_mask_batch,
                bs_actions, bs_actions_arr, bs_obs_batch, bs_mask_batch, cand_lists)
        
    # -------------------------
    # Rollout (MAPPOTrainer-style) + store to buffers
    # -------------------------
    def rollout_episode(self, n_steps: int = 200) -> Dict[str, float]:
        if n_steps is None:
            n_steps = self.cfg.chunk_len
        
        self._maybe_reset_env()
        local_obs, global_obs = self._cur_local_obs, self._cur_global_obs

        thr_sum = 0.0
        thr_last = 0.0
        fair_list = []
        on_ratio_mean_list = []
        served_sum_per_ue = None  # (N,) 누적 rate

        # UE trajectory
        ue_lo, ue_s, ue_a, ue_rtot, ue_nlo, ue_ns, ue_done = [], [], [], [], [], [], []
        ue_mask, ue_next_mask = [], []

        # BS trajectory
        bs_lo, bs_s, bs_a, bs_rtot, bs_rindiv, bs_nlo, bs_ns, bs_done = [], [], [], [], [], [], [], []
        bs_mask, bs_next_mask = [], []

        ep_r_ue = 0.0
        ep_r_bs = 0.0 
        ep_r_bs_mean = 0.0
        done_flag = False

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

            served_rates = info.get("served_rates", None)  # dict {ue_id: rate}
            if served_rates is not None:
                if served_sum_per_ue is None:
                    served_sum_per_ue = np.zeros(self.N_ue, dtype=np.float64)

                step_rates = np.zeros(self.N_ue, dtype=np.float64)
                for ue_id, r in served_rates.items():
                    idx = int(ue_id) - 1
                    if 0 <= idx < self.N_ue:
                        rr = float(r)
                        served_sum_per_ue[idx] += rr
                        step_rates[idx] = rr

                fair_list.append(self.jain_fairness(step_rates))
            
            # =========================
            # On-ratio stats
            # =========================
            power_consumed = info.get("power_consumed", None)  # dict {bs_id: power}

            if power_consumed is not None:
                on_now = np.array(
                    [1.0 if float(power_consumed[b.bs_id]) > 0.0 else 0.0 for b in self.base_stations], 
                    dtype=np.float64)
                on_ratio_mean_list.append(float(np.mean(on_now)))
            else:
                on_feats = info.get("on_feats", None)  # dict {bs_id: on_feat}
                if on_feats is not None:
                    if isinstance(on_feats, dict):
                        vals = np.asarray(list(on_feats.values()), dtype=np.float64)
                    else:
                        vals = np.asarray(on_feats, dtype=np.float64)
                    on_ratio_mean_list.append(float(np.mean(vals)))
                else:
                    on_ratio_mean_list.append(float("nan"))
                

            # reward
            rew_ue = float(info['ue_team_reward'])
            rew_bs = float(info['bs_team_reward'])
            # rew_ue_ind = np.zeros(self.N_ue, dtype=np.float32)
            # served_rates = info.get("served_rates", {})
            # for ue_id, r in served_rates.items():
            #     idx = int(ue_id) - 1
            #     if 0 <= idx < self.N_ue:
            #         rew_ue_ind[idx] = float(r)
            rew_bs_vec = np.asarray(info["bs_rewards"], dtype=np.float32)  # (N_bs, )
            assert rew_bs_vec.shape[0] == self.N_bs
            ep_r_ue += rew_ue
            ep_r_bs += rew_bs
            ep_r_bs_mean += float(np.mean(rew_bs_vec))

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
            # ue_rindiv.append(torch.tensor(rew_ue_ind, dtype=torch.float32, device="cpu"))
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
            bs_rindiv.append(torch.tensor(rew_bs_vec, dtype=torch.float32, device="cpu"))
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
                print("thr:", info.get("total_throughput", 0.0),
                      "| no_req_ratio:", round(no_req_ratio, 3),
                      "| bs_mask_true_mean:", float(np.mean(bs_masks_batch.sum(axis=1))),
                      "| bs_mask_true_min:", int(bs_masks_batch.sum(axis=1).min()),
                      "| cand_mean:", float(np.mean([len(c) for c in cand_lists])),
                      "| cand_min:", int(min(len(c) for c in cand_lists)),
                      "| mean_Q:", float(np.mean(list(Q_u.values()))) if isinstance(Q_u, dict) and len(Q_u) > 0 else 0.0,
                      "| mean_Z:", float(np.mean(list(Z_b.values()))) if isinstance(Z_b, dict) and len(Z_b) > 0 else 0.0,)
            self._decay_eps()

            if done:
                done_flag = True
                break

        T = len(ue_lo)
        if T == 0:
            return {"ep_len": 0.0, "ep_r_ue_sum": 0.0, "ep_r_bs_sum": 0.0, "epsilon": float(self.eps)}

        # stack as (T, N, dim)
        ue_lo = torch.stack(ue_lo, dim=0)
        ue_s = torch.stack(ue_s, dim=0)
        ue_a = torch.stack(ue_a, dim=0)
        # ue_rindiv = torch.stack(ue_rindiv, dim=0)
        ue_rtot = torch.stack(ue_rtot, dim=0)
        ue_nlo = torch.stack(ue_nlo, dim=0)
        ue_ns = torch.stack(ue_ns, dim=0)
        ue_done = torch.stack(ue_done, dim=0)
        ue_mask = torch.stack(ue_mask, dim=0)
        ue_next_mask = torch.stack(ue_next_mask, dim=0)

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
        if T>= self.cfg.seq_len:
            self.buf_ue.push(ue_lo, ue_s, ue_a, ue_rtot, ue_nlo, ue_ns, ue_done, None, ue_mask, ue_next_mask)
            self.buf_bs.push(bs_lo, bs_s, bs_a, bs_rtot, bs_nlo, bs_ns, bs_done, bs_rindiv, bs_mask, bs_next_mask)
        
        self._cur_local_obs, self._cur_global_obs = local_obs, global_obs

        if done_flag:
            self._need_env_reset = True
            self._cur_local_obs, self._cur_global_obs = None, None

        thr_mean = thr_sum / max(1, T)
        fair_ep = self.jain_fairness(served_sum_per_ue) if served_sum_per_ue is not None else float("nan")
        fair_mean_step = float(np.mean(fair_list)) if len(fair_list) > 0 else float("nan")
        on_ratio_mean_ep = float(np.nanmean(on_ratio_mean_list)) if len(on_ratio_mean_list) else float("nan")
        ep_r_bs_mean = ep_r_bs / max(1, T)

        return {"ep_len": float(T),
                "thr_sum": float(thr_sum),
                "thr_mean": float(thr_mean),
                "thr_last": float(thr_last),
                "fair_ep": float(fair_ep),              # 에피소드 누적 rate 기준 Jain
                "fair_mean_step": float(fair_mean_step), # 스텝별 Jain 평균(참고용)
                "on_ratio_mean": on_ratio_mean_ep,
                "ep_r_ue_sum": float(ep_r_ue),
                "ep_r_bs_sum": float(ep_r_bs),
                "ep_r_bs_mean": float(ep_r_bs_mean),
                "epsilon": float(self.eps),
            }  
    
    # -------------------------
    # UE QMIX loss (team reward)
    # -------------------------                                                                                    
    def _loss_ue_qmix(self, batch, *, update_vnorm: bool=True) -> torch.Tensor:
        obs, state, action, r_tot, next_obs, next_state, done, r_indiv, mask, next_mask= batch
        B, L, N, _ = obs.shape
        assert N == self.N_ue
        assert next_mask is not None

        agent_qs, target_qs = [], []

        for i in range(N):
            a_i = action[:, :, i]                    # (B, L)
            s_i = obs[:, :, i, :]                  # (B, L, obs)
            ns_i = next_obs[:, :, i, :]            # (B, L, obs)
            next_mask_i = next_mask[:, :, i, :]     # (B, L, act_dim) 

            h = torch.zeros(B, self.ue_net.hidden_dim, device=self.device)  # (B, hidden_dim)
            h_tgt = torch.zeros_like(h)  # (B, hidden_dim)

            q_seq, tq_seq = [], []

            for t in range(L):
                obs_t = s_i[:, t]  # (B, obs_dim)
                act_t = a_i[:, t]  # (B,)
                next_obs_t = ns_i[:, t]  # (B, obs_dim)
                next_mask_t = next_mask_i[:, t]  # (B, act_dim)

                if t==0:
                    a_prev_1hot = torch.zeros(B, self.ue_act_dim, device=self.device)
                else:
                    a_prev_1hot = one_hot(a_i[:,t-1], self.ue_act_dim)  # (B, A)
                
                # online update at time t
                q_all, h = self.ue_net(obs_t, a_prev_1hot, h)  # (B, A), (B, hidden_dim)
                q_sel = q_all.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)  # (B, )
                q_seq.append(q_sel.unsqueeze(1))  # (B, 1)

                with torch.no_grad():
                    _, h_tgt = self.ue_tgt(obs_t, a_prev_1hot, h_tgt)  # (B, A), (B, hidden_dim)
                    a_curr_1hot = one_hot(act_t, self.ue_act_dim)  # (B, A)
                    # next action selection -> online argmax(ue_net)
                    q_next_online_all, _ = self.ue_net(next_obs_t, a_curr_1hot, h.detach())  # (B, A), (B, hidden_dim)
                    q_next_online_all = apply_mask_q(q_next_online_all, next_mask_t)  # (B, A)
                    next_a = q_next_online_all.argmax(dim=-1, keepdim=True)         # (B, 1)
                    # target Q -> target eval(ue_tgt)
                    q_next_tgt_all, h_tgt_next = self.ue_tgt(next_obs_t, a_curr_1hot, h_tgt)  # (B, A), (B, hidden_dim)
                    q_next_tgt_all = apply_mask_q(q_next_tgt_all, next_mask_t)  # (B, A)
                    tq = q_next_tgt_all.gather(-1, next_a).squeeze(-1)  # (B, )
                    tq_seq.append(tq.unsqueeze(1))  # (B, 1)

                    h_tgt = h_tgt_next

            agent_qs.append(torch.cat(q_seq, dim =1))       # (B, L)
            target_qs.append(torch.cat(tq_seq, dim =1))     # (B, L)
        
        agent_qs = torch.stack(agent_qs, dim=-1)     # (B, L, N)
        target_qs = torch.stack(target_qs, dim=-1)   # (B, L, N)

        q_tot_list, tq_tot_list = [], []
        q_ind_list, tq_ind_list = [], []

        for t in range(L):
            q_tot_t, q_ind_t = self.ue_mix(agent_qs[:, t], state[:, t])
            tq_tot_t, tq_ind_t = self.ue_mix_tgt(target_qs[:, t], next_state[:, t])
            q_tot_list.append(q_tot_t)       # (B, )
            tq_tot_list.append(tq_tot_t)     # (B, )
            #q_ind_list.append(q_ind_t)       # (B, N)
            #tq_ind_list.append(tq_ind_t)     # (B, N)
        q_tot = torch.stack(q_tot_list, dim=1)     # (B, L)
        tq_tot = torch.stack(tq_tot_list, dim=1)   # (B, L)
        #q_ind = torch.stack(q_ind_list, dim=1)     # (B, L, N)
        #tq_ind = torch.stack(tq_ind_list, dim=1)   # (B, L, N)

        done_any = done[:,:,0].float()
        y_tot = r_tot + self.cfg.gamma * (1.0 - done_any) * tq_tot
        #y_ind = r_indiv + self.cfg.gamma * (1.0 - done_any.unsqueeze(-1)) * tq_ind
        if update_vnorm:
            with torch.no_grad():
                self.ue_vnorm.update(y_tot)
                #self.ue_vnorm_ind.update(y_ind)
        y_tot_n = self.ue_vnorm.normalize(y_tot)
        #y_ind_n = self.ue_vnorm_ind.normalize(y_ind)
        loss_tot = F.smooth_l1_loss(q_tot, y_tot_n.detach())
        #loss_ind = F.smooth_l1_loss(q_ind, y_ind_n.detach())

        return loss_tot #+ self.cfg.lambda_ue_ind * loss_ind
    
    # -------------------------
    # BS individual TD loss (tot + indiv reward)
    # -------------------------
    def _loss_bs_qmix(self, batch, *, update_vnorm: bool=True) -> torch.Tensor:
        obs, state, action, r_tot, next_obs, next_state, done, r_indiv, mask, next_mask = batch

        assert r_indiv is not None, "BS buffer must store individual rewards for indiv TD loss"
        B, L, Nb, _ = obs.shape
        assert Nb == self.N_bs
        assert r_indiv.shape == (B, L, Nb)
        assert next_mask is not None, "BS buffer must store next action mask for indiv TD loss"

        agent_qs, target_qs =[], []

        for j in range(Nb):
            a_j = action[:, :, j]                  # (B, L)
            s_j = obs[:, :, j, :]                  # (B, L, obs)
            ns_j = next_obs[:, :, j, :]            # (B, L, obs)
            
            h = torch.zeros(B, self.bs_net.hidden_dim, device=self.device)  # (B, hidden_dim)
            h_tgt = torch.zeros_like(h)  # (B, hidden_dim)

            q_seq, tq_seq = [], []

            for t in range(L):
                obs_t = s_j[:, t]  # (B, obs_dim)
                act_t = a_j[:, t]  # (B,)
                next_obs_t = ns_j[:, t]  # (B, obs_dim)
                next_mask_t = next_mask[:, t, j, :]  # (B, act_dim)

                if t==0:
                    a_prev_1hot = torch.zeros(B, self.bs_act_dim, device=self.device)
                else:
                    a_prev_1hot = one_hot(a_j[:,t-1], self.bs_act_dim)  # (B, A)
                
                # online
                q_all, h = self.bs_net(obs_t, a_prev_1hot, h)  # (B, A), (B, hidden_dim)
                q_all = apply_mask_q(q_all, mask[:,t,j,:]) if mask is not None else q_all
                q_sel = q_all.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)  # (B, )
                q_seq.append(q_sel)  # (B, 1)

                with torch.no_grad():
                    _, h_tgt = self.bs_tgt(obs_t, a_prev_1hot, h_tgt)  # (B, A), (B, hidden_dim)
                    a_curr_1hot = one_hot(act_t, self.bs_act_dim)  # (B, A)
                    # next action selection -> online argmax(bs_net)
                    q_next_online_all, _ = self.bs_net(next_obs_t, a_curr_1hot, h.detach())  # (B, A), (B, hidden_dim)
                    q_next_online_all = apply_mask_q(q_next_online_all, next_mask_t)  # (B, A)
                    next_a = q_next_online_all.argmax(dim=-1, keepdim=True)       # (B, 1)
                    # target Q -> target eval(bs_tgt)
                    q_next_tgt_all, h_tgt_next = self.bs_tgt(next_obs_t, a_curr_1hot, h_tgt)  # (B, A), (B, hidden_dim)
                    q_next_tgt_all = apply_mask_q(q_next_tgt_all, next_mask_t)  # (B, A)
                    tq = q_next_tgt_all.gather(-1, next_a).squeeze(-1)  # (B, )
                    tq_seq.append(tq)  # (B, 1)

                    h_tgt = h_tgt_next

            agent_qs.append(torch.stack(q_seq, dim=1))   # (B,L)
            target_qs.append(torch.stack(tq_seq, dim=1))      # (B,L)

        agent_qs = torch.stack(agent_qs, dim=-1)
        target_qs = torch.stack(target_qs, dim=-1)
        
        q_tot_list, tq_tot_list = [], []
        q_ind_list, tq_ind_list = [], []

        for t in range(L):
            q_tot_t, q_ind_t = self.bs_mix(agent_qs[:,t],state[:,t])
            tq_tot_t, tq_ind_t = self.bs_mix_tgt(target_qs[:,t], next_state[:,t])

            # q_tot_list.append(q_tot_t)
            # tq_tot_list.append(tq_tot_t)
            q_ind_list.append(q_ind_t)
            tq_ind_list.append(tq_ind_t)
        
        # q_tot = torch.stack(q_tot_list, dim=1)      # (B, L)
        # tq_tot = torch.stack(tq_tot_list, dim=1)    # (B, L)
        q_ind = torch.stack(q_ind_list, dim=1)      # (B, L, Nb)
        tq_ind = torch.stack(tq_ind_list, dim=1)    # (B, L, Nb)

        done_any = done[:, :, 0].float()            # (B, L)

        # y_tot = r_tot + self.cfg.gamma * (1.0-done_any) * tq_tot
        y_ind = r_indiv + self.cfg.gamma * (1.0-done_any).unsqueeze(-1) * tq_ind
        if update_vnorm:
            with torch.no_grad():
                self.bs_vnorm_vec.update(y_ind)
        y_ind_n = self.bs_vnorm_vec.normalize(y_ind)

        # loss_tot = F.smooth_l1_loss(q_tot, y_tot_n.detach())
        loss_ind = F.smooth_l1_loss(q_ind, y_ind_n.detach())

        return loss_ind #loss_tot + float(self.cfg.beta_bs_ind) * loss_ind
    
    # -------------------------
    # Update
    # -------------------------
    def update(self) -> Dict[str, float]:
        if len(self.buf_ue) < self.cfg.batch_size or len(self.buf_bs) < self.cfg.batch_size:
            return {}
        
        batch_ue = self.buf_ue.sample(self.cfg.batch_size, self.cfg.seq_len, use_indiv=False)
        batch_bs = self.buf_bs.sample(self.cfg.batch_size, self.cfg.seq_len, use_indiv=True)
    
        loss_ue = self._loss_ue_qmix(batch_ue, update_vnorm=True)
        self.opt_ue.zero_grad()
        loss_ue.backward()
        nn.utils.clip_grad_norm_(list(self.ue_net.parameters()) + list(self.ue_mix.parameters()), self.cfg.grad_clip)
        self.opt_ue.step()
        
        loss_bs = self._loss_bs_qmix(batch_bs, update_vnorm=True)
        self.opt_bs.zero_grad()
        loss_bs.backward()
        nn.utils.clip_grad_norm_(list(self.bs_net.parameters()) + list(self.bs_mix.parameters()), self.cfg.grad_clip)
        self.opt_bs.step()        

        soft_update(self.ue_tgt, self.ue_net, self.cfg.tau)
        soft_update(self.ue_mix_tgt, self.ue_mix, self.cfg.tau)
        soft_update(self.bs_tgt, self.bs_net, self.cfg.tau)
        soft_update(self.bs_mix_tgt, self.bs_mix, self.cfg.tau)

        return {
                "loss_ue": float(loss_ue.item()), 
                "loss_bs": float(loss_bs.item()),
                "epsilon": float(self.eps),
                "ue_v_mean": float(self.ue_vnorm.mean.item()),
                "ue_v_std": float(self.ue_vnorm.std().item()),
                "bs_v_mean": float(self.bs_vnorm_vec.mean.mean().item()),
                "bs_v_std": float(self.bs_vnorm_vec.std().mean().item()),
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
            "ue_mix": self.ue_mix.state_dict(),
            "ue_mix_tgt": self.ue_mix_tgt.state_dict(),
            "bs_net": self.bs_net.state_dict(),
            "bs_tgt": self.bs_tgt.state_dict(),
            "bs_mix": self.bs_mix.state_dict(),
            "bs_mix_tgt": self.bs_mix_tgt.state_dict(),
            "opt_ue": self.opt_ue.state_dict(),
            "opt_bs": self.opt_bs.state_dict(),
            "ue_vnorm": self.ue_vnorm.state_dict(),
            "bs_vnorm_vec": self.bs_vnorm_vec.state_dict(),
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
        self.ue_mix.load_state_dict(checkpoint["ue_mix"])
        self.ue_mix_tgt.load_state_dict(checkpoint["ue_mix_tgt"])
        self.bs_net.load_state_dict(checkpoint["bs_net"])
        self.bs_tgt.load_state_dict(checkpoint["bs_tgt"])
        self.bs_mix.load_state_dict(checkpoint["bs_mix"])
        self.bs_mix_tgt.load_state_dict(checkpoint["bs_mix_tgt"])
        self.opt_ue.load_state_dict(checkpoint["opt_ue"])
        self.opt_bs.load_state_dict(checkpoint["opt_bs"])
        self.ue_vnorm.load_state_dict(checkpoint["ue_vnorm"])
        self.bs_vnorm_vec.load_state_dict(checkpoint["bs_vnorm_vec"])
        self.eps = checkpoint.get("eps", self.cfg.eps_start)
        self.total_env_steps = checkpoint.get("total_env_steps", 0)
        print(f"[LOAD] Model loaded from {path}")