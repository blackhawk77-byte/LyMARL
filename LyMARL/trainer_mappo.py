import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
from typing import Dict, Optional

from LyMARL.networks_mappo import (
    UEActorNetwork,
    BSActorNetwork,
    CentralizedCriticUE,
    CentralizedCriticBS,
    ValueNorm,
    ValueNormVec,
)
from LyMARL.utils_mappo import moving_avg, block_avg_1d


class MAPPOTrainer:
    def __init__(
        self,
        env,
        lr_actor_ue: float = 3e-4,
        lr_actor_bs: float = 3e-4,
        lr_critic_ue: float = 1e-3,
        lr_critic_bs: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef_ue: float = 0.05,
        entropy_coef_bs: float = 0.05,
        value_coef_ue: float = 0.5,
        value_coef_bs: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        minibatch_size: int = 256,
    ):
        self.env = env
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef_ue = float(entropy_coef_ue)
        self.entropy_coef_bs = float(entropy_coef_bs)
        self.value_coef_ue = float(value_coef_ue)
        self.value_coef_bs = float(value_coef_bs)
        self.max_grad_norm = float(max_grad_norm)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actors
        self.ue_actor = UEActorNetwork(env.local_obs_dim, env.action_dim).to(self.device)
        self.ue_actor_optim = optim.Adam(self.ue_actor.parameters(), lr=lr_actor_ue)

        self.bs_actor = BSActorNetwork(env.bs_obs_dim, env.bs_action_dim).to(self.device)
        self.bs_actor_optim = optim.Adam(self.bs_actor.parameters(), lr=lr_actor_bs)

        # Critics
        self.critic_ue = CentralizedCriticUE(env.global_obs_dim).to(self.device)
        self.critic_ue_opt = optim.Adam(self.critic_ue.parameters(), lr=lr_critic_ue)

        self.critic_bs = CentralizedCriticBS(env.global_obs_dim, n_bs=env.n_bs).to(self.device)
        self.critic_bs_opt = optim.Adam(self.critic_bs.parameters(), lr=lr_critic_bs)

        # Value normalization
        self.vn_ue = ValueNorm(device=self.device)
        self.vn_bs = ValueNormVec(dim=env.n_bs, device=self.device)

        self.reset_rollout()

        print(f"[TRAINER] UE agents(shared actor): {len(env.users)}")
        print(f"[TRAINER] BS agents(shared actor): {len(env.base_stations)} | TopK={env.bs_top_k}")
        print(f"[TRAINER] Device: {self.device}")
        print(f"[TRAINER] PPO epochs: {self.n_epochs} | minibatch_size: {self.minibatch_size}")
        print(f"[TRAINER] TWO critics: UE scalar / BS vector(B={env.n_bs})")
        print(f"[TRAINER] UE action includes NO-REQUEST at index {env.no_request_action}\n")

    # =========================================================
    # Save / Load
    # =========================================================
    def save_model(self, path: str, save_optim: bool = False):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        payload = {
            "meta": {
                "local_obs_dim": self.env.local_obs_dim,
                "action_dim": self.env.action_dim,
                "bs_obs_dim": self.env.bs_obs_dim,
                "bs_action_dim": self.env.bs_action_dim,
                "global_obs_dim": self.env.global_obs_dim,
                "n_bs": self.env.n_bs,
                "n_users": self.env.n_agents,
            },
            "ue_actor": self.ue_actor.state_dict(),
            "bs_actor": self.bs_actor.state_dict(),
            "critic_ue": self.critic_ue.state_dict(),
            "critic_bs": self.critic_bs.state_dict(),
            "vn_ue": self.vn_ue.state_dict(),
            "vn_bs": self.vn_bs.state_dict(),
        }

        if save_optim:
            payload.update({
                "ue_actor_optim": self.ue_actor_optim.state_dict(),
                "bs_actor_optim": self.bs_actor_optim.state_dict(),
                "critic_ue_opt": self.critic_ue_opt.state_dict(),
                "critic_bs_opt": self.critic_bs_opt.state_dict(),
            })

        torch.save(payload, path)
        print(f"✅ Model saved: {path}")

    def load_model(self, path: str, load_optim: bool = False, map_location: Optional[str] = None):
        map_location = map_location if map_location is not None else str(self.device)
        payload = torch.load(path, map_location=map_location)

        self.ue_actor.load_state_dict(payload["ue_actor"])
        self.bs_actor.load_state_dict(payload["bs_actor"])
        self.critic_ue.load_state_dict(payload["critic_ue"])
        self.critic_bs.load_state_dict(payload["critic_bs"])
        self.vn_ue.load_state_dict(payload["vn_ue"])
        self.vn_bs.load_state_dict(payload["vn_bs"])

        if load_optim and ("ue_actor_optim" in payload):
            self.ue_actor_optim.load_state_dict(payload["ue_actor_optim"])
            self.bs_actor_optim.load_state_dict(payload["bs_actor_optim"])
            self.critic_ue_opt.load_state_dict(payload["critic_ue_opt"])
            self.critic_bs_opt.load_state_dict(payload["critic_bs_opt"])

        self.ue_actor.eval()
        self.bs_actor.eval()
        self.critic_ue.eval()
        self.critic_bs.eval()

        print(f"✅ Model loaded: {path} (optim={load_optim})")

    # =========================================================
    # Rollout buffer
    # =========================================================
    def reset_rollout(self):
        self.rb = {
            "local_obs": [],
            "ue_masks": [],
            "ue_actions": [],
            "ue_logp": [],

            "bs_obs": [],
            "bs_masks": [],
            "bs_actions": [],
            "bs_logp": [],
            "cand_lists": [],

            "global_obs": [],

            "rew_ue": [],
            "rew_bs": [],

            "v_ue_n": [],
            "nv_ue_n": [],
            "v_bs_n": [],
            "nv_bs_n": [],

            "dones": [],
        }

    @torch.no_grad()
    def select_actions(self, local_obs: Dict[int, np.ndarray], global_obs: np.ndarray):
        users = self.env.users
        global_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        v_ue_n = self.critic_ue(global_t).squeeze(0)
        v_bs_n = self.critic_bs(global_t).squeeze(0)

        # UE actions
        obs_batch = np.stack([local_obs[u.ue_id] for u in users], axis=0).astype(np.float32)
        ue_mask_batch = np.stack([self.env._get_action_mask(u.ue_id) for u in users], axis=0).astype(bool)

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        ue_mask_t = torch.as_tensor(ue_mask_batch, dtype=torch.bool, device=self.device)

        ue_logits = self.ue_actor(obs_t).masked_fill(~ue_mask_t, float("-inf"))
        ue_dist = Categorical(logits=ue_logits)
        ue_actions_t = ue_dist.sample()
        ue_logp_t = ue_dist.log_prob(ue_actions_t)
        ue_ent_t = ue_dist.entropy()

        ue_actions = {u.ue_id: int(ue_actions_t[i].item()) for i, u in enumerate(users)}

        # BS actions
        bs_obs_batch, bs_mask_batch, cand_lists = self.env.build_bs_decision_inputs(ue_actions)
        bs_obs_t = torch.as_tensor(bs_obs_batch, dtype=torch.float32, device=self.device)
        bs_mask_t = torch.as_tensor(bs_mask_batch, dtype=torch.bool, device=self.device)

        bs_logits = self.bs_actor(bs_obs_t).masked_fill(~bs_mask_t, float("-inf"))
        bs_dist = Categorical(logits=bs_logits)
        bs_actions_t = bs_dist.sample()
        bs_logp_t = bs_dist.log_prob(bs_actions_t)
        bs_ent_t = bs_dist.entropy()

        bs_actions = {bs.bs_id: int(bs_actions_t[i].item()) for i, bs in enumerate(self.env.base_stations)}

        return (
            ue_actions,
            ue_logp_t.detach().cpu().numpy().astype(np.float32),
            ue_ent_t.detach().cpu().numpy().astype(np.float32),
            ue_mask_batch,

            bs_actions,
            bs_logp_t.detach().cpu().numpy().astype(np.float32),
            bs_ent_t.detach().cpu().numpy().astype(np.float32),
            bs_obs_batch,
            bs_mask_batch,
            cand_lists,

            float(v_ue_n.item()),
            v_bs_n.detach().cpu().numpy().astype(np.float32),
        )

    def store_step(
        self,
        local_obs, global_obs,
        ue_actions_dict, ue_logp_np, ue_masks_np,
        bs_actions_dict, bs_logp_np, bs_obs_np, bs_masks_np, cand_lists,
        rew_ue: float, rew_bs: np.ndarray,
        v_ue_n: float, nv_ue_n: float,
        v_bs_n: np.ndarray, nv_bs_n: np.ndarray,
        done: bool
    ):
        users = self.env.users
        bss = self.env.base_stations
        B = len(bss)

        ue_obs_step = np.stack([local_obs[u.ue_id] for u in users], axis=0).astype(np.float32)
        ue_act_step = np.array([ue_actions_dict[u.ue_id] for u in users], dtype=np.int64)
        bs_act_step = np.array([bs_actions_dict[bs.bs_id] for bs in bss], dtype=np.int64)

        self.rb["local_obs"].append(ue_obs_step)
        self.rb["ue_masks"].append(ue_masks_np.astype(bool))
        self.rb["ue_actions"].append(ue_act_step)
        self.rb["ue_logp"].append(ue_logp_np)

        self.rb["bs_obs"].append(bs_obs_np.astype(np.float32))
        self.rb["bs_masks"].append(bs_masks_np.astype(bool))
        self.rb["bs_actions"].append(bs_act_step)
        self.rb["bs_logp"].append(bs_logp_np)
        self.rb["cand_lists"].append(cand_lists)

        self.rb["global_obs"].append(np.array(global_obs, dtype=np.float32))

        self.rb["rew_ue"].append(float(rew_ue))
        self.rb["rew_bs"].append(np.array(rew_bs, dtype=np.float32).reshape(B))

        self.rb["v_ue_n"].append(float(v_ue_n))
        self.rb["nv_ue_n"].append(float(nv_ue_n))

        self.rb["v_bs_n"].append(np.array(v_bs_n, dtype=np.float32).reshape(B))
        self.rb["nv_bs_n"].append(np.array(nv_bs_n, dtype=np.float32).reshape(B))

        self.rb["dones"].append(bool(done))

    def _iter_minibatches(self, N: int, batch_size: int):
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            yield idx[start:start + batch_size]

    # =========================================================
    # GAE
    # =========================================================
    def compute_gae_ue(self, rewards, values_n, next_values_n, dones):
        T = len(rewards)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        v_n = torch.tensor(values_n, dtype=torch.float32, device=self.device)
        nv_n = torch.tensor(next_values_n, dtype=torch.float32, device=self.device)

        v = self.vn_ue.denormalize(v_n)
        nv = self.vn_ue.denormalize(nv_n)

        adv = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            done_mask = 1.0 - float(dones[t])
            delta = r_t[t] + self.gamma * nv[t] * done_mask - v[t]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            adv[t] = gae

        ret_raw = adv + v
        return adv, ret_raw

    def compute_gae_bs(self, rewards_bs, values_bs_n, next_values_bs_n, dones):
        T = rewards_bs.shape[0]
        B = rewards_bs.shape[1]

        r = torch.tensor(rewards_bs, dtype=torch.float32, device=self.device)
        v_n = torch.tensor(values_bs_n, dtype=torch.float32, device=self.device)
        nv_n = torch.tensor(next_values_bs_n, dtype=torch.float32, device=self.device)

        v = self.vn_bs.denormalize(v_n)
        nv = self.vn_bs.denormalize(nv_n)

        adv = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        gae = torch.zeros(B, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            done_mask = 1.0 - float(dones[t])
            delta = r[t] + self.gamma * nv[t] * done_mask - v[t]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            adv[t] = gae

        ret_raw = adv + v
        return adv, ret_raw

    # =========================================================
    # PPO Update
    # =========================================================
    def update(self):
        T = len(self.rb["dones"])
        if T == 0:
            return {}

        N = len(self.env.users)
        B = len(self.env.base_stations)
        global_obs = torch.tensor(np.stack(self.rb["global_obs"], axis=0), dtype=torch.float32, device=self.device)
        dones = self.rb["dones"]

        # UE GAE
        adv_ue, ret_ue_raw = self.compute_gae_ue(
            rewards=self.rb["rew_ue"],
            values_n=self.rb["v_ue_n"],
            next_values_n=self.rb["nv_ue_n"],
            dones=dones
        )
        with torch.no_grad():
            self.vn_ue.update(ret_ue_raw)
        ret_ue_n = self.vn_ue.normalize(ret_ue_raw).detach()
        adv_ue = (adv_ue - adv_ue.mean()) / (adv_ue.std() + 1e-8)
        adv_ue = adv_ue.detach()

        # BS GAE
        rew_bs = np.stack(self.rb["rew_bs"], axis=0)
        v_bs_n = np.stack(self.rb["v_bs_n"], axis=0)
        nv_bs_n = np.stack(self.rb["nv_bs_n"], axis=0)

        adv_bs, ret_bs_raw = self.compute_gae_bs(rew_bs, v_bs_n, nv_bs_n, dones)
        with torch.no_grad():
            self.vn_bs.update(ret_bs_raw)
        ret_bs_n = self.vn_bs.normalize(ret_bs_raw).detach()
        adv_bs = (adv_bs - adv_bs.mean()) / (adv_bs.std() + 1e-8)
        adv_bs = adv_bs.detach()

        # UE tensors
        ue_local_obs = torch.tensor(np.stack(self.rb["local_obs"], axis=0), dtype=torch.float32, device=self.device)
        ue_masks = torch.tensor(np.stack(self.rb["ue_masks"], axis=0), dtype=torch.bool, device=self.device)
        ue_actions = torch.tensor(np.stack(self.rb["ue_actions"], axis=0), dtype=torch.long, device=self.device)
        ue_old_logp = torch.tensor(np.stack(self.rb["ue_logp"], axis=0), dtype=torch.float32, device=self.device)

        ue_local_f = ue_local_obs.reshape(T * N, -1)
        ue_masks_f = ue_masks.reshape(T * N, -1)
        ue_actions_f = ue_actions.reshape(T * N)
        ue_old_logp_f = ue_old_logp.reshape(T * N)
        ue_adv_f = adv_ue.repeat_interleave(N)

        # BS tensors
        bs_obs = torch.tensor(np.stack(self.rb["bs_obs"], axis=0), dtype=torch.float32, device=self.device)
        bs_masks = torch.tensor(np.stack(self.rb["bs_masks"], axis=0), dtype=torch.bool, device=self.device)
        bs_actions = torch.tensor(np.stack(self.rb["bs_actions"], axis=0), dtype=torch.long, device=self.device)
        bs_old_logp = torch.tensor(np.stack(self.rb["bs_logp"], axis=0), dtype=torch.float32, device=self.device)

        bs_obs_f = bs_obs.reshape(T * B, -1)
        bs_masks_f = bs_masks.reshape(T * B, -1)
        bs_actions_f = bs_actions.reshape(T * B)
        bs_old_logp_f = bs_old_logp.reshape(T * B)
        bs_adv_f = adv_bs.reshape(T * B)

        losses = {
            "critic_ue": 0.0, "critic_bs": 0.0,
            "actor_ue": 0.0, "actor_bs": 0.0,
            "entropy_ue": 0.0, "entropy_bs": 0.0
        }

        for _ in range(self.n_epochs):
            # Critic UE
            c_ue_epoch, c_ue_cnt = 0.0, 0
            critic_mb = max(32, min(self.minibatch_size, T))
            for mb in self._iter_minibatches(T, critic_mb):
                v_pred_n = self.critic_ue(global_obs[mb])
                loss_v = F.mse_loss(v_pred_n, ret_ue_n[mb])

                self.critic_ue_opt.zero_grad()
                (self.value_coef_ue * loss_v).backward()
                nn.utils.clip_grad_norm_(self.critic_ue.parameters(), self.max_grad_norm)
                self.critic_ue_opt.step()

                c_ue_epoch += float(loss_v.item())
                c_ue_cnt += 1

            # Critic BS
            c_bs_epoch, c_bs_cnt = 0.0, 0
            critic_mb2 = max(32, min(self.minibatch_size, T))
            for mb in self._iter_minibatches(T, critic_mb2):
                v_pred_n = self.critic_bs(global_obs[mb])
                loss_v = F.mse_loss(v_pred_n, ret_bs_n[mb])

                self.critic_bs_opt.zero_grad()
                (self.value_coef_bs * loss_v).backward()
                nn.utils.clip_grad_norm_(self.critic_bs.parameters(), self.max_grad_norm)
                self.critic_bs_opt.step()

                c_bs_epoch += float(loss_v.item())
                c_bs_cnt += 1

            # UE actor
            ue_epoch, ue_ent_epoch, ue_cnt = 0.0, 0.0, 0
            M_ue = T * N
            ue_mb = max(64, min(self.minibatch_size, M_ue))
            for mb in self._iter_minibatches(M_ue, ue_mb):
                logits = self.ue_actor(ue_local_f[mb]).masked_fill(~ue_masks_f[mb], float("-inf"))
                dist = Categorical(logits=logits)

                new_logp = dist.log_prob(ue_actions_f[mb])
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - ue_old_logp_f[mb])
                surr1 = ratio * ue_adv_f[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * ue_adv_f[mb]
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_ent = -entropy.mean()

                self.ue_actor_optim.zero_grad()
                (loss_pi + self.entropy_coef_ue * loss_ent).backward()
                nn.utils.clip_grad_norm_(self.ue_actor.parameters(), self.max_grad_norm)
                self.ue_actor_optim.step()

                ue_epoch += float(loss_pi.item())
                ue_ent_epoch += float(loss_ent.item())
                ue_cnt += 1

            # BS actor
            bs_epoch, bs_ent_epoch, bs_cnt = 0.0, 0.0, 0
            M_bs = T * B
            bs_mb = max(64, min(self.minibatch_size, M_bs))
            for mb in self._iter_minibatches(M_bs, bs_mb):
                logits = self.bs_actor(bs_obs_f[mb]).masked_fill(~bs_masks_f[mb], float("-inf"))
                dist = Categorical(logits=logits)

                new_logp = dist.log_prob(bs_actions_f[mb])
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - bs_old_logp_f[mb])
                surr1 = ratio * bs_adv_f[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * bs_adv_f[mb]
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_ent = -entropy.mean()

                self.bs_actor_optim.zero_grad()
                (loss_pi + self.entropy_coef_bs * loss_ent).backward()
                nn.utils.clip_grad_norm_(self.bs_actor.parameters(), self.max_grad_norm)
                self.bs_actor_optim.step()

                bs_epoch += float(loss_pi.item())
                bs_ent_epoch += float(loss_ent.item())
                bs_cnt += 1

            losses["critic_ue"] += c_ue_epoch / max(1, c_ue_cnt)
            losses["critic_bs"] += c_bs_epoch / max(1, c_bs_cnt)
            losses["actor_ue"] += ue_epoch / max(1, ue_cnt)
            losses["entropy_ue"] += ue_ent_epoch / max(1, ue_cnt)
            losses["actor_bs"] += bs_epoch / max(1, bs_cnt)
            losses["entropy_bs"] += bs_ent_epoch / max(1, bs_cnt)

        for k in losses:
            losses[k] /= self.n_epochs

        self.reset_rollout()
        return losses

    # =========================================================
    # Train / Eval
    # =========================================================
    def train(self, 
              n_episodes: int, 
              steps_per_episode: int, 
              update_interval: int = 128, 
              queue_reset_interval: Optional[int] = None,
              save_npz_path: Optional[str] = None
              ):
        print(f"\n{'='*100}")
        print(" Hetero-MAPPO Training")
        print(f"{'='*100}")
        print(f"Total train episodes: {n_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        print(f"Total train steps: {n_episodes * steps_per_episode}")
        print(f"Queue reset interval: {queue_reset_interval}")
        print(f"Update interval: {update_interval}")
        print(f"Hard constraint during training: {self.env.use_hard_constraint}")
        print(f"{'='*100}\n")

        throughput_history = []
        fairness_history = []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        slot_rates = []
        queue_history = {"Q_u": defaultdict(list), "Z_b": defaultdict(list)}

        ue_team_reward_hist = []
        ue_per_user_reward_hist = []
        bs_reward_vec_hist = []
        bs_reward_mean_hist = []

        global_step =0

        for ep in range(n_episodes):
            local_obs, global_obs = self.env.reset()

            for ep_step in range(steps_per_episode):

                (ue_actions, ue_logp_np, ue_ent_np, ue_masks_np,
                bs_actions, bs_logp_np, bs_ent_np, bs_obs_np, bs_masks_np, cand_lists,
                v_ue_n, v_bs_n_np) = self.select_actions(local_obs, global_obs)

                next_local_obs, next_global_obs, info, done = self.env.step_joint(
                    ue_actions=ue_actions,
                    bs_actions=bs_actions,
                    cand_lists=cand_lists
                )

                with torch.no_grad():
                    next_global_t = torch.as_tensor(next_global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    nv_ue_n = float(self.critic_ue(next_global_t).squeeze(0).item())
                    nv_bs_n = self.critic_bs(next_global_t).squeeze(0).detach().cpu().numpy().astype(np.float32)

                rew_ue = float(info["ue_team_reward"])
                rew_bs = np.array(info["bs_rewards"], dtype=np.float32).reshape(-1)

                queue_reset_now = (
                    queue_reset_interval is not None
                    and queue_reset_interval > 0
                    and (ep_step + 1) % queue_reset_interval == 0
                    and (ep_step + 1) < steps_per_episode
                )
                done_to_store = done or (ep_step == steps_per_episode - 1)
                
                self.store_step(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    ue_actions_dict=ue_actions,
                    ue_logp_np=ue_logp_np,
                    ue_masks_np=ue_masks_np,
                    bs_actions_dict=bs_actions,
                    bs_logp_np=bs_logp_np,
                    bs_obs_np=bs_obs_np,
                    bs_masks_np=bs_masks_np,
                    cand_lists=cand_lists,
                    rew_ue=rew_ue,
                    rew_bs=rew_bs,
                    v_ue_n=float(v_ue_n),
                    nv_ue_n=float(nv_ue_n),
                    v_bs_n=v_bs_n_np,
                    nv_bs_n=nv_bs_n,
                    done=done_to_store
                )

                throughput_history.append(info["total_throughput"])
                rates_this_slot = [info["served_rates"][u.ue_id] for u in self.env.users]
                slot_rates.append(rates_this_slot)
                fairness_history.append(self.env.calculate_jain_fairness(slot_rates))

                for bs_id, power in info["power_consumed"].items():
                    power_history[bs_id].append(power)

                for ue_id, q_val in info["Q_u"].items():
                    queue_history["Q_u"][ue_id].append(q_val)

                for bs_id, zb_val in info["Z_b"].items():
                    queue_history["Z_b"][bs_id].append(zb_val)

                ue_team_reward_hist.append(rew_ue)
                ue_per_user_reward_hist.append([float(info["ue_per_user_rewards"][u.ue_id]) for u in self.env.users])
                bs_reward_vec_hist.append(rew_bs.tolist())
                bs_reward_mean_hist.append(float(rew_bs.mean()))

                local_obs, global_obs = next_local_obs, next_global_obs
                
                if queue_reset_now:
                    local_obs, global_obs = self.env.reset_queues()
                    print(f"[QUEUE RESET] Ep {ep+1} | EpStep {ep_step+1} | "
                    f"GlobalStep {global_step+1}",
                    flush=True)

                if (global_step + 1) % update_interval == 0:
                    losses = self.update()
                    if losses:
                        print(
                            f"[UPDATE] Global Step {global_step+1} | "
                            f"UE_Actor:{losses['actor_ue']:.4f} | BS_Actor:{losses['actor_bs']:.4f} | "
                            f"C_UE:{losses['critic_ue']:.4f} | C_BS:{losses['critic_bs']:.4f} | "
                            f"Ent(UE):{losses['entropy_ue']:.4f} | Ent(BS):{losses['entropy_bs']:.4f}"
                        )

                if (global_step + 1) % 100 == 0:
                    recent_thr = float(np.mean(throughput_history[-100:]))
                    recent_fair = float(fairness_history[-1])
                    ue_team_rew_100 = float(np.mean(ue_team_reward_hist[-100:]))
                    bs_team_rew_100 = float(np.mean(bs_reward_mean_hist[-100:]))
                    no_req_cnt = sum(1 for a in ue_actions.values() if int(a) == self.env.no_request_action)

                    on_parts = []
                    for bs in self.env.base_stations:
                        hist = list(self.env.bs_on_hist[bs.bs_id])
                        on_ratio_100 = float(np.mean(hist[-100:])) if len(hist) > 0 else 0.0
                        on_parts.append(f"BS{bs.bs_id}:{on_ratio_100:.3f}")
                    on_str = " ".join(on_parts)

                    print(
                        f"Global Step {global_step+1:5d} | Ep {ep+1:3d} | EpStep {ep_step+1:4d} | "
                        f"Thr:{recent_thr:.3f} | Fair:{recent_fair:.3f} | "
                        f"ON(100): {on_str} | NO-REQ:{no_req_cnt}/{self.env.n_agents} | "
                        f"UETeamRew(100):{ue_team_rew_100:.3f} | BSTeamRew(100):{bs_team_rew_100:.3f}"
                    )

                global_step += 1

                if done_to_store:
                    print("\n" + "-" * 80)
                    print(f"[EP END] Episode {ep+1}/{n_episodes} | GlobalStep {global_step} | EpStep {ep_step+1}")

                    print("Final Q_u:")
                    for ue_id, q_val in info["Q_u"].items():
                        print(f"  UE{ue_id}: {q_val:.4f}")

                    print("Final Z_b:")
                    for bs_id, z_val in info["Z_b"].items():
                        print(f"  BS{bs_id}: {z_val:.4f}")

                    print("-" * 80 + "\n", flush=True)
                    break

        # 남은 rollout이 있으면 마지막 update
        if len(self.rb["dones"]) > 0:
            losses = self.update()
            if losses:
                print(
                    f"[FINAL UPDATE] GlobalStep {global_step} | "
                    f"UE_Actor:{losses['actor_ue']:.4f} | BS_Actor:{losses['actor_bs']:.4f} | "
                    f"C_UE:{losses['critic_ue']:.4f} | C_BS:{losses['critic_bs']:.4f}"
                )

        results = {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "queue_history": queue_history,

            "ue_team_reward": ue_team_reward_hist,
            "ue_per_user_reward": ue_per_user_reward_hist,
            "bs_reward_vec": bs_reward_vec_hist,
            "bs_reward_mean": bs_reward_mean_hist,
        }

        if save_npz_path is not None:
            self.save_results_npz(results, save_npz_path, tag="train")

        return results

    @torch.no_grad()
    def evaluate(self, 
                 n_episodes: int, 
                 steps_per_episode: int,
                 queue_reset_interval: Optional[int] = None,
                 save_npz_path: Optional[str] = None):
        print(f"\n{'='*84}")
        print(" EVALUATION (No Learning)")
        print(f"{'='*84}")
        print(f"Total eval episodes: {n_episodes}")
        print(f"Hard constraint during evaluation: {self.env.use_hard_constraint}\n")

        self.ue_actor.eval()
        self.bs_actor.eval()
        self.critic_ue.eval()
        self.critic_bs.eval()

        throughput_history = []
        fairness_history = []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        slot_rates = []

        ue_team_reward_hist = []
        ue_per_user_reward_hist = []
        bs_reward_vec_hist = []
        bs_reward_mean_hist = []

        eval_on100_hist = {bs.bs_id: [] for bs in self.env.base_stations}

        global_step = 0

        for ep in range(n_episodes):
            local_obs, global_obs = self.env.reset()

            for ep_step in range(steps_per_episode):
                (ue_actions, ue_logp_np, ue_ent_np, ue_masks_np,
                bs_actions, bs_logp_np, bs_ent_np, bs_obs_np, bs_masks_np, cand_lists,
                v_ue_n, v_bs_n_np) = self.select_actions(local_obs, global_obs)

                next_local_obs, next_global_obs, info, done = self.env.step_joint(
                    ue_actions=ue_actions,
                    bs_actions=bs_actions,
                    cand_lists=cand_lists
                )

                done_to_stop = done or (ep_step == steps_per_episode - 1)
                
                throughput_history.append(info["total_throughput"])
                rates_this_slot = [info["served_rates"][u.ue_id] for u in self.env.users]
                slot_rates.append(rates_this_slot)
                fairness_history.append(self.env.calculate_jain_fairness(slot_rates))

                for bs_id, power in info["power_consumed"].items():
                    power_history[bs_id].append(power)

                ue_team_reward_hist.append(float(info["ue_team_reward"]))
                ue_per_user_reward_hist.append([float(info["ue_per_user_rewards"][u.ue_id]) for u in self.env.users])

                bs_vec = np.array(info["bs_rewards"], dtype=np.float32).reshape(-1)
                bs_reward_vec_hist.append(bs_vec.tolist())
                bs_reward_mean_hist.append(float(np.mean(bs_vec)))

                local_obs, global_obs = next_local_obs, next_global_obs
                
                if queue_reset_interval is not None \
                    and queue_reset_interval > 0 \
                        and (ep_step + 1) % queue_reset_interval == 0 \
                            and (ep_step + 1) < steps_per_episode:
                    local_obs, global_obs = self.env.reset_queues()

                if (global_step + 1) % 100 == 0:
                    recent_thr = float(np.mean(throughput_history[-100:]))
                    recent_fair = float(fairness_history[-1])
                    no_req_cnt = sum(1 for a in ue_actions.values() if int(a) == self.env.no_request_action)

                    on_parts = []
                    for bs in self.env.base_stations:
                        hist = list(self.env.bs_on_hist[bs.bs_id])
                        on_ratio_100 = float(np.mean(hist[-100:])) if len(hist) > 0 else 0.0
                        eval_on100_hist[bs.bs_id].append(on_ratio_100)
                        on_parts.append(f"BS{bs.bs_id}:{on_ratio_100:.3f}")
                    on_str = " ".join(on_parts)

                    print(
                        f"[EVAL] Global Step {global_step+1:5d} | Thr:{recent_thr:.3f} | Fair:{recent_fair:.3f} | "
                        f"ON(100): {on_str} | NO-REQ:{no_req_cnt}/{self.env.n_agents}"
                    )

                if (global_step + 1) % 10000 == 0:
                    thr_10k_mean = float(np.mean(throughput_history[-10000:]))
                    fair_10k_mean = float(np.mean(fairness_history[-10000:]))

                    on10k_parts = []
                    n_blocks_10k = max(1, 10000 // 100)
                    for bs in self.env.base_stations:
                        recent_on100 = eval_on100_hist[bs.bs_id][-n_blocks_10k:]
                        on10k_mean = float(np.mean(recent_on100)) if len(recent_on100) > 0 else 0.0
                        on10k_parts.append(f"BS{bs.bs_id}:{on10k_mean:.3f}")
                    on10k_str = " ".join(on10k_parts)

                    print(
                        f"[EVAL-10K] Global Step {global_step+1:5d} | "
                        f"ThroughputMean(10k):{thr_10k_mean:.3f} | "
                        f"Mean(step-wise Fair(100) over 10k):{fair_10k_mean:.3f} | "
                        f"ON100-Mean(10k): {on10k_str}"
                    )
                global_step += 1
                if done_to_stop:
                    break

        results = {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "ue_team_reward": ue_team_reward_hist,
            "ue_per_user_reward": ue_per_user_reward_hist,
            "bs_reward_vec": bs_reward_vec_hist,
            "bs_reward_mean": bs_reward_mean_hist,
        }

        if save_npz_path is not None:
            self.save_results_npz(results, save_npz_path, tag="eval")

        return results

    # =========================================================
    # NPZ save
    # =========================================================
    def save_results_npz(self, results: Dict, npz_path: str, tag: str = "run"):
        os.makedirs(os.path.dirname(npz_path) if os.path.dirname(npz_path) else ".", exist_ok=True)

        thr = np.asarray(results.get("throughput_history", []), dtype=np.float32)
        fair = np.asarray(results.get("fairness_history", []), dtype=np.float32)

        ue_team = np.asarray(results.get("ue_team_reward", []), dtype=np.float32)
        ue_per_user = np.asarray(results.get("ue_per_user_reward", []), dtype=np.float32)
        bs_mean = np.asarray(results.get("bs_reward_mean", []), dtype=np.float32)
        bs_vec = np.asarray(results.get("bs_reward_vec", []), dtype=np.float32)

        if ue_per_user.ndim == 2 and ue_per_user.shape[0] > 0:
            mean_user_reward_step = ue_per_user.mean(axis=1).astype(np.float32)
            mean_user_reward_ma100 = moving_avg(mean_user_reward_step, 100)
        else:
            mean_user_reward_step = np.asarray([], dtype=np.float32)
            mean_user_reward_ma100 = np.asarray([], dtype=np.float32)

        # block average for reward plotting
        block = 500
        reward_x_500, user_mean_reward_500 = (
            block_avg_1d(mean_user_reward_step, block)
            if mean_user_reward_step.size > 0 else
            (np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32))
        )
        bs_reward_x_500, bs_mean_reward_500 = (
            block_avg_1d(bs_mean, block)
            if bs_mean.size > 0 else
            (np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32))
        )

        power_hist = results.get("power_history", {})
        bs_ids_sorted = sorted(list(power_hist.keys())) if isinstance(power_hist, dict) else []

        power_mat = []
        for bs_id in bs_ids_sorted:
            power_mat.append(np.asarray(power_hist[bs_id], dtype=np.float32))
        power_mat = np.stack(power_mat, axis=0) if len(power_mat) > 0 else np.zeros((0, len(thr)), dtype=np.float32)

        np.savez_compressed(
            npz_path,
            tag=str(tag),
            n_users=int(self.env.n_agents),
            n_bs=int(self.env.n_bs),

            throughput=thr,
            fairness=fair,

            ue_team_reward=ue_team,
            ue_team_reward_step=ue_team,
            ue_per_user_reward=ue_per_user,
            mean_user_reward_step=mean_user_reward_step,
            mean_user_reward_ma100=mean_user_reward_ma100,

            bs_reward_mean=bs_mean,
            bs_reward_mean_step=bs_mean,
            bs_reward_vec=bs_vec,
            bs_reward_vec_step=bs_vec,

            reward_x_500=reward_x_500,
            user_mean_reward_500=user_mean_reward_500,
            bs_reward_x_500=bs_reward_x_500,
            bs_mean_reward_500=bs_mean_reward_500,

            bs_ids=np.asarray(bs_ids_sorted, dtype=np.int32),
            power_mat=power_mat,
        )
        print(f"✅ Saved results npz: {npz_path}")