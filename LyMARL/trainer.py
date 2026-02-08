# trainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from typing import Dict, List

from collections import defaultdict

# local imports: works both as "same-folder" run and as package import
try:
    from norms import ValueNorm, ValueNormVec
    from networks import UEActorNetwork, BSActorNetwork, CentralizedCriticUE, CentralizedCriticBS
    from env import MAPPOEnvironment
except Exception:
    from .norms import ValueNorm, ValueNormVec
    from .networks import UEActorNetwork, BSActorNetwork, CentralizedCriticUE, CentralizedCriticBS
    from .env import MAPPOEnvironment


class MAPPOTrainer:
    def __init__(self,
                 env: MAPPOEnvironment,
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
                 minibatch_size: int = 256):

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

        # ValueNorm
        self.vn_ue = ValueNorm(device=self.device)
        self.vn_bs = ValueNormVec(dim=env.n_bs, device=self.device)

        self.reset_rollout()

        print(f"[TRAINER] UE agents (shared actor): {len(env.users)}")
        print(f"[TRAINER] BS agents (shared actor): {len(env.base_stations)} | TopK={env.bs_top_k}")
        print(f"[TRAINER] Device: {self.device}")
        print(f"[TRAINER] PPO epochs: {self.n_epochs} | minibatch_size: {self.minibatch_size}")
        print(f"[TRAINER] TWO critics: UE(scalar), BS(vector of size B={env.n_bs})")
        print(f"[TRAINER] ✅ UE action includes NO-REQUEST at index {env.no_request_action}\n")

    def reset_rollout(self):
        self.rb = {
            # UE
            "local_obs": [],
            "ue_masks": [],
            "ue_actions": [],
            "ue_logp": [],

            # BS
            "bs_obs": [],
            "bs_masks": [],
            "bs_actions": [],
            "bs_logp": [],
            "cand_lists": [],

            # team/global
            "global_obs": [],

            # rewards
            "rew_ue": [],
            "rew_bs": [],

            # values (normalized)
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

        # critics: UE scalar, BS vector
        v_ue_n = self.critic_ue(global_t).squeeze(0)     # []
        v_bs_n = self.critic_bs(global_t).squeeze(0)     # [B]

        # ---------------- UE actions ----------------
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

        # ---------------- BS actions ----------------
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
            v_bs_n.detach().cpu().numpy().astype(np.float32),  # [B]
        )

    def store_step(self, local_obs, global_obs,
                   ue_actions_dict, ue_logp_np, ue_masks_np,
                   bs_actions_dict, bs_logp_np, bs_obs_np, bs_masks_np, cand_lists,
                   rew_ue: float, rew_bs: np.ndarray,
                   v_ue_n: float, nv_ue_n: float,
                   v_bs_n: np.ndarray, nv_bs_n: np.ndarray,
                   done: bool):

        users = self.env.users
        bss = self.env.base_stations
        N = len(users)
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

    def compute_gae_ue(self, rewards, values_n, next_values_n, dones):
        T = len(rewards)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)           # [T]
        v_n = torch.tensor(values_n, dtype=torch.float32, device=self.device)         # [T]
        nv_n = torch.tensor(next_values_n, dtype=torch.float32, device=self.device)   # [T]

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
        return adv, ret_raw  # [T], [T]

    def compute_gae_bs(self, rewards_bs, values_bs_n, next_values_bs_n, dones):
        """
        rewards_bs: [T, B]
        values_bs_n: [T, B]
        """
        T = rewards_bs.shape[0]
        B = rewards_bs.shape[1]

        r = torch.tensor(rewards_bs, dtype=torch.float32, device=self.device)            # [T,B]
        v_n = torch.tensor(values_bs_n, dtype=torch.float32, device=self.device)         # [T,B]
        nv_n = torch.tensor(next_values_bs_n, dtype=torch.float32, device=self.device)   # [T,B]

        v = self.vn_bs.denormalize(v_n)   # [T,B]
        nv = self.vn_bs.denormalize(nv_n)

        adv = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        gae = torch.zeros(B, dtype=torch.float32, device=self.device)
        for t in reversed(range(T)):
            done_mask = 1.0 - float(dones[t])
            delta = r[t] + self.gamma * nv[t] * done_mask - v[t]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            adv[t] = gae
        ret_raw = adv + v
        return adv, ret_raw  # [T,B], [T,B]

    def update(self):
        T = len(self.rb["dones"])
        if T == 0:
            return {}

        N = len(self.env.users)
        B = len(self.env.base_stations)

        global_obs = torch.tensor(np.stack(self.rb["global_obs"], axis=0), dtype=torch.float32, device=self.device)  # [T, G]
        dones = self.rb["dones"]

        # --- UE GAE ---
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
        adv_ue = adv_ue.detach()  # [T]

        # --- BS GAE ---
        rew_bs = np.stack(self.rb["rew_bs"], axis=0)          # [T,B]
        v_bs_n = np.stack(self.rb["v_bs_n"], axis=0)          # [T,B]
        nv_bs_n = np.stack(self.rb["nv_bs_n"], axis=0)        # [T,B]
        adv_bs, ret_bs_raw = self.compute_gae_bs(rew_bs, v_bs_n, nv_bs_n, dones)
        with torch.no_grad():
            self.vn_bs.update(ret_bs_raw)
        ret_bs_n = self.vn_bs.normalize(ret_bs_raw).detach()  # [T,B]
        adv_bs = (adv_bs - adv_bs.mean()) / (adv_bs.std() + 1e-8)
        adv_bs = adv_bs.detach()  # [T,B]

        # UE tensors
        ue_local_obs = torch.tensor(np.stack(self.rb["local_obs"], axis=0), dtype=torch.float32, device=self.device)  # [T,N,obs]
        ue_masks = torch.tensor(np.stack(self.rb["ue_masks"], axis=0), dtype=torch.bool, device=self.device)          # [T,N,A]
        ue_actions = torch.tensor(np.stack(self.rb["ue_actions"], axis=0), dtype=torch.long, device=self.device)      # [T,N]
        ue_old_logp = torch.tensor(np.stack(self.rb["ue_logp"], axis=0), dtype=torch.float32, device=self.device)     # [T,N]

        ue_local_f = ue_local_obs.reshape(T * N, -1)
        ue_masks_f = ue_masks.reshape(T * N, -1)
        ue_actions_f = ue_actions.reshape(T * N)
        ue_old_logp_f = ue_old_logp.reshape(T * N)
        ue_adv_f = adv_ue.repeat_interleave(N)  # [T*N]

        # BS tensors
        bs_obs = torch.tensor(np.stack(self.rb["bs_obs"], axis=0), dtype=torch.float32, device=self.device)           # [T,B,obs]
        bs_masks = torch.tensor(np.stack(self.rb["bs_masks"], axis=0), dtype=torch.bool, device=self.device)         # [T,B,A]
        bs_actions = torch.tensor(np.stack(self.rb["bs_actions"], axis=0), dtype=torch.long, device=self.device)     # [T,B]
        bs_old_logp = torch.tensor(np.stack(self.rb["bs_logp"], axis=0), dtype=torch.float32, device=self.device)    # [T,B]

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
            # ---------------- Critic UE update ----------------
            c_ue_epoch, c_ue_cnt = 0.0, 0
            critic_bs = max(32, min(self.minibatch_size, T))
            for mb in self._iter_minibatches(T, critic_bs):
                v_pred_n = self.critic_ue(global_obs[mb])          # [mb]
                loss_v = F.mse_loss(v_pred_n, ret_ue_n[mb])

                self.critic_ue_opt.zero_grad()
                (self.value_coef_ue * loss_v).backward()
                nn.utils.clip_grad_norm_(self.critic_ue.parameters(), self.max_grad_norm)
                self.critic_ue_opt.step()

                c_ue_epoch += float(loss_v.item())
                c_ue_cnt += 1

            # ---------------- Critic BS update ----------------
            c_bs_epoch, c_bs_cnt = 0.0, 0
            critic_bs2 = max(32, min(self.minibatch_size, T))
            for mb in self._iter_minibatches(T, critic_bs2):
                v_pred_n = self.critic_bs(global_obs[mb])          # [mb, B]
                loss_v = F.mse_loss(v_pred_n, ret_bs_n[mb])

                self.critic_bs_opt.zero_grad()
                (self.value_coef_bs * loss_v).backward()
                nn.utils.clip_grad_norm_(self.critic_bs.parameters(), self.max_grad_norm)
                self.critic_bs_opt.step()

                c_bs_epoch += float(loss_v.item())
                c_bs_cnt += 1

            # ---------------- UE actor update ----------------
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

            # ---------------- BS actor update ----------------
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

    def train(self, n_steps: int, update_interval: int = 128):
        print(f"\n{'='*100}")
        print(f" Hetero-MAPPO (UE actor + BS actor) with TWO rewards & TWO critics")
        print(f"{'='*100}")
        print(f" Total train steps: {n_steps}")
        print(f" Update interval: {update_interval}")
        print(f" ✅ UE reward(team): total_rate / N")
        print(f" ✅ BS reward(per BS): -c*(on_ratio - rho)^2  (c={self.env.bs_over_penalty})")
        print(f" ✅ UE action includes NO-REQUEST at index {self.env.no_request_action}")
        print(f"{'='*100}\n")

        throughput_history = []
        fairness_history = []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        slot_rates = []

        # ✅ renamed
        queue_history = {"Q_u": defaultdict(list), "Z_b": defaultdict(list)}
        reward_hist = {"ue": [], "bs": []}

        local_obs, global_obs = self.env.reset()

        for step in range(n_steps):
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
            rew_bs = info["bs_rewards"]

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
                done=done
            )

            # metrics
            throughput_history.append(info["total_throughput"])
            rates_this_slot = [info["served_rates"][u.ue_id] for u in self.env.users]
            slot_rates.append(rates_this_slot)
            fairness_history.append(self.env.calculate_jain_fairness(slot_rates))

            for bs_id, power in info["power_consumed"].items():
                power_history[bs_id].append(power)

            # ✅ renamed
            for ue_id, q_val in info["Q_u"].items():
                queue_history["Q_u"][ue_id].append(q_val)
            for bs_id, zb_val in info["Z_b"].items():
                queue_history["Z_b"][bs_id].append(zb_val)

            reward_hist["ue"].append(rew_ue)
            reward_hist["bs"].append(float(np.mean(rew_bs)))

            local_obs, global_obs = next_local_obs, next_global_obs

            if (step + 1) % update_interval == 0:
                losses = self.update()
                if losses:
                    print(f"[UPDATE] Step {step+1} | "
                          f"UE_Actor:{losses['actor_ue']:.4f} | BS_Actor:{losses['actor_bs']:.4f} | "
                          f"C_UE:{losses['critic_ue']:.4f} | C_BS:{losses['critic_bs']:.4f} | "
                          f"Ent(UE):{losses['entropy_ue']:.4f} | Ent(BS):{losses['entropy_bs']:.4f} | "
                          f"VN_UE(mean={self.vn_ue.mean.item():.3f}, std={self.vn_ue.std().item():.3f}) | "
                          f"VN_BS(mean≈{self.vn_bs.mean.mean().item():.3f}, std≈{self.vn_bs.std().mean().item():.3f})")

            if (step + 1) % 100 == 0:
                recent_thr = float(np.mean(throughput_history[-100:]))
                recent_fair = float(fairness_history[-1])

                on_ratios = {}
                for bs in self.env.base_stations:
                    recent_power = power_history[bs.bs_id][-100:]
                    on_ratios[bs.bs_id] = (sum(1 for p in recent_power if p > 0) / len(recent_power)) if len(recent_power) else 0.0

                avg_Q = float(np.mean([queue_history["Q_u"][u.ue_id][-1] for u in self.env.users]))
                avg_Zb = float(np.mean([queue_history["Z_b"][bs.bs_id][-1] for bs in self.env.base_stations]))
                recent_rew_ue = float(np.mean(reward_hist["ue"][-100:]))
                recent_rew_bs = float(np.mean(reward_hist["bs"][-100:]))

                ratio_str = ", ".join([f"BS{b}:{r:.2f}" for b, r in on_ratios.items()])
                cong_str = ", ".join([f"BS{b}:{info['prev_req_ratio'][b]:.2f}" for b in on_ratios.keys()])
                cong_sum = sum(info["prev_req_ratio"].values())
                used_str = ", ".join([f"BS{b}:{info['bs_on_used_in_window'][b]}" for b in on_ratios.keys()])

                # how many chose NO-REQUEST in this step (debug)
                no_req_cnt = sum(1 for a in ue_actions.values() if int(a) == self.env.no_request_action)

                print(f"Step {step+1:5d} | Thr:{recent_thr:.3f} | Fair:{recent_fair:.3f} | "
                      f"NO-REQ:{no_req_cnt}/{self.env.n_agents} | "
                      f"ON:[{ratio_str}] | ReqRatio(t)->Obs(t+1):[{cong_str}] (sum={cong_sum:.2f}) | "
                      f"UsedInWindow:[{used_str}] (t_in_window={info['window_step']%self.env.hard_window_len}) | "
                      f"Q̄:{avg_Q:.2f} | Z̄b:{avg_Zb:.4f} | "
                      f"R_UE:{recent_rew_ue:.3f} | R_BS(avg):{recent_rew_bs:.3f} | "
                      f"rho={info['rho']:.2f} | NoCovCnt:{info['no_coverage_count']}")

        return {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "queue_history": queue_history,
            "reward_hist": reward_hist
        }

    def evaluate(self, n_steps: int):
        print(f"\n{'='*84}")
        print(f"  EVALUATION (No Learning) - UE+BS policies frozen")
        print(f"{'='*84}")
        print(f"  Total eval steps: {n_steps}\n")

        throughput_history = []
        fairness_history = []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        slot_rates = []
        reward_ue_hist, reward_bs_hist = [], []

        fair100_x, fair100_y = [], []

        local_obs, global_obs = self.env.reset()

        for step in range(n_steps):
            (ue_actions, ue_logp_np, ue_ent_np, ue_masks_np,
             bs_actions, bs_logp_np, bs_ent_np, bs_obs_np, bs_masks_np, cand_lists,
             v_ue_n, v_bs_n_np) = self.select_actions(local_obs, global_obs)

            next_local_obs, next_global_obs, info, done = self.env.step_joint(
                ue_actions=ue_actions,
                bs_actions=bs_actions,
                cand_lists=cand_lists
            )

            throughput_history.append(info["total_throughput"])
            rates_this_slot = [info["served_rates"][u.ue_id] for u in self.env.users]
            slot_rates.append(rates_this_slot)

            fairness_history.append(self.env.calculate_jain_fairness(slot_rates))

            for bs_id, power in info["power_consumed"].items():
                power_history[bs_id].append(power)

            reward_ue_hist.append(float(info["ue_team_reward"]))
            reward_bs_hist.append(float(np.mean(info["bs_rewards"])))

            local_obs, global_obs = next_local_obs, next_global_obs

            if (step + 1) % 100 == 0:
                recent_thr = float(np.mean(throughput_history[-100:]))
                recent_fair = float(fairness_history[-1])

                fair100_x.append(step + 1)
                fair100_y.append(recent_fair)

                on_ratios = {}
                for bs in self.env.base_stations:
                    recent_power = power_history[bs.bs_id][-100:]
                    on_ratios[bs.bs_id] = (sum(1 for p in recent_power if p > 0) / len(recent_power)) if len(recent_power) else 0.0
                ratio_str = ", ".join([f"BS{b}:{r:.2f}" for b, r in on_ratios.items()])

                no_req_cnt = sum(1 for a in ue_actions.values() if int(a) == self.env.no_request_action)

                print(f"[EVAL] Step {step+1:5d} | Thr:{recent_thr:.3f} | Fair:{recent_fair:.3f} | "
                      f"NO-REQ:{no_req_cnt}/{self.env.n_agents} | "
                      f"ON:[{ratio_str}] | R_UE:{np.mean(reward_ue_hist[-100:]):.3f} | "
                      f"R_BS(avg):{np.mean(reward_bs_hist[-100:]):.3f}")

        if len(fair100_y) > 0:
            plt.figure(figsize=(12, 4))
            plt.plot(fair100_x, fair100_y, linewidth=3)
            plt.title("EVAL Jain Fairness (sampled every 100 steps)", fontweight="bold", fontsize=16)
            plt.xlabel("Step", fontsize=13)
            plt.ylabel("Jain Fairness", fontsize=13)
            plt.ylim(0.0, 1.05)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            fname = f"eval_fairness_every100_{n_steps}.png"
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            print(f"✓ Eval fairness(100-step) plot saved to '{fname}'")
            plt.show()

        return {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "reward_ue_hist": reward_ue_hist,
            "reward_bs_hist": reward_bs_hist,
            "fairness_100_x": fair100_x,
            "fairness_100_y": fair100_y,
        }

    def plot_results(self, results: Dict, tag: str = "run"):
        """
        Plot (Reward + Fairness)
        - UE reward: avg every 1000 steps
        - BS reward(avg): avg every 1000 steps
        - Jain fairness: avg every 1000 steps
        """
        def window_avg(x: List[float], w: int):
            x = np.asarray(x, dtype=np.float32)
            if len(x) == 0:
                return np.array([]), np.array([])
            xs, ys = [], []
            for start in range(0, len(x), w):
                chunk = x[start:start + w]
                if len(chunk) == 0:
                    continue
                ys.append(float(np.mean(chunk)))
                xs.append(start + len(chunk))
            return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float32)

        window_w = 1000

        if "reward_hist" in results:
            ue_r = results["reward_hist"].get("ue", [])
            bs_r = results["reward_hist"].get("bs", [])
        else:
            ue_r = results.get("reward_ue_hist", [])
            bs_r = results.get("reward_bs_hist", [])

        fair = results.get("fairness_history", [])

        x_ue, y_ue = window_avg(ue_r, window_w)
        x_bs, y_bs = window_avg(bs_r, window_w)
        x_fa, y_fa = window_avg(fair, window_w)

        if len(y_ue) == 0 and len(y_bs) == 0 and len(y_fa) == 0:
            print("No histories found to plot.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 14))

        ax = axes[0]
        if len(y_ue) > 0:
            ax.plot(x_ue, y_ue, linewidth=3)
        ax.set_title(f"UE Reward Trend (avg every {window_w} steps) - {tag}", fontweight="bold", fontsize=16)
        ax.set_xlabel("Step", fontsize=13)
        ax.set_ylabel("UE Reward", fontsize=13)
        ax.grid(alpha=0.3)

        ax = axes[1]
        if len(y_bs) > 0:
            ax.plot(x_bs, y_bs, linewidth=3)
        ax.set_title(f"BS Reward (mean) Trend (avg every {window_w} steps) - {tag}", fontweight="bold", fontsize=16)
        ax.set_xlabel("Step", fontsize=13)
        ax.set_ylabel("BS Reward (avg)", fontsize=13)
        ax.grid(alpha=0.3)

        ax = axes[2]
        if len(y_fa) > 0:
            ax.plot(x_fa, y_fa, linewidth=3)
        ax.set_title(f"Jain Fairness Trend (avg every {window_w} steps) - {tag}", fontweight="bold", fontsize=16)
        ax.set_xlabel("Step", fontsize=13)
        ax.set_ylabel("Jain Fairness", fontsize=13)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        filename = f"metrics_{tag}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Metrics plot saved to '{filename}'")
        plt.show()
