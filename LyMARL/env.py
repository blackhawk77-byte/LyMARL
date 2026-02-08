# env.py
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque

from basestation import BaseStation
from user_equipment import UserEquipment

class MAPPOEnvironment:
    def __init__(self,
                 base_stations: List[BaseStation],
                 users: List['UserEquipment'],
                 V: float = 20.0,
                 power_budget_ratio: float = 0.8,
                 enable_mobility: bool = True,
                 enable_channel_variation: bool = True,
                 lambda_p: float = 2.0,      
                 on_window: int = 100,
                 bs_top_k: int = 5,
                 hard_window_len: int = 1000,
                 bs_over_penalty: float = 50.0):

        self.base_stations = [bs for bs in base_stations if bs.bs_id != 0]
        self.users = users
        self.n_agents = len(users)
        self.n_bs = len(self.base_stations)

        self.V = float(V)
        self.power_budget_ratio = float(power_budget_ratio)
        self.enable_mobility = bool(enable_mobility)
        self.enable_channel_variation = bool(enable_channel_variation)
        self.lambda_p = float(lambda_p)

        self.bs_over_penalty = float(bs_over_penalty)

        self.bs_top_k = int(bs_top_k)
        assert self.bs_top_k >= 1

        self.hard_window_len = int(hard_window_len)
        assert self.hard_window_len >= 1

        # Power (Watt)
        self.P_max = {bs.bs_id: 10 ** (bs.tx_power_dbm / 10) / 1000 for bs in self.base_stations}
        self.P_bar = {bs.bs_id: self.power_budget_ratio * self.P_max[bs.bs_id] for bs in self.base_stations}

        self.Q_u = {u.ue_id: 0.1 for u in users}
        self.Z_b = {bs.bs_id: 0.01 for bs in self.base_stations}
        self.R_max = {u.ue_id: 5.0 for u in users}

        # Environment / channel
        self.noise_dbm = -174 + 10 * np.log10(500e6) + 5
        self.noise_watts = 10 ** (self.noise_dbm / 10) / 1000
        self.mobility_speed = 1.0
        self.area_size = 100
        self.channel_gains = defaultdict(dict)
        self.fading_std = 4.0

        self.timestep = 0

        # =====================================================
        # ✅ UE Action space: [BS0..BS(n_bs-1)] + NO-REQUEST
        #    - last index = NO-REQUEST
        # =====================================================
        self.no_request_action = self.n_bs
        self.action_dim = self.n_bs + 1

        # BS Action space: choose among Top-K candidates + NONE
        self.bs_action_dim = self.bs_top_k + 1  # last index is NONE (BS 끔)

        # Recent ON ratio history
        self.on_window = int(on_window)
        self.bs_on_hist = {bs.bs_id: deque(maxlen=self.on_window) for bs in self.base_stations}

        # congestion feature: previous step request ratio per BS
        self.prev_req_ratio = {bs.bs_id: 0.0 for bs in self.base_stations}

        # previous slot power for interference modeling (cache용)
        self.prev_power = {bs.bs_id: 0.0 for bs in self.base_stations}

        # track ON usage count inside current hard window
        self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}
        self.window_step = 0

        # local obs dim: [Q(1) + rates(n_bs) + Zb(n_bs) + on_ratio(n_bs) + prev_req(n_bs)]
        self.local_obs_dim = 1 + 4 * self.n_bs

        # BS obs dim: [Z_b, on_ratio, prev_req_ratio, remaining_budget_ratio] + score*K
        self.bs_obs_dim = 4 + self.bs_top_k

        # global obs:
        #   per UE: [Q_u, rates(n_bs)] => (1 + n_bs) each
        #   per BS: [Z_b, on_ratio, prev_req_ratio] => 3*n_bs
        self.global_obs_dim = self.n_agents * (1 + self.n_bs) + 3 * self.n_bs

        self._rate_cache = np.zeros((self.n_agents, self.n_bs), dtype=np.float32)
        self.no_coverage_count = 0

        print(f"\n{'='*92}")
        print(f"  MAPPO Env - UE requests + ✅ BS learned scheduling (TopK={self.bs_top_k})")
        print(f"{'='*92}")
        print(f"  #UE={self.n_agents} | #BS={self.n_bs} | UE_action_dim={self.action_dim} (incl. NO-REQUEST) | BS_action_dim={self.bs_action_dim}")
        print(f"  V={self.V} | power_budget_ratio(ρ)={self.power_budget_ratio} | lambda_p={self.lambda_p} (unused in UE reward)")
        print(f"  ✅ NEW rewards:")
        print(f"    UE reward(team): mean throughput (sum rate / N)")
        print(f"    BS reward(per BS): -c * (on_ratio - ρ)^2  (c={self.bs_over_penalty})")
        print(f"  local_obs_dim={self.local_obs_dim}")
        print(f"  bs_obs_dim={self.bs_obs_dim}  (Zb,on,req,remaining_budget + K*score)")
        print(f"  hard_window_len={self.hard_window_len}  (for remaining budget ratio feature)")
        print(f"  global_obs_dim={self.global_obs_dim}  (UE:[Q+rates], BS:[Zb+on+req])")
        print(f"{'='*92}\n")

    #reset은 시작할 때 한번만 사용
    def reset(self):
        self.timestep = 0
        self.no_coverage_count = 0

        for user in self.users:
            user.position = np.array([np.random.uniform(10, 90), np.random.uniform(10, 90)])

        self.update_channel_gains(0)

        self.Q_u = {u.ue_id: 0.1 for u in self.users}
        self.Z_b = {bs.bs_id: 0.01 for bs in self.base_stations}
        self.R_max = {u.ue_id: 5.0 for u in self.users}

        self.bs_on_hist = {bs.bs_id: deque(maxlen=self.on_window) for bs in self.base_stations}
        self.prev_req_ratio = {bs.bs_id: 0.0 for bs in self.base_stations}
        self.prev_power = {bs.bs_id: 0.0 for bs in self.base_stations}

        self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}
        self.window_step = 0

        self.update_max_rates()
        return self._get_observations()

    # ------------------------
    # dynamics
    # ------------------------
    def update_user_positions(self):
        if not self.enable_mobility:
            return
        for user in self.users:
            dx, dy = np.random.normal(0, self.mobility_speed, 2)
            new_x = np.clip(user.position[0] + dx, 5, self.area_size - 5)
            new_y = np.clip(user.position[1] + dy, 5, self.area_size - 5)
            user.position = np.array([new_x, new_y])

    def update_channel_gains(self, t: int):
        if not self.enable_channel_variation:
            for u in self.users:
                for bs in self.base_stations:
                    self.channel_gains[u.ue_id][bs.bs_id] = 1.0
            return

        for u in self.users:
            for bs in self.base_stations:
                if t == 0:
                    fading_db = np.random.normal(0, self.fading_std)
                else:
                    prev_db = 10 * np.log10(self.channel_gains[u.ue_id][bs.bs_id] + 1e-10)
                    fading_db = 0.9 * prev_db + np.random.normal(0, self.fading_std * np.sqrt(1 - 0.9**2))
                self.channel_gains[u.ue_id][bs.bs_id] = 10 ** (fading_db / 10)

    # ------------------------
    # PHY / rate
    # ------------------------
    def calculate_achievable_rate(self, user_id: int, bs_id: int) -> float:
        """
        Returns rate [Gbps]
        - decision/cache용: "이전 슬롯 prev_power" 기반 간섭 사용
        """
        user = next(u for u in self.users if u.ue_id == user_id)
        bs = next(b for b in self.base_stations if b.bs_id == bs_id)

        if not bs.can_serve(user.position):
            return 0.0

        dist = max(1, bs.distance_to(user.position))
        rx_dbm = bs.receive_power(dist)

        gain = self.channel_gains.get(user_id, {}).get(bs_id, 1.0)
        rx_dbm += 10 * np.log10(gain + 1e-12)
        rx_watts = 10 ** (rx_dbm / 10) / 1000

        interference = 0.0
        for other_bs in self.base_stations:
            if other_bs.bs_id == bs_id:
                continue
            prev_p = float(self.prev_power.get(other_bs.bs_id, 0.0))
            if prev_p <= 0.0:
                continue

            other_dist = max(1, other_bs.distance_to(user.position))
            other_rx_dbm = other_bs.receive_power(other_dist)

            other_gain = self.channel_gains.get(user_id, {}).get(other_bs.bs_id, 1.0)
            other_rx_dbm += 10 * np.log10(other_gain + 1e-12)
            other_rx_watts = 10 ** (other_rx_dbm / 10) / 1000

            denom = max(float(self.P_max.get(other_bs.bs_id, 1e-12)), 1e-12)
            power_scale = prev_p / denom  # typically 0 or 1 (켜져있으면 1, 아니면 0)
            interference += other_rx_watts * power_scale

        sinr = rx_watts / (self.noise_watts + interference)
        rate_bps = bs.bandwidth * np.log2(1 + sinr)
        return max(0.0, float(rate_bps / 1e9))

    def calculate_scheduled_rate(self, user_id: int, serving_bs_id: int, tx_power_map: Dict[int, float]) -> float:
        """
        결정 이후의 실제 rate 계산
        - tx_power_map: {bs_id: transmit_power_watts} (ON이면 Pmax, OFF이면 0)
        - 간섭은 '이번 슬롯 ON된 BS들의 tx_power_map' 기준으로 계산
        """
        user = next(u for u in self.users if u.ue_id == user_id)
        bs = next(b for b in self.base_stations if b.bs_id == serving_bs_id)

        if not bs.can_serve(user.position):
            return 0.0

        dist = max(1, bs.distance_to(user.position))
        rx_dbm = bs.receive_power(dist)

        gain = self.channel_gains.get(user_id, {}).get(serving_bs_id, 1.0)
        rx_dbm += 10 * np.log10(gain + 1e-12)
        rx_watts = 10 ** (rx_dbm / 10) / 1000

        interference = 0.0
        for other_bs in self.base_stations:
            if other_bs.bs_id == serving_bs_id:
                continue

            p_now = float(tx_power_map.get(other_bs.bs_id, 0.0))
            if p_now <= 0.0:
                continue

            other_dist = max(1, other_bs.distance_to(user.position))
            other_rx_dbm = other_bs.receive_power(other_dist)

            other_gain = self.channel_gains.get(user_id, {}).get(other_bs.bs_id, 1.0)
            other_rx_dbm += 10 * np.log10(other_gain + 1e-12)
            other_rx_watts = 10 ** (other_rx_dbm / 10) / 1000

            denom = max(float(self.P_max.get(other_bs.bs_id, 1e-12)), 1e-12)
            power_scale = p_now / denom  # ON이면 1, OFF면 0
            interference += other_rx_watts * power_scale

        sinr = rx_watts / (self.noise_watts + interference)
        rate_bps = bs.bandwidth * np.log2(1 + sinr)
        return max(0.0, float(rate_bps / 1e9))

    def compute_aux_rate(self, u_id: int) -> float:
        """r* = min{R_max, V/Q}"""
        Q_u = self.Q_u[u_id]
        return min(self.R_max[u_id], self.V / max(Q_u, 1e-6))

    def update_max_rates(self):
        """Compute R_max and cache UE×BS rates for current state."""
        rates = np.zeros((self.n_agents, self.n_bs), dtype=np.float32)
        for ui, user in enumerate(self.users):
            max_rate = 0.0
            for bi, bs in enumerate(self.base_stations):
                r = self.calculate_achievable_rate(user.ue_id, bs.bs_id)
                rates[ui, bi] = float(r)
                if r > max_rate:
                    max_rate = r
            self.R_max[user.ue_id] = max_rate if max_rate > 0 else 1.0
        self._rate_cache = rates

    # ------------------------
    # features
    # ------------------------
    def _get_bs_on_features(self) -> List[float]:
        feats = []
        for bs in self.base_stations:
            hist = self.bs_on_hist[bs.bs_id]
            feats.append(0.0 if len(hist) == 0 else float(sum(hist) / len(hist)))
        return feats

    def _get_bs_congestion_features(self) -> List[float]:
        return [float(self.prev_req_ratio.get(bs.bs_id, 0.0)) for bs in self.base_stations]

    # ------------------------
    # observations
    # ------------------------
    def _get_local_observation_by_index(self, ui: int) -> np.ndarray:
        ue = self.users[ui]
        obs = []
        obs.append(float(self.Q_u[ue.ue_id]))
        obs.extend(self._rate_cache[ui, :].tolist())

        for bs in self.base_stations:
            obs.append(float(self.Z_b[bs.bs_id]))

        obs.extend(self._get_bs_on_features())
        obs.extend(self._get_bs_congestion_features())

        result = np.array(obs, dtype=np.float32)
        assert len(result) == self.local_obs_dim, f"UE obs dim mismatch: {len(result)} vs {self.local_obs_dim}"
        return result

    def _get_global_observation(self) -> np.ndarray:
        obs = []
        for ui, ue in enumerate(self.users):
            obs.append(float(self.Q_u[ue.ue_id]))
            obs.extend(self._rate_cache[ui, :].tolist())

        for bs in self.base_stations:
            obs.append(float(self.Z_b[bs.bs_id]))
        obs.extend(self._get_bs_on_features())
        obs.extend(self._get_bs_congestion_features())

        result = np.array(obs, dtype=np.float32)
        assert len(result) == self.global_obs_dim, f"Global obs dim mismatch: {len(result)} vs {self.global_obs_dim}"
        return result

    def _get_observations(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        local_obs = {}
        for ui, ue in enumerate(self.users):
            local_obs[ue.ue_id] = self._get_local_observation_by_index(ui)
        global_obs = self._get_global_observation()
        return local_obs, global_obs

    # ------------------------
    # masks (UE)
    # ------------------------
    def _get_action_mask(self, ue_id: int) -> np.ndarray:
        """
        mask length = action_dim = n_bs + 1
        - [0..n_bs-1] : selectable BSs (coverage-based)
        - [n_bs]      : NO-REQUEST (always allowed)
        """
        user = next(u for u in self.users if u.ue_id == ue_id)
        mask = np.zeros(self.action_dim, dtype=bool)

        # BS choices
        for i, bs in enumerate(self.base_stations):
            mask[i] = bool(bs.can_serve(user.position))

        # ✅ NO-REQUEST always allowed
        mask[self.no_request_action] = True

        # (optional) if no BS coverage, we still allow NO-REQUEST only (already True)
        if not mask[:self.n_bs].any():
            self.no_coverage_count += 1

        return mask

    # ------------------------
    # BS decision inputs
    # ------------------------
    def _ue_index(self, ue_id: int) -> int:
        for i, u in enumerate(self.users):
            if u.ue_id == ue_id:
                return i
        raise KeyError(f"UE id {ue_id} not found")

    def build_bs_decision_inputs(self, ue_actions: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}

        # ✅ UE may choose NO-REQUEST -> then it sends no request
        for ue_id, a in ue_actions.items():
            a = int(a)
            if a == self.no_request_action:
                continue
            # valid BS action
            if not (0 <= a < self.n_bs):
                continue
            bs_id = self.base_stations[a].bs_id
            bs_requests[bs_id].append(ue_id)

        bs_obs_batch = np.zeros((self.n_bs, self.bs_obs_dim), dtype=np.float32)
        bs_mask_batch = np.zeros((self.n_bs, self.bs_action_dim), dtype=bool)
        cand_lists: List[List[int]] = []

        on_feats = self._get_bs_on_features()
        cong_feats = self._get_bs_congestion_features()

        for bi, bs in enumerate(self.base_stations):
            reqs = bs_requests[bs.bs_id]

            scored = []
            for ue_id in reqs:
                ui = self._ue_index(ue_id)
                rate = float(self._rate_cache[ui, bi])
                if rate <= 0.0:
                    continue
                score = float(self.Q_u[ue_id] * rate)  # ✅ score = Q*rate
                scored.append((score, ue_id))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[: self.bs_top_k]

            cand = [ue_id for (score, ue_id) in top]
            scores = [score for (score, ue_id) in top]

            while len(cand) < self.bs_top_k:
                cand.append(-1)
                scores.append(0.0)

            cand_lists.append(cand)

            Zb = float(self.Z_b[bs.bs_id])
            on_r = float(on_feats[bi])
            prev_req = float(cong_feats[bi])

            budget = int(round(self.power_budget_ratio * self.hard_window_len))
            used = int(self.bs_on_used_in_window.get(bs.bs_id, 0))
            if budget <= 0:
                remaining_budget_ratio = 0.0
            else:
                remaining_budget_ratio = max(0.0, (budget - used) / float(budget))

            obs = [Zb, on_r, prev_req, float(remaining_budget_ratio)]
            for k in range(self.bs_top_k):
                obs.append(float(scores[k]))

            bs_obs_batch[bi, :] = np.array(obs, dtype=np.float32)

            for k in range(self.bs_top_k):
                bs_mask_batch[bi, k] = (cand[k] >= 0)
            bs_mask_batch[bi, self.bs_top_k] = True  # NONE always allowed

            if not any(bs_mask_batch[bi, :self.bs_top_k]):
                bs_mask_batch[bi, :self.bs_top_k] = False
                bs_mask_batch[bi, self.bs_top_k] = True

        return bs_obs_batch, bs_mask_batch, cand_lists

    # ------------------------
    # step (JOINT)
    # ------------------------
    def step_joint(self, ue_actions: Dict[int, int], bs_actions: Dict[int, int], cand_lists: List[List[int]]):
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}

        # UE may choose NO-REQUEST
        for ue_id, action in ue_actions.items():
            action = int(action)
            assert 0 <= action < self.action_dim, f"Invalid UE action {action}"
            if action == self.no_request_action:
                continue
            bs_id = self.base_stations[action].bs_id
            bs_requests[bs_id].append(ue_id)

        # congestion 기록 (observed at next step)
        for bs in self.base_stations:
            self.prev_req_ratio[bs.bs_id] = len(bs_requests[bs.bs_id]) / max(1, self.n_agents)

        # BS selects UE
        bs_selections: Dict[int, Optional[int]] = {}
        for bi, bs in enumerate(self.base_stations):
            a_b = int(bs_actions[bs.bs_id])  # 0..K or K(NONE)
            if a_b == self.bs_top_k:
                bs_selections[bs.bs_id] = None
                continue

            cand = cand_lists[bi]
            if not (0 <= a_b < self.bs_top_k):
                bs_selections[bs.bs_id] = None
                continue

            ue_id = cand[a_b]
            if ue_id < 0:
                bs_selections[bs.bs_id] = None
                continue

            if ue_id not in bs_requests[bs.bs_id]:
                bs_selections[bs.bs_id] = None
                continue

            ui = self._ue_index(ue_id)
            if float(self._rate_cache[ui, bi]) <= 0.0:
                bs_selections[bs.bs_id] = None
            else:
                bs_selections[bs.bs_id] = ue_id

        # 이번 슬롯 ON/OFF 확정
        tx_power_map_now: Dict[int, float] = {}
        for bs in self.base_stations:
            sel = bs_selections[bs.bs_id]
            tx_power_map_now[bs.bs_id] = float(self.P_max[bs.bs_id]) if (sel is not None) else 0.0

        # "스케줄링 이후" 실제 rate 계산 (현재 슬롯 간섭 포함)
        served_rates = {u.ue_id: 0.0 for u in self.users}               # per UE
        bs_served_rate = {bs.bs_id: 0.0 for bs in self.base_stations}   # per BS (reward용)

        for bs in self.base_stations:
            sel = bs_selections[bs.bs_id]
            if sel is None:
                continue
            rate = self.calculate_scheduled_rate(sel, bs.bs_id, tx_power_map_now)
            served_rates[sel] = max(served_rates[sel], rate)
            bs_served_rate[bs.bs_id] = float(rate)

        total_rate = float(sum(served_rates.values()))

        power_consumed = {bs.bs_id: float(tx_power_map_now[bs.bs_id]) for bs in self.base_stations}

        # update hard-window ON usage counter
        self.window_step += 1
        for bs in self.base_stations:
            if power_consumed[bs.bs_id] > 0.0:
                self.bs_on_used_in_window[bs.bs_id] += 1

        if self.window_step % self.hard_window_len == 0:
            self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}

        # ON history
        for bs in self.base_stations:
            self.bs_on_hist[bs.bs_id].append(1.0 if power_consumed[bs.bs_id] > 0 else 0.0)

        # update prev_power for next-step interference (cache용)
        self.prev_power = power_consumed.copy()

        # save old queues for dummy stats
        old_Q_u = self.Q_u.copy()
        old_Z_b = self.Z_b.copy()

        # queue updates (same dynamics, only names changed)
        for u in self.users:
            aux_rate = self.compute_aux_rate(u.ue_id)
            actual_rate = served_rates[u.ue_id]
            self.Q_u[u.ue_id] = max(1e-12, self.Q_u[u.ue_id] + (aux_rate - actual_rate))

        for bs in self.base_stations:
            power = power_consumed[bs.bs_id]
            budget = self.P_bar[bs.bs_id]
            self.Z_b[bs.bs_id] = max(0.001, self.Z_b[bs.bs_id] + (power - budget))

        # UE Reward
        eta = 1.0
        mean_Q = float(np.mean([self.Q_u[u.ue_id] for u in self.users]))
        ue_team_reward = float(total_rate / max(1, self.n_agents) - eta * mean_Q)

        # BS reward
        on_feats = self._get_bs_on_features()
        rho = self.power_budget_ratio
        c = self.bs_over_penalty

        alpha = 3.0
        beta = 1.0

        bs_rewards = []
        for bi, bs in enumerate(self.base_stations):
            served_rate_i = float(bs_served_rate[bs.bs_id])

            on_i = float(on_feats[bi])
            over = max(0.0, on_i - rho)

            on_now = 1.0 if power_consumed[bs.bs_id] > 0.0 else 0.0
            r_i = alpha * served_rate_i - c * (over ** 2) - beta * float(self.Z_b[bs.bs_id]) * on_now
            bs_rewards.append(float(r_i))

        bs_rewards = np.array(bs_rewards, dtype=np.float32)
        bs_team_reward = float(np.mean(bs_rewards))

        # Transition to next state
        self.timestep += 1
        self.update_user_positions()
        self.update_channel_gains(self.timestep)
        self.update_max_rates()

        local_obs, global_obs = self._get_observations()

        info = {
            "total_throughput": total_rate,
            "power_consumed": power_consumed,
            "served_rates": served_rates,

            # ✅ renamed
            "Q_u": self.Q_u.copy(),
            "Z_b": self.Z_b.copy(),

            "bs_selections": bs_selections,
            "bs_requests": {bs_id: len(reqs) for bs_id, reqs in bs_requests.items()},
            "prev_req_ratio": self.prev_req_ratio.copy(),
            "ue_team_reward": ue_team_reward,
            "bs_rewards": bs_rewards.copy(),
            "bs_team_reward": bs_team_reward,

            # dummy stats
            "total_QR_dummy": float(sum(old_Q_u[u.ue_id] * served_rates[u.ue_id] for u in self.users)),
            "total_ZP_dummy": float(sum(old_Z_b[bs.bs_id] * power_consumed[bs.bs_id] for bs in self.base_stations)),

            "no_coverage_count": int(self.no_coverage_count),
            "bs_on_used_in_window": self.bs_on_used_in_window.copy(),
            "window_step": int(self.window_step),
            "on_feats": on_feats,
            "rho": float(rho),

            # helpful: NO-REQUEST index
            "ue_no_request_action": int(self.no_request_action),
        }

        done = False
        return local_obs, global_obs, info, done

    # metrics (수정 가능)
    def calculate_jain_fairness(self, rate_history: List) -> float:
        recent = rate_history if len(rate_history) < 100 else rate_history[-100:]
        if not recent:
            return 0.0
        rate_array = np.array(recent)
        per_user_avg = rate_array.mean(axis=0)
        sum_rates = per_user_avg.sum()
        sum_squared = (per_user_avg ** 2).sum()
        n_users = len(per_user_avg)
        if sum_squared < 1e-12:
            return 0.0
        return float((sum_rates ** 2) / (n_users * sum_squared))