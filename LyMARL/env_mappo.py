import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque

from LyMARL.basestation import BaseStation
from LyMARL.user_equipment import UserEquipment


class MAPPOEnvironment:
    """
    Environment for UE-BS heterogeneous MAPPO.

    - Training: soft constraint behavior only
    - Evaluation: optional hard constraint can be enabled
    """
    def __init__(
        self,
        base_stations: List[BaseStation],
        users: List[UserEquipment],
        V: float = 20.0,
        power_budget_ratio: float = 0.6,
        enable_mobility: bool = True,
        enable_channel_variation: bool = True,
        on_window: int = 1000,
        bs_top_k: int = 5,
        hard_window_len: int = 10000,
        use_hard_constraint: bool = False,
    ):
        self.base_stations = [bs for bs in base_stations if bs.bs_id != 0]
        self.users = users
        self.n_agents = len(users)
        self.n_bs = len(self.base_stations)

        self.V = float(V)
        self.power_budget_ratio = float(power_budget_ratio)
        self.enable_mobility = bool(enable_mobility)
        self.enable_channel_variation = bool(enable_channel_variation)

        self.bs_top_k = int(bs_top_k)
        assert self.bs_top_k >= 1

        self.hard_window_len = int(hard_window_len)
        assert self.hard_window_len >= 1

        self.use_hard_constraint = bool(use_hard_constraint)

        # Power (Watt)
        self.P_max = {bs.bs_id: 10 ** (bs.tx_power_dbm / 10) / 1000 for bs in self.base_stations}
        self.P_bar = {bs.bs_id: self.power_budget_ratio * self.P_max[bs.bs_id] for bs in self.base_stations}

        # Hard constraint allowance:
        # each BS can be ON at most power_budget_ratio * hard_window_len times per window
        self.hard_on_limit = {
            bs.bs_id: int(np.floor(self.power_budget_ratio * self.hard_window_len))
            for bs in self.base_stations
        }

        # Queues
        self.Q_u = {u.ue_id: 1.0 for u in users}
        self.Z_b = {bs.bs_id: 1.0 for bs in self.base_stations}
        self.R_max = {u.ue_id: 5.0 for u in users}

        # Channel / mobility
        self.noise_dbm = -174 + 10 * np.log10(500e6) + 5
        self.noise_watts = 10 ** (self.noise_dbm / 10) / 1000
        self.mobility_speed = 1.0
        self.area_size = 100
        self.channel_gains = defaultdict(dict)
        self.fading_std = 4.0

        self.timestep = 0

        # UE action: [BS0..BS(n_bs-1)] + NO-REQUEST
        self.no_request_action = self.n_bs
        self.action_dim = self.n_bs + 1

        # BS action: Top-K candidates + NONE
        self.bs_action_dim = self.bs_top_k + 1

        # Recent ON ratio history
        self.on_window = int(on_window)
        self.bs_on_hist = {bs.bs_id: deque(maxlen=self.on_window) for bs in self.base_stations}

        # Congestion logging only
        self.prev_req_ratio = {bs.bs_id: 0.0 for bs in self.base_stations}

        # Previous-slot power for interference estimation in decision phase
        self.prev_power = {bs.bs_id: 0.0 for bs in self.base_stations}

        # Hard window usage
        self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}
        self.window_step = 0

        # Fast lookup maps
        self.user_map = {u.ue_id: u for u in self.users}
        self.bs_map = {bs.bs_id: bs for bs in self.base_stations}
        self.ue_id_to_index = {u.ue_id: i for i, u in enumerate(self.users)}
        self.bs_id_to_index = {bs.bs_id: i for i, bs in enumerate(self.base_stations)}

        # Observation dimensions
        # UE local: [Q_u] + rates(n_bs) + Z_b(n_bs)
        self.local_obs_dim = 1 + 2 * self.n_bs

        # BS local: [Z_b] + top-K scores
        self.bs_obs_dim = 1 + self.bs_top_k

        # Global:
        # per UE: [Q_u, rates(n_bs)] => n_agents * (1 + n_bs)
        # per BS: [Z_b] => n_bs
        self.global_obs_dim = self.n_agents * (1 + self.n_bs) + self.n_bs

        self._rate_cache = np.zeros((self.n_agents, self.n_bs), dtype=np.float32)
        self.no_coverage_count = 0

        print(f"\n{'='*96}")
        print(" MAPPO Environment")
        print(f"{'='*96}")
        print(f"#UE={self.n_agents} | #BS={self.n_bs} | UE_action_dim={self.action_dim} | BS_action_dim={self.bs_action_dim}")
        print(f"V={self.V} | power_budget_ratio={self.power_budget_ratio}")
        print(f"Hard constraint enabled: {self.use_hard_constraint}")
        print(f"local_obs_dim={self.local_obs_dim} | bs_obs_dim={self.bs_obs_dim} | global_obs_dim={self.global_obs_dim}")
        print(f"{'='*96}\n")

    def set_hard_constraint(self, enabled: bool):
        self.use_hard_constraint = bool(enabled)

    def reset(self):
        self.timestep = 0
        self.no_coverage_count = 0

        for user in self.users:
            user.position = np.array([np.random.uniform(10, 90), np.random.uniform(10, 90)])

        self.update_channel_gains(0)

        self.Q_u = {u.ue_id: 1.0 for u in self.users}
        self.Z_b = {bs.bs_id: 1.0 for bs in self.base_stations}
        self.R_max = {u.ue_id: 5.0 for u in self.users}

        self.bs_on_hist = {bs.bs_id: deque(maxlen=self.on_window) for bs in self.base_stations}
        self.prev_req_ratio = {bs.bs_id: 0.0 for bs in self.base_stations}
        self.prev_power = {bs.bs_id: 0.0 for bs in self.base_stations}

        self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}
        self.window_step = 0

        self.update_max_rates()
        return self._get_observations()
    
    def reset_queues(self):
        self.Q_u = {u.ue_id: 1.0 for u in self.users}
        self.Z_b = {bs.bs_id: 1.0 for bs in self.base_stations}
        self.R_max = {u.ue_id: 5.0 for u in self.users}
        self.update_max_rates()
        return self._get_observations()
    
    # =========================================================
    # Dynamics
    # =========================================================
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

    # =========================================================
    # PHY / Rate
    # =========================================================
    def calculate_achievable_rate(self, user_id: int, bs_id: int) -> float:
        """
        Rate used for decision / cache.
        Interference is estimated using previous-slot BS power.
        Returns rate in Gbps.
        """
        user = self.user_map[user_id]
        bs = self.bs_map[bs_id]

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
            power_scale = prev_p / denom
            interference += other_rx_watts * power_scale

        sinr = rx_watts / (self.noise_watts + interference)
        rate_bps = bs.bandwidth * np.log2(1 + sinr)
        return max(0.0, float(rate_bps / 1e9))

    def calculate_scheduled_rate(self, user_id: int, serving_bs_id: int, tx_power_map: Dict[int, float]) -> float:
        """
        Actual rate after scheduling decision.
        Interference is computed from current-slot tx_power_map.
        Returns rate in Gbps.
        """
        user = self.user_map[user_id]
        bs = self.bs_map[serving_bs_id]

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
            power_scale = p_now / denom
            interference += other_rx_watts * power_scale

        sinr = rx_watts / (self.noise_watts + interference)
        rate_bps = bs.bandwidth * np.log2(1 + sinr)
        return max(0.0, float(rate_bps / 1e9))

    def compute_aux_rate(self, u_id: int) -> float:
        """
        Auxiliary rate term for queue update:
        r* = min{R_max, V / Q}
        """
        Q_u = self.Q_u[u_id]
        return min(self.R_max[u_id], self.V / max(Q_u, 1e-6))

    def update_max_rates(self):
        """
        Compute R_max and cache UE-BS achievable rates for the current state.
        """
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

    # =========================================================
    # Features / Observations
    # =========================================================
    def _get_bs_on_features(self) -> List[float]:
        feats = []
        for bs in self.base_stations:
            hist = self.bs_on_hist[bs.bs_id]
            feats.append(0.0 if len(hist) == 0 else float(sum(hist) / len(hist)))
        return feats

    def _get_local_observation_by_index(self, ui: int) -> np.ndarray:
        ue = self.users[ui]
        obs = [float(self.Q_u[ue.ue_id])]
        obs.extend(self._rate_cache[ui, :].tolist())

        for bs in self.base_stations:
            obs.append(float(self.Z_b[bs.bs_id]))

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

        result = np.array(obs, dtype=np.float32)
        assert len(result) == self.global_obs_dim, f"Global obs dim mismatch: {len(result)} vs {self.global_obs_dim}"
        return result

    def _get_observations(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        local_obs = {}
        for ui, ue in enumerate(self.users):
            local_obs[ue.ue_id] = self._get_local_observation_by_index(ui)

        global_obs = self._get_global_observation()
        return local_obs, global_obs

    # =========================================================
    # Masks / Decision Inputs
    # =========================================================
    def _get_action_mask(self, ue_id: int) -> np.ndarray:
        """
        mask length = n_bs + 1
        [0..n_bs-1]: selectable BSs based on coverage
        [n_bs]: NO-REQUEST, always valid
        """
        user = self.user_map[ue_id]
        mask = np.zeros(self.action_dim, dtype=bool)

        for i, bs in enumerate(self.base_stations):
            mask[i] = bool(bs.can_serve(user.position))

        mask[self.no_request_action] = True

        if not mask[:self.n_bs].any():
            self.no_coverage_count += 1

        return mask

    def build_bs_decision_inputs(self, ue_actions: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """
        Build per-BS observations and masks from UE requests.
        BS sees:
        [Z_b, score_1, ..., score_K]
        where score = Q_u * achievable_rate
        """
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}

        for ue_id, a in ue_actions.items():
            a = int(a)
            if a == self.no_request_action:
                continue
            if not (0 <= a < self.n_bs):
                continue

            bs_id = self.base_stations[a].bs_id
            bs_requests[bs_id].append(ue_id)

        bs_obs_batch = np.zeros((self.n_bs, self.bs_obs_dim), dtype=np.float32)
        bs_mask_batch = np.zeros((self.n_bs, self.bs_action_dim), dtype=bool)
        cand_lists: List[List[int]] = []

        for bi, bs in enumerate(self.base_stations):
            reqs = bs_requests[bs.bs_id]

            scored = []
            for ue_id in reqs:
                ui = self.ue_id_to_index[ue_id]
                rate = float(self._rate_cache[ui, bi])
                if rate <= 0.0:
                    continue

                score = float(self.Q_u[ue_id] * rate)
                scored.append((score, ue_id))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:self.bs_top_k]

            cand = [ue_id for (score, ue_id) in top]
            scores = [score for (score, ue_id) in top]

            while len(cand) < self.bs_top_k:
                cand.append(-1)
                scores.append(0.0)

            cand_lists.append(cand)

            obs = [float(self.Z_b[bs.bs_id])]
            obs.extend([float(s) for s in scores])
            bs_obs_batch[bi, :] = np.array(obs, dtype=np.float32)

            for k in range(self.bs_top_k):
                bs_mask_batch[bi, k] = (cand[k] >= 0)

            bs_mask_batch[bi, self.bs_top_k] = True  # NONE always valid

        return bs_obs_batch, bs_mask_batch, cand_lists

    # =========================================================
    # Step
    # =========================================================
    def step_joint(self, ue_actions: Dict[int, int], bs_actions: Dict[int, int], cand_lists: List[List[int]]):
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}

        for ue_id, action in ue_actions.items():
            action = int(action)
            assert 0 <= action < self.action_dim, f"Invalid UE action {action}"

            if action == self.no_request_action:
                continue

            bs_id = self.base_stations[action].bs_id
            bs_requests[bs_id].append(ue_id)

        # Congestion logging
        for bs in self.base_stations:
            self.prev_req_ratio[bs.bs_id] = len(bs_requests[bs.bs_id]) / max(1, self.n_agents)

        # BS selects one UE or NONE
        bs_selections: Dict[int, Optional[int]] = {}
        for bi, bs in enumerate(self.base_stations):
            a_b = int(bs_actions[bs.bs_id])

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

            ui = self.ue_id_to_index[ue_id]
            if float(self._rate_cache[ui, bi]) <= 0.0:
                bs_selections[bs.bs_id] = None
            else:
                bs_selections[bs.bs_id] = ue_id

        # Optional hard constraint enforcement for evaluation
        if self.use_hard_constraint:
            for bs in self.base_stations:
                used = self.bs_on_used_in_window[bs.bs_id]
                limit = self.hard_on_limit[bs.bs_id]
                if used >= limit:
                    bs_selections[bs.bs_id] = None

        # Current-slot ON/OFF and tx power
        tx_power_map_now: Dict[int, float] = {}
        for bs in self.base_stations:
            sel = bs_selections[bs.bs_id]
            tx_power_map_now[bs.bs_id] = float(self.P_max[bs.bs_id]) if (sel is not None) else 0.0

        # Actual scheduled rates
        served_rates = {u.ue_id: 0.0 for u in self.users}
        bs_served_rate = {bs.bs_id: 0.0 for bs in self.base_stations}

        for bs in self.base_stations:
            sel = bs_selections[bs.bs_id]
            if sel is None:
                continue

            rate = self.calculate_scheduled_rate(sel, bs.bs_id, tx_power_map_now)
            served_rates[sel] = max(served_rates[sel], rate)
            bs_served_rate[bs.bs_id] = float(rate)

        total_rate = float(sum(served_rates.values()))
        power_consumed = {bs.bs_id: float(tx_power_map_now[bs.bs_id]) for bs in self.base_stations}

        # Update hard window usage
        self.window_step += 1
        for bs in self.base_stations:
            if power_consumed[bs.bs_id] > 0.0:
                self.bs_on_used_in_window[bs.bs_id] += 1

        if self.window_step % self.hard_window_len == 0:
            self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}

        # ON history
        for bs in self.base_stations:
            self.bs_on_hist[bs.bs_id].append(1.0 if power_consumed[bs.bs_id] > 0.0 else 0.0)

        # Store current-slot power for next-slot decision interference
        self.prev_power = power_consumed.copy()

        old_Q_u = self.Q_u.copy()
        old_Z_b = self.Z_b.copy()

        # Queue updates
        for u in self.users:
            aux_rate = self.compute_aux_rate(u.ue_id)
            actual_rate = served_rates[u.ue_id]
            self.Q_u[u.ue_id] = max(1e-12, self.Q_u[u.ue_id] + (aux_rate - actual_rate))

        for bs in self.base_stations:
            power = power_consumed[bs.bs_id]
            budget = self.P_bar[bs.bs_id]
            self.Z_b[bs.bs_id] = max(0.001, self.Z_b[bs.bs_id] + (power - budget))

        # UE team reward
        ue_team_reward = float(np.mean([
            old_Q_u[u.ue_id] * served_rates[u.ue_id]
            for u in self.users
        ]))

        # Per-user reward for logging
        ue_per_user_rewards = {
            u.ue_id: float(old_Q_u[u.ue_id] * served_rates[u.ue_id])
            for u in self.users
        }

        # BS rewards
        bs_rewards = []

        for bi, bs in enumerate(self.base_stations):
            bs_id = bs.bs_id
            selected_ue = bs_selections[bs_id]

            served_rate_i = float(bs_served_rate[bs_id])
            P_tx_i = float(self.P_max[bs_id])

            eps_log = 1e-3

            if selected_ue is None:
                rate_reward = float(np.log(eps_log))
                energy_penalty = 0.0
            else:
                rate_reward = float(np.log(max(served_rate_i, eps_log)))
                energy_penalty = P_tx_i * float(old_Z_b[bs_id])

            r_i = rate_reward - energy_penalty
            
            bs_rewards.append(float(r_i))

        bs_rewards = np.array(bs_rewards, dtype=np.float32)
        bs_team_reward = float(np.mean(bs_rewards))

        # Move to next state
        self.timestep += 1
        self.update_user_positions()
        self.update_channel_gains(self.timestep)
        self.update_max_rates()

        local_obs, global_obs = self._get_observations()

        info = {
            "total_throughput": total_rate,
            "power_consumed": power_consumed,
            "served_rates": served_rates,

            "Q_u": self.Q_u.copy(),
            "Z_b": self.Z_b.copy(),

            "ue_team_reward": ue_team_reward,
            "ue_per_user_rewards": ue_per_user_rewards,
            "bs_rewards": bs_rewards.copy(),
            "bs_team_reward": bs_team_reward,

            "bs_selections": bs_selections,
            "bs_requests": {bs_id: len(reqs) for bs_id, reqs in bs_requests.items()},
            "prev_req_ratio": self.prev_req_ratio.copy(),

            "total_QR_dummy": float(sum(old_Q_u[u.ue_id] * served_rates[u.ue_id] for u in self.users)),
            "total_ZP_dummy": float(sum(old_Z_b[bs.bs_id] * power_consumed[bs.bs_id] for bs in self.base_stations)),

            "no_coverage_count": int(self.no_coverage_count),
            "bs_on_used_in_window": self.bs_on_used_in_window.copy(),
            "window_step": int(self.window_step),

            "ue_no_request_action": int(self.no_request_action),
            "hard_constraint_enabled": bool(self.use_hard_constraint),
            "hard_on_limit": self.hard_on_limit.copy(),
        }

        done = False
        return local_obs, global_obs, info, done

    # =========================================================
    # Metric
    # =========================================================
    def calculate_jain_fairness(self, rate_history: List) -> float:
        """
        Jain's fairness computed from the most recent up to 100 slots.
        """
        fair_window = 1000
        recent = rate_history if len(rate_history) < fair_window else rate_history[-fair_window:]
        
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