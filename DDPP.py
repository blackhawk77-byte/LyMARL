import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import defaultdict

from basestation import BaseStation, SmallCellBaseStation
from user_equipment import UserEquipment
from core import generate_triangle_coverage

################################################
## BASELINE DDPP ALGORITHM IMPLEMENTATION
################################################

class DDPPAlgorithm:
    """
    Q_u(t+1) = [Q_u(t) + r*(t) - R_u(t)]_+
    Z_b(t+1) = [Z_b(t) + P_b(t) - P_bar]_+
    Score (UE->BS request) = Q_u × R - Z_b × P_max
    
    calculate_achievable_rate: R(SINR)계산인데, (SNR계산은 안해도 됨)
    결정 단계(user association/bs scheduling/gamma_max update)는 이전 슬롯 ON 간섭원 기준, 
    실제 서비스(queue update)는 현재 슬롯 ON 간섭원 기준으로 함.
    """

    def __init__(self,
                 base_stations: List[BaseStation],
                 users: List[UserEquipment],
                 V: float = 20.0,
                 power_budget_ratio: float = 0.7,
                 max_slots: int = 1000,
                 enable_mobility: bool = True,
                 enable_channel_variation: bool = True,
                 seed: int = None):

        self.users = users
        self.base_stations = [bs for bs in base_stations if bs.bs_id != 0]
        self.V = V
        self.power_budget_ratio = power_budget_ratio
        self.max_slots = max_slots
        self.enable_mobility = enable_mobility
        self.enable_channel_variation = enable_channel_variation
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.prev_power = {bs.bs_id: 0.0 for bs in self.base_stations}

        self.P_max = {
            bs.bs_id: 10 ** (bs.tx_power_dbm / 10) / 1000  # [W] # 20 dBm -> 0.1 W
            for bs in self.base_stations
        }

        self.P_bar = {
            bs.bs_id: self.power_budget_ratio * self.P_max[bs.bs_id]  # [W]
            for bs in self.base_stations
        }

        self.Q_u = {ue.ue_id: 0.1 for ue in users}
        self.Z_b = {bs.bs_id: 0.01 for bs in self.base_stations}
        self.gamma_max = {ue.ue_id: 5.0 for ue in users}

        # ==========================================
        # Tracking
        # ==========================================
        self.associations_history = []
        self.bs_status_history = []
        self.throughput_history = []
        self.power_history = defaultdict(list)
        self.queue_history = {'Q': defaultdict(list), 'Z': defaultdict(list)}
        self.user_rate_history = defaultdict(list)
        self.fairness_history = []
        self.slot_rates = []

        # ==========================================
        # Environment
        # ==========================================
        self.noise_dbm = -174 + 10 * np.log10(500e6) + 5
        self.noise_watts = 10 ** (self.noise_dbm / 10) / 1000
        self.mobility_speed = 1.0
        self.area_size = 100
        self.channel_gains = defaultdict(dict)
        self.fading_std = 4.0

    # ==========================================
    # Environment Dynamics
    # ==========================================
    def update_user_positions(self):
        if not self.enable_mobility:
            return
        for user in self.users:
            dx, dy = self.rng.normal(0, self.mobility_speed, 2)
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
                    fading_db = self.rng.normal(0, self.fading_std)
                else:
                    prev_db = 10 * np.log10(self.channel_gains[u.ue_id][bs.bs_id] + 1e-10)
                    fading_db = 0.9 * prev_db + self.rng.normal(0, self.fading_std * np.sqrt(1 - 0.9**2))
                self.channel_gains[u.ue_id][bs.bs_id] = 10 ** (fading_db / 10)

    # ==========================================
    # PHY Layer
    # ==========================================
    def calculate_achievable_rate(self, user_id: int, bs_id: int, interferer_status: dict = None) -> float:
        """ Returns: rate [Gbps] """
        user = next(u for u in self.users if u.ue_id == user_id)
        bs = next(b for b in self.base_stations if b.bs_id == bs_id)

        dist = max(1.0, bs.distance_to(user.position))
        rx_dbm = bs.receive_power(dist)

        # small-scale fading gain (linear) 
        gain = self.channel_gains.get(user_id, {}).get(bs_id, 1.0)
        rx_dbm += 10 * np.log10(gain + 1e-12)

        # dBm -> W
        rx_watts = 10 ** (rx_dbm / 10) / 1000

        # interference
        interference = 0.0
        for other_bs in self.base_stations:
            if other_bs.bs_id == bs_id:
                continue

            # 간섭원 선택 기준
            if interferer_status is None:
                # 이전 슬롯 ON만 간섭원
                is_on = (self.prev_power.get(other_bs.bs_id, 0.0) > 0.0)
            else:
                # 현재 슬롯 ON만 간섭원
                is_on = (interferer_status.get(other_bs.bs_id, 0) == 1)

            if not is_on:
                continue

            other_dist = max(1.0, other_bs.distance_to(user.position))
            other_rx_dbm = other_bs.receive_power(other_dist)
            # interference 쪽에는 channel gain 안 붙임
            interference += 10 ** (other_rx_dbm / 10) / 1000

        sinr = rx_watts / (self.noise_watts + interference)
        rate_bps = bs.bandwidth * np.log2(1.0 + sinr)
        return max(0.0, rate_bps / 1e9)

    # ==========================================
    # DPP Core
    # ==========================================
    def compute_aux_rate(self, u_id: int) -> float:
        """r* = min{gamma_max, V/Q_u} [Gbps]"""
        Q_u = self.Q_u[u_id]
        return min(self.gamma_max[u_id], self.V / max(Q_u, 1e-6))  # 상한 gamma_max

    def user_association(self, t: int) -> dict:
        """Score = Q_u × R_tilde - Z_b × P_max"""
        associations = {}

        for user in self.users:
            best_bs = None
            best_score = -np.inf

            for bs in self.base_stations:
                # 결정 단계는 prev_power 기반 간섭 -> R_tilde 
                R_tilde = self.calculate_achievable_rate(user.ue_id, bs.bs_id)
                power = self.P_max[bs.bs_id]
                score = self.Q_u[user.ue_id] * R_tilde - self.Z_b[bs.bs_id] * power

                if score > best_score:
                    best_score = score
                    best_bs = bs.bs_id

            associations[user.ue_id] = best_bs if best_score > 0 else None

        return associations

    def bs_scheduling(self, associations: dict) -> tuple:
        """
        R = Q_u × R_tilde
        threshold = Z_b × P_max
        if R > threshold => ON and serve best UE
        """
        bs_status = {}
        scheduled_users = {}

        proposers = defaultdict(list)
        for ue_id, bs_id in associations.items():
            if bs_id is not None:
                proposers[bs_id].append(ue_id)

        for bs in self.base_stations:
            if not proposers[bs.bs_id]:
                bs_status[bs.bs_id] = 0
                scheduled_users[bs.bs_id] = None
                continue

            best_score_qr = 0.0
            best_ue = None

            for ue_id in proposers[bs.bs_id]:
                # 결정 단계는 prev_power 기반 간섭 -> R_tilde 
                R_tilde = self.calculate_achievable_rate(ue_id, bs.bs_id)
                score_qr = self.Q_u[ue_id] * R_tilde
                if score_qr > best_score_qr:
                    best_score_qr = score_qr
                    best_ue = ue_id

            power = self.P_max[bs.bs_id]
            threshold = self.Z_b[bs.bs_id] * power

            if best_score_qr > threshold:
                bs_status[bs.bs_id] = 1
                scheduled_users[bs.bs_id] = best_ue
            else:
                bs_status[bs.bs_id] = 0
                scheduled_users[bs.bs_id] = None

        return bs_status, scheduled_users


    def update_queues(self, scheduled_users: dict, bs_status: dict) -> dict:
        """Queue updates + returns actual served rates R(SINR)"""
        actual_rates = {u.ue_id: 0.0 for u in self.users}

        for bs_id, ue_id in scheduled_users.items():
            if ue_id is not None and bs_status[bs_id] == 1:
                # actual rate는 "현재 슬롯 ON(bs_status)"을 간섭원으로 사용
                actual_rate = self.calculate_achievable_rate(ue_id, bs_id, interferer_status=bs_status)
                actual_rates[ue_id] = actual_rate
                self.user_rate_history[ue_id].append(actual_rate)

        for user in self.users:
            aux_rate = self.compute_aux_rate(user.ue_id)
            served_rate = actual_rates[user.ue_id]
            self.Q_u[user.ue_id] = max(1e-12, self.Q_u[user.ue_id] + (aux_rate - served_rate))

        for bs in self.base_stations:
            power = self.P_max[bs.bs_id] if bs_status[bs.bs_id] == 1 else 0.0
            budget = self.P_bar[bs.bs_id]
            self.Z_b[bs.bs_id] = max(0.001, self.Z_b[bs.bs_id] + (power - budget))

        return actual_rates

    def update_max_rates(self):
        for user in self.users:
            max_R_tilde = 0.0
            for bs in self.base_stations:
                # R_max는 결정/관측과 같은 기준(prev 간섭)으로 계산
                R_tilde = self.calculate_achievable_rate(user.ue_id, bs.bs_id)
                max_R_tilde = max(max_R_tilde, R_tilde)
            self.gamma_max[user.ue_id] = max_R_tilde if max_R_tilde > 0 else 1.0

    # ==========================================
    # Metrics
    # ==========================================
    def calculate_jain_fairness(self, window: int = 200) -> float:
        recent_slots = self.slot_rates if len(self.slot_rates) < window else self.slot_rates[-window:]
        if not recent_slots:
            return 0.0
        rate_array = np.asarray(recent_slots, dtype=np.float32)  # (time, users)
        if rate_array.ndim != 2:
            return 0.0
        per_user_avg = rate_array.mean(axis=0)
        sum_rates = per_user_avg.sum()
        sum_squared = (per_user_avg ** 2).sum()
        n_users = len(per_user_avg)
        if sum_squared < 1e-12:
            return 0.0
        return (sum_rates ** 2) / (n_users * sum_squared)

    # ==========================================
    # Simulation
    # ==========================================
    def run_slot(self, t: int):
        # NOTE: 결정(association/scheduling/gamma_max)은 prev_power(t-1) 간섭 기준 -> R_tilde
        # queue update(실제 서비스): current bs_status(t) 간섭 기반 -> R
        self.update_user_positions()
        self.update_channel_gains(t)
        self.update_max_rates()

        associations = self.user_association(t)
        bs_status, scheduled = self.bs_scheduling(associations)
        actual_rates = self.update_queues(scheduled, bs_status)

        # prev_power 업데이트 -> 다음 슬롯 R_tilde 계산에 사용
        self.prev_power = {bs_id: (bs_status.get(bs_id, 0) * self.P_max[bs_id]) for bs_id in self.P_max}

        self.associations_history.append(scheduled)
        self.bs_status_history.append(bs_status)
        self.throughput_history.append(sum(actual_rates.values()))

        for bs_id, status in bs_status.items():
            power_watts = status * self.P_max[bs_id]
            self.power_history[bs_id].append(power_watts)

        for ue_id in self.Q_u:
            self.queue_history['Q'][ue_id].append(self.Q_u[ue_id])
        for bs_id in self.Z_b:
            self.queue_history['Z'][bs_id].append(self.Z_b[bs_id])

        self.slot_rates.append([actual_rates.get(u.ue_id, 0.0) for u in self.users])
        self.fairness_history.append(self.calculate_jain_fairness(window=100))

    def run_simulation(self):
        print(f"\n{'='*60}")
        print(f"  Pure DPP Algorithm (Decision: prev ON, Actual: current ON)")
        print(f"{'='*60}")
        print(f"  V = {self.V}")
        print(f"  Power budget ratio = {self.power_budget_ratio}")
        print(f"  Total slots = {self.max_slots}")
        print(f"{'='*60}\n")
        self.recent_fair_list = []

        for t in range(self.max_slots):

            if hasattr(self, "V_schedule_fn"):
                self.V = self.V_schedule_fn(t)
            if t in [0, 10000, 20000, 30000]:
                print(f"[DDPP] t={t:6d} | V={self.V}")
            self.run_slot(t)
            
            if (t + 1) % 100 == 0:
                recent_thr = float(np.mean(self.throughput_history[-100:]))
                recent_fair = float(self.calculate_jain_fairness(window=100)) # 100 슬롯 단위로 JFI 저장
                self.recent_fair_list.append(recent_fair)

                on_ratios = {}
                for bs in self.base_stations:
                    on_count = sum(1 for s in self.bs_status_history[-100:] if s.get(bs.bs_id, 0) == 1)
                    on_ratios[bs.bs_id] = on_count / 100

                ratio_str = ', '.join([f'BS{b}:{r:.2f}' for b, r in on_ratios.items()])
                print(f"Slot {t+1:4d} | Thr: {recent_thr:.3f} Gbps | "
                      f"Fair(JFI@100): {recent_fair:.3f} | ON: [{ratio_str}]")
        print(f"\n{'='*60}")
        overall_thr = float(np.mean(self.throughput_history))
        overall_fair = float(np.mean(self.recent_fair_list)) # ep 전체 슬롯 기준 JFI
        print(f"  Avg Throughput: {overall_thr:.3f} Gbps")
        print(f"  JFI (avg over 100 slots): {overall_fair:.4f}")

        print(f"\n  Power Budget Check:")
        for bs in self.base_stations:
            avg_power = np.mean(self.power_history[bs.bs_id])
            budget = self.P_bar[bs.bs_id]
            on_ratio = sum(1 for p in self.power_history[bs.bs_id] if p > 0) / len(self.power_history[bs.bs_id])
            print(f"    BS {bs.bs_id}: {avg_power:.4f}W / {budget:.4f}W | "
                  f"ON={on_ratio:.3f} (target={self.power_budget_ratio})")

        print(f"\n  Queue Value Ranges:")
        q_vals = [self.Q_u[u.ue_id] for u in self.users]
        z_vals = [self.Z_b[bs.bs_id] for bs in self.base_stations]
        rtilde_vals = []
        w_vals = []
        for u in self.users:
            for bs in self.base_stations:
                R_tilde = self.calculate_achievable_rate(u.ue_id, bs.bs_id)
                rtilde_vals.append(R_tilde)
                w_vals.append(self.Q_u[u.ue_id] * R_tilde - self.Z_b[bs.bs_id] * self.P_max[bs.bs_id])
        print(f"    Q_u: [{min(q_vals):.4f}, {max(q_vals):.4f}]")
        print(f"    Z_b: [{min(z_vals):.6f}, {max(z_vals):.6f}]")
        print(f"    R_tilde: [{min(rtilde_vals):.4f}, {max(rtilde_vals):.4f}] Gbps")
        print(f"    W: [{min(w_vals):.4f}, {max(w_vals):.4f}]")
        print(f"{'='*60}\n")
        
    def plot_results(self):
        bs_ids = sorted([bs.bs_id for bs in self.base_stations])
        T = len(self.bs_status_history)
        if T == 0:
            print("No simulation data to plot.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # =========================================================
        # (1) ON ratio per 10,000 slots 
        # =========================================================
        ax = axes[0]
        block = 10000

        x_points = []
        block_ratios = {bs_id: [] for bs_id in bs_ids}

        for start in range(0, T, block):
            chunk = self.bs_status_history[start:start + block]
            if len(chunk) == 0:
                continue
            end_step = start + len(chunk)
            x_points.append(end_step)

            for bs_id in bs_ids:
                on_count = sum(1 for s in chunk if s.get(bs_id, 0) == 1)
                block_ratios[bs_id].append(on_count / len(chunk))

        for bs_id in bs_ids:
            ax.plot(x_points, block_ratios[bs_id], label=f'BS{bs_id}')

        ax.set_title(f'ON Ratio per BS (every {block} slots)', fontweight='bold')
        ax.set_xlabel('Slot (end of each block)')
        ax.set_ylabel('ON ratio')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

        # =========================================================
        # (2) Local ON ratio per BS (last 10,000 slots) 
        # =========================================================
        ax = axes[1]
        window = 1000
        last_window = 10000

        start_idx = max(0, T - last_window)
        status_slice = self.bs_status_history[start_idx:]
        T_slice = len(status_slice)

        for bs_id in bs_ids:
            local_ratios = []
            x_pts = []
            for i in range(0, T_slice, window):
                chunk = status_slice[i:i + window]
                if len(chunk) == 0:
                    continue
                on_count = sum(1 for s in chunk if s.get(bs_id, 0) == 1)
                ratio = on_count / len(chunk)
                local_ratios.append(ratio)
                global_step = start_idx + i + len(chunk)
                x_pts.append(global_step)

            ax.plot(x_pts, local_ratios, label=f'BS{bs_id}')

        ax.set_title(f'Local ON Ratio per BS (last {last_window} slots, window={window})', fontweight='bold')
        ax.set_xlabel('Slot')
        ax.set_ylabel('ON ratio (per 1000 slots)')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

        # =========================================================
        # (3) Throughput Trend (last 10,000 slots) 
        # =========================================================
        ax = axes[2]
        T_thr = len(self.throughput_history)
        start_idx_thr = max(0, T_thr - last_window)
        thr_slice = self.throughput_history[start_idx_thr:]

        avg_vals = []
        x_pts = []
        for i in range(0, len(thr_slice), window):
            chunk = thr_slice[i:i + window]
            if len(chunk) == 0:
                continue
            avg_vals.append(np.mean(chunk))
            global_step = start_idx_thr + i + len(chunk)
            x_pts.append(global_step)

        ax.plot(x_pts, avg_vals, linewidth=2, color='orange', label=f'Avg Throughput (window={window})')

        ax.set_title(f'Throughput Trend (last {last_window} slots)', fontweight='bold')
        ax.set_xlabel('Slot')
        ax.set_ylabel('Throughput [Gbps]')
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig('dpp_summary_modified.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    area_size = 100
    num_users = 20

    sbs_positions = generate_triangle_coverage(area_size, 35)
    sbs_list = [SmallCellBaseStation(i + 1, pos, 10, 35) for i, pos in enumerate(sbs_positions)]
    users = [UserEquipment(i + 1, (np.random.uniform(10, 90), np.random.uniform(10, 90)))
             for i in range(num_users)]

    dpp = DDPPAlgorithm(sbs_list, users, V=10, power_budget_ratio=0.6, max_slots=100000)
    dpp.run_simulation()
    dpp.plot_results()
