from __future__ import annotations

import os
import sys
import random
import json
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt

import numpy as np
import torch

# -------------------------------------------------
# path
# -------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from LyMARL.basestation import SmallCellBaseStation
from LyMARL.user_equipment import UserEquipment
from LyMARL.core import generate_triangle_coverage
from LyMARL.env_mappo import MAPPOEnvironment
from LyMARL.trainer_mappo import MAPPOTrainer

from benchmark.qmix.HeteroQMIXAgent import HeteroQMIXAgent, HeteroQMIXcfg
from benchmark.qplex.HeteroQPLEXAgent import HeteroQPLEXAgent, HeteroQPLEXcfg

# -------------------------------------------------
# Utils
# -------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_float(x, default=0.0):
    if x is None:
        return float(default)
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)

def pick_metric(d: Dict[str, Any], candidates: List[str], default=0.0):
    for key in candidates:
        if key in d:
            return to_float(d[key], default)
    return float(default)
    
def summarize_metric(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if len(arr) > 0 else 0.0,
        "std": float(np.std(arr)) if len(arr) > 0 else 0.0,
        "min": float(np.min(arr)) if len(arr) > 0 else 0.0,
        "25%": float(np.percentile(arr, 25)) if len(arr) > 0 else 0.0,
        "50%": float(np.percentile(arr, 50)) if len(arr) > 0 else 0.0,
        "75%": float(np.percentile(arr, 75)) if len(arr) > 0 else 0.0,   
        "max": float(np.max(arr)) if len(arr) > 0 else 0.0,}

def moving_average(x, window = 1000):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    if window <= 1:
        return x
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode='same')

def compute_scr_from_power_history(
    power_history: Dict[int, List[float]],
    power_budget_ratio: float,
    hard_window_len: int,
) -> float:
    """
    SCR = average fraction of slots where each BS still has positive remaining budget.
    Budget is reset every hard_window_len slots.
    """
    scr_bs = []

    for bs_id, hist in power_history.items():
        if len(hist) == 0:
            continue

        limit = int(np.floor(power_budget_ratio * hard_window_len))
        used = 0
        alive_flags = []

        for t, p in enumerate(hist):
            # reset at the beginning of each hard window
            if t % hard_window_len == 0:
                used = 0

            # remaining budget is positive before this slot if used < limit
            alive_flags.append(1.0 if used < limit else 0.0)

            if float(p) > 0.0:
                used += 1

        scr_bs.append(float(np.mean(alive_flags)))

    return float(np.mean(scr_bs)) if len(scr_bs) > 0 else 0.0


def compute_block_jfi_from_rate_history(
    rate_history: List[np.ndarray],
    block_size: int = 1000,
) -> float:
    """
    Compute JFI for each block, then average over blocks.

    Example:
    T=100000, block_size=1000
    -> compute 100 JFI values
    -> final JFI = mean of 100 block JFIs
    """
    if len(rate_history) == 0:
        return 0.0

    rate_array = np.asarray(rate_history, dtype=np.float64)  # shape: (T, N_ue)
    T = rate_array.shape[0]

    jfi_blocks = []

    for start in range(0, T, block_size):
        end = min(start + block_size, T)
        block = rate_array[start:end]

        if block.shape[0] == 0:
            continue

        per_user_avg = block.mean(axis=0)

        numerator = float(np.square(np.sum(per_user_avg)))
        denominator = float(len(per_user_avg) * np.sum(np.square(per_user_avg)) + 1e-12)

        jfi = numerator / denominator if denominator > 0 else 0.0
        jfi_blocks.append(jfi)

    return float(np.mean(jfi_blocks)) if len(jfi_blocks) > 0 else 0.0


def compute_block_on_ratio_from_power_history(
    power_history: Dict[int, List[float]],
    block_size: int = 1000,
) -> float:
    """
    Compute ON-ratio for each block, then average over blocks.

    For each 1000-step block:
        BS-level ON-ratio -> average across BSs
    Then:
        final ON-ratio = average across blocks
    """
    lengths = [len(hist) for hist in power_history.values() if len(hist) > 0]
    if len(lengths) == 0:
        return 0.0

    T = min(lengths)
    block_on_ratios = []

    for start in range(0, T, block_size):
        end = min(start + block_size, T)

        bs_vals = []

        for bs_id, hist in power_history.items():
            block = hist[start:end]

            if len(block) == 0:
                continue

            on_ratio_bs = np.mean([1.0 if float(p) > 0.0 else 0.0 for p in block])
            bs_vals.append(float(on_ratio_bs))

        if len(bs_vals) > 0:
            block_on_ratios.append(float(np.mean(bs_vals)))

    return float(np.mean(block_on_ratios)) if len(block_on_ratios) > 0 else 0.0

# -------------------------------------------------
# Environment builder
# -------------------------------------------------
def _sample_positions_uniform(n: int, low: float=10.0, high: float = 90.0) -> List[Tuple[float, float]]:
    pts = np.random.uniform(low=low, high=high, size=(n, 2))
    return [(float(x), float(y)) for x, y in pts]

def build_env(n_ue: int, n_bs: int, bs_top_k: int, power_budget_ratio: float,
              V: float, enable_mobility: bool, enable_channel_variation: bool,
              hard_window_len: int, on_window: int):
    tri_pos = generate_triangle_coverage()
    bs_pos = list(tri_pos[:min(len(tri_pos), n_bs)])
    if len(bs_pos) < n_bs:
        bs_pos += _sample_positions_uniform(n_bs - len(bs_pos))

    base_stations = [
        SmallCellBaseStation(bs_id = i+1, position=bs_pos[i], beam_limit=np.inf, coverage_radius=np.inf)
        for i in range(n_bs)
    ]

    # UE positions (env.reset() will randomize again)
    ue_pos = _sample_positions_uniform(n_ue)
    users = [
        UserEquipment(ue_id = i+1, position=ue_pos[i]) for i in range(n_ue)
    ]
    env = MAPPOEnvironment(
        base_stations=base_stations,
        users=users,
        V=V,
        power_budget_ratio=power_budget_ratio,
        enable_mobility=enable_mobility,
        enable_channel_variation=enable_channel_variation,
        on_window=on_window,
        bs_top_k=bs_top_k,
        hard_window_len=hard_window_len,
    )
    return env

# -------------------------------------------------
# Agent / Trainer loaders
# -------------------------------------------------
def load_qmix_agent(env, ckpt_path: str, device: str = "cuda"):
    cfg = HeteroQMIXcfg()
    agent = HeteroQMIXAgent(env, cfg, device=device)
    agent.load(ckpt_path)
    agent.eps = 0.0  # evaluation mode
    return agent

def load_qplex_agent(env, ckpt_path: str, device: str = "cuda"):
    cfg = HeteroQPLEXcfg()
    agent = HeteroQPLEXAgent(env, cfg, device=device)
    agent.load(ckpt_path)
    agent.eps = 0.0  # evaluation mode
    return agent

def load_mappo_trainer(env, ckpt_path: str, device: str = "cuda"):
    trainer = MAPPOTrainer(env)
    checkpoint = torch.load(ckpt_path, map_location=trainer.device)

    trainer.ue_actor.load_state_dict(checkpoint["ue_actor"])
    trainer.bs_actor.load_state_dict(checkpoint["bs_actor"])
    trainer.critic_ue.load_state_dict(checkpoint["critic_ue"])
    trainer.critic_bs.load_state_dict(checkpoint["critic_bs"])

    if "vn_ue" in checkpoint:
        trainer.vn_ue.load_state_dict(checkpoint["vn_ue"])
    if "vn_bs" in checkpoint:
        trainer.vn_bs.load_state_dict(checkpoint["vn_bs"])
    return trainer

# -------------------------------------------------
# Single episode eval
# rollout / evaluate 리턴 키가 다를 수 있으니 후보 키를 넓게 잡음
# -------------------------------------------------
@torch.no_grad()
def eval_one_episode_value_agent(
    agent,
    rollout_horizon: int = 100000,
    eval_epsilon: float = 0.1,
    fairness_block_size: int = 1000,
    on_block_size: int = 100,
) -> Dict[str, Any]:
    agent.eps = float(eval_epsilon)

    agent._need_env_reset = True
    agent._maybe_reset_env()
    local_obs, global_obs = agent._cur_local_obs, agent._cur_global_obs

    throughput_history = []
    rate_history = []

    power_history = {bs.bs_id: [] for bs in agent.env.base_stations}

    reward_ue_hist = []
    reward_bs_hist = []

    for _ in range(rollout_horizon):
        (
            ue_actions, ue_actions_arr, ue_obs_batch, ue_masks_batch,
            bs_actions, bs_actions_arr, bs_obs_batch, bs_masks_batch, cand_lists
        ) = agent.select_actions(local_obs, global_obs, eps_override=eval_epsilon)

        next_local_obs, next_global_obs, info, done = agent.env.step_joint(
            ue_actions=ue_actions,
            bs_actions=bs_actions,
            cand_lists=cand_lists,
        )

        # 1) Throughput 저장
        thr = float(info.get("total_throughput", 0.0))
        throughput_history.append(thr)

        # 2) User별 served rate 저장
        served_rates = info.get("served_rates", None)

        if served_rates is not None:
            step_rates = np.zeros(agent.N_ue, dtype=np.float64)

            for ue_id, r in served_rates.items():
                idx = int(ue_id) - 1
                if 0 <= idx < agent.N_ue:
                    step_rates[idx] = float(r)

            rate_history.append(step_rates)

        # 3) BS별 power 저장
        power_consumed = info.get("power_consumed", {})

        for bs in agent.env.base_stations:
            p = float(power_consumed.get(bs.bs_id, 0.0))
            power_history[bs.bs_id].append(p)

        # 4) Reward는 plot용으로만 저장
        reward_ue_hist.append(float(info.get("ue_team_reward", 0.0)))
        reward_bs_hist.append(float(info.get("bs_team_reward", 0.0)))

        local_obs, global_obs = next_local_obs, next_global_obs

        if done:
            break

    # 여기서부터 eval 끝난 뒤 한 번만 계산
    throughput = float(np.mean(throughput_history)) if len(throughput_history) > 0 else 0.0

    fairness = compute_block_jfi_from_rate_history(
        rate_history=rate_history,
        block_size=fairness_block_size,
    )

    on_ratio = compute_block_on_ratio_from_power_history(
        power_history=power_history,
        block_size=on_block_size,
    )

    scr = compute_scr_from_power_history(
        power_history=power_history,
        power_budget_ratio=agent.env.power_budget_ratio,
        hard_window_len=agent.env.hard_window_len,
    )

    agent._cur_local_obs, agent._cur_global_obs = local_obs, global_obs

    return {
        "throughput": throughput,
        "fairness": float(fairness),
        "on_ratio": float(on_ratio),
        "scr": float(scr),

        # raw histories for npz save
        "throughput_history": np.asarray(throughput_history, dtype=np.float64),
        "rate_history": np.asarray(rate_history, dtype=np.float64),  # shape: (T, N_ue)
        "power_history": {
            int(bs_id): np.asarray(hist, dtype=np.float64)
            for bs_id, hist in power_history.items()
        },

        "reward_ue_hist": reward_ue_hist,
        "reward_bs_hist": reward_bs_hist,
    }

@torch.no_grad()
def eval_one_ep_qmix(
    agent,
    rollout_horizon: int = 100000,
    eval_epsilon: float = 0.1,
    fairness_block_size: int = 1000,
    on_block_size: int = 100,
) -> Dict[str, Any]:
    return eval_one_episode_value_agent(
        agent,
        rollout_horizon=rollout_horizon,
        eval_epsilon=eval_epsilon,
        fairness_block_size=fairness_block_size,
        on_block_size=on_block_size
    )


@torch.no_grad()
def eval_one_episode_qplex(
    agent,
    rollout_horizon: int = 100000,
    eval_epsilon: float = 0.1,
    fairness_block_size: int = 1000,
    on_block_size: int = 100,
) -> Dict[str, Any]:
    return eval_one_episode_value_agent(
        agent,
        rollout_horizon=rollout_horizon,
        eval_epsilon=eval_epsilon,
        fairness_block_size=fairness_block_size,
        on_block_size=on_block_size
    )

@torch.no_grad()
def eval_one_episode_mappo(trainer, rollout_horizon: int = 100000, eval_epsilon: float = 0.0) -> Dict[str, Any]:
    out = trainer.evaluate(n_steps=rollout_horizon)

    throughput = float(np.mean(out["throughput_history"])) if len(out["throughput_history"]) > 0 else 0.0

    # fairness_history already uses env.calculate_jain_fairness()
    fairness = float(out["fairness_history"][-1]) if len(out["fairness_history"]) > 0 else 0.0

    power_history = out["power_history"]

    on_vals = []
    for bs_id, hist in power_history.items():
        if len(hist) == 0:
            continue
        on_ratio_bs = sum(1 for p in hist if p > 0) / len(hist)
        on_vals.append(on_ratio_bs)
    on_ratio = float(np.mean(on_vals)) if len(on_vals) > 0 else 0.0

    scr = compute_scr_from_power_history(
        power_history=power_history,
        power_budget_ratio=trainer.env.power_budget_ratio,
        hard_window_len=trainer.env.hard_window_len,
    )

    reward_ue_hist = list(out.get("ue_team_reward", out.get("reward_ue_hist", [])))
    reward_bs_hist = list(out.get("bs_reward_mean", out.get("reward_bs_hist", [])))

    return {
        "throughput": throughput,
        "fairness": fairness,
        "on_ratio": on_ratio,
        "scr": float(scr),
        "reward_ue_hist": reward_ue_hist,
        "reward_bs_hist": reward_bs_hist,
    }

# -------------------------------------------------
# Multi-episode evaluator
# -------------------------------------------------
def evaluate_model(
    model_name: str, 
    ckpt_path: str,
    env_kwargs: Dict[str, Any],
    seeds: List[int], 
    rollout_horizon: int = 100000,
    seed: int=0,
    device: str = "cuda",
) -> Dict[str, Any]:
    
    reward_ue_hists, reward_bs_hists = [], []
    
    throughput_list, fairness_list, on_ratio_list, scr_list = [], [], [], []
    
    for seed in seeds:
        set_seed(seed)
        env = build_env(**env_kwargs)
        env.set_hard_constraint(True)

        # if model_name.lower() == "qmix":
        #     model = load_qmix_agent(env, ckpt_path, device=device)
        #     eval_fn = eval_one_ep_qmix
        if model_name.lower() == "qplex":
            model = load_qplex_agent(env, ckpt_path, device=device)
            eval_fn = eval_one_episode_qplex
        # elif model_name.lower() == "mappo":
        #     model = load_mappo_trainer(env, ckpt_path, device=device)
        #     eval_fn = eval_one_episode_mappo
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        ep_metrics = eval_fn(
            model,
            rollout_horizon=rollout_horizon,
            eval_epsilon=0.1,
            fairness_block_size=1000,
            on_block_size=100,
        )

        save_eval_npz(
            save_dir=os.path.join(REPO_ROOT, "results"),
            model_name=model_name,
            seed=seed,
            ep_metrics=ep_metrics,
        )

        throughput_list.append(ep_metrics["throughput"])
        fairness_list.append(ep_metrics["fairness"])
        on_ratio_list.append(ep_metrics["on_ratio"])
        scr_list.append(ep_metrics["scr"])
        
        reward_ue_hists.append(ep_metrics["reward_ue_hist"])
        reward_bs_hists.append(ep_metrics["reward_bs_hist"])

        print(
            f"[{model_name.upper()}][seed={seed}] "
            f"Throughput: {ep_metrics['throughput']:.4f} | "
            f"Fairness: {ep_metrics['fairness']:.4f} | "
            f"ON Ratio: {ep_metrics['on_ratio']:.4f} | "
            f"SCR: {ep_metrics['scr']:.4f}"
        )
    
    result = {
        "model_name": model_name,
        "ckpt_path": ckpt_path,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "rollout_horizon": rollout_horizon,
        "throughput": summarize_metric(throughput_list),
        "fairness": summarize_metric(fairness_list),
        "on_ratio": summarize_metric(on_ratio_list),
        "scr": summarize_metric(scr_list),
        "device": device,
        "reward_ue_hists": reward_ue_hists,
        "reward_bs_hists": reward_bs_hists,
    }

    return result

def save_eval_npz(
    save_dir: str,
    model_name: str,
    seed: int,
    ep_metrics: Dict[str, Any],
):
    os.makedirs(save_dir, exist_ok=True)

    power_history = ep_metrics["power_history"]

    save_path = os.path.join(
        save_dir,
        f"{model_name.upper()}_eval_seed{seed}.npz"
    )

    np.savez(
        save_path,

        # scalar metrics
        throughput=np.float64(ep_metrics["throughput"]),
        fairness=np.float64(ep_metrics["fairness"]),
        on_ratio=np.float64(ep_metrics["on_ratio"]),
        scr=np.float64(ep_metrics["scr"]),

        # raw histories
        throughput_history=np.asarray(ep_metrics["throughput_history"], dtype=np.float64),
        rate_history=np.asarray(ep_metrics["rate_history"], dtype=np.float64),

        # BS별 power history
        power_bs1=np.asarray(power_history.get(1, []), dtype=np.float64),
        power_bs2=np.asarray(power_history.get(2, []), dtype=np.float64),
        power_bs3=np.asarray(power_history.get(3, []), dtype=np.float64),

        # optional reward histories
        reward_ue_hist=np.asarray(ep_metrics["reward_ue_hist"], dtype=np.float64),
        reward_bs_hist=np.asarray(ep_metrics["reward_bs_hist"], dtype=np.float64),
    )

    print(f"Saved eval npz: {save_path}")

def print_summary_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 100)
    print(
        f"{'Model':<10}"
        f"{'Throughput(mean±std)':<24}"
        f"{'JFI-1000(mean±std)':<24}"
        f"{'ON-100(mean±std)':<24}"
        f"{'SCR(mean±std)':<24}"
    )
    print("-" * 100)

    for r in results:
        thr_str = f"{r['throughput']['mean']:.4f}±{r['throughput']['std']:.4f}"
        fair_str = f"{r['fairness']['mean']:.4f}±{r['fairness']['std']:.4f}"
        on_str = f"{r['on_ratio']['mean']:.4f}±{r['on_ratio']['std']:.4f}"
        scr_str = f"{r['scr']['mean']:.4f}±{r['scr']['std']:.4f}"

        print(
            f"{r['model_name']:<10}"
            f"{thr_str:<24}"
            f"{fair_str:<24}"
            f"{on_str:<24}"
            f"{scr_str:<24}"
        )
    print("=" * 100 + "\n")

def print_latex_table_rows(results: List[Dict[str, Any]]):
    print("\nLaTeX table rows:")
    for r in results:
        method = r["model_name"].upper()

        thr = f"{r['throughput']['mean']:.3f}$\\pm${r['throughput']['std']:.3f}"
        jfi = f"{r['fairness']['mean']:.3f}$\\pm${r['fairness']['std']:.3f}"
        on = f"{r['on_ratio']['mean']:.3f}$\\pm${r['on_ratio']['std']:.3f}"
        scr = f"{r['scr']['mean']:.3f}$\\pm${r['scr']['std']:.3f}"

        print(f"{method} & {thr} & {jfi} & {on} & {scr} \\\\")

def plot_reward_histories(results: List[Dict[str, Any]], 
                          reward_key: str,
                          save_path: str,
                          y_label: str,
                          ma_window: int = 1000):
    plt.figure(figsize=(12, 6))
    for r in results:
        arr = np.asarray(r[reward_key], dtype=np.float64)  # (n_seeds, T)
        mean_curve = arr.mean(axis=0)
        mean_curve_ma = moving_average(mean_curve, window=ma_window)

        x=np.arange(1, len(mean_curve_ma) + 1)
        rawline, = plt.plot(x, mean_curve, alpha=0.2)
        plt.plot(x, mean_curve_ma, linewidth=2, color=rawline.get_color(), label=r['model_name'].upper())
    
    plt.xlabel("Eval Step")
    plt.ylabel(y_label)
    plt.title("Reward History Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved reward history plot to {save_path}")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_kwargs = dict(
        n_ue=20,
        n_bs=3,
        bs_top_k=5,
        power_budget_ratio=0.6,
        V=5.0,
        enable_mobility=True,
        enable_channel_variation=True,
        hard_window_len=10000,
        on_window=1000,
    )

    ckpts = {
        # "qmix": os.path.join(REPO_ROOT, "results", "QMIX.pt"),``
        "qplex": os.path.join(REPO_ROOT, "results", "QPLEX.pt"),
        # "mappo": os.path.join(REPO_ROOT, "results", "LyMARL.pt"),
    }

    seeds = [0, 80, 1000] 
    results = []

    for model_name, ckpt_path in ckpts.items():
        res = evaluate_model(
            model_name=model_name,
            ckpt_path=ckpt_path,
            env_kwargs=env_kwargs,
            seeds=seeds,
            rollout_horizon=100000,
            device=device,
        )
        results.append(res)

    print_summary_table(results)
    print_latex_table_rows(results)
    
    save_path = os.path.join(REPO_ROOT, "results", "eval_compare_3models.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved evaluation results to {save_path}")

    # plot_reward_histories(results, reward_key="reward_ue_hists", y_label="UE Reward",
    #                       save_path=os.path.join(REPO_ROOT, "results", "reward_ue_ma1000.png"))
    # plot_reward_histories(results, reward_key="reward_bs_hists", y_label="BS Reward",
    #                       save_path=os.path.join(REPO_ROOT, "results", "reward_bs_ma1000.png"))

if __name__ == "__main__":
    main()