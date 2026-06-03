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

# -------------------------------------------------
# Environment builder
# -------------------------------------------------
def _sample_positions_uniform(n: int, low: float=10.0, high: float = 90.0) -> List[Tuple[float, float]]:
    pts = np.random.uniform(low=low, high=high, size=(n, 2))
    return [(float(x), float(y)) for x, y in pts]

def build_env(n_ue: int, n_bs: int, bs_top_k: int, power_budget_ratio: float,
              V: float, enable_mobility: bool, enable_channel_variation: bool,
              hard_window_len: int, on_window: int, bs_over_penalty: float):
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
        bs_over_penalty=bs_over_penalty,
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
def eval_one_ep_qmix(agent, rollout_horizon: int = 200) -> Dict[str, Any]:
    out = agent.rollout_episode(n_steps=rollout_horizon)
    ep_len = max(1.0, float(out["ep_len"]))
    n_bs = float(agent.env.n_bs) if hasattr(agent.env, "n_bs") else float(len(agent.env.base_stations))

    return {
        "throughput": float(out["thr_mean"]),
        "fairness": float(out["fair_mean"]),
        "on_ratio": float(out["on_ratio_mean"]),
        "reward_ue": float(out["ep_r_ue_sum"]) / ep_len,
        "reward_bs": float(out["ep_r_bs_sum"]) / ep_len / n_bs,
        "reward_ue_hist": list(out.get("reward_ue_hist", [])),
        "reward_bs_hist": list(out.get("reward_bs_hist", [])),
    }

@torch.no_grad()
def eval_one_episode_qplex(agent, rollout_horizon: int = 200) -> Dict[str, Any]:
    out = agent.rollout_episode(n_steps=rollout_horizon)
    ep_len = max(1.0, float(out["ep_len"]))
    n_bs = float(agent.env.n_bs) if hasattr(agent.env, "n_bs") else float(len(agent.env.base_stations))

    return {
        "throughput": float(out["thr_mean"]),
        "fairness": float(out["fair_mean"]),
        "on_ratio": float(out["on_ratio_mean"]),
        "reward_ue": float(out["ep_r_ue_sum"]) / ep_len,
        "reward_bs": float(out["ep_r_bs_sum"]) / ep_len / n_bs,
        "reward_ue_hist": list(out.get("reward_ue_hist", [])),
        "reward_bs_hist": list(out.get("reward_bs_hist", [])),
    }

@torch.no_grad()
def eval_one_episode_mappo(trainer, rollout_horizon: int = 200) -> Dict[str, Any]:
    out = trainer.evaluate(n_steps=rollout_horizon)

    throughput = float(np.mean(out["throughput_history"])) if len(out["throughput_history"]) > 0 else 0.0
    fairness = float(np.mean(out["fairness_history"])) if len(out["fairness_history"]) > 0 else 0.0

    power_history = out["power_history"]
    on_vals = []
    for bs_id, hist in power_history.items():
        if len(hist) == 0:
            continue
        on_ratio_bs = sum(1 for p in hist if p > 0) / len(hist)
        on_vals.append(on_ratio_bs)
    on_ratio = float(np.mean(on_vals)) if len(on_vals) > 0 else 0.0

    reward_ue = float(np.mean(out["reward_ue_hist"])) if len(out["reward_ue_hist"]) > 0 else 0.0
    reward_bs = float(np.mean(out["reward_bs_hist"])) if len(out["reward_bs_hist"]) > 0 else 0.0

    return {
        "throughput": throughput,
        "fairness": fairness,
        "on_ratio": on_ratio,
        "reward_ue": reward_ue,
        "reward_bs": reward_bs,
        "reward_ue_hist": list(out.get("reward_ue_hist", [])),
        "reward_bs_hist": list(out.get("reward_bs_hist", [])),
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
    
    throughput_list, fairness_list, on_ratio_list, reward_ue_list, reward_bs_list = [], [], [], [], []
    
    for seed in seeds:
        set_seed(seed)
        env = build_env(**env_kwargs)

        if model_name.lower() == "qmix":
            model = load_qmix_agent(env, ckpt_path, device=device)
            eval_fn = eval_one_ep_qmix
        elif model_name.lower() == "qplex":
            model = load_qplex_agent(env, ckpt_path, device=device)
            eval_fn = eval_one_episode_qplex
        elif model_name.lower() == "mappo":
            model = load_mappo_trainer(env, ckpt_path, device=device)
            eval_fn = eval_one_episode_mappo
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        ep_metrics = eval_fn(model, rollout_horizon=rollout_horizon)

        throughput_list.append(ep_metrics["throughput"])
        fairness_list.append(ep_metrics["fairness"])
        on_ratio_list.append(ep_metrics["on_ratio"])
        reward_ue_list.append(ep_metrics["reward_ue"])
        reward_bs_list.append(ep_metrics["reward_bs"])
        reward_ue_hists.append(ep_metrics["reward_ue_hist"])
        reward_bs_hists.append(ep_metrics["reward_bs_hist"])

        print(
            f"[{model_name.upper()}][seed={seed}] "
            f"Throughput: {ep_metrics['throughput']:.4f} | "
            f"Fairness: {ep_metrics['fairness']:.4f} | "
            f"ON Ratio: {ep_metrics['on_ratio']:.4f} | "
            f"Reward UE: {ep_metrics['reward_ue']:.4f} | "
            f"Reward BS: {ep_metrics['reward_bs']:.4f}"
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
        "reward_ue": summarize_metric(reward_ue_list),
        "reward_bs": summarize_metric(reward_bs_list),
        "device": device,
        "reward_ue_hists": reward_ue_hists,
        "reward_bs_hists": reward_bs_hists,
    }

    return result

def print_summary_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 100)
    print(
        f"{'Model':<10}"
        f"{'Throughput(mean±std)':<24}"
        f"{'Fairness(mean±std)':<24}"
        f"{'OnRatio(mean±std)':<24}"
        f"{'Reward UE(mean±std)':<24}"
        f"{'Reward BS(mean±std)':<24}"
    )
    print("-" * 100)

    for r in results:
        thr_str = f"{r['throughput']['mean']:.4f}±{r['throughput']['std']:.4f}"
        fair_str = f"{r['fairness']['mean']:.4f}±{r['fairness']['std']:.4f}"
        on_str = f"{r['on_ratio']['mean']:.4f}±{r['on_ratio']['std']:.4f}"
        rew_ue_str = f"{r['reward_ue']['mean']:.4f}±{r['reward_ue']['std']:.4f}"
        rew_bs_str = f"{r['reward_bs']['mean']:.4f}±{r['reward_bs']['std']:.4f}"
        print(
            f"{r['model_name']:<10}"
            f"{thr_str:<24}"
            f"{fair_str:<24}"
            f"{on_str:<24}"
            f"{rew_ue_str:<24}"
            f"{rew_bs_str:<24}"
        )
    print("=" * 100 + "\n")

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
        on_window=100,
        bs_over_penalty=100.0,
    )

    ckpts = {
        "qmix": os.path.join(REPO_ROOT, "checkpoints", "heteroqmix_queue.pt"),
        "qplex": os.path.join(REPO_ROOT, "checkpoints", "heteroqplex_queue.pt"),
        "mappo": os.path.join(REPO_ROOT, "checkpoints", "mappo_50k.pt"),
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
    
    save_path = os.path.join(REPO_ROOT, "results", "eval_compare_3models.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved evaluation results to {save_path}")

    plot_reward_histories(results, reward_key="reward_ue_hists", y_label="UE Reward",
                          save_path=os.path.join(REPO_ROOT, "results", "reward_ue_ma1000.png"))
    plot_reward_histories(results, reward_key="reward_bs_hists", y_label="BS Reward",
                          save_path=os.path.join(REPO_ROOT, "results", "reward_bs_ma1000.png"))

if __name__ == "__main__":
    main()