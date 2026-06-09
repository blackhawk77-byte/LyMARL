"""
qplextest.py
"""
from __future__ import annotations

import argparse
import random
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # /home/.../LyMARL
sys.path.insert(0, REPO_ROOT)

from dataclasses import asdict
from typing import List, Tuple
import numpy as np 
import torch
import matplotlib.pyplot as plt
import time
import csv

from LyMARL.basestation import SmallCellBaseStation
from LyMARL.user_equipment import UserEquipment
from LyMARL.core import generate_triangle_coverage
from LyMARL.env_mappo import MAPPOEnvironment
from benchmark.qplex.HeteroQPLEXAgent import HeteroQPLEXAgent, HeteroQPLEXcfg

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _sample_positions_uniform(n: int, low: float=10.0, high: float = 90.0) -> List[Tuple[float, float]]:
    pts = np.random.uniform(low=low, high=high, size=(n, 2))
    return [(float(x), float(y)) for x, y in pts]

def save_logs_csv(logs, path = "./results/train_logs/train_log_qplex.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = set()
    for x in logs:
        keys |= set(x.keys())
    keys = sorted(list(keys))

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for x in logs:
            w.writerow(x)

def moving_avg(x, w: int = 100):
    x = np.asarray(x, dtype=float)
    if w is None or w<=1 or len(x) <w:
        return x, np.arange(len(x))
    k = np.ones(w, dtype=float) / float(w)
    y = np.convolve(x, k, mode='valid')
    idx = np.arange(w-1, w-1+len(y))
    return y, idx

def plot_train_metrics(logs, agent, window: int=1000, save_dir: str = "./results/plots"):
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # =========================
    # (A) logs 기반: thr/on/fair + loss
    # =========================
    step_cursor = 0
    x_roll, thr_mean, on_ratio_100, fair_100 = [], [], [], []
    x_upd, loss_ue, loss_bs = [], [], []

    for row in logs:
        typ = row.get("type", "")
        if typ in ["rollout", "eval"]:
            ep_len = int(float(row.get("ep_len", 0.0)))
            step_cursor += max(0, ep_len)
            x_roll.append(step_cursor)
            thr_mean.append(float(row.get("thr_mean", float("nan"))))
            on_ratio_100.append(float(row.get("on_ratio_mean", float("nan"))))
            fair_100.append(float(row.get("fair_mean", float("nan"))))
        elif typ == "update":
            x_upd.append(step_cursor)
            loss_ue.append(float(row.get("loss_ue", float("nan"))))
            loss_bs.append(float(row.get("loss_bs", float("nan"))))

    def _plot(x, y, title, fname):
        if len(x) == 0:
            return
        plt.figure(figsize=(10, 4))
        plt.plot(x, y, alpha=0.35, label="raw")
        y_ma, idx = moving_avg(y, window)
        if len(y_ma) > 0:
            plt.plot(np.asarray(x)[idx], y_ma, linewidth=2, label=f"MA{window}")
        plt.title(title)
        plt.xlabel("env steps (approx)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(save_dir, f"{fname}_{ts}.png")
        plt.savefig(out, dpi=180)
        plt.close()
        print(f"[plot] saved: {out}")

    _plot(x_roll, thr_mean, "Throughput mean (thr_mean)", "thr_mean_qplex")
    _plot(x_roll, on_ratio_100, "BS on ratio mean (on_ratio_mean)", "on_ratio_mean_qplex")
    _plot(x_roll, fair_100, "Jain fairness (fair_ep)", "fair_ep_qplex")
    _plot(x_upd, loss_ue, "Loss (UE) - QPLEX", "loss_ue_qplex")
    _plot(x_upd, loss_bs, "Loss (BS) - QPLEX", "loss_bs_qplex")

    # =========================
    # (B) agent 기반: UE/BS reward history
    # =========================
    if not hasattr(agent, "step_history"):
        return
    steps = np.asarray(agent.step_history)
    ue_rewards = np.asarray(agent.reward_history_ue, dtype = float)
    bs_rewards = np.asarray(agent.reward_history_bs, dtype = float)

    T = min(len(steps), len(ue_rewards), len(bs_rewards))
    if T <= 0:
        return
    steps = steps[:T]
    ue_rewards = ue_rewards[:T]
    bs_rewards = bs_rewards[:T]

    ue_ma, ue_idx = moving_avg(ue_rewards, window)
    bs_ma, bs_idx = moving_avg(bs_rewards, window)

    fig, axes = plt.subplots(2,1, figsize=(14,8), sharex=True)

    # =========================
    # (1) UE Team Reward
    # =========================
    ax0 = axes[0]
    ax0.plot(steps, ue_rewards, alpha =0.3, label="UE team reward (raw)")
    ax0.plot(steps[ue_idx] ,ue_ma,linewidth =2, label =f"UE team reward (MA{window})")
    ax0.set_ylabel("UE team Reward")
    ax0.legend()
    ax0.grid(True)

    # =========================
    # (2) BS Team Reward
    # =========================
    ax1 = axes[1]
    ax1.plot(steps, bs_rewards, alpha=0.3, label="BS team reward (raw)")
    ax1.plot(steps[bs_idx], bs_ma, linewidth=2, label=f"BS team reward (MA{window})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("BS team Reward")
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"reward_plot_qplex_{ts}.png")
    plt.savefig(out_path,dpi=200)
    print(f"[plot_reward] saved to: {out_path}")
    plt.close(fig)


# -------------------------
# Env builder
# -------------------------
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

# -------------------------
# Train / Eval
# -------------------------
def run_train(args):
    env = build_env(
        n_ue=args.n_ue,
        n_bs=args.n_bs,
        bs_top_k=args.bs_top_k,
        power_budget_ratio=args.power_budget_ratio,
        V=args.V,
        enable_mobility=args.enable_mobility,
        enable_channel_variation=args.enable_channel_variation,
        hard_window_len=args.hard_window_len,
        on_window=args.on_window,
        bs_over_penalty=args.bs_over_penalty
    )

    cfg = HeteroQPLEXcfg(
        hidden_dim=args.hidden_dim,
        lr = args.lr,
        gamma=args.gamma,
        tau=args.tau,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        capacity_episodes=args.capacity_episodes,
        update_interval_steps=args.update_interval_steps,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        n_heads=args.n_heads,
    )
    
    agent = HeteroQPLEXAgent(env=env, cfg=cfg, device=args.device)

    print("\n[QPLEX TEST] Config:")
    print({
        "env":{
            "n_ue": args.n_ue,
            "n_bs": args.n_bs,
            "bs_top_k": args.bs_top_k,
            "power_budget_ratio": args.power_budget_ratio,
            "V": args.V,
            "enable_mobility": args.enable_mobility,
            "enable_channel_variation": args.enable_channel_variation,
            "hard_window_len": args.hard_window_len,
            "on_window": args.on_window,
            "bs_over_penalty": args.bs_over_penalty
        },
        "agent": asdict(cfg),
    })
    print()
    
    logs = agent.train(n_env_steps=args.n_env_steps, rollout_horizon=args.chunk_len)
    
    if args.save_ckpt:
        agent.save(args.ckpt_path)
        print(f"[checkpoint] saved to: {args.ckpt_path}")
    
    rollouts = [x for x in logs if x.get("type") == "rollout"]
    updates = [x for x in logs if x.get("type") == "update"]
    if rollouts:
        thr_all_mean = float(np.mean(agent.thr_history)) if len(agent.thr_history) > 0 else float("nan")
        fair_all_mean = float(np.mean(agent.fair_history_100step)) if len(agent.fair_history_100step) > 0 else float("nan")
        on_all_mean = float(np.mean(agent.on_ratio_history)) if len(agent.on_ratio_history) > 0 else float("nan")
        last = rollouts[-1]
        print(
            f"[DONE] env_steps={agent.total_env_steps} | last_ep_len={last['ep_len']:.0f} "
            f"| last_ep_r_ue_sum={last.get('ep_r_ue_sum', float('nan')):.3f} "
            f"| last_ep_r_bs_sum={last.get('ep_r_bs_sum', float('nan')):.3f} "
            f"| epsilon={last['epsilon']:.3f}"
            f"| thr_mean={thr_all_mean:.3f} "
            f"| fair_mean={fair_all_mean:.3f} "
            f"| on_ratio={on_all_mean:.3f} "
        )
    if updates:
        last_u = updates[-1]
        print(
            f"[DONE] last_update loss= (ue={last_u.get('loss_ue', float('nan')):.4f}, "
            f"bs={last_u.get('loss_bs', float('nan')):.4f}) " 
            f"| epsilon={last_u.get('epsilon', float('nan')):.3f}"
        )
    plot_train_metrics(logs, agent, save_dir="./results/plots", window=100)
    save_logs_csv(logs, path = "./results/train_logs/train_log_qplex.csv")

@torch.no_grad()
def run_eval(args):
    env = build_env(
        n_ue=args.n_ue,
        n_bs=args.n_bs,
        bs_top_k=args.bs_top_k,
        power_budget_ratio=args.power_budget_ratio,
        V=args.V,
        enable_mobility=args.enable_mobility,
        enable_channel_variation=args.enable_channel_variation,
        hard_window_len=args.hard_window_len,
        on_window=args.on_window,
        bs_over_penalty=args.bs_over_penalty
    )
    cfg = HeteroQPLEXcfg(
        hidden_dim=args.hidden_dim,
        lr = args.lr,
        gamma=args.gamma,
        tau=args.tau,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        capacity_episodes=args.capacity_episodes,
        update_interval_steps=args.update_interval_steps,
        eps_start=0.00,
        eps_end=0.00,
        eps_decay=1.0,
        n_heads=args.n_heads,
    )
    agent = HeteroQPLEXAgent(env=env, cfg=cfg, device=args.device)
    
    if args.load_ckpt:
        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {args.ckpt_path}")
        agent.load(args.ckpt_path)
        print(f"[EVAL] loaded {args.ckpt_path}")
    
    agent.eps = args.eval_epsilon
    logs = []
    eval_horizon = 50000
    print(f"\n[EVAL] episodes={args.episodes} | horizon={eval_horizon} | epsilon={args.eval_epsilon}\n")

    for ep_i in range(args.episodes):
        steps_done = 0
        rollout_idx = 0
        while steps_done < eval_horizon:
            steps_to_run = min(args.chunk_len, eval_horizon - steps_done)
            out = agent.rollout_episode(n_steps=steps_to_run)
            
            logs.append({
                "type": "eval",
                "episode": ep_i,
                **out
            })
            steps_done += int(out.get("ep_len", 0))
            rollout_idx += 1

        print(
            f"  ep={ep_i:03d} | len={out.get('ep_len', 0.0):.0f} "
            f"| r_ue_sum={out.get('ep_r_ue_sum', float('nan')):.3f} "
            f"| r_bs_sum={out.get('ep_r_bs_sum', float('nan')):.3f} "
            f"| epsilon={out.get('epsilon', float('nan')):.3f}"
            f"| thr_mean={out.get('thr_mean', float('nan')):.3f} "
            f"| fair_mean={out.get('fair_mean', float('nan')):.3f}"
            f"| on_ratio={out.get('on_ratio_mean', float('nan')):.3f} "
        )
    save_logs_csv(logs, path="./results/eval_logs/eval_log_qplex.csv")
    plot_train_metrics(logs, agent, save_dir="./results/eval_plots", window=100)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="eval")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # env
    parser.add_argument("--n_ue", type=int, default=20)
    parser.add_argument("--n_bs", type=int, default=3)
    parser.add_argument("--bs_top_k", type=int, default=5)
    parser.add_argument("--power_budget_ratio", type=float, default=0.6)
    parser.add_argument("--V", type=float, default=5.0)
    parser.add_argument("--enable_mobility", action="store_true", default=True)
    parser.add_argument("--enable_channel_variation", action="store_true", default=True)
    parser.add_argument("--hard_window_len", type=int, default=10000)
    parser.add_argument("--on_window", type=int, default=100)
    parser.add_argument("--bs_over_penalty", type=float, default=100.0)
    
    # rollout/train
    parser.add_argument("--n_env_steps", type=int, default=50000)
    parser.add_argument("--chunk_len", type=int, default=HeteroQPLEXcfg.chunk_len)
    
    # agent
    parser.add_argument("--hidden_dim", type=int, default=HeteroQPLEXcfg.hidden_dim)
    parser.add_argument("--lr", type=float, default=HeteroQPLEXcfg.lr)
    parser.add_argument("--gamma", type=float, default=HeteroQPLEXcfg.gamma)
    parser.add_argument("--tau", type=float, default=HeteroQPLEXcfg.tau)
    parser.add_argument("--grad_clip", type=float, default=HeteroQPLEXcfg.grad_clip)
    parser.add_argument("--batch_size", type=int, default=HeteroQPLEXcfg.batch_size)
    parser.add_argument("--seq_len", type=int, default=HeteroQPLEXcfg.seq_len)
    parser.add_argument("--capacity_episodes", type=int, default=HeteroQPLEXcfg.capacity_episodes)
    parser.add_argument("--update_interval_steps", type=int, default=HeteroQPLEXcfg.update_interval_steps)
    parser.add_argument("--eps_start", type=float, default=HeteroQPLEXcfg.eps_start)
    parser.add_argument("--eps_end", type=float, default=HeteroQPLEXcfg.eps_end)
    parser.add_argument("--eps_decay", type=float, default=HeteroQPLEXcfg.eps_decay)
    parser.add_argument("--n_heads", type=int, default=HeteroQPLEXcfg.n_heads)

    # ckpt
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/heteroqplex_queue.pt")
    parser.add_argument("--save_ckpt", action="store_true", default=True)
    parser.add_argument("--load_ckpt", action="store_true", default=True)  
    
    # eval
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval_epsilon", type=float, default=0.05)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
