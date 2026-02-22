"""
Usage examples:
  python qmixtest.py --n_env_steps 5000 --rollout_horizon 200 --device cuda
  python qmixtest.py --mode eval --episodes 5 --eval_epsilon 0.05
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
from basestation import SmallCellBaseStation
from user_equipment import UserEquipment
from core import generate_triangle_coverage
from LyMARL.env import MAPPOEnvironment
from LyMARL.trainer import MAPPOTrainer
from benchmark.HeteroQMIXAgent import HeteroQMIXAgent, HeteroQMIXcfg
import matplotlib.pyplot as plt
import random
import time
import csv

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _sample_positions_uniform(n: int, low: float=10.0, high: float = 90.0) -> List[Tuple[float, float]]:
    pts = np.random.uniform(low=low, high=high, size=(n, 2))
    return [(float(x), float(y)) for x, y in pts]

def save_logs_csv(logs, path = "./results/train_logs/train_log.csv"):
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
    x = np.arange(w-1, w-1+len(y))
    return y, x

def plot_train_metrics(logs, agent, window: int=100, save_dir: str = "./results/plots"):
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # =========================
    # (A) logs 기반: thr/on/fair + loss
    # =========================
    step_cursor = 0
    x_roll, thr_mean, on_ratio, fair_ep = [], [], [], []
    x_upd, loss, loss_ue, loss_bs = [], [], [], []

    for row in logs:
        typ = row.get("type", "")
        if typ == "rollout":
            ep_len = int(float(row.get("ep_len", 0.0)))
            step_cursor += max(0, ep_len)
            x_roll.append(step_cursor)
            thr_mean.append(float(row.get("thr_mean", float("nan"))))
            on_ratio.append(float(row.get("on_ratio_mean", float("nan"))))
            fair_ep.append(float(row.get("fair_ep", float("nan"))))
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

    _plot(x_roll, thr_mean, "Throughput mean (thr_mean)", "thr_mean")
    _plot(x_roll, on_ratio, "BS on ratio mean (on_ratio_mean)", "on_ratio_mean")
    _plot(x_roll, fair_ep, "Jain fairness (fair_ep)", "fair_ep")
    _plot(x_upd, loss_ue, "Loss (UE)", "loss_ue")
    _plot(x_upd, loss_bs, "Loss (BS)", "loss_bs")

    # =========================
    # (B) agent 기반: UE/BS reward history
    # =========================
    steps = np.asarray(agent.step_history)
    ue_rewards = np.asarray(agent.reward_history_ue, dtype = float)
    bs_rewards = np.asarray(agent.reward_history_bs, dtype = float)

    T = min(len(steps), len(ue_rewards), len(bs_rewards))
    if T <= 0:
        return
    steps = steps[:T]
    ue_rewards = ue_rewards[:T]
    bs_rewards = bs_rewards[:T]

    ue_ma, ue_steps = moving_avg(ue_rewards,window)
    fig, axes = plt.subplots(2,1, figsize=(14,8), sharex=True)
    # =========================
    # (1) UE Team Reward
    # =========================
    ax0 = axes[0]
    ax0.plot(steps, ue_rewards, alpha =0.3, label="UE team reward (raw)")
    ax0.plot(ue_steps ,ue_ma,linewidth =2, label =f"UE team reward (MA{window})")
    ax0.set_ylabel("UE team Reward")
    ax0.legend()
    ax0.grid(True)

    # =========================
    # (2) BS Team Reward
    # =========================
    ax1 = axes[1]
    # n_bs = bs_rewards.shape[1]
    # for i in range(n_bs):
    #     bs_i = bs_rewards[:,i]
    #     bs_ma, bs_steps = moving_avg(bs_i, window)
    #     ax1.plot(steps, bs_i, alpha=0.15)
    #     ax1.plot(bs_steps, bs_ma, linewidth =1.0, label =f"BS{i} (MA{window})")
    ax1.plot(steps, bs_rewards, alpha=0.3, label="BS team reward (raw)")
    bs_ma, bs_steps = moving_avg(bs_rewards, window)
    ax1.plot(bs_steps, bs_ma, linewidth=2, label=f"BS team reward (MA{window})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("BS team Reward")
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"reward_plot_{ts}.png")
    plt.savefig(out_path,dpi=200)
    print(f"[plot_reward] saved to: {out_path}")
    #plt.show()
    plt.close(fig)


def build_env(n_ue: int, n_bs: int, bs_top_k: int, power_budget_ratio: float,
              V: float, enable_mobility: bool, enable_channel_variation: bool,
              hard_window_len: int, on_window: int, bs_over_penalty: float):
    # BS positions: triangle template + fill remainder uniformly
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

    cfg = HeteroQMIXcfg(
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
    )
    
    agent = HeteroQMIXAgent(env=env, cfg=cfg, log_dir="./results/train_logs", device=args.device)
    print("\n[QMIX TEST] Config:")
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
        }
    )
    print()
    logs = agent.train(n_env_steps=args.n_env_steps, rollout_horizon=args.rollout_horizon)
    if args.save_ckpt:
        agent.save(args.ckpt_path)
        print(f"[checkpoint] saved to: {args.ckpt_path}")
    rollouts = [x for x in logs if x.get("type") == "rollout"]
    updates = [x for x in logs if x.get("type") == "update"]
    if rollouts:
        last = rollouts[-1]
        print(
            f"[DONE] env_steps={agent.total_env_steps} | last_ep_len={last['ep_len']:.0f} "
            f"| last_ep_r_ue_sum={last['ep_r_ue_sum']:.3f} | last_ep_r_bs_sum={last['ep_r_bs_sum']:.3f} "
            f"| epsilon={last['epsilon']:.3f}"
            f"| thr_mean={last.get('thr_mean', float('nan')):.3f} "
            f"| fair_ep={last.get('fair_ep', float('nan')):.3f}"
            f"| on_ratio={last.get('on_ratio_mean', float('nan')):.3f} "
        )
    if updates:
        last_u = updates[-1]
        print(
            f"[DONE] last_update loss= (ue={last_u['loss_ue']:.4f}, bs={last_u['loss_bs']:.4f}) " #{last_u['loss']:.4f}
            f"| epsilon={last_u['epsilon']:.3f}"
        )
    plot_train_metrics(logs, agent, save_dir="./results/plots", window=100)
    save_logs_csv(logs)

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
    cfg = HeteroQMIXcfg(
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
        eps_decay=1.0,
    )
    agent = HeteroQMIXAgent(env=env, cfg=cfg, log_dir="./results/eval_logs", device=args.device)
    
    if args.load_ckpt:
        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {args.ckpt_path}")
        agent.load(args.ckpt_path, load_optimizer=False)
        print(f"[EVAL] loaded {args.ckpt_path}")
    
    agent.eps = args.eval_epsilon
    logs = []
    print(f"\n[EVAL] episodes={args.episodes} | horizon={args.rollout_horizon} | epsilon={args.eval_epsilon}\n")

    for ep_i in range(args.episodes):
        out = agent.rollout_episode(n_steps=args.rollout_horizon)
        log = {
            "type": "eval",
            "episode": ep_i,
            **out
        }
        logs.append(log)
        print(
            f"  ep={ep_i:03d} | len={out['ep_len']:.0f} | r_ue_sum={out['ep_r_ue_sum']:.3f} "
            f"| r_bs_sum={out['ep_r_bs_sum']:.3f} | epsilon={out['epsilon']:.3f}"
        )
    save_logs_csv(logs, path="./results/eval_logs/eval_log.csv")
    plot_train_metrics(logs, agent, save_dir="./results/eval_plots", window=100)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # env
    parser.add_argument("--n_ue", type=int, default=20)
    parser.add_argument("--n_bs", type=int, default=3)
    parser.add_argument("--bs_top_k", type=int, default=5)
    parser.add_argument("--power_budget_ratio", type=float, default=0.6)
    parser.add_argument("--V", type=float, default=5.0)
    parser.add_argument("--enable_mobility", action="store_true", default=True)
    parser.add_argument("--enable_channel_variation", action="store_true", default=True)
    parser.add_argument("--hard_window_len", type=int, default=1000)
    parser.add_argument("--on_window", type=int, default=100)
    parser.add_argument("--bs_over_penalty", type=float, default=50.0)
    # rollout/train
    parser.add_argument("--rollout_horizon", type=int, default=200)
    parser.add_argument("--n_env_steps", type=int, default=50000)
    # agent
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--capacity_episodes", type=int, default=10000)
    parser.add_argument("--update_interval_steps", type=int, default=128)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.9995)

    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/heteroqmix_queue.pt")
    parser.add_argument("--save_ckpt", action="store_true", default=True)
    parser.add_argument("--load_ckpt", action="store_true", default=False)  
    parser.add_argument("--no_load_optimizer", action="store_true", default=False)
    # eval
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval_epsilon", type=float, default=0.05)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()