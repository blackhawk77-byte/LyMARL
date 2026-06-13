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


def save_qplex_npz(agent, logs, path: str):
    """
    Save QPLEX training/evaluation metrics to .npz.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rollout_logs = [x for x in logs if x.get("type") in ["rollout", "eval"]]
    update_logs = [x for x in logs if x.get("type") == "update"]

    np.savez(
        path,

        # step-wise histories from agent
        thr_history=np.asarray(getattr(agent, "thr_history", []), dtype=np.float32),
        fair_history=np.asarray(getattr(agent, "fair_history_100step", []), dtype=np.float32),
        on_ratio_history=np.asarray(getattr(agent, "on_ratio_history", []), dtype=np.float32),
        reward_history_ue=np.asarray(getattr(agent, "reward_history_ue", []), dtype=np.float32),
        reward_history_bs=np.asarray(getattr(agent, "reward_history_bs", []), dtype=np.float32),
        step_history=np.asarray(getattr(agent, "step_history", []), dtype=np.int64),

        # rollout/eval summary logs
        rollout_thr_mean=np.asarray([x.get("thr_mean", np.nan) for x in rollout_logs], dtype=np.float32),
        rollout_fair_mean=np.asarray([x.get("fair_mean", np.nan) for x in rollout_logs], dtype=np.float32),
        rollout_on_ratio_mean=np.asarray([x.get("on_ratio_mean", np.nan) for x in rollout_logs], dtype=np.float32),
        rollout_ep_r_ue_sum=np.asarray([x.get("ep_r_ue_sum", np.nan) for x in rollout_logs], dtype=np.float32),
        rollout_ep_r_bs_sum=np.asarray([x.get("ep_r_bs_sum", np.nan) for x in rollout_logs], dtype=np.float32),
        rollout_ep_len=np.asarray([x.get("ep_len", np.nan) for x in rollout_logs], dtype=np.float32),

        # update losses
        loss_ue=np.asarray([x.get("loss_ue", np.nan) for x in update_logs], dtype=np.float32),
        loss_bs=np.asarray([x.get("loss_bs", np.nan) for x in update_logs], dtype=np.float32),
    )

    print(f"✅ NPZ saved to: {os.path.abspath(path)}")

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
    
    logs = agent.train(
        n_env_steps=args.n_env_steps, 
        rollout_horizon=args.chunk_len
    )
    
    # save model
    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
    agent.save(args.ckpt_path)

    # save train npz
    save_qplex_npz(agent, logs, args.train_npz_path)

    print(f"\n✅ Model saved to: {os.path.abspath(args.ckpt_path)}")
    print(f"✅ Train results saved to: {os.path.abspath(args.train_npz_path)}")
    print("✅ Training completed!\n")


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

    # hard constraint only for evaluation
    env.set_hard_constraint(True)

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
        eps_start=args.eval_epsilon,
        eps_end=args.eval_epsilon,
        eps_decay=1.0,
        n_heads=args.n_heads,
    )

    agent = HeteroQPLEXAgent(env=env, cfg=cfg, device=args.device)
    
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt_path}")
    
    agent.load(args.ckpt_path)
    print(f"[EVAL] loaded model from: {args.ckpt_path}")
    
    agent.eps = args.eval_epsilon

    logs = []
    eval_horizon = args.eval_steps

    print(f"\n[EVAL] episodes={args.episodes} | horizon={eval_horizon} | epsilon={args.eval_epsilon}\n")

    for ep_i in range(args.episodes):
        steps_done = 0
        last_out = None

        while steps_done < eval_horizon:
            steps_to_run = min(args.chunk_len, eval_horizon - steps_done)
            out = agent.rollout_episode(n_steps=steps_to_run)

            logs.append({
                "type": "eval",
                "episode": ep_i,
                **out
            })

            steps_done += int(out.get("ep_len", 0))
            last_out = out

        print(
            f"  ep={ep_i:03d} | len={steps_done:.0f} "
            f"| r_ue_sum={last_out.get('ep_r_ue_sum', float('nan')):.3f} "
            f"| r_bs_sum={last_out.get('ep_r_bs_sum', float('nan')):.3f} "
            f"| thr_mean={last_out.get('thr_mean', float('nan')):.3f} "
            f"| fair_mean={last_out.get('fair_mean', float('nan')):.3f} "
            f"| on_ratio={last_out.get('on_ratio_mean', float('nan')):.3f}"
        )

    save_qplex_npz(agent, logs, args.eval_npz_path)

    print(f"\n✅ Eval results saved to: {os.path.abspath(args.eval_npz_path)}")
    print("✅ Evaluation completed!\n")
    

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

    # save paths
    parser.add_argument("--ckpt_path", type=str, default="./results/QPLEX.pt")
    parser.add_argument("--train_npz_path", type=str, default="./results/npz/QPLEX_train.npz")
    parser.add_argument("--eval_npz_path", type=str, default="./results/npz/QPLEX_eval.npz")
    parser.add_argument("--save_ckpt", action="store_true", default=True)
    parser.add_argument("--load_ckpt", action="store_true", default=True)

    # eval
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=50000)
    parser.add_argument("--eval_epsilon", type=float, default=0.2)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
