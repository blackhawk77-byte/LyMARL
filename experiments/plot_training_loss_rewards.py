import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}
bs_colors = ["#000000", "#C00000", "#005AB5"]  # black, deep red, deep blue
user_color = "#1B4F72"
# ============================================================
# Config
# ============================================================
NPZ_PATH = "LyMARL.npz"   # npz 파일 경로
SAVE_DIR = "plots"

MA_WINDOW = 1000             # 1000-step moving average
DPI = 400
WARMUP_STEP = 1000

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================
def moving_average_valid(x, window=1000):
    """
    Valid moving average.

    For window=1000:
    - The first averaged value uses samples 1~1000.
    - Therefore, the first x-axis time step is 1000.
    """
    x = np.asarray(x, dtype=float).squeeze()

    if x.ndim != 1:
        raise ValueError(f"moving_average_valid expects 1D array, got shape {x.shape}")

    if window <= 1:
        return x

    if len(x) < window:
        raise ValueError(f"Input length {len(x)} is smaller than moving average window {window}")

    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / window


def setup_journal_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.8,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })


def save_fig(fig, name):
    png_path = os.path.join(SAVE_DIR, f"{name}.png")

    fig.savefig(png_path)

    print(f"[SAVED] {png_path}")


def get_first_existing(data, keys):
    for k in keys:
        if k in data.files:
            return k, data[k]
    return None, None


def make_x_axis_after_ma(original_length, ma_window):
    """
    If original data is logged every step:
        first MA value corresponds to time step = ma_window.
    """
    return np.arange(ma_window, original_length + 1)


def set_common_x_axis(ax, max_step, start_step=1000):
    """
    Start from start_step and display x-axis labels as 5k, 10k, ...
    """
    base_ticks = [WARMUP_STEP, 10000, 20000, 30000, 40000, 50000]
    ticks = [t for t in base_ticks if start_step <= t <= max_step]

    ax.set_xlim(start_step, max_step)
    ax.set_xticks(ticks)

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x/1000)}k")
    )


# ============================================================
# Load data
# ============================================================
setup_journal_style()

data = np.load(NPZ_PATH, allow_pickle=True)

print("[INFO] Loaded keys:")
for k in data.files:
    print("  -", k)


# ============================================================
# 1) User team reward plot
# ============================================================
user_key, user_reward = get_first_existing(
    data,
    [
        "ue_team_reward_step",
        "ue_team_reward",
        "mean_user_reward_step",
        "user_team_reward",
        "user_reward",
    ]
)

if user_reward is not None:
    user_reward = np.asarray(user_reward, dtype=float).squeeze()

    if user_reward.ndim != 1:
        raise ValueError(f"{user_key} should be 1D, but got shape {user_reward.shape}")

    user_ma = moving_average_valid(user_reward, MA_WINDOW)

    # 첫 moving average 값은 time step = 1000
    x = make_x_axis_after_ma(len(user_reward), MA_WINDOW)
    
    mask = x >= WARMUP_STEP

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    ax.plot(
        x[mask],
        user_ma[mask],
        label="User Team Reward",
        color=user_color,
        linewidth=2.6,
    )

    set_common_x_axis(ax, max_step=x[-1], start_step=WARMUP_STEP)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average User Team Reward")

    # 기존 그림처럼 x/y grid 모두 표시하되 약하게
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        loc="upper right",
        frameon=False,
    )

    save_fig(fig, f"user_team_reward_ma{MA_WINDOW}_from{MA_WINDOW}")
    plt.close(fig)

else:
    print("[SKIP] User reward key not found.")

# ============================================================
# User reward plot: mean ± std across UEs
# ============================================================
ue_per_key, ue_per_user_reward = get_first_existing(
    data,
    [
        "ue_per_user_reward",
        "user_per_user_reward",
        "per_user_reward",
        "ue_rewards",
    ]
)

if ue_per_user_reward is not None:
    ue_per_user_reward = np.asarray(ue_per_user_reward, dtype=float)

    if ue_per_user_reward.ndim != 2:
        raise ValueError(
            f"{ue_per_key} should be 2D, but got shape {ue_per_user_reward.shape}"
        )

    # Expected shape: (T, U)
    # 만약 shape이 (U, T)면 자동 transpose
    if ue_per_user_reward.shape[0] < ue_per_user_reward.shape[1] and ue_per_user_reward.shape[0] <= 100:
        ue_per_user_reward = ue_per_user_reward.T

    T, U = ue_per_user_reward.shape

    ue_ma_list = []
    for u in range(U):
        ue_ma_list.append(moving_average_valid(ue_per_user_reward[:, u], MA_WINDOW))

    ue_ma = np.stack(ue_ma_list, axis=1)

    # 첫 moving average 값은 time step = 1000
    x = make_x_axis_after_ma(T, MA_WINDOW)

    ue_mean = np.mean(ue_ma, axis=1)
    ue_std = np.std(ue_ma, axis=1)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    ax.plot(
        x,
        ue_mean,
        label="Mean User Reward",
    )

    ax.fill_between(
        x,
        ue_mean - ue_std,
        ue_mean + ue_std,
        alpha=0.18,
        label="Std. across UEs",
    )

    set_common_x_axis(ax, max_step=x[-1], start_step=WARMUP_STEP)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average User Reward")

    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        loc="upper right",
        frameon=False,
    )

    save_fig(fig, f"user_reward_mean_std_ma{MA_WINDOW}_from{MA_WINDOW}")
    plt.close(fig)

else:
    print("[SKIP] Per-user reward key not found.")
    
# ============================================================
# 2) BS reward plot: individual BS curves
# ============================================================

bs_key, bs_reward = get_first_existing(
    data,
    [
        "bs_reward_vec_step",
        "bs_reward_vec",
        "bs_rewards",
        "bs_reward",
    ]
)

if bs_reward is not None:
    bs_reward = np.asarray(bs_reward, dtype=float)

    if bs_reward.ndim != 2:
        raise ValueError(f"{bs_key} should be 2D, but got shape {bs_reward.shape}")

    # Expected shape: (T, B)
    # 만약 shape이 (B, T) 형태라면 자동 transpose
    if bs_reward.shape[0] < bs_reward.shape[1] and bs_reward.shape[0] <= 10:
        bs_reward = bs_reward.T

    T, B = bs_reward.shape

    bs_ma_list = []
    for b in range(B):
        bs_ma_list.append(moving_average_valid(bs_reward[:, b], MA_WINDOW))

    bs_ma = np.stack(bs_ma_list, axis=1)

    # 첫 moving average 값은 time step = 1000
    x = make_x_axis_after_ma(T, MA_WINDOW)

    mask = x >= WARMUP_STEP
    
    bs_ids = data["bs_ids"] if "bs_ids" in data.files else np.arange(1, B + 1)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for b in range(B):
        ax.plot(
            x[mask],
            bs_ma[mask, b],
            label=f"BS{bs_ids[b]} Reward",
            color=bs_colors[b % len(bs_colors)],
            linewidth=2.6,
        )

    set_common_x_axis(ax, max_step=x[-1], start_step=WARMUP_STEP)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average BS Reward")

    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        loc="lower right",
        frameon=False,
    )

    save_fig(fig, f"bs_reward_individual_ma{MA_WINDOW}_from{MA_WINDOW}")
    plt.close(fig)


    # ========================================================
    # 3) BS reward plot: mean ± std across BSs
    # ========================================================
    bs_mean = np.mean(bs_ma, axis=1)
    bs_std = np.std(bs_ma, axis=1)
    

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    ax.plot(
        x[mask],
        bs_mean[mask],
        label="Mean BS Reward",
    )

    ax.fill_between(
        x[mask],
        bs_mean[mask] - bs_std[mask],
        bs_mean[mask] + bs_std[mask],
        alpha=0.18,
        label="Std. across BSs",
    )

    set_common_x_axis(ax, max_step=x[-1], start_step=WARMUP_STEP)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average BS Reward")

    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        loc="lower right",
        frameon=False,
    )

    save_fig(fig, f"bs_reward_mean_std_ma{MA_WINDOW}_from{MA_WINDOW}")
    plt.close(fig)

else:
    print("[SKIP] BS reward vector key not found.")


# ============================================================
# 4) Training loss plots
# ============================================================
# npz 안에 loss key가 있으면 자동으로 그림.
# 없으면 skip.
loss_groups = {
    "actor_loss": [
        ("UE Actor Loss", ["ue_actor_loss", "UE_Actor", "ue_actor_losses"]),
        ("BS Actor Loss", ["bs_actor_loss", "BS_Actor", "bs_actor_losses"]),
    ],
    "critic_loss": [
        ("UE Critic Loss", ["critic_ue_loss", "C_UE", "ue_critic_loss", "c_ue"]),
        ("BS Critic Loss", ["critic_bs_loss", "C_BS", "bs_critic_loss", "c_bs"]),
    ],
    "entropy": [
        ("UE Entropy", ["ue_entropy", "Ent_UE", "entropy_ue"]),
        ("BS Entropy", ["bs_entropy", "Ent_BS", "entropy_bs"]),
    ],
}

update_step_key, update_steps = get_first_existing(
    data,
    ["update_steps", "loss_steps", "train_update_steps", "global_update_steps"]
)

for group_name, items in loss_groups.items():
    curves = []

    for label, candidate_keys in items:
        key, arr = get_first_existing(data, candidate_keys)

        if arr is not None:
            arr = np.asarray(arr, dtype=float).squeeze()

            if arr.ndim == 1:
                curves.append((label, key, arr))
            else:
                print(f"[SKIP] {key} is not 1D. Shape: {arr.shape}")

    if len(curves) == 0:
        print(f"[SKIP] No keys found for {group_name}.")
        continue

    min_len = min(len(arr) for _, _, arr in curves)

    if update_steps is not None:
        x_loss = np.asarray(update_steps).squeeze()[:min_len]
    else:
        x_loss = np.arange(1, min_len + 1)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for label, key, arr in curves:
        arr = arr[:min_len]

        # loss는 update 단위라 reward처럼 1000-window를 쓰면 너무 클 수 있음
        # 그래서 자동으로 적당히 smoothing
        loss_ma_window = min(20, max(1, len(arr) // 20))

        if len(arr) >= loss_ma_window and loss_ma_window > 1:
            arr_ma = moving_average_valid(arr, loss_ma_window)
            x_plot = x_loss[loss_ma_window - 1:]
            plot_label = f"{label}"
        else:
            arr_ma = arr
            x_plot = x_loss
            plot_label = label

        ax.plot(
            x_plot,
            arr_ma,
            label=plot_label,
        )

    ax.set_xlabel("Update Step" if update_steps is not None else "Update Index")
    ax.set_ylabel(group_name.replace("_", " ").title())

    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        loc="best",
        frameon=False,
    )

    save_fig(fig, f"{group_name}_training_curve")
    plt.close(fig)


print("[DONE] Plot generation completed.")