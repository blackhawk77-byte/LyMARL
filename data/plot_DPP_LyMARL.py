# plot_hard_compare_3figs_v3.py
# ------------------------------------------------------------
# Requested final tweaks:
# - Remove ALL titles.
# - Do NOT annotate bar values on top.
# - Keep Times New Roman.
# - Fig1: ON-ratio compare (bigger fonts), legend order consistent.
# - Fig2: Fairness + Throughput combined (1x2), black-only styling:
#     LyMARL: empty black bar
#     DPP   : empty black bar with hatch
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
})

##################################파일 이름 바꿔서########################################
DPP_NPZ = "dpp_hard_seed0_100k_metrics.npz"
LM_NPZ  = "eval_hard_seed0_100k_metrics.npz"  # LyMARL hard

OUTDIR = "plots_hard_compare"
os.makedirs(OUTDIR, exist_ok=True)

dpp = np.load(DPP_NPZ, allow_pickle=True)
lm  = np.load(LM_NPZ, allow_pickle=True)

def _require_keys(npz, keys, name):
    missing = [k for k in keys if k not in npz.files]
    if missing:
        raise KeyError(f"[{name}] missing keys: {missing}. Available: {npz.files}")

_require_keys(dpp, ["bs_ids", "onratio_x", "onratio_100", "throughput_mean", "fairness_mean_excl_alloff"], "DPP")
_require_keys(lm,  ["bs_ids", "onratio_x", "onratio_100", "throughput_mean", "fairness_mean"], "LyMARL")

bs_ids_dpp = dpp["bs_ids"].astype(int).tolist()
bs_ids_lm  = lm["bs_ids"].astype(int).tolist()
all_bs_ids = sorted(list(set(bs_ids_dpp).intersection(set(bs_ids_lm))))
if len(all_bs_ids) == 0:
    raise RuntimeError("No common BS ids between DPP and LyMARL npz files.")

def align_onratio(npz, bs_ids_target):
    bs_ids = npz["bs_ids"].astype(int).tolist()
    x = npz["onratio_x"].astype(int)
    y = npz["onratio_100"].astype(np.float32)
    col_map = [bs_ids.index(b) for b in bs_ids_target]
    return x, y[:, col_map]

x_dpp, on_dpp = align_onratio(dpp, all_bs_ids)
x_lm,  on_lm  = align_onratio(lm,  all_bs_ids)

thr_dpp = float(dpp["throughput_mean"])
thr_lm  = float(lm["throughput_mean"])
fair_dpp = float(dpp["fairness_mean_excl_alloff"])
fair_lm  = float(lm["fairness_mean"])

# ============================================================
# Fig 1) ON-ratio compare (bigger fonts, no title)
# ============================================================
base_colors = ["black", "blue", "red", "green", "purple", "orange"]
bs_color = {bs_id: base_colors[i % len(base_colors)] for i, bs_id in enumerate(all_bs_ids)}

FIG1_LABEL = 20
FIG1_TICKS = 20
FIG1_LEGEND = 14

fig, ax = plt.subplots(figsize=(11, 5.8))
line_handles = {}

for bi, bs_id in enumerate(all_bs_ids):
    c = bs_color[bs_id]
    h, = ax.plot(x_dpp, on_dpp[:, bi], linestyle="--", linewidth=2.3, color=c, label=f"DPP BS{bs_id}")
    line_handles[f"DDPP BS{bs_id}"] = h

for bi, bs_id in enumerate(all_bs_ids):
    c = bs_color[bs_id]
    h, = ax.plot(x_lm, on_lm[:, bi], linestyle="-", linewidth=2.3, color=c, label=f"LyMARL BS{bs_id}")
    line_handles[f"LyMARL BS{bs_id}"] = h

ax.set_xlabel("Time Step", fontsize=FIG1_LABEL)
ax.set_ylabel("BS ON-ratio", fontsize=FIG1_LABEL)
ax.set_ylim(0.0, 1.05)
ax.grid(alpha=0.3)
ax.tick_params(axis="both", labelsize=FIG1_TICKS)

legend_labels = [f"DDPP BS{b}" for b in all_bs_ids] + [f"LyMARL BS{b}" for b in all_bs_ids]
legend_handles = [line_handles[lbl] for lbl in legend_labels]
ax.legend(legend_handles, legend_labels, loc="upper right", ncol=2, fontsize=FIG1_LEGEND)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "onratio_100_compare.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Fig 2) Fairness + Throughput combined (no titles, no value text)
# ============================================================
def draw_two_bar(ax, lm_val, dpp_val, ylabel, ylim=None):
    x0, x1 = 0.0, 0.55   # close spacing
    width = 0.35

    ax.bar(x0, lm_val, width=width, color="none", edgecolor="black", linewidth=1.4, label="LyMARL")
    ax.bar(x1, dpp_val, width=width, color="none", edgecolor="black", linewidth=1.4, hatch="///", label="DDPP")

    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["LyMARL", "DDPP"], fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

fig, axes = plt.subplots(1, 2, figsize=(11, 7.2))

draw_two_bar(axes[0], lm_val=fair_lm, dpp_val=fair_dpp, ylabel="Jain Fairness Index (JFI)", ylim=(0.0, 1.05))
draw_two_bar(axes[1], lm_val=thr_lm, dpp_val=thr_dpp, ylabel="Total Throughput (Gbps)", ylim=None)

# one legend only
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc="upper left", fontsize=20, frameon=True)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fairness_throughput_mean_bars.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved figures to: {OUTDIR}/")
print(" - onratio_100_compare.png")
print(" - fairness_throughput_mean_bars.png")