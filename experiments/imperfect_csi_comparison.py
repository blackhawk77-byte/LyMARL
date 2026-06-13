import numpy as np
import os

# ============================================================
# File paths
# ============================================================
perfect_path = "LyMARL_eval.npz"

imperfect_paths = {
    0.01: "LyMARL_eval_imperfect_csi_0.01.npz",
    0.05: "LyMARL_eval_imperfect_csi_0.05.npz",
    0.10: "LyMARL_eval_imperfect_csi_0.1.npz",
}

# ============================================================
# Helper
# ============================================================
def load_mean_throughput(npz_path, warmup=0):
    data = np.load(npz_path, allow_pickle=True)

    if "throughput_history" in data:
        thr = np.asarray(data["throughput_history"], dtype=np.float64)
    elif "throughput" in data:
        thr = np.asarray(data["throughput"], dtype=np.float64)
    else:
        raise KeyError(f"Throughput key not found in {npz_path}. Available keys: {data.files}")

    if warmup > 0:
        thr = thr[warmup:]

    return float(np.mean(thr)), float(np.std(thr)), len(thr)


# ============================================================
# Compute performance retention
# ============================================================
warmup = 0  # 필요하면 10000으로 바꿔서 초기 10k slot 제외 가능

perfect_mean, perfect_std, perfect_n = load_mean_throughput(perfect_path, warmup=warmup)

print("=" * 80)
print("Perfect CSI baseline")
print("=" * 80)
print(f"File        : {perfect_path}")
print(f"Mean Thr.   : {perfect_mean:.4f} Gbps")
print(f"Std Thr.    : {perfect_std:.4f} Gbps")
print(f"Samples     : {perfect_n}")
print()

print("=" * 80)
print("Imperfect CSI throughput retention")
print("=" * 80)
print(f"{'sigma_e^2':>10} | {'Mean Thr. (Gbps)':>18} | {'Retention (%)':>15} | {'Drop (%)':>10}")
print("-" * 80)

retentions = {}

for sigma_e2, path in imperfect_paths.items():
    imp_mean, imp_std, imp_n = load_mean_throughput(path, warmup=warmup)

    retention = 100.0 * imp_mean / perfect_mean
    drop = 100.0 - retention
    retentions[sigma_e2] = retention

    print(f"{sigma_e2:10.2f} | {imp_mean:18.4f} | {retention:15.2f} | {drop:10.2f}")

print("-" * 80)

# 논문 문장용 출력
r001 = retentions[0.01]
r005 = retentions[0.05]
r010 = retentions[0.10]

print()
print("Paper sentence:")
print(
    f"The results show that LyMARL maintains "
    f"{r001:.2f}%, {r005:.2f}%, and {r010:.2f}% "
    f"of its perfect-CSI throughput performance under "
    f"$\\sigma_e^2 = 0.01$, $0.05$, and $0.1$, respectively."
)