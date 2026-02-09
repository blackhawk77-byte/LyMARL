# LyMARL: Lyapunov-Guided Multi-Agent Reinforcement Learning Framework for Energy-Aware Radio Resource Management

This repository provides an implementation of **LyMARL**, a Lyapunov-guided multi-agent reinforcement learning framework for **joint user association (UA) and base-station (BS) activation** in multi-cell wireless networks under **finite-horizon energy constraints**.

LyMARL bridges **Lyapunov-based stochastic network optimization** and **multi-agent reinforcement learning (MARL)** to achieve:
- high system throughput,
- improved user fairness, and
- strict compliance with finite-time BS energy budgets.

---

## 1. System Model

We consider a multi-cell wireless network consisting of:
- **B base stations (BSs)** and
- **U user equipments (UEs)**

operating over a **finite time horizon** of $T$ time slots.

### Network Operation
- Each BS can serve **at most one user per slot**, and only when it is **active (ON)**.
- Each UE can be associated with **at most one BS per slot**.
- BS activation incurs a fixed transmit power cost $P_{\max}$.

### Channel Model
- Downlink transmission with:
  - path-loss attenuation,
  - log-normal shadowing,
  - small-scale fading.
- Achievable rates are computed from SINR and depend on:
  - user association,
  - BS activation,
  - inter-cell interference.
- User mobility and channel variation can be enabled.

### Energy Constraint
Each BS is subject to a **finite-horizon energy budget**:
```math
\sum_{t=1}^{T} y_b(t) P_{\max} \le P^{\text{total}}_b,
\quad y_b(t) \in \{0,1\}
```

This finite-time constraint introduces a fundamental tradeoff:
- activating more BSs improves throughput and fairness,
- but accelerates energy depletion.

---

## 2. DDPP: Distributed Drift-Plus-Penalty Baseline

### Overview
This repository includes a **pure Lyapunov-based baseline**, referred to as **DDPP (Distributed Drift-Plus-Penalty)**.

DDPP solves the UA and BS activation problem using **Lyapunov stochastic optimization**, without any learning.

### Virtual Queues
- **User fairness queue** $Q_u(t)$: enforces long-term rate fairness.
- **BS power queue** $Z_b(t)$: regulates average BS power consumption.

Queue updates follow:
```math
Q_u(t+1) = [Q_u(t) + \gamma_u(t) - r_u(t)]^+ \\
Z_b(t+1) = [Z_b(t) + y_b(t)P_{\max} - P^{\text{avg}}_b]^+
```

### Slot-wise Control
At each slot, DDPP:
1. Updates auxiliary rate variables $\gamma_u(t)$,
2. Computes weighted user–BS scores,
3. Performs greedy distributed matching:
   - UEs request BSs with positive surplus,
   - BSs activate only if requests exist.

### Limitations
While DDPP guarantees queue stability and asymptotic optimality,  
it enforces energy constraints **only in a long-term average sense**.

In finite-horizon settings, this leads to:
- aggressive BS activation in early stages,
- forced deactivation later,
- degraded throughput and fairness.

DDPP therefore serves as a **baseline reference** for LyMARL.

---

## 3. LyMARL: Lyapunov-Guided MARL

### Key Idea
**LyMARL overcomes the finite-horizon limitation of DDPP** by integrating Lyapunov virtual queues into a **Multi-Agent Proximal Policy Optimization (MAPPO)** framework.

Instead of passive Lyapunov control, LyMARL introduces:
- **proactive BS agents** that learn energy-aware activation policies,
- while preserving **fully decentralized execution**.

---

### Agent Design

#### User Agents
- Generate association requests.
- Observe:
  - fairness backlog $Q_u(t)$,
  - predicted achievable rates,
  - BS energy and load indicators.
- Share a **team reward** promoting throughput and fairness.

#### BS Agents
- Decide whether to activate and which user to schedule.
- Observe:
  - power queue $Z_b(t)$,
  - activation history,
  - top-K requesting users.
- Receive **per-BS rewards** balancing:
  - throughput,
  - activation regularity,
  - power budget compliance.

---

### Centralized Training with Decentralized Execution (CTDE)

LyMARL follows the CTDE paradigm:
- Actors operate on local observations during execution.
- Centralized critics access the global system state to mitigate partial observability.
- Two critics are used:
  - a UE-side critic (scalar),
  - a BS-side critic (vector-valued).

---

### Training Algorithm

LyMARL is trained using MAPPO with:
- Generalized Advantage Estimation (GAE),
- PPO clipping,
- entropy regularization,
- value and advantage normalization.

At each update:
1. Rollouts are collected over a fixed horizon,
2. Advantages and returns are computed for UE and BS objectives,
3. Critics are updated via mean squared error loss,
4. Actors are updated via PPO.

---

## 4. Configuration

All system, channel, Lyapunov, and training parameters are specified in a single YAML file
