# ================================
# tab_rl/logics.py
# ================================
import numpy as np
import pandas as pd


# =========================================================
# ENVIRONMENT BUILDING
# =========================================================
def prepare_rl_environment(df: pd.DataFrame, n_temp_bins: int = 8, n_humidity_bins: int = 8):

    df = df.copy().dropna(subset=["Temperature", "Humidity", "Crop_Type", "Crop_Yield"])
    df = df[df["Crop_Yield"] > 0]

    # ensure ints (IMPORTANT: avoid float bin sizes from Streamlit sliders)
    n_temp_bins = int(n_temp_bins)
    n_humidity_bins = int(n_humidity_bins)

    # bin edges
    temp_edges = np.linspace(
        df["Temperature"].min(),
        df["Temperature"].max(),
        n_temp_bins + 1
    )

    hum_edges = np.linspace(
        df["Humidity"].min(),
        df["Humidity"].max(),
        n_humidity_bins + 1
    )

    # bin assignment
    df["temp_bin"] = np.digitize(df["Temperature"], temp_edges[1:-1])
    df["hum_bin"] = np.digitize(df["Humidity"], hum_edges[1:-1])

    # clip safety (VERY IMPORTANT for RL indexing)
    df["temp_bin"] = df["temp_bin"].clip(0, n_temp_bins - 1)
    df["hum_bin"] = df["hum_bin"].clip(0, n_humidity_bins - 1)

    # state encoding
    df["state"] = df["temp_bin"] * n_humidity_bins + df["hum_bin"]

    # action encoding
    crop_types = sorted(df["Crop_Type"].unique().tolist())
    crop_to_idx = {c: i for i, c in enumerate(crop_types)}
    df["action"] = df["Crop_Type"].map(crop_to_idx)

    n_states = n_temp_bins * n_humidity_bins
    n_actions = len(crop_types)

    # reward matrix
    R = np.zeros((n_states, n_actions))

    for (s, a), grp in df.groupby(["state", "action"]):
        R[int(s), int(a)] = grp["Crop_Yield"].mean()

    R_max = R.max() if R.max() > 0 else 1.0
    R_norm = R / R_max

    # labels
    temp_labels = [
        f"{temp_edges[i]:.1f}-{temp_edges[i+1]:.1f}°C"
        for i in range(n_temp_bins)
    ]

    hum_labels = [
        f"{hum_edges[i]:.0f}-{hum_edges[i+1]:.0f}%"
        for i in range(n_humidity_bins)
    ]

    return (
        R, R_norm, crop_types,
        n_states, n_actions,
        temp_labels, hum_labels,
        n_temp_bins, n_humidity_bins,
        R_max
    )


# =========================================================
# Q-LEARNING
# =========================================================
def run_q_learning(
    R_norm,
    n_states,
    n_actions,
    alpha=0.1,
    gamma=0.9,
    epsilon_start=0.3,
    episodes=500,
    seed=42
):
    np.random.seed(seed)

    Q = np.zeros((n_states, n_actions))

    ep_rewards = []
    eps_hist = []
    q_delta = []

    epsilon = epsilon_start
    epsilon_min = 0.01
    decay = (epsilon_start - epsilon_min) / max(episodes, 1)

    prev_Q = Q.copy()

    for ep in range(episodes):
        state = np.random.randint(n_states)
        total = 0.0

        for _ in range(50):

            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])

            reward = R_norm[state, action]
            next_state = np.random.randint(n_states)

            # Q update
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total += reward

        epsilon = max(epsilon_min, epsilon - decay)

        ep_rewards.append(total)
        eps_hist.append(epsilon)

        if ep % 10 == 0:
            q_delta.append(float(np.mean(np.abs(Q - prev_Q))))
            prev_Q = Q.copy()

    return Q, ep_rewards, eps_hist, q_delta


# =========================================================
# POLICY HELPERS
# =========================================================
def get_optimal_policy(Q, crop_types):
    idx = np.argmax(Q, axis=1)
    return [crop_types[i] for i in idx], idx


def policy_improvement_ratio(R, optimal_idx):
    learned = np.mean([R[s, optimal_idx[s]] for s in range(len(optimal_idx))])

    rand_vals = []
    for s in range(len(optimal_idx)):
        nz = np.where(R[s] > 0)[0]
        if len(nz):
            rand_vals.append(R[s, np.random.choice(nz)])
        else:
            rand_vals.append(0.0)

    return learned, np.mean(rand_vals)


def crop_state_distribution(optimal_crops, crop_types, n_temp_bins, n_humidity_bins):
    grid = np.array([crop_types.index(c) for c in optimal_crops])
    return grid.reshape(n_temp_bins, n_humidity_bins)


# =========================================================
# MAIN MODEL
# =========================================================
def build_rl_model(
    df,
    n_temp_bins: int = 8,
    n_humidity_bins: int = 8,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.3,
    episodes=500
):

    R, R_norm, crop_types, n_states, n_actions, temp_labels, hum_labels, nt, nh, R_max = \
        prepare_rl_environment(df, n_temp_bins, n_humidity_bins)

    Q, ep_rewards, eps_hist, q_delta = run_q_learning(
        R_norm, n_states, n_actions,
        alpha, gamma, epsilon, episodes
    )

    optimal_crops, optimal_idx = get_optimal_policy(Q, crop_types)

    learned, random_mean = policy_improvement_ratio(R, optimal_idx)

    policy_grid = crop_state_distribution(
        optimal_crops, crop_types, nt, nh
    )

    improvement_pct = (learned - random_mean) / max(random_mean, 1e-6) * 100



    return {
        "Q": Q,
        "R": R,
        "R_max": R_max,
        "ep_rewards": ep_rewards,
        "eps_hist": eps_hist,
        "q_delta": q_delta,
        "optimal_crops": optimal_crops,
        "optimal_idx": optimal_idx,
        "policy_grid": policy_grid,
        "crop_types": crop_types,
        "temp_labels": temp_labels,
        "hum_labels": hum_labels,
        "nt": nt,
        "nh": nh,
        "learned_mean": float(learned),
        "random_mean": float(random_mean),
        "improvement_pct": float(improvement_pct),
        "converged": bool(len(q_delta) and q_delta[-1] < 1e-3)
    }
    
def moving_average(x, window=10):
    x = np.array(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")