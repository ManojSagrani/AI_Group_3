# ================================
# tab_rl/logics.py
# ================================
import numpy as np
import pandas as pd
import time

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
import numpy as np


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

    # =========================================================
    # REPRODUCIBILITY
    # =========================================================
    np.random.seed(seed)

    # =========================================================
    # INITIALISE Q TABLE
    # =========================================================
    Q = np.zeros((n_states, n_actions))

    # =========================================================
    # TRACKING VARIABLES
    # =========================================================
    ep_rewards = []
    eps_hist = []
    q_delta = []

    # =========================================================
    # EPSILON DECAY
    # =========================================================
    epsilon = epsilon_start

    epsilon_min = 0.01

    # smoother exponential decay
    epsilon_decay = 0.995

    # =========================================================
    # PREVIOUS Q TABLE
    # =========================================================
    prev_Q = Q.copy()

    # =========================================================
    # TRAINING LOOP
    # =========================================================
    for ep in range(episodes):

        # random starting state
        state = np.random.randint(n_states)

        total_reward = 0.0

        # =====================================================
        # EPISODE STEPS
        # =====================================================
        for step in range(50):

            # -------------------------------------------------
            # ε-GREEDY ACTION SELECTION
            # -------------------------------------------------
            if np.random.rand() < epsilon:

                # exploration
                action = np.random.randint(n_actions)

            else:

                # exploitation
                action = np.argmax(Q[state])

            # -------------------------------------------------
            # GET REWARD
            # -------------------------------------------------
            reward = R_norm[state, action]

            # random transition
            next_state = np.random.randint(n_states)

            # -------------------------------------------------
            # Q-LEARNING UPDATE
            # -------------------------------------------------
            old_q = Q[state, action]

            td_target = reward + gamma * np.max(Q[next_state])

            td_error = td_target - old_q

            Q[state, action] = old_q + alpha * td_error

            # -------------------------------------------------
            # MOVE TO NEXT STATE
            # -------------------------------------------------
            state = next_state

            # accumulate reward
            total_reward += reward

        # =====================================================
        # EPSILON DECAY
        # =====================================================
        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        # =====================================================
        # STORE HISTORY
        # =====================================================
        ep_rewards.append(float(total_reward))

        eps_hist.append(float(epsilon))

        # =====================================================
        # Q-TABLE CHANGE TRACKING
        # =====================================================
        if ep % 10 == 0:

            delta = np.mean(np.abs(Q - prev_Q))

            q_delta.append(float(delta))

            prev_Q = Q.copy()

    # =========================================================
    # RETURN RESULTS
    # =========================================================
    return (
        Q,
        ep_rewards,
        eps_hist,
        q_delta
    )


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

    # =========================
    # START TIMER
    # =========================
    start_time = time.time()

    # =========================
    # PREP ENVIRONMENT
    # =========================
    R, R_norm, crop_types, n_states, n_actions, temp_labels, hum_labels, nt, nh, R_max = \
        prepare_rl_environment(df, n_temp_bins, n_humidity_bins)

    # =========================
    # TRAIN Q-LEARNING
    # =========================
    Q, ep_rewards, eps_hist, q_delta = run_q_learning(
        R_norm,
        n_states,
        n_actions,
        alpha,
        gamma,
        epsilon,
        episodes
    )

    # =========================
    # POLICY EXTRACTION
    # =========================
    optimal_crops, optimal_idx = get_optimal_policy(Q, crop_types)

    learned, random_mean = policy_improvement_ratio(R, optimal_idx)

    policy_grid = crop_state_distribution(
        optimal_crops,
        crop_types,
        nt,
        nh
    )

    # =========================
    # IMPROVEMENT %
    # =========================
    improvement_pct = (
        (learned - random_mean)
        / max(random_mean, 1e-6)
    ) * 100

    # =========================
    # RL PERFORMANCE METRICS
    # =========================

    # Mean Absolute Error
    mae = np.mean(np.abs(Q - R_norm))

    # Approximate R²
    ss_res = np.sum((R_norm - Q) ** 2)
    ss_tot = np.sum((R_norm - np.mean(R_norm)) ** 2)

    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Action prediction accuracy
    predicted_actions = np.argmax(Q, axis=1)
    optimal_actions = np.argmax(R_norm, axis=1)

    accuracy = np.mean(predicted_actions == optimal_actions)

    # =========================
    # CONVERGENCE DETECTION
    # =========================

    converged = False
    convergence_score = 0.0

    if len(q_delta) >= 50:

        # Examine last 50 Q-updates
        recent_delta = np.array(q_delta[-50:])

        mean_delta = np.mean(recent_delta)
        std_delta = np.std(recent_delta)

        # Stable + tiny updates = convergence
        if mean_delta < 1e-3 and std_delta < 1e-4:
            converged = True

        # Confidence score
        convergence_score = 1 / (1 + mean_delta + std_delta)

    # =========================
    # TRAINING TIME
    # =========================
    train_time = time.time() - start_time

    # =========================
    # RETURN RESULTS
    # =========================
    return {

        # Core RL Outputs
        "Q": Q,
        "R": R,
        "R_max": R_max,

        # Training History
        "ep_rewards": ep_rewards,
        "eps_hist": eps_hist,
        "q_delta": q_delta,

        # Policy
        "optimal_crops": optimal_crops,
        "optimal_idx": optimal_idx,
        "policy_grid": policy_grid,

        # Labels
        "crop_types": crop_types,
        "temp_labels": temp_labels,
        "hum_labels": hum_labels,

        # Grid Dimensions
        "nt": nt,
        "nh": nh,

        # RL Reward Metrics
        "learned_mean": float(learned),
        "random_mean": float(random_mean),
        "improvement_pct": float(improvement_pct),

        # Performance Metrics
        "mae": float(mae),
        "r2": float(r2),
        "accuracy": float(accuracy),

        # Training Stats
        "train_time": float(train_time),
        "n_iter": int(episodes),

        # Convergence
        "converged": bool(converged),
        "convergence_score": float(convergence_score)
    }
    
def moving_average(x, window=10):
    x = np.array(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")