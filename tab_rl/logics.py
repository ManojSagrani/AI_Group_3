import numpy as np
import pandas as pd


def prepare_rl_environment(df: pd.DataFrame, n_temp_bins: int = 8, n_humidity_bins: int = 8):
    """
    Discretise (Temperature, Humidity) into a grid of states.
    Each (state, action) pair maps to the mean Crop_Yield observed in the data.

    Returns
    -------
    R              : raw reward matrix  [n_states × n_actions]
    R_norm         : rewards normalised to [0, 1]
    crop_types     : ordered list of crop names (action labels)
    n_states       : total number of states
    n_actions      : number of unique crops
    temp_labels    : bin labels for Temperature axis
    hum_labels     : bin labels for Humidity axis
    n_temp_bins    : (echo)
    n_humidity_bins: (echo)
    R_max          : max raw reward (used to de-normalise later)
    """
    df = df.copy().dropna(subset=["Temperature", "Humidity", "Crop_Type", "Crop_Yield"])
    df = df[df["Crop_Yield"] > 0]  # exclude failed-crop observations

    temp_edges = np.linspace(df["Temperature"].min(), df["Temperature"].max(), n_temp_bins + 1)
    hum_edges = np.linspace(df["Humidity"].min(), df["Humidity"].max(), n_humidity_bins + 1)

    # np.digitize returns bins 1..n; subtract 1 for 0-based and clip
    df["temp_bin"] = (np.digitize(df["Temperature"], temp_edges[1:-1])).clip(0, n_temp_bins - 1)
    df["hum_bin"] = (np.digitize(df["Humidity"], hum_edges[1:-1])).clip(0, n_humidity_bins - 1)
    df["state"] = df["temp_bin"] * n_humidity_bins + df["hum_bin"]

    crop_types = sorted(df["Crop_Type"].unique().tolist())
    crop_to_idx = {c: i for i, c in enumerate(crop_types)}
    df["action"] = df["Crop_Type"].map(crop_to_idx)

    n_states = n_temp_bins * n_humidity_bins
    n_actions = len(crop_types)

    R = np.zeros((n_states, n_actions))
    for (s, a), grp in df.groupby(["state", "action"]):
        s, a = int(s), int(a)
        if 0 <= s < n_states and 0 <= a < n_actions:
            R[s, a] = grp["Crop_Yield"].mean()

    R_max = R.max() if R.max() > 0 else 1.0
    R_norm = R / R_max

    temp_labels = [f"{temp_edges[i]:.1f}–{temp_edges[i+1]:.1f}°C" for i in range(n_temp_bins)]
    hum_labels = [f"{hum_edges[i]:.0f}–{hum_edges[i+1]:.0f}%" for i in range(n_humidity_bins)]

    return R, R_norm, crop_types, n_states, n_actions, temp_labels, hum_labels, n_temp_bins, n_humidity_bins, R_max


def run_q_learning(R_norm, n_states, n_actions, alpha=0.1, gamma=0.9,
                   epsilon_start=0.3, episodes=500, seed=42):
    """
    Tabular Q-Learning with linear epsilon decay.

    State transitions are stochastic (uniform random next-state) because the
    environment (weather) is exogenous and cannot be controlled by the farmer.

    Bellman update:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

    Returns Q-table, per-episode reward, epsilon trace, and Q-change trace.
    """
    np.random.seed(seed)
    Q = np.zeros((n_states, n_actions))
    episode_rewards = []
    epsilon_history = []
    q_delta_history = []   # mean |ΔQ| every 10 episodes → convergence proxy

    epsilon = epsilon_start
    epsilon_min = 0.01
    epsilon_decay = (epsilon_start - epsilon_min) / max(episodes, 1)

    prev_Q = Q.copy()

    for ep in range(episodes):
        state = np.random.randint(0, n_states)
        total_reward = 0.0

        for _ in range(50):  # steps per episode
            if np.random.random() < epsilon:
                action = np.random.randint(0, n_actions)         # explore
            else:
                action = int(np.argmax(Q[state]))                # exploit

            reward = R_norm[state, action]
            next_state = np.random.randint(0, n_states)          # stochastic env

            # Bellman update
            td_target = reward + gamma * np.max(Q[next_state])
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        episode_rewards.append(total_reward)
        epsilon_history.append(epsilon)

        if ep % 10 == 0:
            q_delta_history.append(float(np.mean(np.abs(Q - prev_Q))))
            prev_Q = Q.copy()

    return Q, episode_rewards, epsilon_history, q_delta_history


def moving_average(data, window=20):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def get_optimal_policy(Q, crop_types):
    """Return list of best crop names (one per state) and their indices."""
    optimal_idx = np.argmax(Q, axis=1)
    optimal_crops = [crop_types[i] for i in optimal_idx]
    return optimal_crops, optimal_idx


def policy_improvement_ratio(R, optimal_idx):
    """Compare mean yield of learned policy vs pure random selection."""
    learned = np.mean([R[s, optimal_idx[s]] for s in range(len(optimal_idx))])
    rand_vals = []
    for s in range(len(optimal_idx)):
        non_zero = np.where(R[s] > 0)[0]
        if len(non_zero):
            rand_vals.append(R[s, np.random.choice(non_zero)])
        else:
            rand_vals.append(0.0)
    random_mean = np.mean(rand_vals)
    return learned, random_mean


def crop_state_distribution(optimal_crops, crop_types, n_temp_bins, n_humidity_bins):
    """Build a 2-D grid of the winning crop index for heatmap rendering."""
    n_states = n_temp_bins * n_humidity_bins
    grid = np.array([crop_types.index(c) for c in optimal_crops]).reshape(n_temp_bins, n_humidity_bins)
    return grid
