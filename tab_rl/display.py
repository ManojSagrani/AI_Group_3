import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tab_rl import logics


CROP_PALETTE = [
    "#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#FFC107", "#3F51B5"
]


def render(df: pd.DataFrame, alpha: float, gamma: float, epsilon: float,
           episodes: int, n_temp_bins: int, n_humidity_bins: int):

    st.markdown("## Reinforcement Learning — Q-Learning Crop Optimisation")

    # ── Methodology ──────────────────────────────────────────────────────────
    with st.expander("📖 Methodology & MDP Formulation", expanded=False):
        st.markdown("""
**Problem framing as a Markov Decision Process (MDP)**

| MDP Component | Definition in this system |
|---|---|
| **State** *s* | Discretised (Temperature × Humidity) environmental condition |
| **Action** *a* | Crop type selected for planting |
| **Reward** *r(s,a)* | Mean historical Crop Yield for that (condition, crop) pair |
| **Transition** *P(s'|s,a)* | Stochastic — weather conditions change independently of crop choice |
| **Objective** | Learn policy *π(s)* that maximises expected cumulative discounted yield |

**Algorithm — Q-Learning (off-policy TD control)**

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha \\Big[ r + \\gamma \\max_{a'} Q(s',a') - Q(s,a) \\Big]$$

The agent begins with full exploration (high ε) and gradually exploits its learned knowledge
as ε decays linearly toward 0.01. After convergence the **greedy policy** π(s) = argmax_a Q(s,a)
returns the empirically best crop for every environmental condition in the grid.
        """)

    # ── Training Parameters Summary ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Learning Rate α", f"{alpha:.3f}")
    col2.metric("Discount Factor γ", f"{gamma:.2f}")
    col3.metric("Initial Epsilon ε", f"{epsilon:.2f}")
    col4.metric("Episodes", f"{episodes:,}")

    st.markdown("---")

    # ── Prepare environment ───────────────────────────────────────────────────
    env = logics.prepare_rl_environment(df, n_temp_bins, n_humidity_bins)
    R, R_norm, crop_types, n_states, n_actions, temp_labels, hum_labels, nt, nh, R_max = env

    n_crops = len(crop_types)
    crop_colors = {c: CROP_PALETTE[i % len(CROP_PALETTE)] for i, c in enumerate(crop_types)}

    st.info(f"**Environment:** {n_states} states ({nt}×{nh} grid) · {n_actions} actions ({', '.join(crop_types)})")

    # ── Run Training ─────────────────────────────────────────────────────────
    if st.button("🚀 Train Q-Learning Agent", type="primary", use_container_width=True):
        progress = st.progress(0, text="Training…")

        Q, ep_rewards, eps_hist, q_delta = logics.run_q_learning(
            R_norm, n_states, n_actions, alpha, gamma, epsilon, episodes
        )
        progress.progress(1.0, text="Done!")

        optimal_crops, optimal_idx = logics.get_optimal_policy(Q, crop_types)
        learned_mean, random_mean = logics.policy_improvement_ratio(R, optimal_idx)
        policy_grid = logics.crop_state_distribution(optimal_crops, crop_types, nt, nh)

        st.session_state.rl_results = {
            "Q": Q, "ep_rewards": ep_rewards, "eps_hist": eps_hist,
            "q_delta": q_delta, "optimal_crops": optimal_crops,
            "optimal_idx": optimal_idx, "policy_grid": policy_grid,
            "learned_mean": learned_mean, "random_mean": random_mean,
            "R": R, "R_norm": R_norm,
        }
        st.success("✅ Training complete!")

    # ── Display Results ───────────────────────────────────────────────────────
    if "rl_results" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Train Q-Learning Agent**.")
        return

    res = st.session_state.rl_results
    Q = res["Q"]
    ep_rewards = res["ep_rewards"]
    eps_hist = res["eps_hist"]
    q_delta = res["q_delta"]
    optimal_crops = res["optimal_crops"]
    policy_grid = res["policy_grid"]
    learned_mean = res["learned_mean"]
    random_mean = res["random_mean"]
    R = res["R"]

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("### Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Learned Policy Avg Yield", f"{learned_mean:.1f} t/ha")
    m2.metric("Random Policy Avg Yield", f"{random_mean:.1f} t/ha")
    improvement = (learned_mean - random_mean) / max(random_mean, 1e-6) * 100
    m3.metric("Policy Improvement", f"+{improvement:.1f}%")
    m4.metric("Q-Table Convergence", f"{q_delta[-1]:.5f}" if q_delta else "N/A", delta="→ 0 = converged")

    st.markdown("---")

    # ── Learning Curves ───────────────────────────────────────────────────────
    st.markdown("### Learning Curves")
    lc1, lc2 = st.columns(2)

    with lc1:
        st.markdown("**Episode Reward (with 30-episode moving average)**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ep_rewards, alpha=0.3, color="#2196F3", linewidth=0.8, label="Raw")
        smooth = logics.moving_average(ep_rewards, 30)
        offset = len(ep_rewards) - len(smooth)
        ax.plot(range(offset, len(ep_rewards)), smooth, color="#0D47A1", linewidth=2, label="MA-30")
        ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with lc2:
        st.markdown("**Epsilon Decay (Exploration → Exploitation)**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(eps_hist, color="#FF9800", linewidth=2)
        ax.fill_between(range(len(eps_hist)), eps_hist, alpha=0.15, color="#FF9800")
        ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon (ε)")
        ax.set_ylim(0, max(eps_hist) * 1.1); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Q-Table Convergence ───────────────────────────────────────────────────
    if q_delta:
        st.markdown("**Q-Table Mean Absolute Change (convergence diagnostic)**")
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.semilogy(range(0, len(q_delta) * 10, 10), q_delta, color="#4CAF50", linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Episode"); ax.set_ylabel("|ΔQ| (log scale)"); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Optimal Policy Heatmap ────────────────────────────────────────────────
    st.markdown("### Optimal Policy Map — Best Crop per Environmental Condition")
    st.caption("Each cell shows the crop the trained agent would recommend given that (Temperature, Humidity) combination.")

    cmap = mcolors.ListedColormap(CROP_PALETTE[:len(crop_types)])
    bounds = list(range(len(crop_types) + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(policy_grid, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells with crop names
    for i in range(nt):
        for j in range(nh):
            crop_name = crop_types[policy_grid[i, j]]
            ax.text(j, i, crop_name[:4], ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    ax.set_xticks(range(nh)); ax.set_xticklabels(hum_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(nt)); ax.set_yticklabels(temp_labels, fontsize=7)
    ax.set_xlabel("Humidity"); ax.set_ylabel("Temperature")
    ax.set_title("Optimal Crop Policy π(s) = argmax_a Q(s,a)", fontsize=12)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=CROP_PALETTE[i], label=c) for i, c in enumerate(crop_types)]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.18, 1),
              fontsize=8, framealpha=0.9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Q-Value Heatmaps per Crop ─────────────────────────────────────────────
    st.markdown("### Q-Value Maps per Crop Action")
    st.caption("Each heatmap shows how valuable (expected yield) the agent thinks planting that crop is, across all environmental states.")

    cols_per_row = 3
    crop_chunks = [crop_types[i:i+cols_per_row] for i in range(0, len(crop_types), cols_per_row)]

    for chunk in crop_chunks:
        cols = st.columns(len(chunk))
        for col, crop in zip(cols, chunk):
            idx = crop_types.index(crop)
            q_grid = Q[:, idx].reshape(nt, nh) * R_max  # de-normalise for display
            with col:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(q_grid, ax=ax, cmap="YlOrRd", fmt=".0f",
                            annot=(nt * nh <= 64), annot_kws={"size": 6},
                            xticklabels=[h.split("–")[0] for h in hum_labels],
                            yticklabels=[t.split("–")[0] for t in temp_labels])
                ax.set_title(crop, fontsize=10, fontweight="bold")
                ax.set_xlabel("Humidity bin", fontsize=7)
                ax.set_ylabel("Temp bin", fontsize=7)
                ax.tick_params(labelsize=6)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Reward Table Overview ─────────────────────────────────────────────────
    st.markdown("### Raw Reward Table (Mean Yield per Condition)")
    reward_df = pd.DataFrame(R, columns=crop_types)
    reward_df.index.name = "State"
    st.dataframe(reward_df.round(2).style.background_gradient(cmap="YlGn", axis=None),
                 use_container_width=True, height=250)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    dl1, dl2 = st.columns(2)
    with dl1:
        policy_df = pd.DataFrame({
            "State": range(n_states),
            "Temp Bin": [temp_labels[s // nh] for s in range(n_states)],
            "Humidity Bin": [hum_labels[s % nh] for s in range(n_states)],
            "Optimal Crop": optimal_crops,
            "Expected Yield (t/ha)": [R[s, res["optimal_idx"][s]] for s in range(n_states)],
        })
        st.download_button("📥 Download Optimal Policy CSV", policy_df.to_csv(index=False).encode(),
                           "rl_optimal_policy.csv", "text/csv", use_container_width=True)
    with dl2:
        q_df = pd.DataFrame(Q * R_max, columns=crop_types)
        q_df.index.name = "State"
        st.download_button("📥 Download Q-Table CSV", q_df.to_csv().encode(),
                           "rl_q_table.csv", "text/csv", use_container_width=True)

    # ── Ethical Considerations ────────────────────────────────────────────────
    with st.expander("⚠️ Ethical Considerations & Limitations"):
        st.markdown("""
- **Data representativeness**: The reward table is derived from historical records that may not reflect
  future climate extremes or data-scarce regions.
- **Oversimplification**: Temperature and Humidity alone do not capture all agronomic factors
  (disease pressure, market prices, labour availability).
- **Stochastic transitions**: Real weather is correlated over time; the i.i.d. next-state assumption
  may overestimate the policy's generalisation to unseen conditions.
- **Equity**: Automated recommendations without local agronomist validation may harm smallholder
  farmers who lack the resources to act on suboptimal suggestions.
- **Transparency**: All Q-values and reward tables are fully downloadable, supporting auditability.
        """)
