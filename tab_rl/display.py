# ================================
# tab_rl/display.py (CLEAN VERSION)
# ================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tab_rl import logics


CROP_PALETTE = [
    "#2196F3", "#4CAF50", "#FF9800", "#F44336",
    "#9C27B0", "#00BCD4", "#FF5722", "#8BC34A"
]


def render(df, alpha, gamma, epsilon, episodes):

    st.title("🌾 RL Crop Optimisation (Q-Learning)")

    # =========================
    # TRAIN MODEL
    # =========================
    if st.button("🚀 Train Model", use_container_width=True):

        with st.spinner("Training Q-learning agent..."):
            res = logics.build_rl_model(
                df,
                alpha,
                gamma,
                epsilon,
                episodes
            )

        st.session_state.rl_results = res
        st.success("Training complete")

    # =========================
    # CHECK RESULTS
    # =========================
    if "rl_results" not in st.session_state:
        st.info("Click **Train Model** to begin.")
        return

    res = st.session_state.rl_results

    Q = res["Q"]
    ep_rewards = res["ep_rewards"]
    eps_hist = res["eps_hist"]
    q_delta = res["q_delta"]

    crop_types = res["crop_types"]
    policy_grid = res["policy_grid"]
    temp_labels = res["temp_labels"]
    hum_labels = res["hum_labels"]
    nt, nh = res["nt"], res["nh"]
    R_max = res["R_max"]

    # =========================
    # KPI METRICS
    # =========================
    st.markdown("## 📊 Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Learned Yield", f"{res['learned_mean']:.2f}")
    col2.metric("Random Yield", f"{res['random_mean']:.2f}")
    col3.metric("Improvement %", f"{res['improvement_pct']:.2f}%")

    st.markdown("---")

    # =========================
    # CONVERGENCE
    # =========================
    st.subheader("📉 Convergence")

    if len(q_delta) > 0:
        fig, ax = plt.subplots()
        ax.plot(q_delta, color="green")
        ax.set_xlabel("Iterations (x10 episodes)")
        ax.set_ylabel("|ΔQ|")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    # =========================
    # LEARNING CURVES
    # =========================
    st.subheader("📈 Learning Curves")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(ep_rewards, alpha=0.4, label="Reward")

        if len(ep_rewards) > 20:
            smooth = logics.moving_average(ep_rewards, 30)
            ax.plot(
                range(len(ep_rewards) - len(smooth), len(ep_rewards)),
                smooth,
                label="MA-30"
            )

        ax.set_title("Episode Rewards")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots()
        ax.plot(eps_hist, color="orange")
        ax.fill_between(range(len(eps_hist)), eps_hist, alpha=0.2)
        ax.set_title("Epsilon Decay")
        ax.set_xlabel("Episode")
        ax.set_ylabel("ε")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # =========================
    # POLICY HEATMAP
    # =========================
    st.subheader("🌾 Optimal Crop Policy")

    cmap = mcolors.ListedColormap(CROP_PALETTE[:len(crop_types)])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(policy_grid, cmap=cmap, aspect="auto")

    for i in range(nt):
        for j in range(nh):
            crop = crop_types[policy_grid[i, j]]
            ax.text(j, i, crop[:4], ha="center", va="center",
                    fontsize=7, color="white")

    ax.set_xticks(range(nh))
    ax.set_xticklabels(hum_labels, rotation=45, fontsize=7)

    ax.set_yticks(range(nt))
    ax.set_yticklabels(temp_labels, fontsize=7)

    ax.set_xlabel("Humidity")
    ax.set_ylabel("Temperature")
    ax.set_title("Policy π(s) = argmax Q(s,a)")

    st.pyplot(fig)
    plt.close()

    # =========================
    # Q-TABLE HEATMAPS
    # =========================
    st.subheader("🔥 Q-Value Maps")

    for i, crop in enumerate(crop_types):

        q_grid = Q[:, i].reshape(nt, nh) * R_max

        fig, ax = plt.subplots()
        sns.heatmap(q_grid, ax=ax, cmap="YlOrRd")

        ax.set_title(crop)
        ax.set_xlabel("Humidity")
        ax.set_ylabel("Temperature")

        st.pyplot(fig)
        plt.close()

    # =========================
    # INSIGHTS
    # =========================
    st.subheader("🧾 Insights")

    if res["converged"]:
        st.success("Model converged successfully")
    else:
        st.warning("Model may need more training")

    if res["improvement_pct"] > 0:
        st.success("Policy outperforms random baseline")
    else:
        st.error("Policy underperforms random baseline")
        
        
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
