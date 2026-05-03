import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
import os

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


# --- PAGE CONFIG ---
st.set_page_config(page_title="Crop Recommendation System", layout="wide", initial_sidebar_state="expanded")

st.title("Crop Recommendation System")
st.markdown("### AI Optimal Crop Selection based on Soil and Weather Data")

# --- SIDEBAR ---
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("RL Parameters")
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.5, 0.99, 0.9)
epsilon = st.sidebar.slider("Exploration (ε %)", 0, 100, 20) / 100.0  # Epsilon from 100
episodes = st.sidebar.number_input("Episodes", 100, 5000, 500)

if uploaded_file is not None:
    if 'df_logics' in locals():
        df = df_logics.load_csv(uploaded_file)
    else:
        uploaded_file.seek(0)  # 🔥 critical fix
        df = pd.read_csv(uploaded_file)
               
    df['Date'] = pd.to_datetime(df['Date'])
            
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔢 Numeric", "🔤 Text", "📅 Date", "🤖 RL Model"])
    st.sidebar.success(f"Loaded: {uploaded_file.name}")

    # Standard Tabs (1-4)
    with tab1:
        if 'df_display' in locals(): df_display.overview(df)
        else: st.write(df.head())
    with tab2:
        if 'numeric_display' in locals(): numeric_display.numeric_series(df)
    with tab3:
        if 'text_display' in locals(): text_display.text_series(df)
    with tab4:
        if 'date_display' in locals(): date_display.datetime_series(df)
    with tab5:
        st.header("TD-Learning Value Map")
        st.info("Epsilon-Decay: it learns randomly at first, then shifts to picking high-yield regions.")
        
        # 1. SETUP REDUCED GRID
        GRID_SIZE = 8 
        n_states = GRID_SIZE * GRID_SIZE
        
        # Define labels based on data range
        x_col, y_col = 'Temperature', 'Humidity' 
        if x_col in df.columns and y_col in df.columns:
            x_bins = np.linspace(df[x_col].min(), df[x_col].max(), GRID_SIZE).round(1)
            y_bins = np.linspace(df[y_col].min(), df[y_col].max(), GRID_SIZE).round(1)
        else:
            x_bins = [f"R{i}" for i in range(GRID_SIZE)]
            y_bins = [f"R{i}" for i in range(GRID_SIZE)]

        # 2. TRAINING EXECUTION
        if st.button("🚀 Run RL Training with Epsilon-Decay"):
            V = np.zeros(n_states)
            logs = []
            reward_history = []
            
            # Use the epsilon from your sidebar as the STARTING point
            current_epsilon = epsilon 
            
            df_mapped = df.copy()
            df_mapped['state_idx'] = np.linspace(0, n_states - 1, len(df)).astype(int)

            progress_bar = st.progress(0)

            for ep in range(1, episodes + 1):
                state = random.randint(0, n_states - 1)
                ep_reward = 0
                
                for _ in range(30): # Steps per episode
                    row, col = divmod(state, GRID_SIZE)
                    possible_moves = []
                    if col > 0: possible_moves.append(-1)
                    if col < GRID_SIZE - 1: possible_moves.append(1)
                    if row > 0: possible_moves.append(-GRID_SIZE)
                    if row < GRID_SIZE - 1: possible_moves.append(GRID_SIZE)
                    
                    # --- EPSILON-GREEDY STRATEGY ---
                    if random.random() > current_epsilon:
                        # Exploit: Pick neighbor with highest V-value
                        move_values = [V[state + m] for m in possible_moves]
                        n_s = state + possible_moves[np.argmax(move_values)]
                    else:
                        # Explore: Random move
                        n_s = state + random.choice(possible_moves)
                    
                    # Reward lookup
                    state_data = df_mapped[df_mapped['state_idx'] == n_s]
                    reward = state_data['Crop_Yield'].mean() if not state_data.empty else 0
                    
                    # TD Update (Bellman Equation)
                    V[state] += alpha * (reward + gamma * V[n_s] - V[state])
                    
                    state = n_s
                    ep_reward += reward
                
                # --- EPSILON DECAY ---
                # Gradually reduce randomness by 1% each episode, down to a floor of 0.01
                current_epsilon = max(0.01, current_epsilon * 0.99)
                
                reward_history.append(ep_reward)
                if ep % 50 == 0 or ep == 1:
                    logs.append({
                        "Episode": ep, 
                        "Total Reward": round(ep_reward, 2), 
                        "Epsilon": round(current_epsilon, 3)
                    })
                
                progress_bar.progress(ep / episodes)

            st.session_state.summary_data = {
                "V": V.reshape(GRID_SIZE, GRID_SIZE),
                "logs": pd.DataFrame(logs),
                "history": reward_history
            }
            st.success("Training Complete!")

        # 3. DISPLAY RESULTS
        if 'summary_data' in st.session_state:
            data = st.session_state.summary_data
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Learning Log")
                st.dataframe(data["logs"], use_container_width=True)
            with c2:
                st.subheader("Reward Progression (Upward Trend)")
                st.line_chart(data["history"])

            st.subheader("Learned Value Heatmap (Optimal Growing Zones)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(data["V"], annot=True, fmt=".1f", cmap="YlGn", xticklabels=x_bins, yticklabels=y_bins, ax=ax)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            st.pyplot(fig)
 
    