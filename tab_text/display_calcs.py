import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- 1. PAGE CONFIG & SIDEBAR ---
st.set_page_config(page_title="Crop RL Dashboard", layout="wide")

st.sidebar.header("🕹️ RL Control Panel")
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1, help="How much new info overrides old info.")
gamma = st.sidebar.slider("Discount Factor (γ)", 0.5, 0.99, 0.95, help="Importance of future rewards.")
episodes = st.sidebar.number_input("Training Episodes", 100, 5000, 1000)

st.title("🌱 Crop Yield Temporal Difference Learning")

# --- 2. DATA INITIALIZATION ---
if 'df' not in st.session_state:
    # Creating a 144-row dataset for a perfect 12x12 grid
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=144),
        "Crop_Type": ["Wheat", "Corn"] * 72,
        "Soil_Type": ["Sandy", "Loamy"] * 72,
        "Soil_pH": np.random.uniform(5, 8, 144),
        "Temperature": np.random.uniform(15, 35, 144),
        "Humidity": np.random.uniform(30, 90, 144),
        "Wind_Speed": np.random.uniform(0, 20, 144),
        "N": np.random.uniform(10, 100, 144),
        "P": np.random.uniform(10, 100, 144),
        "K": np.random.uniform(10, 100, 144),
        "Crop_Yield": np.random.uniform(2, 10, 144),
        "Soil_Quality": np.random.uniform(0, 1, 144),
    }
    st.session_state.df = pd.DataFrame(data)

df = st.session_state.df

# --- 3. PREPROCESSING ---
df_encoded = pd.get_dummies(df, columns=["Crop_Type", "Soil_Type"])
numeric_cols = ["Soil_pH","Temperature","Humidity","Wind_Speed","N","P","K","Crop_Yield","Soil_Quality"]
df_encoded[numeric_cols] = (df_encoded[numeric_cols] - df_encoded[numeric_cols].mean()) / df_encoded[numeric_cols].std()

temporal_grid = df_encoded.sort_values("Date").set_index("Date")
grid_size = int(len(temporal_grid)**0.5)
n_states = grid_size * grid_size

# --- 4. RL LOGIC ---
def get_possible_actions(state):
    row, col = divmod(state, grid_size)
    actions = []
    if col > 0: actions.append(0) # Left
    if col < grid_size - 1: actions.append(1) # Right
    if row > 0: actions.append(2) # Up
    if row < grid_size - 1: actions.append(3) # Down
    return actions

def next_state(state, action):
    if action == 0: return state - 1
    if action == 1: return state + 1
    if action == 2: return state - grid_size
    if action == 3: return state + grid_size
    return state

def get_reward(state):
    return temporal_grid.iloc[state]['Crop_Yield'] if state < len(temporal_grid) else 0

def run_training():
    V = np.zeros(n_states)
    prog_bar = st.progress(0)
    for ep in range(episodes):
        state = 0
        done = False
        for _ in range(50): # Limit steps per episode
            actions = get_possible_actions(state)
            action = random.choice(actions)
            n_state = next_state(state, action)
            reward = get_reward(n_state)
            V[state] += alpha * (reward + gamma * V[n_state] - V[state])
            state = n_state
        if ep % (max(1, episodes // 20)) == 0:
            prog_bar.progress(ep / episodes)
    prog_bar.empty()
    return V

# --- 5. UI LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Data Preview")
    st.dataframe(df.head(15), height=400)
    if st.button("🚀 Run TD(0) Training"):
        st.session_state.V = run_training()
        st.success("Training Complete!")

with col2:
    st.subheader("Value Function Heatmap")
    if 'V' in st.session_state:
        V_grid = st.session_state.V.reshape(grid_size, grid_size)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(V_grid, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Click the 'Run TD(0) Training' button to visualize results.")
