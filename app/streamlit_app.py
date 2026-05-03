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

# Attempting imports from your custom modules
try:
    from tab_df import display as df_display, logics as df_logics
    from tab_text import display as text_display
    from tab_numeric import display as numeric_display
    from tab_date import display as date_display
except ImportError:
    st.error("Custom modules not found. Check your folder structure.")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crop Recommendation System", layout="wide", initial_sidebar_state="expanded")

st.title("Crop Recommendation System")
st.markdown("### - AI Optimal Crop Selection based on Soil and Weather Data")

# --- SIDEBAR & UPLOAD ---
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("RL Parameters")
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.5, 0.99, 0.9)
episodes = st.sidebar.number_input("Episodes", 100, 5000, 500)

if uploaded_file:
    # Use your logic to load CSV
    df = pd.read_csv(uploaded_file)
    if 'df_logics' in locals():
        df = df_logics.load_csv(uploaded_file)
    
    df['Date'] = pd.to_datetime(df['Date'])
    st.sidebar.success(f"Loaded: {uploaded_file.name}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔢 Numeric", "🔤 Text", "📅 Date", "🤖 RL Yield Model"
    ])

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

    # --- TAB 5: RL LOGIC + DOWNLOAD ---
    with tab5:
        st.header("Temporal Difference (TD) Learning Grid")
        
        # 1. Preprocessing
        cat_cols = [col for col in ["Crop_Type", "Soil_Type"] if col in df.columns]
        df_encoded = pd.get_dummies(df, columns=cat_cols).sort_values("Date")
        
        grid_size = int(len(df_encoded)**0.5)
        n_states = grid_size * grid_size
        
        # 2. RL Functions
        def get_possible_actions(s):
            row, col = divmod(s, grid_size)
            actions = []
            if col > 0: actions.append(0)
            if col < grid_size - 1: actions.append(1)
            if row > 0: actions.append(2)
            if row < grid_size - 1: actions.append(3)
            return actions

        def next_state(s, a):
            if a == 0: return s - 1
            if a == 1: return s + 1
            if a == 2: return s - grid_size
            if a == 3: return s + grid_size
            return s

        def get_reward(s):
            return df_encoded.iloc[s]["Crop_Yield"] if "Crop_Yield" in df_encoded.columns else 0

        # 3. Training Execution
        if st.button("🚀 Run RL Training"):
            V = np.zeros(n_states)
            bar = st.progress(0)
            
            for ep in range(episodes):
                state = 0
                for _ in range(50):
                    actions = get_possible_actions(state)
                    if not actions: break
                    n_s = next_state(state, random.choice(actions))
                    reward = get_reward(n_s)
                    V[state] += alpha * (reward + gamma * V[n_s] - V[state])
                    state = n_s
                if ep % (max(1, episodes // 10)) == 0:
                    bar.progress(ep / episodes)
            bar.empty()
            
            # Store in session state for persistency
            st.session_state.trained_V = V.reshape(grid_size, grid_size)
            st.success("Training Complete!")

        # 4. Display & Export
        if 'trained_V' in st.session_state:
            V_grid = st.session_state.trained_V
            
            # Visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(V_grid, annot=True, cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
            
            # Download Button
            v_df = pd.DataFrame(V_grid)
            csv_data = v_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Learned Value Table (CSV)",
                data=csv_data,
                file_name="learned_crop_values.csv",
                mime="text/csv",
            )
else:
    st.info("Please upload a CSV file to begin.")
