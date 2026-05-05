import os
import sys
import streamlit as st
import pandas as pd

# ── Path resolution ───────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

DATASET_DEFAULT = os.path.join(BASE_DIR, "..", "crop_yield_dataset.csv")

# ── Module imports ────────────────────────────────────────────────────────────
from tab_df import display as df_display
from tab_numeric import display as numeric_display
from tab_text import display as text_display
from tab_date import display as date_display
from tab_rl import display as rl_display
from tab_ann import display as ann_display
from tab_genai import display as genai_display

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropAI — Intelligent Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metric card styling */
    div[data-testid="metric-container"] {
        background: rgba(33, 150, 243, 0.08);
        border: 1px solid rgba(33, 150, 243, 0.2);
        border-radius: 8px;
        padding: 12px;
    }
    /* Tab font */
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
    }
    /* Sidebar header */
    section[data-testid="stSidebar"] h2 {
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 CropAI — Intelligent Crop Recommendation System")
st.markdown(
    "*Integrating Reinforcement Learning · Artificial Neural Networks · Generative AI "
    "to optimise agricultural crop selection from soil and weather data.*"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wheat.png", width=80)
    st.markdown("## 🌾 CropAI Settings")

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")
    use_default = os.path.isfile(DATASET_DEFAULT)
    upload_label = "Upload a CSV file" if not use_default else "Upload a custom CSV (or use bundled)"
    uploaded_file = st.file_uploader(upload_label, type=["csv"])

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded: {uploaded_file.name} ({len(df_raw):,} rows)")
    elif use_default:
        df_raw = pd.read_csv(DATASET_DEFAULT)
        st.success(f"✅ Using bundled dataset ({len(df_raw):,} rows)")
    else:
        df_raw = None
        st.warning("Upload a CSV file to begin.")

    st.markdown("---")

    # ── RL Parameters ─────────────────────────────────────────────────────────
    st.markdown("### 🤖 RL — Q-Learning Parameters")
    rl_alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01,
                         help="How aggressively Q-values update from new rewards.")
    rl_gamma = st.slider("Discount Factor (γ)", 0.5, 0.99, 0.9, 0.01,
                         help="Weight of future vs. immediate rewards.")
    rl_epsilon = st.slider("Initial Exploration (ε)", 0.05, 1.0, 0.3, 0.05,
                           help="Probability of random action at episode start.")
    rl_episodes = st.number_input("Training Episodes", 100, 5000, 500, step=100)
    rl_temp_bins = st.selectbox("Temperature Bins", [6, 8, 10], index=1)
    rl_hum_bins = st.selectbox("Humidity Bins", [6, 8, 10], index=1)

    st.markdown("---")

    # ── ANN Parameters ────────────────────────────────────────────────────────
    st.markdown("### 🧠 ANN — MLP Parameters")
    ann_layers_str = st.text_input(
        "Hidden Layer Sizes (comma-separated)", "128,64,32",
        help="E.g. '128,64,32' → three hidden layers of 128, 64, and 32 neurons."
    )
    try:
        ann_hidden = tuple(int(x.strip()) for x in ann_layers_str.split(",") if x.strip())
        if not ann_hidden:
            ann_hidden = (128, 64, 32)
    except ValueError:
        ann_hidden = (128, 64, 32)
        st.warning("Invalid layer sizes, using default (128, 64, 32).")

    ann_activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"], index=0)
    ann_alpha = st.select_slider("L2 Regularisation (α)", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-4,
                                 format_func=lambda x: f"{x:.0e}")
    ann_max_iter = st.number_input("Max Training Iterations", 100, 2000, 500, step=100)

    st.markdown("---")

    # ── GenAI Parameters ──────────────────────────────────────────────────────
    st.markdown("### ✨ GenAI — RAG Settings")

    # Auto-load API key from .env if present
    _env_path = os.path.join(BASE_DIR, ".env")
    _default_key = ""
    if os.path.isfile(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                if _line.startswith("OPENAI_API_KEY="):
                    _default_key = _line.split("=", 1)[1].strip()
                    break

    genai_api_key = st.text_input(
        "OpenAI API Key", value=_default_key, type="password",
        help="Loaded automatically from .env if present."
    )
    genai_n_similar = st.slider("RAG Context Records (k)", 3, 10, 5,
                                help="Number of similar historical records sent as context to GPT.")

    st.markdown("---")
    st.caption("AI Group 3 · CropAI Project · 2026")

# ── Guard: no data ────────────────────────────────────────────────────────────
if df_raw is None:
    st.info("👈 Please upload a CSV file using the sidebar to get started.")
    st.stop()

# ── Preprocessing ─────────────────────────────────────────────────────────────
df = df_raw.copy()
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# ── Tabs ──────────────────────────────────────────────────────────────────────
TAB_LABELS = [
    "📊 Overview",
    "🔢 Numeric",
    "🔤 Text",
    "📅 Date",
    "🤖 Reinforcement Learning",
    "🧠 Neural Network",
    "✨ GenAI Advisor",
]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(TAB_LABELS)

with tab1:
    df_display.overview(df)

with tab2:
    numeric_display.numeric_serie(df)

with tab3:
    text_display.text_serie(df)

with tab4:
    date_display.datetime_serie(df)

with tab5:
    rl_display.render(
        df,
        alpha=rl_alpha,
        gamma=rl_gamma,
        epsilon=rl_epsilon,
        episodes=int(rl_episodes),
        n_temp_bins=rl_temp_bins,
        n_humidity_bins=rl_hum_bins,
    )

with tab6:
    ann_display.render(
        df,
        hidden_layers=ann_hidden,
        activation=ann_activation,
        alpha=ann_alpha,
        max_iter=int(ann_max_iter),
    )

with tab7:
    genai_display.render(
        df,
        api_key=genai_api_key,
        n_similar=genai_n_similar,
    )
