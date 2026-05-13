import os
import sys
from openai import api_key
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from dotenv import load_dotenv
import time



# Load .env file
load_dotenv()

# Read API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    div[data-testid="metric-container"] {
        background: rgba(33, 150, 243, 0.08);
        border: 1px solid rgba(33, 150, 243, 0.2);
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 CropAI — Intelligent Crop Recommendation System")
st.markdown("*Integrating RL · ANN · GenAI for optimal crop recommendations.*")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wheat.png", width=80)
    st.markdown("## 🌾 CropAI Settings")

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
    st.markdown("### ⚙️ Preprocessing Parameters")
    n_temp_bins = st.slider("Temperature bins", 2, 20, 8, step=1)
    n_humidity_bins = st.slider("Humidity bins", 2, 20, 8, step=1)

    st.markdown("---")

    # RL PARAMETERS
    st.markdown("### 🤖 RL — Q-Learning Parameters")
    rl_alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01)
    rl_gamma = st.slider("Discount Factor (γ)", 0.5, 0.99, 0.9, 0.01)
    rl_epsilon = st.slider("Exploration (ε)", 0.05, 1.0, 0.3, 0.05)
    rl_episodes = st.number_input("Training Episodes", 100, 5000, 500, step=100)

    st.markdown("---")

    # ANN PARAMETERS
    st.markdown("### 🧠 ANN — MLP Parameters")
    ann_layers_str = st.text_input("Hidden Layers", "128,64,32")
    try:
        ann_hidden = tuple(int(x.strip()) for x in ann_layers_str.split(",") if x.strip())
    except ValueError:
        ann_hidden = (128, 64, 32)
    ann_activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"], index=0)
    ann_alpha = st.select_slider("L2 Regularisation (α)", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
    ann_max_iter = st.number_input("Max Iterations", 100, 2000, 500, step=100)

if df_raw is None:
    st.info("👈 Upload a CSV to continue.")
    st.stop()

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
    "📈 Model Comparison",   # ← NEW TAB
    "✨ GenAI Advisor",
]

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(TAB_LABELS)

with tab1:
    df_display.overview(df)

with tab2:
    numeric_display.numeric_serie(df)

with tab3:
    text_display.text_serie(df)

with tab4:
    date_display.datetime_serie(df)

with tab5:
    rl_results = rl_display.render(
        df,
        alpha=rl_alpha,
        gamma=rl_gamma,
        epsilon=rl_epsilon,
        episodes=int(rl_episodes)
       
    )

with tab6:
    ann_results = ann_display.render(
        df,
        hidden_layers=ann_hidden,
        activation=ann_activation,
        alpha=ann_alpha,
        max_iter=int(ann_max_iter),
        n_temp_bins=int(n_temp_bins),
        n_humidity_bins=int(n_humidity_bins)
    )

# ── NEW TAB:  Comparison ─────────────────────────────────────────────────
# ──  COMPARISON TAB ──────────────────────────────────────────────────────
with tab7:
    st.subheader("📊 Research-Grade Model Comparison (RL vs ANN)")

    rl = st.session_state.get("rl_results")
    ann = st.session_state.get("ann_results")

    # ---------------- SAFETY ----------------
    if rl is None or ann is None:
        st.warning("⚠️ Please train BOTH RL and ANN models first.")
        st.stop()

    if not isinstance(rl, dict) or not isinstance(ann, dict):
        st.error("❌ Invalid model outputs.")
        st.stop()

    st.markdown("## 📌 Performance + Efficiency Overview")

    col1, col2 = st.columns(2)

    # ---------------- RL ----------------
    with col1:
        st.markdown("### 🤖 Reinforcement Learning")

        rl_mae = rl.get("mae", 0)
        rl_r2 = rl.get("r2", None)
        rl_acc = rl.get("accuracy", None)
        rl_time = rl.get("train_time", 0)
        rl_iter = rl.get("n_iter", 0)
        rl_conv = rl.get("converged", False)

        st.metric("MAE", f"{rl_mae:.3f}")
        if rl_r2 is not None:
            st.metric("R²", f"{rl_r2:.3f}")
        if rl_acc is not None:
            st.metric("Accuracy", f"{rl_acc*100:.2f}%")

        st.metric("Training Time (s)", f"{rl_time:.3f}")
        st.metric("Iterations", rl_iter)
        st.metric("Converged", "✅ Yes" if rl_conv else "❌ No")

    # ---------------- ANN ----------------
    with col2:
        st.markdown("### 🧠 Artificial Neural Network")

        ann_mae = ann.get("mae", 0)
        ann_rmse = ann.get("rmse", None)
        ann_r2 = ann.get("r2", None)
        ann_acc = ann.get("accuracy", None)
        ann_time = ann.get("train_time", 0)
        ann_iter = ann.get("n_iter", 0)
        ann_conv = ann.get("converged", False)

        st.metric("MAE", f"{ann_mae:.3f}")
        if ann_rmse is not None:
            st.metric("RMSE", f"{ann_rmse:.3f}")
        if ann_r2 is not None:
            st.metric("R²", f"{ann_r2:.3f}")
        if ann_acc is not None:
            st.metric("Accuracy", f"{ann_acc*100:.2f}%")

        st.metric("Training Time (s)", f"{ann_time:.3f}")
        st.metric("Iterations", ann_iter)
        st.metric("Converged", "✅ Yes" if ann_conv else "❌ No")

    # ---------------- NORMALISED SCORE ----------------
    st.markdown("---")
    st.markdown("## 🏆 Overall Model Score (Research Metric)")

    def score_model(m):
        score = 0

        # error (lower is better)
        if "mae" in m:
            score += max(0, 1 - m["mae"])

        # r2 (higher is better)
        if m.get("r2") is not None:
            score += m["r2"]

        # convergence bonus
        if m.get("converged"):
            score += 0.5

        # speed bonus
        t = m.get("train_time", 1)
        score += 1 / (1 + t)

        return score

    rl_score = score_model(rl)
    ann_score = score_model(ann)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("RL Overall Score", f"{rl_score:.3f}")

    with col2:
        st.metric("ANN Overall Score", f"{ann_score:.3f}")

    # ---------------- WINNER ----------------
    st.markdown("## 🥇 Final Verdict")

    if rl_score > ann_score:
        st.success("🏆 Reinforcement Learning performs better overall")
    elif ann_score > rl_score:
        st.success("🏆 ANN performs better overall")
    else:
        st.info("🤝 Both models perform similarly")

    # ---------------- INSIGHT SUMMARY ----------------
    st.markdown("## 🧾 Key Insights")

    insights = []

    if rl_conv and not ann_conv:
        insights.append("RL converged while ANN did not")
    if ann_conv and not rl_conv:
        insights.append("ANN converged while RL did not")

    if rl_time < ann_time:
        insights.append("RL trained faster")
    else:
        insights.append("ANN trained faster")

    if rl_mae < ann_mae:
        insights.append("RL has lower prediction error (MAE)")
    else:
        insights.append("ANN has lower prediction error (MAE)")

    for i in insights:
        st.write("• " + i)        
with tab8:
  genai_results = genai_display.render(
        df,
        api_key=OPENAI_API_KEY.strip(),
        n_similar=5
    )