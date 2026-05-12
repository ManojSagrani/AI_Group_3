import os
import sys
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

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
        episodes=int(rl_episodes),
    )

with tab6:
    ann_results = ann_display.render(
        df,
        hidden_layers=ann_hidden,
        activation=ann_activation,
        alpha=ann_alpha,
        max_iter=int(ann_max_iter),
    )

# ── NEW TAB: Model Comparison ─────────────────────────────────────────────────
with tab7:
    st.subheader("📈 Compare Reinforcement Learning vs MLP Model")

    if "rl_results" not in locals() or "ann_results" not in locals():
        st.warning("⚠️ Train both models first in their respective tabs.")
    else:
        st.markdown("### 🔍 Performance Metrics")

        # Example structure if both return predictions
        try:
            y_true = rl_results["y_true"]
            y_pred_rl = rl_results["y_pred"]
            y_pred_ann = ann_results["y_pred"]

            if pd.api.types.is_numeric_dtype(y_true):
                mae_rl = mean_absolute_error(y_true, y_pred_rl)
                mae_ann = mean_absolute_error(y_true, y_pred_ann)
                r2_rl = r2_score(y_true, y_pred_rl)
                r2_ann = r2_score(y_true, y_pred_ann)

                st.metric("RL MAE", f"{mae_rl:.3f}")
                st.metric("ANN MAE", f"{mae_ann:.3f}")
                st.metric("RL R²", f"{r2_rl:.3f}")
                st.metric("ANN R²", f"{r2_ann:.3f}")
            else:
                acc_rl = accuracy_score(y_true, y_pred_rl)
                acc_ann = accuracy_score(y_true, y_pred_ann)
                f1_rl = f1_score(y_true, y_pred_rl, average='macro')
                f1_ann = f1_score(y_true, y_pred_ann, average='macro')

                st.metric("RL Accuracy", f"{acc_rl*100:.2f}%")
                st.metric("ANN Accuracy", f"{acc_ann*100:.2f}%")
                st.metric("RL F1-score", f"{f1_rl:.3f}")
                st.metric("ANN F1-score", f"{f1_ann:.3f}")

        except Exception as e:
            st.error(f"Comparison failed: {e}")

with tab8:
    genai_display.render(df)
