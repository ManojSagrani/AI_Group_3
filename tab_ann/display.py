import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tab_ann import logics

FEATURE_COLS = logics.FEATURE_COLS
FEATURE_DEFAULTS = {
    "N": 60.0, "P": 45.0, "K": 35.0, "Soil_pH": 6.5,
    "Temperature": 22.0, "Humidity": 65.0, "Wind_Speed": 8.0, "Soil_Quality": 50.0
}


def _architecture_str(hidden_layers):
    sizes = [f"Input ({len(FEATURE_COLS)}+)"] + [f"Dense({n}) + ReLU" for n in hidden_layers] + ["Output"]
    return " → ".join(sizes)


def render(df: pd.DataFrame, hidden_layers: tuple, activation: str,
           alpha: float, max_iter: int):

    st.markdown("## Artificial Neural Network — Multi-Layer Perceptron")

    with st.expander("📖 Methodology & Architecture", expanded=False):
        st.markdown(f"""
**Two supervised learning tasks on the crop yield dataset:**

| Task | Type | Target | Metric |
|---|---|---|---|
| Yield Prediction | Regression | `Crop_Yield` (continuous) | R², RMSE, MAE |
| Crop Classification | Multi-class | `Crop_Type` (10 classes) | Accuracy, F1 |

**Network Architecture**
`{_architecture_str(hidden_layers)}`

- **Solver**: Adam (adaptive learning rate)
- **Regularisation**: L2 penalty α = {alpha}
- **Early stopping**: validation fraction 10 %, patience 25 iterations
- **Feature scaling**: StandardScaler (zero mean, unit variance) applied before training

Categorical features (`Soil_Type`, `Crop_Type`) are one-hot encoded.
Permutation-based feature importance is computed post-training on the held-out test set.
        """)

    task = st.radio("Select Task", ["Yield Prediction (Regression)", "Crop Classification"],
                    horizontal=True)
    st.markdown("---")

    regression_mode = task.startswith("Yield")

    if st.button("🧠 Train Neural Network", type="primary", use_container_width=True):
        with st.spinner("Preprocessing & training…"):
            if regression_mode:
                X, y, feat_names = logics.preprocess_regression(df)
                model, scaler, X_te_s, y_te, y_pred, metrics = logics.train_regressor(
                    X, y, hidden_layers, activation, alpha, max_iter
                )
                imp_df = logics.feature_importance(model, X_te_s, y_te, feat_names)
                st.session_state.ann_reg = {
                    "model": model, "scaler": scaler, "feat_names": feat_names,
                    "y_te": y_te, "y_pred": y_pred, "metrics": metrics, "imp_df": imp_df,
                }
            else:
                X, y, feat_names, le = logics.preprocess_classification(df)
                model, scaler, X_te_s, y_te, y_pred, y_proba, metrics = logics.train_classifier(
                    X, y, le, hidden_layers, activation, alpha, max_iter
                )
                imp_df = logics.feature_importance(model, X_te_s, y_te, feat_names)
                st.session_state.ann_cls = {
                    "model": model, "scaler": scaler, "feat_names": feat_names,
                    "le": le, "y_te": y_te, "y_pred": y_pred, "y_proba": y_proba,
                    "metrics": metrics, "imp_df": imp_df,
                }
        st.success("✅ Training complete!")

    # ── Regression Results ────────────────────────────────────────────────────
    if regression_mode and "ann_reg" in st.session_state:
        res = st.session_state.ann_reg
        m = res["metrics"]

        st.markdown("### Regression Performance")
        st.caption("Target is log₁p(Crop_Yield) — metrics are on the log scale; scatter plot shows de-transformed t/ha values.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score", f"{m['r2']:.4f}", help="1.0 = perfect fit on log scale")
        c2.metric("RMSE (log scale)", f"{m['rmse']:.4f}")
        c3.metric("MAE (log scale)", f"{m['mae']:.4f}")
        c4.metric("Iterations", f"{m['n_iter']}", delta="converged" if m["converged"] else "max reached")

        import numpy as _np  # local import to avoid module-level unused warning
        y_te_orig = _np.expm1(res["y_te"])
        y_pred_orig = _np.expm1(res["y_pred"])

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Training Loss Curve**")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(res["model"].loss_curve_, color="#2196F3", linewidth=2, label="Train loss")
            ax.set_xlabel("Iteration"); ax.set_ylabel("MSE Loss (log scale)")
            ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_b:
            st.markdown("**Actual vs Predicted Yield (t/ha)**")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(y_te_orig, y_pred_orig, alpha=0.25, s=8, color="#4CAF50")
            lims = [min(y_te_orig.min(), y_pred_orig.min()),
                    max(y_te_orig.max(), y_pred_orig.max())]
            ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
            ax.set_xlabel("Actual Yield (t/ha)"); ax.set_ylabel("Predicted Yield (t/ha)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Permutation Feature Importance**")
        fig, ax = plt.subplots(figsize=(10, max(3, len(res["imp_df"]) * 0.35)))
        imp = res["imp_df"].head(15)
        colors = ["#2196F3" if v >= 0 else "#F44336" for v in imp["Importance"]]
        ax.barh(imp["Feature"], imp["Importance"], xerr=imp["Std"],
                color=colors, capsize=3, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean decrease in R²"); ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Residuals (on log scale — homoscedasticity check)
        st.markdown("**Residual Distribution (log scale)**")
        residuals = res["y_te"] - res["y_pred"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].hist(residuals, bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
        axes[0].axvline(0, color="red", linestyle="--"); axes[0].set_title("Residual Histogram")
        axes[0].set_xlabel("Residual log(actual) − log(predicted)")
        axes[1].scatter(res["y_pred"], residuals, alpha=0.2, s=6, color="#FF9800")
        axes[1].axhline(0, color="red", linestyle="--"); axes[1].set_title("Residuals vs Fitted")
        axes[1].set_xlabel("Predicted (log scale)"); axes[1].set_ylabel("Residual")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Live prediction
        _live_prediction_regression(res)

    # ── Classification Results ────────────────────────────────────────────────
    if not regression_mode and "ann_cls" in st.session_state:
        res = st.session_state.ann_cls
        m = res["metrics"]
        le = res["le"]

        st.markdown("### Classification Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        c2.metric("Macro F1", f"{m['report']['macro avg']['f1-score']:.4f}")
        c3.metric("Iterations", f"{m['n_iter']}", delta="converged" if m["converged"] else "max reached")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Training Loss Curve**")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(res["model"].loss_curve_, color="#2196F3", linewidth=2, label="Train loss")
            if hasattr(res["model"], "validation_scores_") and res["model"].validation_scores_:
                ax2 = ax.twinx()
                ax2.plot(res["model"].validation_scores_, color="#FF9800", linewidth=1.5,
                         linestyle="--", label="Val accuracy")
                ax2.set_ylabel("Validation Accuracy", color="#FF9800")
            ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
            ax.legend(loc="upper left"); ax.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_b:
            st.markdown("**Confusion Matrix**")
            cm = m["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=le.classes_, yticklabels=le.classes_,
                        linewidths=0.5)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.tick_params(axis="x", rotation=45); ax.tick_params(axis="y", rotation=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Per-Class Classification Report**")
        report_df = pd.DataFrame(m["report"]).T
        report_df = report_df.drop(index=["accuracy"], errors="ignore")
        st.dataframe(report_df.round(4).style.background_gradient(cmap="Blues", subset=["f1-score", "precision", "recall"]),
                     use_container_width=True)

        st.markdown("**Permutation Feature Importance**")
        fig, ax = plt.subplots(figsize=(10, max(3, len(res["imp_df"].head(15)) * 0.35)))
        imp = res["imp_df"].head(15)
        ax.barh(imp["Feature"], imp["Importance"], xerr=imp["Std"],
                color="#4CAF50", capsize=3, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean decrease in Accuracy"); ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Live prediction
        _live_prediction_classification(res, le)

    if not regression_mode and "ann_cls" not in st.session_state and regression_mode is False:
        if "ann_reg" not in st.session_state:
            st.info("Configure parameters in the sidebar and click **Train Neural Network**.")

    # ── Ethical Considerations ────────────────────────────────────────────────
    with st.expander("⚠️ Ethical Considerations & Limitations"):
        st.markdown("""
- **Data leakage risk**: Features `Crop_Type` and `Soil_Type` are one-hot encoded; including
  `Crop_Type` in the regression task effectively gives the model the answer for some patterns.
  Interpret feature importance accordingly.
- **Distribution shift**: A model trained on 2014–2024 data may degrade under climate change
  scenarios that push conditions outside the training distribution.
- **Yield vs. profitability**: Maximising yield does not always maximise farmer income
  (commodity prices, input costs, and market access are excluded from this model).
- **Model uncertainty**: Confidence intervals are not provided by sklearn's MLP;
  production systems should use Bayesian Neural Networks or conformal prediction.
- **Fairness**: The dataset does not record farm size, ownership, or farmer demographics —
  recommendations may inadvertently favour large-scale, input-intensive farming.
        """)


def _live_prediction_regression(res):
    st.markdown("---")
    st.markdown("### 🔮 Live Yield Prediction")
    st.caption("Enter field conditions to get an instant ANN yield estimate.")

    with st.form("reg_pred_form"):
        cols = st.columns(4)
        vals = {}
        for i, feat in enumerate(FEATURE_COLS):
            with cols[i % 4]:
                vals[feat] = st.number_input(feat, value=float(FEATURE_DEFAULTS[feat]), format="%.2f")

        # Pad extra features (one-hot columns) with zero
        full_vals = [vals[f] for f in FEATURE_COLS]
        n_extra = len(res["feat_names"]) - len(FEATURE_COLS)
        full_vals += [0.0] * n_extra

        submitted = st.form_submit_button("Predict Yield")
        if submitted:
            import numpy as _np
            pred_log, _ = logics.predict_single(res["model"], res["scaler"], full_vals)
            pred = float(_np.expm1(pred_log))
            st.success(f"🌾 Predicted Crop Yield: **{pred:.2f} t/ha**")


def _live_prediction_classification(res, le):
    st.markdown("---")
    st.markdown("### 🔮 Live Crop Recommendation")
    st.caption("Enter field conditions to get the ANN's top crop recommendation.")

    with st.form("cls_pred_form"):
        cols = st.columns(4)
        vals = {}
        for i, feat in enumerate(FEATURE_COLS):
            with cols[i % 4]:
                vals[feat] = st.number_input(feat, value=float(FEATURE_DEFAULTS[feat]), format="%.2f")

        full_vals = [vals[f] for f in FEATURE_COLS]
        n_extra = len(res["feat_names"]) - len(FEATURE_COLS)
        full_vals += [0.0] * n_extra

        submitted = st.form_submit_button("Recommend Crop")
        if submitted:
            pred_idx, proba = logics.predict_single(
                res["model"], res["scaler"], full_vals
            )
            pred_crop = le.inverse_transform([int(pred_idx)])[0]
            st.success(f"🌱 Recommended Crop: **{pred_crop}**")

            if proba is not None:
                st.markdown("**Confidence per class:**")
                proba_df = pd.DataFrame({
                    "Crop": le.classes_,
                    "Confidence": (proba * 100).round(2)
                }).sort_values("Confidence", ascending=False)
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.barh(proba_df["Crop"], proba_df["Confidence"],
                        color=["#4CAF50" if c == pred_crop else "#2196F3" for c in proba_df["Crop"]])
                ax.set_xlabel("Confidence (%)"); ax.invert_yaxis()
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout(); st.pyplot(fig); plt.close()
