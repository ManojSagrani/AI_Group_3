import streamlit as st
import pandas as pd
import json
from tab_genai import logics

FEATURE_COLS = logics.RAG_FEATURES
DEFAULTS = {
    "N": 60.0, "P": 45.0, "K": 35.0, "Soil_pH": 6.5,
    "Temperature": 22.0, "Humidity": 65.0, "Wind_Speed": 8.0, "Soil_Quality": 50.0
}


def render(df: pd.DataFrame, api_key: str, n_similar: int):
    st.markdown("## Generative AI — RAG-Enhanced Crop Advisor")

    with st.expander("📖 Methodology: Retrieval-Augmented Generation (RAG)", expanded=False):
        st.markdown("""
**How it works**

1. **User Input** — Enter current field conditions (soil nutrients, pH, temperature, humidity).
2. **Retrieval (RAG)** — The system finds the `k` most similar historical records in the dataset
   using L2 distance on normalised feature vectors. These records serve as *grounding evidence*.
3. **Augmented Prompt** — The retrieved records are injected into the prompt alongside the
   user's conditions, giving the language model concrete empirical context.
4. **Generation** — GPT-4o-mini (OpenAI) plays the role of an expert agronomist and produces
   a structured JSON recommendation covering crop selection, yield forecast, practical advice,
   risk factors, and sustainability considerations.
5. **Structured Output** — The response is enforced via `response_format: json_object` to
   guarantee machine-readable, parseable output every time.

**Why RAG over pure generation?**
Pure LLM generation relies on parametric knowledge that may be outdated or hallucinated.
RAG anchors the response to *actual data points from this specific dataset*, dramatically
improving factual accuracy and reducing hallucination risk for domain-specific recommendations.
        """)

    if not api_key or api_key.strip() == "":
        st.error("⚠️ No OpenAI API key found. Enter one in the sidebar under **GenAI Settings**.")
        return

    st.markdown("### Field Condition Input")
    st.caption("Enter your field's current soil and weather measurements.")

    with st.form("genai_input_form"):
        region = st.text_input("Region / Field Name", value="Farmland Plot A")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            N = st.number_input("Nitrogen N (kg/ha)", 0.0, 200.0, DEFAULTS["N"], step=1.0)
            P = st.number_input("Phosphorus P (kg/ha)", 0.0, 150.0, DEFAULTS["P"], step=1.0)
        with c2:
            K = st.number_input("Potassium K (kg/ha)", 0.0, 200.0, DEFAULTS["K"], step=1.0)
            Soil_pH = st.number_input("Soil pH", 3.0, 10.0, DEFAULTS["Soil_pH"], step=0.1)
        with c3:
            Temperature = st.number_input("Temperature (°C)", -10.0, 60.0, DEFAULTS["Temperature"], step=0.5)
            Humidity = st.number_input("Humidity (%)", 0.0, 100.0, DEFAULTS["Humidity"], step=1.0)
        with c4:
            Wind_Speed = st.number_input("Wind Speed (km/h)", 0.0, 50.0, DEFAULTS["Wind_Speed"], step=0.5)
            Soil_Quality = st.number_input("Soil Quality Index (0–100)", 0.0, 100.0, DEFAULTS["Soil_Quality"], step=1.0)

        submitted = st.form_submit_button("🌾 Get AI Crop Recommendation", type="primary", use_container_width=True)

    if not submitted:
        return

    conditions = {
        "N": N, "P": P, "K": K, "Soil_pH": Soil_pH,
        "Temperature": Temperature, "Humidity": Humidity,
        "Wind_Speed": Wind_Speed, "Soil_Quality": Soil_Quality
    }

    # ── RAG: Show Retrieved Records ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Retrieved Similar Records (RAG Context)")
    st.caption(f"Top {n_similar} historically closest field conditions from the dataset, used as grounding evidence.")

    with st.spinner("Searching dataset for similar records…"):
        try:
            similar = logics.find_similar_records(df, conditions, n=n_similar)
            st.dataframe(similar.style.background_gradient(cmap="YlGn", subset=["Crop_Yield"]),
                         use_container_width=True)
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            return

    # ── GenAI Recommendation ──────────────────────────────────────────────────
    st.markdown("### AI Recommendation")

    with st.spinner("Consulting the AI agronomist… (this may take 5–15 seconds)"):
        try:
            rec, _ = logics.get_recommendation(
                api_key.strip(), conditions, df, region, n_similar
            )
        except json.JSONDecodeError as e:
            st.error(f"The model returned malformed JSON. Try again. ({e})")
            return
        except Exception as e:
            st.error(f"API error: {e}")
            return

    # ── Primary Recommendation Card ───────────────────────────────────────────
    crop = rec.get("recommended_crop", "Unknown")
    conf = rec.get("confidence", 0.0)
    yield_range = rec.get("expected_yield_range", "N/A")
    summary = rec.get("summary", "")

    col_hero, col_meta = st.columns([1, 2])
    with col_hero:
        st.markdown(f"""
<div style="background: linear-gradient(135deg, #1B5E20, #4CAF50); padding: 24px; border-radius: 12px;
            text-align: center; color: white;">
  <div style="font-size: 48px;">🌱</div>
  <div style="font-size: 28px; font-weight: 700; margin: 8px 0;">{crop}</div>
  <div style="font-size: 14px; opacity: 0.85;">Recommended Crop</div>
  <hr style="border-color: rgba(255,255,255,0.3); margin: 12px 0;">
  <div style="font-size: 22px; font-weight: 600;">{conf*100:.0f}%</div>
  <div style="font-size: 12px; opacity: 0.85;">Confidence</div>
  <div style="margin-top: 8px; font-size: 14px;">📦 {yield_range}</div>
</div>
""", unsafe_allow_html=True)

    with col_meta:
        st.markdown(f"**Executive Summary**\n\n{summary}")

        alts = rec.get("alternative_crops", [])
        if alts:
            st.markdown("**Alternative Crops**")
            for alt in alts:
                st.markdown(f"- **{alt.get('crop', '?')}**: {alt.get('rationale', '')}")

    # ── Detailed Sections ─────────────────────────────────────────────────────
    st.markdown("---")
    tab_tech, tab_prac, tab_risk, tab_sust = st.tabs(
        ["🔬 Technical Reasoning", "🌾 Practical Advice", "⚠️ Risk Factors", "♻️ Sustainability"]
    )

    with tab_tech:
        st.markdown(rec.get("technical_reasoning", "N/A"))

    with tab_prac:
        st.markdown(rec.get("practical_advice", "N/A"))

    with tab_risk:
        risks = rec.get("risk_factors", [])
        if risks:
            for r in risks:
                st.warning(r)
        else:
            st.info("No specific risk factors identified.")

    with tab_sust:
        st.markdown(rec.get("sustainability_note", "N/A"))

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    with st.expander("🔍 Raw JSON Response"):
        st.json(rec)

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        "📥 Download Recommendation (JSON)",
        data=json.dumps({"conditions": conditions, "region": region,
                         "similar_records": similar.to_dict(orient="records"),
                         "recommendation": rec}, indent=2),
        file_name="crop_recommendation.json",
        mime="application/json",
        use_container_width=True,
    )

    # ── Ethical Considerations ────────────────────────────────────────────────
    with st.expander("⚠️ Ethical Considerations & Limitations"):
        st.markdown("""
- **Hallucination risk**: Although RAG grounds the model in dataset evidence, LLMs can still
  fabricate plausible-sounding but incorrect agronomic claims. Always cross-validate with a
  qualified agronomist before acting on recommendations.
- **Data scope**: This dataset covers a limited set of crops, soils, and climate zones.
  Recommendations for conditions outside the training distribution may be unreliable.
- **Cost & access**: Dependence on a commercial API (OpenAI) creates a barrier for
  smallholder farmers in low-connectivity or low-income regions.
- **Privacy**: Field condition data entered here is sent to OpenAI's servers. Do not enter
  commercially sensitive GPS coordinates or proprietary agronomic data.
- **Accountability**: AI-generated recommendations should support, not replace, human expertise.
  Final planting decisions carry economic risk and must remain with the farmer.
        """)
