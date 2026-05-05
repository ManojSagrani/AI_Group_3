import json
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cdist

RAG_FEATURES = ["N", "P", "K", "Soil_pH", "Temperature", "Humidity", "Wind_Speed", "Soil_Quality"]

SYSTEM_PROMPT = """
You are Dr. Amara Osei, a senior Agronomist and Precision Agriculture Scientist at the Global Institute
for Sustainable Crop Systems. You hold a PhD in Soil Science and have 20 years of field experience across
diverse agro-climatic zones. You specialise in translating soil chemistry and microclimate data into
actionable, data-driven crop recommendations.

Your core principles:
1. Every recommendation must be grounded in the provided numerical data — no generic advice.
2. Explain the mechanistic "why": how specific nutrient ratios, pH, temperature, and humidity
   create the physiological conditions that favour or disfavour each crop.
3. Cross-reference your reasoning with the historical records provided as context.
4. Quantify uncertainty where data coverage is limited.
5. Balance immediate yield maximisation with long-term soil health and sustainability.
6. Use precise agricultural terminology: cation exchange capacity, vapour pressure deficit,
   nitrogen use efficiency, rooting depth, etc.
""".strip()

OUTPUT_SCHEMA = """\
Return ONLY a valid JSON object with exactly this structure — no preamble, no markdown:
{
  "recommended_crop": "<primary crop name>",
  "confidence": <float 0.0–1.0>,
  "expected_yield_range": "<min>–<max> tonnes/ha",
  "technical_reasoning": "<200+ word scientific justification citing the specific data values provided>",
  "practical_advice": "<planting calendar, fertilisation strategy, irrigation regime, pest/disease watch>",
  "risk_factors": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "alternative_crops": [
    {"crop": "<name>", "rationale": "<one sentence>"},
    {"crop": "<name>", "rationale": "<one sentence>"}
  ],
  "sustainability_note": "<soil health impact, crop rotation suggestion, carbon footprint note>",
  "summary": "<2-sentence executive takeaway for the farmer>"
}"""


def find_similar_records(df: pd.DataFrame, query: dict, n: int = 5) -> pd.DataFrame:
    """Retrieve the n most similar historical records using L2 distance on normalised features."""
    df_clean = df.dropna(subset=RAG_FEATURES + ["Crop_Type", "Crop_Yield"]).copy()

    X = df_clean[RAG_FEATURES].values.astype(float)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    X_norm = (X - mins) / ranges

    q_vec = np.array([float(query.get(f, 0.0)) for f in RAG_FEATURES])
    q_norm = (q_vec - mins) / ranges

    dists = cdist(q_norm.reshape(1, -1), X_norm, metric="euclidean")[0]
    top_idx = np.argsort(dists)[:n]

    out = df_clean.iloc[top_idx][RAG_FEATURES + ["Crop_Type", "Soil_Type", "Crop_Yield"]].copy()
    out.insert(0, "Similarity Rank", range(1, n + 1))
    out["Distance"] = dists[top_idx].round(4)
    out["Crop_Yield"] = out["Crop_Yield"].round(2)
    return out.reset_index(drop=True)


def _format_conditions(cond: dict) -> str:
    unit_map = {
        "N": "kg/ha", "P": "kg/ha", "K": "kg/ha",
        "Soil_pH": "(unitless)", "Temperature": "°C",
        "Humidity": "%", "Wind_Speed": "km/h", "Soil_Quality": "/100"
    }
    lines = [f"  • {k}: {cond.get(k, 'N/A')} {unit_map.get(k, '')}" for k in RAG_FEATURES]
    return "\n".join(lines)


def build_user_prompt(conditions: dict, similar_records: pd.DataFrame,
                      region: str = "Unknown Region") -> str:
    return f"""FIELD ASSESSMENT REQUEST — Region: {region}

Current Soil & Environmental Conditions:
{_format_conditions(conditions)}

Historical Performance Records (RAG Context — {len(similar_records)} closest matches):
{similar_records.to_string(index=False)}

Using the field conditions above and the historical crop performance data as empirical context,
provide a precision crop recommendation.

{OUTPUT_SCHEMA}
""".strip()


def get_recommendation(api_key: str, conditions: dict, df: pd.DataFrame,
                       region: str = "Unknown Region", n_similar: int = 5,
                       model: str = "gpt-4o-mini") -> tuple[dict, pd.DataFrame]:
    """
    Call OpenAI with RAG context and return (parsed_recommendation, similar_records).
    Raises json.JSONDecodeError or openai.APIError on failure.
    """
    similar = find_similar_records(df, conditions, n=n_similar)
    user_prompt = build_user_prompt(conditions, similar, region)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.15,
        max_tokens=1800,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    recommendation = json.loads(raw)
    return recommendation, similar
