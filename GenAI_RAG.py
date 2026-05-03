def get_row_as_summary(df, row_index):
    """Maps specific column headers to a readable string for the prompt."""
    row = df.iloc[row_index]
    
    # Adjust the keys (e.g., 'N', 'ph') to match your exact CSV/Excel headers
    summary = (
        f"Nitrogen: {row['N']}, Phosphorus: {row['P']}, Potassium: {row['K']}, "
        f"Soil pH: {row['ph']}, Temperature: {row['temperature']}°C, "
        f"Humidity: {row['humidity']}%, Rainfall: {row['rainfall']}mm"
    )
    return summary, row.get('location', 'Unknown Region')

def get_row_as_summary(df, row_index):
    """Maps specific column headers to a readable string for the prompt."""
    row = df.iloc[row_index]
    
    # Adjust the keys (e.g., 'N', 'ph') to match your exact CSV/Excel headers
    summary = (
        f"Nitrogen: {row['N']}, Phosphorus: {row['P']}, Potassium: {row['K']}, "
        f"Soil pH: {row['ph']}, Temperature: {row['temperature']}°C, "
        f"Humidity: {row['humidity']}%, Rainfall: {row['rainfall']}mm"
    )
    return summary, row.get('location', 'Unknown Region')


# 1. Role description
ROLE_DESCRIPTION = """
You are a senior Agronomist and Soil Scientist. Your mission is to recommend the optimal crop for a specific plot of land based on soil composition (N, P, K, pH) and environmental data (Temperature, Humidity, Rainfall). 

Core rules:
- Base recommendations strictly on the provided agricultural data.
- Balance high yield potential with soil health and sustainability.
- Explain the "why" behind the choice using the specific data points provided.

Humanization rules:
- Sound like a professional advisor: authoritative yet practical.
- Use natural phrasing common in agriculture (e.g., "nutrient uptake," "water requirements").
- Avoid generic gardening advice; focus on professional-grade farming logic.
""".strip()

# 2. Output format instruction
OUTPUT_FORMAT_INSTRUCTION = """
Return ONLY a valid JSON object with exactly this structure. No preamble or markdown.

{
  "technical": "<scientific reasoning for the recommendation>",
  "practical": "<advice on planting and maintenance for the farmer>",
  "summary": "<1-sentence takeaway of the recommended crop>"
}

Hard constraints:
- Key names are fixed: "technical", "practical", "summary".
- Use the unit metrics provided in the data (e.g., Celsius, mm, pH levels).
""".strip()

# 3. Per-style instructions
TECHNICAL_STYLE_INSTRUCTION = "Focus on soil chemistry, nutrient balance, and climate tolerances. Use precise scientific terminology."
PRACTICAL_STYLE_INSTRUCTION = "Focus on the day-to-day farming operations, timing, and resource management needed for this specific crop."

# 4. Updated Backend Function
def get_crop_recommendation_system_prompt(data_summary: str) -> str:
    return f"""
{ROLE_DESCRIPTION}

Current Environmental and Soil Data:
{data_summary}

{OUTPUT_FORMAT_INSTRUCTION}

- Draft 1 (style: "technical"): {TECHNICAL_STYLE_INSTRUCTION}
- Draft 2 (style: "practical"): {PRACTICAL_STYLE_INSTRUCTION}
""".strip()

def get_crop_recommendation_user_prompt(location_context: str) -> str:
    return f"Provide a crop recommendation for the following region: '{location_context}'"

# Example of data imported from a previous step (like a CSV or DataFrame)
sample_data_summary = "Nitrogen: 90, Phosphorus: 42, Potassium: 43, pH: 6.5, Temp: 20.8°C, Humidity: 82%, Rainfall: 202mm"
sample_location = "Coastal Lowlands - Region A"

# Generate prompts
system_prompt = get_crop_recommendation_system_prompt(sample_data_summary)
user_prompt = get_crop_recommendation_user_prompt(sample_location)

# Execute via OpenAI
client = make_openai_client(api_key=OPENAI_API_KEY)

responses_result = client.responses.create(
    model="gpt-4o-mini",
    instructions=system_prompt,
    input=user_prompt,
)

print(responses_result.output_text)

