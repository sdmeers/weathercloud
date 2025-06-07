# app.py – Streamlit × Vertex AI weather chatbot
# ----------------------------------------------
# pip install streamlit google-cloud-aiplatform vertexai requests

import json
import requests
import streamlit as st
import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
    Tool,
    FunctionDeclaration,
    Part,
    Content,
)

# ── Vertex AI init ────────────────────────────────────────────────────
vertexai.init(project="weathercloud-460719", location="us-central1")
MODEL_ID = "gemini-2.0-flash-lite-001"            # universally available
# ---------------------------------------------------------------------

# ── System prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a weather-station analyst.

When you need data, call **query_weather** with
`range_param` and optional `fields`. Legal ranges:

• latest · today · yesterday · last24h · last7days
• day=N, week=N, month=N, year=YYYY
• ISO-8601 interval (e.g. 2025-01-01T00:00Z/2025-02-01T00:00Z)

After the tool returns JSON:
1. Pick columns that match the user request
2. Do max / min / mean / sum
3. Round to 1 decimal place
4. Answer in one short sentence + unit (°C, mm, hPa, m s⁻¹)
5. If no data, apologise briefly

Never reveal code, tool calls or raw JSON.
"""

# ── Weather MCP tool implementation ───────────────────────────────────
def query_weather(range_param: str = "latest", fields: list | None = None) -> str:
    """Call the MCP server and return JSON-encoded text."""
    if fields is None:
        fields = ["timestamp_UTC", "temperature", "humidity",
                  "pressure", "wind_speed"]

    payload = {
        "name": "queryWeather",
        "arguments": {"range": range_param, "fields": fields},
    }

    try:
        r = requests.post(
            "https://weather-mcp-728445650450.europe-west2.run.app",
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        response_data = r.json()
        
        # Debug logging
        st.write("**DEBUG: Weather API Response:**")
        st.json(response_data)
        
        return json.dumps(response_data, indent=2)
    except Exception as exc:
        error_msg = f"Weather API Error: {str(exc)}"
        st.error(error_msg)
        return json.dumps({"error": str(exc)})

# Vertex-AI function declaration
weather_function = FunctionDeclaration(
    name="query_weather",
    description="Retrieve weather-station data by time range.",
    parameters={
        "type": "object",
        "properties": {
            "range_param": {
                "type": "string",
                "description": "latest, today, yesterday, last24h, last7days, "
                               "day=N, week=N, month=N, year=YYYY, or ISO-8601 interval",
                "default": "latest",
            },
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Weather fields to return.",
                "default": ["timestamp_UTC", "temperature", "humidity",
                            "pressure", "wind_speed"],
            },
        },
        "required": ["range_param"],
    },
)

weather_tool = Tool(function_declarations=[weather_function])

# Load Gemini with the tool attached and system instruction
model = GenerativeModel(
    MODEL_ID, 
    tools=[weather_tool], 
    system_instruction=SYSTEM_PROMPT
)

# ── Streamlit UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="Weather-Station Chatbot", page_icon="🌤️")
st.title("Weather-Station Chatbot 🌤️")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat loop ─────────────────────────────────────────────────────────
user_msg = st.chat_input("Ask me about the weather…")
if user_msg:
    # show user bubble & store
    st.chat_message("user").markdown(user_msg)
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # build conversation: history only (no system role)
    convo: list[Content] = []
    for m in st.session_state.messages:
        role = "user" if m["role"] == "user" else "model"
        convo.append(Content(role=role, parts=[Part.from_text(m["content"])]))

    # First pass
    try:
        first = model.generate_content(convo)
    except Exception as exc:
        err = f"Error: {exc}"
        st.chat_message("assistant").markdown(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
        st.stop()

    # Extract any tool calls
    calls = [p.function_call for p in first.candidates[0].content.parts
             if p.function_call]

    if calls:
        # fulfil call(s) then second pass
        st.write("**DEBUG: Tool calls detected:**")
        st.json([{"name": fc.name, "args": dict(fc.args)} for fc in calls])
        
        for fc in calls:
            if fc.name == "query_weather":
                rng  = fc.args.get("range_param", "latest")
                flds = fc.args.get("fields", ["timestamp_UTC", "temperature",
                                              "humidity", "pressure", "wind_speed"])
                
                st.write(f"**DEBUG: Calling weather API with range='{rng}', fields={flds}**")
                data_json = query_weather(rng, flds)

                convo.extend([
                    Content(role="model", parts=[Part.from_function_call(fc)]),
                    Content(role="function", parts=[
                        Part.from_function_response(
                            name="query_weather",
                            response={"content": data_json},
                        )
                    ]),
                ])

        try:
            second = model.generate_content(convo)
            assistant_reply = second.text
        except Exception as exc:
            assistant_reply = f"Error: {exc}"
    else:
        assistant_reply = first.text

    # show & cache reply
    st.chat_message("assistant").markdown(assistant_reply)
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )