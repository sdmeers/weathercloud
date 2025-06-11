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
You are a weather-station analyst that provides precise, helpful weather information.

IMPORTANT: You must ALWAYS call the query_weather tool to get real-time data. Never guess or provide cached information.

When asked about weather data, call **query_weather** with:
- `range_param`: Time range for the data
- `fields`: Relevant weather fields (optional)  
- `operation`: Aggregation operation (optional, defaults to "raw")

**Legal ranges:**
• latest, today, yesterday, last24h, last7days
• day=N, week=N, month=N, year=YYYY
• ISO-8601 interval (e.g. 2025-01-01T00:00Z/2025-02-01T00:00Z)

**Available operations:**
• raw: Return all data points (default)
• max: Maximum values (for "highest", "warmest", "maximum")
• min: Minimum values (for "lowest", "coldest", "minimum") 
• mean: Average values (for "average", "typical")
• sum: Total values (for "total rainfall", "total")
• count: Number of records

**Field mapping:**
• Temperature questions → "temperature" field
• Pressure questions → "pressure" field  
• Humidity questions → "humidity" field
• Rain/rainfall total questions → "rain" field
• Rain rate questions → "rain_rate" field
• Light/brightness questions → "luminance" field
• Wind speed questions → "wind_speed" field
• Wind direction questions → "wind_direction" field

**Examples of operation selection:**
• "What was the maximum temperature yesterday?" → operation="max", fields=["temperature"]
• "How much rain fell in January?" → operation="sum", fields=["rain"]
• "What was the average humidity last week?" → operation="mean", fields=["humidity"]
• "What's the current rain rate?" → operation="raw", range_param="latest", fields=["rain_rate"]
• "What was the highest wind speed today?" → operation="max", fields=["wind_speed"]
• "How bright was it yesterday?" → operation="max", fields=["luminance"]

**Response format handling:**
- Raw data (operation="raw"): Returns {"data": [...], "_metadata": {...}}
- Aggregated data: Returns {field1: value1, field2: value2, "_metadata": {...}}
- Always check for and use the "_metadata.units" section for accurate unit information
- For raw data, iterate through the "data" array to extract readings
- For aggregated data, use the direct field values in the response

**Response format:**
1. Extract the key information from the returned data structure
2. Use the units provided in "_metadata.units" for accurate unit information
3. Round numbers to 1 decimal place when presenting to user
4. If no data available, apologize briefly and suggest checking the time range

**Important:**
- Always use the most appropriate operation for the user's question
- Never reveal raw JSON, tool calls, or technical details to the user
- Keep responses conversational and helpful
- The response always includes a _metadata section with units - use these for accurate unit information
- Handle both raw data format (with "data" array) and aggregated data format (direct values)
"""

# ── Weather MCP tool implementation ───────────────────────────────────
def query_weather(range_param: str = "latest", fields: list | None = None, operation: str = "raw") -> str:
    """Call the MCP server and return JSON-encoded text."""
    if fields is None:
        fields = ["timestamp_UTC", "temperature", "humidity", "pressure", "wind_speed"]

    payload = {
        "name": "queryWeather",
        "arguments": {
            "range": range_param, 
            "fields": fields,
            "operation": operation
        },
    }

    try:
        r = requests.post(
            "https://weather-mcp-728445650450.europe-west2.run.app",
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        response_data = r.json()
        
        # Debug logging (can be toggled)
        if st.session_state.get("debug_mode", False):
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
    description="Retrieve weather-station data by time range with optional aggregation.",
    parameters={
        "type": "object",
        "properties": {
            "range_param": {
                "type": "string",
                "description": "Time range: latest, today, yesterday, last24h, last7days, "
                               "day=N, week=N, month=N, year=YYYY, or ISO-8601 interval",
                "default": "latest",
            },
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Weather fields: temperature, humidity, pressure, rain, rain_rate, luminance, wind_speed, wind_direction",
                "default": ["timestamp_UTC", "temperature", "humidity", "pressure", "wind_speed"],
            },
            "operation": {
                "type": "string",
                "enum": ["raw", "max", "min", "mean", "sum", "count"],
                "description": "Aggregation operation: raw (no aggregation), max, min, mean, sum, count",
                "default": "raw",
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

# Add debug toggle
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

with st.sidebar:
    st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

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
        
        if st.session_state.debug_mode:
            st.write("**DEBUG: First response received**")
        
        # Check if response has function calls or text
        if first.candidates and first.candidates[0].content.parts:
            parts = first.candidates[0].content.parts
            text_parts = [p.text for p in parts if hasattr(p, 'text') and p.text]
            function_parts = [p.function_call for p in parts if hasattr(p, 'function_call') and p.function_call]
            
            if st.session_state.debug_mode:
                st.write(f"**DEBUG: Found {len(text_parts)} text parts and {len(function_parts)} function calls**")
                if text_parts:
                    st.write(f"**DEBUG: Text parts:** {text_parts}")
        
    except Exception as exc:
        err = f"Error generating response: {exc}"
        st.chat_message("assistant").markdown(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
        st.stop()

    # Extract any tool calls
    calls = [p.function_call for p in first.candidates[0].content.parts
             if hasattr(p, 'function_call') and p.function_call]
    
    if st.session_state.debug_mode:
        st.write(f"**DEBUG: Found {len(calls)} tool calls**")

    if calls:
        # fulfil call(s) then second pass
        if st.session_state.debug_mode:
            st.write("**DEBUG: Tool calls detected:**")
            st.json([{"name": fc.name, "args": dict(fc.args)} for fc in calls])
        
        # Add the model's response with function calls to conversation
        convo.append(Content(role="model", parts=first.candidates[0].content.parts))
        
        # Process each function call
        function_responses = []
        for fc in calls:
            if fc.name == "query_weather":
                rng = fc.args.get("range_param", "latest")
                flds_raw = fc.args.get("fields", ["timestamp_UTC", "temperature", "humidity", "pressure", "wind_speed"])
                op = fc.args.get("operation", "raw")
                
                # Convert RepeatedComposite to regular Python list
                if hasattr(flds_raw, '__iter__') and not isinstance(flds_raw, str):
                    flds = list(flds_raw)
                else:
                    flds = flds_raw if isinstance(flds_raw, list) else [flds_raw]
                
                if st.session_state.debug_mode:
                    st.write(f"**DEBUG: Calling weather API with range='{rng}', fields={flds}, operation='{op}'**")
                
                data_json = query_weather(rng, flds, op)
                
                # Add function response to conversation
                function_responses.append(
                    Part.from_function_response(
                        name="query_weather",
                        response={"content": data_json},
                    )
                )
        
        # Add all function responses to conversation
        if function_responses:
            convo.append(Content(role="function", parts=function_responses))
        
        # Second pass with function results
        try:
            second = model.generate_content(convo)
            assistant_reply = second.text if second.text else "I couldn't process the weather data properly."
            
            if st.session_state.debug_mode:
                st.write("**DEBUG: Second response generated successfully**")
                
        except Exception as exc:
            assistant_reply = f"Error processing weather data: {exc}"
            if st.session_state.debug_mode:
                st.write(f"**DEBUG: Error in second pass: {exc}")
    else:
        # No function calls, extract text from parts
        if first.candidates and first.candidates[0].content.parts:
            text_parts = [p.text for p in first.candidates[0].content.parts if hasattr(p, 'text') and p.text]
            assistant_reply = ' '.join(text_parts) if text_parts else "I need to check the weather data for you. Let me try again."
        else:
            assistant_reply = "I couldn't generate a proper response. Please try asking about the weather again."

    # show & cache reply
    st.chat_message("assistant").markdown(assistant_reply)
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )