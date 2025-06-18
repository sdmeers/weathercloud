# app.py – Streamlit × Vertex AI weather chatbot
# ----------------------------------------------
# pip install streamlit google-cloud-aiplatform vertexai requests

import json
import requests
from datetime import datetime
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
MODEL_ID = "gemini-2.0-flash-lite-001"

# ── System prompt ─────────────────────────────────────────────────────
def get_system_prompt():
    """Generate system prompt with current date context."""
    now = datetime.now()
    current_date = now.strftime("%A, %B %d, %Y")
    current_month = now.month
    current_year = now.year
    current_month_name = now.strftime("%B")
    
    return f"""
    You are a weather-station analyst that provides precise, helpful weather information.

    **CURRENT DATE CONTEXT:**
    Today is {current_date} (Month: {current_month}, Year: {current_year}).
    Use this information to interpret all relative time references accurately.

    IMPORTANT: You must ALWAYS call the query_weather tool to get real-time data. Never guess or provide cached information.

    When asked about weather data, call **query_weather** with:
    - `range_param`: Time range for the data
    - `fields`: Relevant weather fields (optional)  
    - `operation`: Aggregation operation (optional, defaults to "raw")

    **Legal ranges:**
    • latest, today, yesterday, last24h, last7days
    • day=N, week=N, month=N (current year), year=YYYY
    • ISO-8601 interval (e.g. 2025-01-01T00:00Z/2025-02-01T00:00Z)

    **Relative time interpretation (calculate based on current date above):**
    • "this month" → month={current_month} (current month)
    • "last month" → month={current_month-1 if current_month > 1 else 12} (previous month, handle year rollover)
    • "next month" → month={current_month+1 if current_month < 12 else 1} (next month, handle year rollover)
    • "this year" → year={current_year}
    • "last year" → year={current_year-1}
    • "next year" → year={current_year+1}
    • **IMPORTANT: When only a month name is given (e.g., "June", "January"), ALWAYS assume current year ({current_year})**
    • When month/year not specified, assume current year ({current_year})
    • For cross-year month references (like "last month" in January or "next month" in December), 
    use ISO-8601 intervals to specify the correct year

    **For specific time period combinations:**
    • **GENERAL RULE: For current or future time periods, use simple format (year=YYYY, month=N, week=N, day=N)**
    • **Use ISO-8601 intervals ONLY for completed past periods**

    **Examples by time period:**
    • **Years:**
    - "2025" (current year) → use "year=2025" (not ISO interval)
    - "2026" (future year) → use "year=2026" (not ISO interval)
    - "2024" (completed past year) → use "2024-01-01T00:00Z/2025-01-01T00:00Z" OR "year=2024" (both work)

    • **Months:**
    - "June 2025" (current month) → use "month=6" (not ISO interval)
    - "July 2025" (future month) → use "month=7" (not ISO interval)  
    - "May 2025" (completed past month) → use "2025-05-01T00:00Z/2025-06-01T00:00Z" OR "month=5" (both work)

    • **General principle:** The API handles boundaries intelligently with simple formats (year=N, month=N), 
    so prefer these over ISO intervals unless you need precise historical date ranges

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

    **For questions asking "when" or "which day" with superlatives:**
    When users ask questions like:
    • "When was the wettest day this month?"
    • "Which day had the highest temperature?"
    • "What date was the windiest day?"

    Use operation="raw" instead of aggregation operations, then analyze the raw data to find:
    1. The maximum/minimum value
    2. The corresponding timestamp for that value

    **Examples of "when/which day" queries:**
    • "When was the wettest day this month?" → operation="raw", fields=["timestamp_UTC", "rain"], range_param="month=6"
    • "Which day had the highest temperature this year?" → operation="raw", fields=["timestamp_UTC", "temperature"], range_param="year=2025"  
    • "What date was the windiest day last week?" → operation="raw", fields=["timestamp_UTC", "wind_speed"], range_param="last7days"

    **Response format for date-specific queries:**
    1. Analyze the raw data to find the maximum/minimum value
    2. Identify the timestamp when this occurred
    3. Present both the value AND the date/time to the user
    4. Format the date in a user-friendly way (e.g., "June 15th" or "Tuesday, June 15th")

    **Examples of operation selection (based on current date context):**
    • "What was the maximum temperature yesterday?" → operation="max", fields=["temperature"], range_param="yesterday"
    • "How much rain fell in January?" → operation="sum", fields=["rain"], range_param="month=1" (assumes current year {current_year})
    • "What was the maximum temperature in June?" → operation="max", fields=["temperature"], range_param="month=6" (assumes current year {current_year})
    • "What was the maximum temperature in June 2025?" → operation="max", fields=["temperature"], range_param="month=6" (current month, use month=6)
    • "What was the maximum temperature in May 2025?" → operation="max", fields=["temperature"], range_param="month=5" (past month, month=5 works fine)
    • "What was the maximum temperature in July 2025?" → operation="max", fields=["temperature"], range_param="month=7" (future month, use month=7)
    • "What was the total rainfall in 2025?" → operation="sum", fields=["rain"], range_param="year=2025" (current year, use year=2025)
    • "What was the total rainfall in 2026?" → operation="sum", fields=["rain"], range_param="year=2026" (future year, use year=2026)
    • "What was the total rainfall in 2024?" → operation="sum", fields=["rain"], range_param="year=2024" (past year, year=2024 works fine)
    • "What was the maximum temperature this month?" → operation="max", fields=["temperature"], range_param="month={current_month}" ({current_month_name} {current_year})
    • "What was the maximum temperature last month?" → operation="max", fields=["temperature"], range_param="month={current_month-1 if current_month > 1 else 12}" (handle year rollover if needed)
    • "What was the maximum temperature next month?" → operation="max", fields=["temperature"], range_param="month={current_month+1 if current_month < 12 else 1}" (handle year rollover if needed)
    • "What was the average humidity last week?" → operation="mean", fields=["humidity"], range_param="last7days"
    • "What's the current rain rate?" → operation="raw", range_param="latest", fields=["rain_rate"]
    • "What was the highest wind speed today?" → operation="max", fields=["wind_speed"], range_param="today"
    • "Total rainfall this year?" → operation="sum", fields=["rain"], range_param="year={current_year}"
    • "Total rainfall last year?" → operation="sum", fields=["rain"], range_param="year={current_year-1}"

    **Response format handling:**
    - Raw data (operation="raw"): Returns {{"data": [...], "_metadata": {{...}}}}
    - Aggregated data: Returns {{field1: value1, field2: value2, "_metadata": {{...}}}}
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
    - Pay special attention to year rollovers when dealing with "last month" in January or "next month" in December
    """

SYSTEM_PROMPT = get_system_prompt()

# ── Weather MCP tool implementation ───────────────────────────────────
def query_weather(range_param: str = "latest", fields: list | None = None, operation: str = "raw") -> str:
    """Call the MCP server and return JSON-encoded text."""
    # Ensure fields is always a standard Python list for JSON serialization
    if fields is None:
        fields_for_payload = ["timestamp_UTC", "temperature", "humidity", "pressure", "wind_speed"]
    elif not isinstance(fields, list):
        # Handle cases where fields might be a RepeatedComposite or other non-list type
        fields_for_payload = list(fields)
    else:
        fields_for_payload = fields

    payload = {
        "name": "queryWeather",
        "arguments": {
            "range": range_param,  
            "fields": fields_for_payload,
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

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    @import url('https://fonts.googleapis.com/css?family=Raleway');

    /* Hide default Streamlit UI elements */
    header[data-testid="stHeader"],
    .stDeployButton,
    div[data-testid="stToolbar"],
    .viewerBadge_link__1S137,
    .stStatus,
    div[data-testid="stStatusWidget"],
    .stApp > footer,
    .stSidebar,
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Top nav bar - FORCE BLACK BACKGROUND */
    .w3-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 50px;
        background-color: black !important;
        color: white !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 20px;
        z-index: 9999;
        font-size: 16px;
        font-family: "Raleway", sans-serif;
    }

    .w3-bar-item {
        display: flex;
        align-items: center;
        gap: 8px;
        color: white !important;
    }

    .w3-bar-item a {
        color: white !important;
        text-decoration: none;
    }

    .w3-bar-item a:hover {
        color: #ccc !important;
    }

    .w3-bar-item i {
        color: white !important;
    }

    /* FORCE WHITE BACKGROUND EVERYWHERE EXCEPT NAV */
    html, body, .stApp, .main, div, section, 
    div[data-testid="stAppViewContainer"],
    div[data-testid="stMain"],
    div[data-testid="stBottom"],
    div[data-testid="stChatFloatingInputContainer"] {
        background-color: white !important;
        color: black !important;
        font-family: "Raleway", sans-serif !important;
    }

    .block-container {
        background-color: white !important;
        color: black !important;
        padding-top: 70px !important;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Title styling */
    .stApp h1 {
        color: black !important;
        font-family: "Raleway", sans-serif !important;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background-color: white !important;
        color: black !important;
        border: none !important;
        margin-bottom: 1rem;
    }

    div[data-testid="stChatMessage"] * {
        background-color: transparent !important;
        color: black !important;
    }

    /* SIMPLE CHAT INPUT STYLING */
    div[data-testid="stChatInput"] {
        background-color: #f8f9fa !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 25px !important;
        max-width: 600px !important;
        margin: 0 auto !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
    }

    div[data-testid="stChatInput"] {
        background-color: #f8f9fa !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 25px !important;
        max-width: 600px !important;
        margin: 0 auto !important;
        padding: 3px !important;                  /* Small padding instead of 0 */
        display: flex !important;
        align-items: center !important;
        overflow: hidden !important;              /* Ensures no overflow */
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #6c757d !important;
    }
            
    div[data-testid="stChatInput"] textarea {
    color: #212529 !important;
        caret-color: #212529 !important;          /* Ensures cursor is visible too */
    }

    div[data-testid="stChatInput"] textarea:focus {
        color: #212529 !important;
    }

    /* FIXED SEND BUTTON SIZING */
    button[data-testid="stChatInputSubmitButton"] {
        background-color: #007bff !important;
        border: none !important;
        border-radius: 50% !important;
        width: 30px !important;                   /* Smaller button */
        height: 30px !important;                  /* Smaller button */
        margin: 5px 8px !important;               /* Reduce margins */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
    }

    button[data-testid="stChatInputSubmitButton"] svg {
        color: white !important;
        width: 16px !important;
        height: 16px !important;
    }

    /* Ensure all text is visible */
    .stMarkdown, .stText, p, div, span {
        color: black !important;
    }
</style>

<div class="w3-bar w3-top w3-black w3-large">
    <span class="w3-bar-item w3-left"><i class="fa-solid fa-magnifying-glass-chart"></i> <a href="https://weather-dashboard-728445650450.europe-west2.run.app/" target="_self">Weather Summary</a></span>
    <span class="w3-bar-item w3-right"><i class="fa-solid fa-database"></i> <a href="https://display-weather-data-728445650450.europe-west2.run.app/" target="_self">Weather data</a></span>
</div>
""", unsafe_allow_html=True)

st.title("Weather-Station Chatbot 🌤️")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ── Chat loop ─────────────────────────────────────────────────────────
user_msg = st.chat_input("Ask me about the weather…")
if user_msg:
    # Show user message and store
    st.chat_message("user").markdown(user_msg)
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # Trim conversation history to prevent context buildup
    MAX_CONVERSATION_TURNS = 4
    if len(st.session_state.messages) > MAX_CONVERSATION_TURNS * 2:
        # Start fresh - keep only the current user message
        current_user_msg = st.session_state.messages[-1]
        st.session_state.messages = [current_user_msg]
        convo: list[Content] = [Content(role="user", parts=[Part.from_text(user_msg)])]
    else:
        # Build conversation normally
        convo: list[Content] = []
        for m in st.session_state.messages:
            role = "user" if m["role"] == "user" else "model"
            convo.append(Content(role=role, parts=[Part.from_text(m["content"])]))

    # First pass
    try:
        first = model.generate_content(convo)
    except Exception as exc:
        err = f"Error generating response: {exc}"
        st.chat_message("assistant").markdown(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
        st.stop()

    # Extract any tool calls
    calls = [p.function_call for p in first.candidates[0].content.parts
            if hasattr(p, 'function_call') and p.function_call]

    if calls:
        # Add the model's response with function calls to conversation
        convo.append(Content(role="model", parts=first.candidates[0].content.parts))
        
        # Process each function call
        function_responses = []
        for fc in calls:
            if fc.name == "query_weather":
                rng = fc.args.get("range_param", "latest")
                op = fc.args.get("operation", "raw")
                
                # Retrieve fields from function call arguments
                flds = fc.args.get("fields")
                
                # Ensure 'fields' is a proper Python list for JSON serialization
                if flds is None:
                    flds_for_api = ["timestamp_UTC", "temperature", "humidity", "pressure", "wind_speed"]
                elif hasattr(flds, '__iter__') and not isinstance(flds, (str, list)):
                    # Handle RepeatedComposite and similar iterable non-list types
                    flds_for_api = list(flds)
                elif isinstance(flds, str):
                    # Handle string representation of a list
                    try:
                        flds_for_api = json.loads(flds.replace("'", "\""))
                    except json.JSONDecodeError:
                        flds_for_api = [flds]
                else:
                    flds_for_api = flds
                
                data_json = query_weather(rng, flds_for_api, op)
                
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
        except Exception as exc:
            assistant_reply = f"Error processing weather data: {exc}"
    else:
        # No function calls, extract text from parts
        if first.candidates and first.candidates[0].content.parts:
            text_parts = [p.text for p in first.candidates[0].content.parts if hasattr(p, 'text') and p.text]
            assistant_reply = ' '.join(text_parts) if text_parts else "I need to check the weather data for you. Let me try again."
        else:
            assistant_reply = "I couldn't generate a proper response. Please try asking about the weather again."

    # Show and store reply
    st.chat_message("assistant").markdown(assistant_reply)
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )   