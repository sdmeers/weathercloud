import sys
import importlib.metadata
import google.cloud.aiplatform as gap        # add this back

# ─── print Vertex-AI SDK version to Cloud Run logs ─────────────────────
print(
    "==== Vertex-AI SDK:",
    importlib.metadata.version("google-cloud-aiplatform"),
    file=sys.stderr,
    flush=True,
)

# ─── core imports ──────────────────────────────────────────────────────
import streamlit as st
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Tool

# Show which SDK version actually got installed
st.write("Vertex AI SDK version:", gap.__version__)   # <--- leave this for now

# 1) Initialise Vertex AI
vertexai.init(project="weathercloud-460719", location="europe-west2")

# 2) MCP tool
weather_tool = Tool.from_mcp_url(
    "https://weather-mcp-728445650450.europe-west2.run.app"
)

# 3) Gemma 7-B IT model with the tool attached
model = GenerativeModel("gemma-7b-it", tools=[weather_tool])

# ──────────────────────────────────  Streamlit chat loop  ──────────────────────────────────
st.title("Weather-Station Chatbot 🌤️")

if "hist" not in st.session_state:
    st.session_state.hist = []

msg = st.chat_input("Ask me about the weather…")
if msg:
    st.session_state.hist.append({"role": "user", "parts": [msg]})
    resp = model.generate_content(st.session_state.hist)
    st.session_state.hist.append({"role": "model", "parts": [resp.text]})

for m in st.session_state.hist:
    st.chat_message(m["role"]).markdown(m["parts"][0])