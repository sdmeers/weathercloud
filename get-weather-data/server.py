"""
Minimal MCP-compatible wrapper for get_weather_data().
No external SDK required.
"""

import json, re
from flask import Flask, request, jsonify
from werkzeug.test import EnvironBuilder
from flask import Request as FlaskRequest
from main import get_weather_data

app = Flask(__name__)

# ─── schema used for GET / ────────────────────────────────────────────
ALL_FIELDS = [
    "timestamp_UTC", "temperature", "humidity", "pressure", "rain",
    "rain_rate", "luminance", "wind_speed", "wind_direction",
]

TIME_RANGE_PATTERN = r"^((latest|first|all|" \
                     r"today|yesterday|last24h|last7days|" \
                     r"week|month|year)|" \
                     r"day=\d{1,3}|week=\d{1,2}|month=\d{1,2}|year=\d{4})$"

SCHEMA = {
    "name": "weatherstation",
    "version": "1",
    "actions": {
        "queryWeather": {
            "description": "Return Firestore weather readings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "range":  {"type": "string", "pattern": TIME_RANGE_PATTERN},
                    "start":  {"type": "string"},
                    "end":    {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string", "enum": ALL_FIELDS},
                    },
                },
                "required": ["range"],
                "additionalProperties": False,
            },
        }
    },
}

# ─── helper to reuse Cloud-Function code unchanged ────────────────────
def _make_flask_request(body: dict) -> FlaskRequest:
    builder = EnvironBuilder(
        path="/", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body),
    )
    return FlaskRequest(builder.get_environ())

# ─── routes ───────────────────────────────────────────────────────────
@app.get("/")
def describe():
    """Return tool metadata (MCP self-description)."""
    return jsonify(SCHEMA)

@app.post("/")
def call_tool():
    """Accept OpenAI-style tool call and proxy to get_weather_data()."""
    payload = request.get_json(silent=True) or {}
    if payload.get("name") != "queryWeather":
        return jsonify({"error": "Unknown or missing tool name"}), 400

    fake_req = _make_flask_request(payload.get("arguments", {}))
    resp, status, _ = get_weather_data(fake_req)
    return resp, status

# ─── entry-point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Cloud Run sets PORT=8080
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
