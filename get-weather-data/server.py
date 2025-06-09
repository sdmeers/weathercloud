"""
Enhanced MCP-compatible wrapper for get_weather_data() with aggregation support.
"""

import json, re
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.test import EnvironBuilder
from flask import Request as FlaskRequest
from main import get_weather_data

app = Flask(__name__)

ALL_FIELDS = [
    "timestamp_UTC", "temperature", "humidity", "pressure", "rain",
    "rain_rate", "luminance", "wind_speed", "wind_direction",
]

TIME_RANGE_PATTERN = r"^((latest|first|all|" \
                     r"today|yesterday|last24h|last7days|" \
                     r"week|month|year)|" \
                     r"day=\d{1,3}|week=\d{1,2}|month=\d{1,2}|year=\d{4})$"

AGGREGATION_OPERATIONS = ["raw", "max", "min", "mean", "sum", "count"]

SCHEMA = {
    "name": "weatherstation",
    "version": "2",
    "actions": {
        "queryWeather": {
            "description": "Return Firestore weather readings with optional aggregation.",
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
                    "operation": {
                        "type": "string",
                        "enum": AGGREGATION_OPERATIONS,
                        "default": "raw",
                        "description": "Aggregation operation: raw (no aggregation), max, min, mean, sum, count"
                    },
                },
                "required": ["range"],
                "additionalProperties": False,
            },
        }
    },
}

def _make_flask_request(body: dict) -> FlaskRequest:
    """Helper to reuse Cloud-Function code unchanged."""
    builder = EnvironBuilder(
        path="/", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body),
    )
    return FlaskRequest(builder.get_environ())

def _aggregate_data(data, operation, fields):
    """Apply aggregation operation to the data."""
    if not data or operation == "raw":
        return data
    
    try:
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(data)
        
        # Filter to requested fields (excluding timestamp for aggregations)
        numeric_fields = [f for f in fields if f != "timestamp_UTC" and f in df.columns]
        
        if not numeric_fields:
            return {"error": "No numeric fields available for aggregation"}
        
        # Apply aggregation
        if operation == "max":
            result = df[numeric_fields].max().round(1).to_dict()
        elif operation == "min":
            result = df[numeric_fields].min().round(1).to_dict()
        elif operation == "mean":
            result = df[numeric_fields].mean().round(1).to_dict()
        elif operation == "sum":
            result = df[numeric_fields].sum().round(1).to_dict()
        elif operation == "count":
            result = {"count": len(df)}
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        # Add metadata
        result["_metadata"] = {
            "operation": operation,
            "record_count": len(df),
            "time_range": f"{df['timestamp_UTC'].min()} to {df['timestamp_UTC'].max()}" if 'timestamp_UTC' in df.columns else "unknown"
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Aggregation failed: {str(e)}"}

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

    args = payload.get("arguments", {})
    operation = args.get("operation", "raw")
    fields = args.get("fields", ALL_FIELDS)

    # Get raw data from existing function
    fake_req = _make_flask_request(args)
    resp, status, _ = get_weather_data(fake_req)
    
    if status != 200:
        return resp, status
    
    # Parse the response if it's JSON
    try:
        if hasattr(resp, 'get_json'):
            data = resp.get_json()
        else:
            data = json.loads(resp) if isinstance(resp, str) else resp
    except:
        return jsonify({"error": "Failed to parse weather data"}), 500
    
    # Apply aggregation if requested
    if operation != "raw":
        data = _aggregate_data(data, operation, fields)
    
    return jsonify(data)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))