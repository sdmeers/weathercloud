"""
Enhanced MCP-compatible wrapper for get_weather_data() with aggregation support.
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

def _convert_units(value, field):
    """Convert values to more readable units."""
    if field == "rain_rate":
        # Convert mm/s to mm/hr
        return value * 3600
    elif field == "wind_speed":
        # Convert m/s to mph
        return value * 2.23694
    else:
        return value

def _aggregate_data(data, operation, fields):
    """Apply aggregation operation to the data using pure Python."""
    if not data or operation == "raw":
        # Apply unit conversions to raw data
        if isinstance(data, list):
            converted_data = []
            for record in data:
                converted_record = record.copy()
                for field in ["rain_rate", "wind_speed"]:
                    if field in converted_record and converted_record[field] is not None:
                        try:
                            converted_record[field] = round(_convert_units(float(converted_record[field]), field), 1)
                        except (ValueError, TypeError):
                            pass
                converted_data.append(converted_record)
            return converted_data
        return data
    
    try:
        # Ensure data is a list of dictionaries
        if not isinstance(data, list) or not data:
            return {"error": "No data available for aggregation"}
        
        # Filter to requested numeric fields (excluding timestamp)
        numeric_fields = [f for f in fields if f != "timestamp_UTC"]
        
        if not numeric_fields:
            return {"error": "No numeric fields available for aggregation"}
        
        # Extract values for each field with unit conversion
        field_values = {}
        for field in numeric_fields:
            values = []
            for record in data:
                if field in record and record[field] is not None:
                    try:
                        raw_value = float(record[field])
                        converted_value = _convert_units(raw_value, field)
                        values.append(converted_value)
                    except (ValueError, TypeError):
                        continue
            if values:
                field_values[field] = values
        
        if not field_values:
            return {"error": "No valid numeric data found"}
        
        # Apply aggregation
        result = {}
        if operation == "max":
            for field, values in field_values.items():
                result[field] = round(max(values), 1)
        elif operation == "min":
            for field, values in field_values.items():
                result[field] = round(min(values), 1)
        elif operation == "mean":
            for field, values in field_values.items():
                result[field] = round(sum(values) / len(values), 1)
        elif operation == "sum":
            for field, values in field_values.items():
                result[field] = round(sum(values), 1)
        elif operation == "count":
            result = {"count": len(data)}
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        # Add metadata with unit information
        timestamps = [record.get('timestamp_UTC') for record in data if record.get('timestamp_UTC')]
        result["_metadata"] = {
            "operation": operation,
            "record_count": len(data),
            "time_range": f"{min(timestamps)} to {max(timestamps)}" if timestamps else "unknown",
            "units": {
                "temperature": "°C",
                "humidity": "%",
                "pressure": "hPa",
                "rain": "mm",
                "rain_rate": "mm/hr",
                "wind_speed": "mph",
                "luminance": "lux"
            }
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