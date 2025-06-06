"""
weather-mcp / server.py
DIY MCP server that wraps the get_weather_data Cloud Function.
"""

import json
import re
from flask import Request
from mcp.server import MCPServer, Action, Field
import functions_framework           # fabricate Request objects
from main import get_weather_data     # your Cloud Function

# ──────────────────────────────────────────────────────────────────────────────
# 1. Parameter schema
# ──────────────────────────────────────────────────────────────────────────────
ALL_POSSIBLE_FIELDS = [
    "timestamp_UTC", "temperature", "humidity", "pressure", "rain",
    "rain_rate", "luminance", "wind_speed", "wind_direction",
]

TIME_RANGE_PATTERN = re.compile(
    r"""^(
          (latest|first|all|
           today|yesterday|
           last24h|last7days|
           week|month|year)            # plain keywords
          |
          day=\d{1,3}                 # day=<1-366>
          |week=\d{1,2}               # week=<1-53>
          |month=\d{1,2}              # month=<1-12>
          |year=\d{4}                 # year=<YYYY>
        )$""",
    re.VERBOSE,
).pattern

PARAMETERS = {
    "range":  Field(
        str,
        pattern=TIME_RANGE_PATTERN,
        description=(            "Time window keyword (e.g. \"latest\", \"week\", \"week=23\", "
            "\"year=2024\")."
        ),
    ),
    "start":  Field(str, required=False, description="UTC ISO start timestamp"),
    "end":    Field(str, required=False, description="UTC ISO end timestamp"),
    "fields": Field(
        list[str],
        required=False,
        items=Field(str, enum=ALL_POSSIBLE_FIELDS),
        description="Optional list of field names to include.",
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. MCP Action
# ──────────────────────────────────────────────────────────────────────────────
class QueryWeather(Action):
    name = "queryWeather"
    description = "Return weather-station readings from Firestore."
    parameters = PARAMETERS

    def run(self, **kwargs):
        """Forward the call to get_weather_data() unchanged."""
        fake_request = functions_framework.Request(
            path="/",
            headers={"content-type": "application/json"},
            body=json.dumps(kwargs).encode(),
        )
        response, status, _ = get_weather_data(fake_request)
        if status != 200:
            # Bubble up Cloud Function errors as MCP errors
            raise RuntimeError(response.get_json().get("error"))
        return response.get_json()    # must be JSON-serialisable


# ──────────────────────────────────────────────────────────────────────────────
# 3. Server bootstrap
# ──────────────────────────────────────────────────────────────────────────────
server = MCPServer(name="weatherstation", actions=[QueryWeather()])

if __name__ == "__main__":
    # Expose HTTP on port 8080 (Cloud Run default), plus stdio/SSE.
    server.serve_http(host="0.0.0.0", port=8080)