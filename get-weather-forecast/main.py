""" main.py — Cloud Function Help / Manual Page
===========================================

NAME
    get_weather_forecast – Retrieve structured weather forecast data from Firestore.

SYNOPSIS
    Deployed as a Google Cloud Function or Cloud Run service that accepts
    **HTTPS POST** requests with a JSON body.  
    CORS pre-flight (**OPTIONS**) is supported.

    ENDPOINT
        https://<REGION>-<PROJECT>.cloudfunctions.net/get_weather_forecast

    HEADERS
        Content-Type:  application/json
        (No auth required by default – control with IAM / an API Gateway.)

JSON REQUEST SHAPES
    1. Time range by keyword
        {
          "range": "<keyword>",          # see RANGE KEYWORDS below
          "fields": ["temperature", ...] # optional – choose columns
        }

    2. Explicit UTC window
        {
          "start": "YYYY-MM-DDTHH:MM:SSZ",
          "end":   "YYYY-MM-DDTHH:MM:SSZ",
          "fields": ["temperature", ...] # optional
        }

RANGE KEYWORDS
    latest          Return the single most-recent document
    first           Return the very first document on record
    all             Return *all* data in the collection
    today           Local day in Europe/London
    yesterday       Previous local day
    last24h         Rolling 24-hour window (now minus 24 h)
    last7days       Rolling 7-day window
    week            ISO week containing “now”
    month           Calendar month containing “now”
    year            Calendar year containing “now”
    day=<1-366>     Ordinal day of current year
    week=<1-53>     ISO week number of current year
    month=<1-12>    Calendar month number of current year
    year=<YYYY>     An absolute year (±100 y from current)

FIELD SELECTION
    • If *fields* is omitted, **all** columns listed in `ALL_POSSIBLE_FIELDS`
      are returned.  
    • `time` is always included even if you omit it.

    Valid field names (case-sensitive):
        time, temperature, humidity, pressure, rain_total,
        prob_of_rain, wind_speed

EXAMPLE REQUESTS
    # Curl – latest forecast, all fields
    curl -X POST \
         -H "Content-Type: application/json" \
         -d '{"range":"latest"}' \
         https://…/get_weather_forecast

    # Curl – yesterday’s humidity and temperature only
    curl -X POST \
         -H "Content-Type: application/json" \
         -d '{"range":"yesterday", "fields":["temperature","humidity"]}' \
         https://…/get_weather_forecast

RESPONSE
    Success (HTTP 200) – JSON array of documents, ascending by timestamp
        [
          {
            "time": "2025-06-05T14:20:33.123456+00:00",
            "temperature": 22.8,
            "humidity": 54.7,
            …
          },
          …
        ]

ERRORS
    400 Bad Request   • Missing/invalid JSON  
                      • Unknown range keyword or field name  
                      • Malformed start/end timestamps
    500 Internal      • Firestore unavailable or unexpected exception

PERFORMANCE NOTES
    • Queries are bounded by the supplied time window and projected
      field list; narrower requests return faster.  
    • `latest` and `first` execute with a Firestore *limit 1* optimized query.  
    • Log timings are emitted at INFO level for observability.

ENVIRONMENT
    • Firestore database-ID expected: “weatherdata”.  
    • Service operates in Europe/London local time for relative ranges.
    • Dependencies: Flask, google-cloud-firestore, python-dateutil.

COPYRIGHT
    © 2025 Steven Meers. MIT License.
"""

import json
import logging
from datetime import datetime, timedelta, date
from dateutil import tz
from flask import Request, jsonify
from google.cloud import firestore
from google.cloud.exceptions import GoogleCloudError
from typing import Optional, Tuple, Dict, Any
import calendar
import inspect
import time

# --- Configuration ---
FIRESTORE_DATABASE_ID = "weatherdata"
try:
    db = firestore.Client(database=FIRESTORE_DATABASE_ID)
except Exception as e:
    logging.critical(f"Failed to initialize Firestore client with database '{FIRESTORE_DATABASE_ID}': {e}")
    db = None

LONDON_TZ = tz.gettz('Europe/London')
if not LONDON_TZ:
    logging.critical("Failed to get 'Europe/London' timezone. Please ensure dateutil is properly configured.")
    LONDON_TZ = tz.tzutc()

# Define ALL possible fields in your Firestore documents.
# This list is used as the default if no specific fields are requested.
ALL_POSSIBLE_FIELDS = [
    'time', 'temperature', 'humidity', 'pressure', 'rain_total',
    'prob_of_rain', 'wind_speed'
]


# --- Helper Function: get_time_range ---
def get_time_range(arg: str, current_local_time: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
    start_utc_dt = None
    end_utc_dt = None

    if current_local_time.tzinfo is None:
        current_local_time = current_local_time.replace(tzinfo=LONDON_TZ)
    current_utc_time = current_local_time.astimezone(tz.tzutc())
    
    if arg == "latest" or arg == "first" or arg == "all":
        return None, None
    elif arg == "today":
        start_of_today_local = current_local_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_today_local = start_of_today_local + timedelta(days=1) - timedelta(microseconds=1)
        start_utc_dt = start_of_today_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_today_local.astimezone(tz.tzutc())
    elif arg == "last24h":
        start_utc_dt = current_utc_time - timedelta(hours=24)
        end_utc_dt = current_utc_time
    elif arg == "yesterday":
        yesterday_local = current_local_time - timedelta(days=1)
        start_of_yesterday_local = yesterday_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_yesterday_local = start_of_yesterday_local + timedelta(days=1) - timedelta(microseconds=1)
        start_utc_dt = start_of_yesterday_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_yesterday_local.astimezone(tz.tzutc())
    elif arg == "last7days":
        start_utc_dt = current_utc_time - timedelta(days=7)
        end_utc_dt = current_utc_time
    elif arg == "week":
        start_of_current_week_local = current_local_time - timedelta(days=current_local_time.isoweekday() - 1)
        start_of_current_week_local = start_of_current_week_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_current_week_local = start_of_current_week_local + timedelta(days=7) - timedelta(microseconds=1)
        start_utc_dt = start_of_current_week_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_current_week_local.astimezone(tz.tzutc())
    elif arg == "month":
        start_of_month_local = current_local_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        num_days_in_month = calendar.monthrange(current_local_time.year, current_local_time.month)[1]
        end_of_month_local = current_local_time.replace(day=num_days_in_month, hour=23, minute=59, second=59, microsecond=999999)
        start_utc_dt = start_of_month_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_month_local.astimezone(tz.tzutc())
    elif arg == "year":
        start_of_year_local = current_local_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_of_year_local = current_local_time.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
        start_utc_dt = start_of_year_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_year_local.astimezone(tz.tzutc())
    elif "=" in arg:
        try:
            key, value_str = arg.split('=', 1)
            value = int(value_str)
            if key == "day":
                if not 1 <= value <= 366:
                    raise ValueError(f"Day number {value} out of range (1-366).")
                start_of_year_local = current_local_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                target_day_local = start_of_year_local + timedelta(days=value - 1)
                start_of_day_local = target_day_local.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day_local = start_of_day_local + timedelta(days=1) - timedelta(microseconds=1)
                start_utc_dt = start_of_day_local.astimezone(tz.tzutc())
                end_utc_dt = end_of_day_local.astimezone(tz.tzutc())
            elif key == "week":
                if not 1 <= value <= 53:
                    raise ValueError(f"Week number {value} out of range (1-53).")
                year = current_local_time.year
                d = date(year, 1, 4) 
                first_iso_monday_of_year = d - timedelta(days=d.isoweekday() - 1)
                start_of_week_local_date = first_iso_monday_of_year + timedelta(weeks=value - 1)
                start_of_week_local = datetime(start_of_week_local_date.year, start_of_week_local_date.month, start_of_week_local_date.day, 0, 0, 0, tzinfo=LONDON_TZ)
                end_of_week_local = start_of_week_local + timedelta(days=7) - timedelta(microseconds=1)
                start_utc_dt = start_of_week_local.astimezone(tz.tzutc())
                end_utc_dt = end_of_week_local.astimezone(tz.tzutc())
            elif key == "month":
                if not 1 <= value <= 12:
                    raise ValueError(f"Month number {value} out of range (1-12).")
                start_of_month_local = current_local_time.replace(month=value, day=1, hour=0, minute=0, second=0, microsecond=0)
                num_days_in_month = calendar.monthrange(current_local_time.year, value)[1]
                end_of_month_local = start_of_month_local.replace(day=num_days_in_month, hour=23, minute=59, second=59, microsecond=999999)
                start_utc_dt = start_of_month_local.astimezone(tz.tzutc())
                end_utc_dt = end_of_month_local.astimezone(tz.tzutc())
            elif key == "year":
                if not (current_local_time.year - 100 <= value <= current_local_time.year + 100):
                    raise ValueError(f"Year {value} is far out of typical range.")
                start_of_year_local = datetime(value, 1, 1, 0, 0, 0, tzinfo=LONDON_TZ)
                end_of_year_local = datetime(value, 12, 31, 23, 59, 59, 999999, tzinfo=LONDON_TZ)
                start_utc_dt = start_of_year_local.astimezone(tz.tzutc())
                end_utc_dt = end_of_year_local.astimezone(tz.tzutc())
            else:
                raise ValueError(f"Unknown date range key: {key}")
        except (ValueError, IndexError):
            raise ValueError(f"Invalid date range argument format: {arg}")
    else:
        raise ValueError(f"Unknown date range argument: {arg}")
    return start_utc_dt, end_utc_dt

# --- Cloud Function Entry Point: get_weather_forecast ---
def get_weather_forecast(request: Request):
    start_request_time = time.time()
    logging.info(f"Function received request at {start_request_time:.2f}s")

    headers = {'Access-Control-Allow-Origin': '*'}
    if request.method == 'OPTIONS':
        return ('', 204, headers)

    if db is None:
        logging.error("Firestore client is not initialized. Cannot process request.")
        return (jsonify({'error': 'Internal server configuration error: Firestore client unavailable'}), 500, headers)

    try:
        parse_json_start = time.time()
        request_json = request.get_json(silent=True)
        if not request_json:
            logging.warning("Request has no JSON payload or it's invalid.")
            return (jsonify({'error': 'Request must be JSON'}), 400, headers)
        logging.info(f"Received JSON payload: {request_json}")
        parse_json_end = time.time()
        logging.info(f"JSON parsing took: {parse_json_end - parse_json_start:.4f}s")

        # NEW: Determine requested fields
        # If 'fields' are provided, use them. Otherwise, use all defined fields.
        # Ensure 'time' is always included, as it's crucial for time-based queries.
        requested_fields = request_json.get('fields', ALL_POSSIBLE_FIELDS)
        if 'time' not in requested_fields:
            requested_fields.insert(0, 'time') # Add it at the beginning for consistency
        
        # Validate that requested_fields are known/valid to prevent unexpected behavior
        for field in requested_fields:
            if field not in ALL_POSSIBLE_FIELDS:
                return (jsonify({'error': f"Invalid field requested: {field}"}), 400, headers)
        
        logging.info(f"Requested fields for Firestore query: {requested_fields}")


        start_utc_query = None
        end_utc_query = None
        limit_query = None
        order_direction = firestore.Query.ASCENDING # Use the Firestore enum for clarity

        if "range" in request_json:
            range_arg = request_json["range"]
            logging.info(f"Received range argument: '{range_arg}'")
            current_time = datetime.now(LONDON_TZ)
            start_utc_query, end_utc_query = get_time_range(range_arg, current_time)

            if range_arg == "latest":
                limit_query = 1
                order_direction = firestore.Query.DESCENDING
            elif range_arg == "first":
                limit_query = 1
                order_direction = firestore.Query.ASCENDING
            elif range_arg == "all":
                pass # No limit, no specific time range for 'all'
            else:
                if start_utc_query is None or end_utc_query is None:
                    raise ValueError("Invalid range argument resulted in null start/end times.")
        
        elif "start" in request_json and "end" in request_json:
            start_str = request_json["start"]
            end_str = request_json["end"]
            logging.info(f"Received start/end arguments: '{start_str}' to '{end_str}'")
            try:
                start_utc_query = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=tz.tzutc())
                end_utc_query = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=tz.tzutc())
            except ValueError:
                return (jsonify({'error': 'Invalid start/end datetime format. Expected YYYY-MM-DDTHH:MM:SSZ'}), 400, headers)
        else:
            return (jsonify({'error': 'Invalid request. Expected "range" or "start"/"end" in JSON body.'}), 400, headers)

        collection_ref = db.collection('weather-forecasts')
        query = collection_ref.order_by('time', direction=order_direction)

        if start_utc_query:
            query = query.where('time', '>=', start_utc_query)
        if end_utc_query:
            query = query.where('time', '<=', end_utc_query)
        if limit_query:
            query = query.limit(limit_query)
        
        # NEW: Apply .select() to only retrieve specified fields
        query = query.select(requested_fields)


        # --- START Firestore Query Timing ---
        firestore_query_start = time.time()
        docs_stream = query.stream()
        
        results = []
        doc_count = 0
        for doc in docs_stream:
            doc_count += 1
            data = doc.to_dict()
            # Convert Firestore Timestamp objects (which are Python datetime) to ISO strings
            if 'time' in data and isinstance(data['time'], datetime):
                data['time'] = data['time'].isoformat()
            results.append(data)
        
        firestore_query_end = time.time()
        logging.info(f"Firestore query and stream iteration took: {firestore_query_end - firestore_query_start:.4f}s")
        logging.info(f"Successfully retrieved {doc_count} documents.")
        # --- END Firestore Query Timing ---

        # --- START JSON Serialization Timing ---
        json_serialize_start = time.time()
        json_response = jsonify(results) # jsonify returns a Flask Response object directly
        json_serialize_end = time.time()
        logging.info(f"JSON serialization took: {json_serialize_end - json_serialize_start:.4f}s")
        # --- END JSON Serialization Timing ---

        end_request_time = time.time()
        logging.info(f"Total function execution time: {end_request_time - start_request_time:.4f}s")

        return (jsonify(results), 200, headers) # Changed from json_response to jsonify(results) for consistency

    except ValueError as ve:
        logging.warning(f"Argument parsing error: {ve}")
        return (jsonify({'error': str(ve)}), 400, headers)
    except GoogleCloudError as gce:
        logging.error(f"Firestore error: {str(gce)}")
        return (jsonify({'error': 'Database error occurred', 'details': str(gce)}), 500, headers)
    except Exception as e:
        # Log the full traceback for unexpected errors
        logging.error(f"Unexpected error in get_weather_forecast: {str(e)}", exc_info=True)
        return (jsonify({'error': 'Internal server error', 'details': str(e)}), 500, headers)