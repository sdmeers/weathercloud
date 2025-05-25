import json
import logging
from datetime import datetime, timedelta, date
from dateutil import tz
from flask import Request, jsonify # Import jsonify for easier JSON responses
from google.cloud import firestore
from google.cloud.exceptions import GoogleCloudError
from typing import Optional, Tuple, Dict, Any
import calendar # For monthrange

# --- Configuration ---
# Initialize Firestore client globally to be reused across invocations
# Ensure this matches the Database ID you created in Firestore.
FIRESTORE_DATABASE_ID = "weatherdata"

try:
    db = firestore.Client(database=FIRESTORE_DATABASE_ID)
except Exception as e:
    logging.critical(f"Failed to initialize Firestore client with database '{FIRESTORE_DATABASE_ID}': {e}")
    db = None

# Define the target timezone for London, which correctly handles DST
LONDON_TZ = tz.gettz('Europe/London')
if not LONDON_TZ:
    logging.critical("Failed to get 'Europe/London' timezone. Please ensure dateutil is properly configured.")
    # Fallback to UTC if timezone cannot be loaded, though this will affect "local" queries
    LONDON_TZ = tz.tzutc()


# --- Helper Function: get_time_range ---
def get_time_range(arg: str, current_local_time: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Determines the start and end UTC datetime objects based on a string argument,
    relative to a provided current_local_time.
    Returns (start_utc, end_utc) or (None, None) for 'latest', 'first', 'all'.
    """
    start_utc_dt = None
    end_utc_dt = None

    # Ensure current_local_time is timezone-aware
    if current_local_time.tzinfo is None:
        current_local_time = current_local_time.replace(tzinfo=LONDON_TZ)

    current_utc_time = current_local_time.astimezone(tz.tzutc())
    
    # Handle specific string commands
    if arg == "latest" or arg == "first" or arg == "all":
        return None, None # These are handled by query limits/ordering later

    elif arg == "today":
        start_of_today_local = current_local_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_today_local = start_of_today_local + timedelta(days=1) - timedelta(microseconds=1)
        start_utc_dt = start_of_today_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_today_local.astimezone(tz.tzutc())

    elif arg == "last24h":
        start_utc_dt = current_utc_time - timedelta(hours=24)
        end_utc_dt = current_utc_time # Up to the current moment

    elif arg == "yesterday":
        yesterday_local = current_local_time - timedelta(days=1)
        start_of_yesterday_local = yesterday_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_yesterday_local = start_of_yesterday_local + timedelta(days=1) - timedelta(microseconds=1)
        start_utc_dt = start_of_yesterday_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_yesterday_local.astimezone(tz.tzutc())

    elif arg == "last7days":
        # Last 7 full 24-hour periods from the current moment
        start_utc_dt = current_utc_time - timedelta(days=7)
        end_utc_dt = current_utc_time

    elif arg == "week":
        # Current local week (ISO week: Monday start)
        # isoweekday() returns 1 for Monday, 7 for Sunday
        start_of_current_week_local = current_local_time - timedelta(days=current_local_time.isoweekday() - 1)
        start_of_current_week_local = start_of_current_week_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_current_week_local = start_of_current_week_local + timedelta(days=7) - timedelta(microseconds=1)
        start_utc_dt = start_of_current_week_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_current_week_local.astimezone(tz.tzutc())
        
    elif arg == "month":
        # Current local month
        start_of_month_local = current_local_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        num_days_in_month = calendar.monthrange(current_local_time.year, current_local_time.month)[1]
        end_of_month_local = current_local_time.replace(day=num_days_in_month, hour=23, minute=59, second=59, microsecond=999999)
        start_utc_dt = start_of_month_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_month_local.astimezone(tz.tzutc())

    elif arg == "year":
        # Current local year
        start_of_year_local = current_local_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_of_year_local = current_local_time.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
        start_utc_dt = start_of_year_local.astimezone(tz.tzutc())
        end_utc_dt = end_of_year_local.astimezone(tz.tzutc())

    elif "=" in arg:
        try:
            key, value_str = arg.split('=', 1)
            value = int(value_str)

            if key == "day": # nth day of the current local year
                if not 1 <= value <= 366: # Handle leap years
                    raise ValueError(f"Day number {value} out of range (1-366).")
                
                start_of_year_local = current_local_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                target_day_local = start_of_year_local + timedelta(days=value - 1)
                
                start_of_day_local = target_day_local.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day_local = start_of_day_local + timedelta(days=1) - timedelta(microseconds=1)
                
                start_utc_dt = start_of_day_local.astimezone(tz.tzutc())
                end_utc_dt = end_of_day_local.astimezone(tz.tzutc())

            elif key == "week": # nth ISO week of the current local year
                if not 1 <= value <= 53:
                    raise ValueError(f"Week number {value} out of range (1-53).")
                
                # Calculate the start and end of the ISO week in the local timezone
                # ISO week 1 starts on the Monday of the week containing January 4th.
                year = current_local_time.year
                d = date(year, 1, 4) # Jan 4 is always in week 1
                # Adjust d to the Monday of the week it belongs to
                first_iso_monday_of_year = d - timedelta(days=d.isoweekday() - 1)
                
                start_of_week_local_date = first_iso_monday_of_year + timedelta(weeks=value - 1)
                
                start_of_week_local = datetime(start_of_week_local_date.year, start_of_week_local_date.month, start_of_week_local_date.day, 0, 0, 0, tzinfo=LONDON_TZ)
                end_of_week_local = start_of_week_local + timedelta(days=7) - timedelta(microseconds=1)
                
                start_utc_dt = start_of_week_local.astimezone(tz.tzutc())
                end_utc_dt = end_of_week_local.astimezone(tz.tzutc())

            elif key == "month": # nth month of the current local year
                if not 1 <= value <= 12:
                    raise ValueError(f"Month number {value} out of range (1-12).")
                
                start_of_month_local = current_local_time.replace(month=value, day=1, hour=0, minute=0, second=0, microsecond=0)
                num_days_in_month = calendar.monthrange(current_local_time.year, value)[1]
                end_of_month_local = start_of_month_local.replace(day=num_days_in_month, hour=23, minute=59, second=59, microsecond=999999)
                
                start_utc_dt = start_of_month_local.astimezone(tz.tzutc())
                end_utc_dt = end_of_month_local.astimezone(tz.tzutc())

            elif key == "year": # specific year
                # Year can be specified. Assume no more than 100 years in past/future for basic validation
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


# --- Cloud Function Entry Point: get_weather_data ---
def get_weather_data(request: Request):
    """
    Cloud Function to fetch weather data from Firestore based on time range criteria.

    Parameters expected in JSON body:
        {
            "range": "string_argument"  OR
            "start": "YYYY-MM-DDTHH:MM:SSZ",
            "end": "YYYY-MM-DDTHH:MM:SSZ"
        }

    Returns:
        JSON: A JSON object containing a list of weather data dictionaries.
    """
    headers = {'Access-Control-Allow-Origin': '*'} # Allow CORS for client applications

    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        return ('', 204, headers)

    if db is None:
        logging.error("Firestore client is not initialized. Cannot process request.")
        return (jsonify({'error': 'Internal server configuration error: Firestore client unavailable'}), 500, headers)

    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return (jsonify({'error': 'Request must be JSON'}), 400, headers)

        start_utc_query = None
        end_utc_query = None
        limit_query = None
        order_direction = firestore.Query.ASCENDING # Default order for most queries

        if "range" in request_json:
            range_arg = request_json["range"]
            logging.info(f"Received range argument: '{range_arg}'")

            current_time = datetime.now(LONDON_TZ) # Get current time in London timezone
            start_utc_query, end_utc_query = get_time_range(range_arg, current_time)

            if range_arg == "latest":
                limit_query = 1
                order_direction = firestore.Query.DESCENDING
            elif range_arg == "first":
                limit_query = 1
                order_direction = firestore.Query.ASCENDING
            elif range_arg == "all":
                pass # No limits, just order by timestamp by default
            else:
                # For all other range strings, we apply the calculated UTC range filters
                if start_utc_query is None or end_utc_query is None:
                    # This should ideally not happen if get_time_range is well-behaved
                    # for non-special string arguments.
                    raise ValueError("Invalid range argument resulted in null start/end times.")

        elif "start" in request_json and "end" in request_json:
            start_str = request_json["start"]
            end_str = request_json["end"]
            logging.info(f"Received start/end arguments: '{start_str}' to '{end_str}'")
            try:
                # Expecting UTC ISO format from client for direct range queries
                start_utc_query = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=tz.tzutc())
                end_utc_query = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=tz.tzutc())
            except ValueError:
                return (jsonify({'error': 'Invalid start/end datetime format. Expected YYYY-MM-DDTHH:MM:SSZ'}), 400, headers)
        else:
            return (jsonify({'error': 'Invalid request. Expected "range" or "start"/"end" in JSON body.'}), 400, headers)

        # Build Firestore query
        collection_ref = db.collection('weather-readings')
        query = collection_ref.order_by('timestamp_UTC', direction=order_direction)

        if start_utc_query:
            query = query.where('timestamp_UTC', '>=', start_utc_query)
        if end_utc_query:
            query = query.where('timestamp_UTC', '<=', end_utc_query)
        if limit_query:
            query = query.limit(limit_query)

        docs_stream = query.stream()
        
        results = []
        for doc in docs_stream:
            data = doc.to_dict()
            # Convert Firestore Timestamp objects to ISO strings for JSON serialization
            if 'timestamp_UTC' in data and isinstance(data['timestamp_UTC'], firestore.Timestamp):
                data['timestamp_UTC'] = data['timestamp_UTC'].isoformat()
            
            results.append(data)
        
        logging.info(f"Successfully retrieved {len(results)} documents.")
        return (jsonify(results), 200, headers)

    except ValueError as ve:
        logging.warning(f"Argument parsing error: {ve}")
        return (jsonify({'error': str(ve)}), 400, headers)
    except GoogleCloudError as gce:
        logging.error(f"Firestore error: {str(gce)}")
        return (jsonify({'error': 'Database error occurred', 'details': str(gce)}), 500, headers)
    except Exception as e:
        logging.error(f"Unexpected error in get_weather_data: {str(e)}", exc_info=True)
        return (jsonify({'error': 'Internal server error', 'details': str(e)}), 500, headers)