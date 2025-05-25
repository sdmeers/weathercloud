import json
import logging
from datetime import datetime
from dateutil import tz
from flask import Request
from google.cloud import firestore
from google.cloud.exceptions import GoogleCloudError

# --- Configuration ---
# Initialize Firestore client globally to be reused across invocations
# Ensure this matches the Database ID you created in Firestore.
FIRESTORE_DATABASE_ID = "weatherdata"

try:
    db = firestore.Client(database=FIRESTORE_DATABASE_ID)
except Exception as e:
    logging.critical(f"Failed to initialize Firestore client with database '{FIRESTORE_DATABASE_ID}': {e}")
    # If the client fails to initialize, the function won't work.
    db = None

# Define the target timezone for London, which correctly handles DST
LONDON_TZ = tz.gettz('Europe/London')

# --- Entry Point: store_weather_data ---
def store_weather_data(request: Request):
    """
    Cloud Function to receive and store weather data from Raspberry Pi Pico.
    
    Expected JSON format from client:
    {
        "temperature": 22.5,
        "humidity": 65.2,
        "pressure": 1013.25,
        "rain": 0.0,
        "rain_rate": 0.0,
        "luminance": 5000,
        "wind_speed": 5.2,
        "wind_direction": 180.0,
        "timestamp": "YYYY-MM-DDTHH:MM:SSZ" // UTC timestamp from Pico
    }
    """

    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*', # Adjust to your specific domain in production
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {'Access-Control-Allow-Origin': '*'} # Adjust in production

    if db is None:
        logging.error("Firestore client is not initialized. Cannot process request.")
        return ({'error': 'Internal server configuration error: Firestore client unavailable'}, 500, headers)

    try:
        # 1. Validate request method
        if request.method != 'POST':
            logging.warning(f"Invalid request method: {request.method}")
            return ({'error': 'Only POST method allowed'}, 405, headers)
        
        # 2. Get JSON data from request
        if not request.is_json:
            logging.warning("Request is not JSON.")
            return ({'error': 'Request must be JSON'}, 400, headers)
        
        weather_data = request.get_json()
        
        if not weather_data:
            logging.warning("No JSON data provided in request.")
            return ({'error': 'No data provided'}, 400, headers)
        
        # 3. Validate required fields
        required_client_fields = ['temperature', 'humidity', 'pressure', 'rain', 'rain_rate', 
                                  'luminance', 'wind_speed', 'wind_direction', 'timestamp']
        missing_fields = [field for field in required_client_fields if field not in weather_data]
        
        if missing_fields:
            logging.warning(f"Missing required fields: {', '.join(missing_fields)}")
            return ({'error': f'Missing required fields: {", ".join(missing_fields)}'}, 400, headers)
                
        # 4. Process timestamps and create document ID
        client_utc_timestamp_str = weather_data.get('timestamp')

        if client_utc_timestamp_str and LONDON_TZ:
            try:
                # Parse the incoming UTC string into a timezone-aware UTC datetime object
                utc_time_dt = datetime.strptime(client_utc_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                utc_time_dt = utc_time_dt.replace(tzinfo=tz.tzutc()) 

                # Convert the UTC datetime object to the local London timezone
                local_time_dt = utc_time_dt.astimezone(LONDON_TZ)

                # --- Logging to confirm the conversion ---
                logging.info(f"Original UTC string: {client_utc_timestamp_str}")
                logging.info(f"Parsed UTC datetime object: {utc_time_dt} (tzinfo: {utc_time_dt.tzinfo})")
                logging.info(f"Local London datetime object: {local_time_dt} (tzinfo: {local_time_dt.tzinfo})")
                # ----------------------------------------

                # Store UTC as a native Firestore Timestamp (primary time field)
                weather_data["timestamp_UTC"] = utc_time_dt 
                
                # Store local time as an ISO string for clear visual distinction and explicit offset
                weather_data["timestamp_local_iso_str"] = local_time_dt.isoformat()

                # Derive doc_id from the London-adjusted timestamp (local time)
                doc_id_timestamp_part = local_time_dt.strftime("%Y-%m-%dT%H-%M-%S")
                doc_id = f"reading_{doc_id_timestamp_part}"

            except ValueError as e:
                logging.warning(f"Could not parse client timestamp '{client_utc_timestamp_str}'. Error: {e}")
                return ({'error': f'Invalid timestamp format: {client_utc_timestamp_str}'}, 400, headers)
            except Exception as e:
                logging.error(f"Unexpected error processing timestamp '{client_utc_timestamp_str}': {e}")
                return ({'error': 'Internal server error during timestamp processing'}, 500, headers)
        else:
            logging.warning("Client 'timestamp' field is missing or LONDON_TZ is not defined.")
            return ({'error': 'Timestamp is required'}, 400, headers)

        # 5. Validate and convert numeric fields
        numeric_fields = ['temperature', 'humidity', 'pressure', 'rain',
                          'rain_rate', 'luminance', 'wind_speed', 'wind_direction']
        for field in numeric_fields:
            if field in weather_data:
                try:
                    weather_data[field] = float(weather_data[field])
                except (ValueError, TypeError):
                    logging.warning(f"Field '{field}' with value '{weather_data[field]}' is not a valid number.")
                    return ({'error': f'Field {field} must be a number. Received: {weather_data[field]}'}, 400, headers)
        
        # 6. Remove the original 'timestamp' field from the document as it's now split
        # We assume the original 'timestamp' is only for parsing and not needed in the final doc
        if 'timestamp' in weather_data:
            del weather_data['timestamp']

        # 7. Store in Firestore
        collection_ref = db.collection('weather-readings')
        doc_ref = collection_ref.document(doc_id)
        doc_ref.set(weather_data)
        
        logging.info(f"Successfully stored weather data with ID: {doc_id} and data: {weather_data}")
        
        return ({
            'status': 'success',
            'message': 'Weather data stored successfully',
            'document_id': doc_id,
            'timestamp_local_display': weather_data['timestamp_local_iso_str'],
            'timestamp_UTC_display': weather_data['timestamp_UTC'].isoformat()
        }, 200, headers)
        
    except GoogleCloudError as e:
        logging.error(f"Firestore error: {str(e)}")
        return ({'error': 'Database error occurred', 'details': str(e)}, 500, headers)
    except Exception as e:
        logging.error(f"Unexpected error in store_weather_data: {str(e)}", exc_info=True)
        return ({'error': 'Internal server error', 'details': str(e)}, 500, headers)