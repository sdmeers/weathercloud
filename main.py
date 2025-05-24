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
    # Depending on GCP environment, this might cause deployment to fail or invocations to error out immediately.
    db = None 

# Define the target timezone for deriving day, week, month, year
LONDON_TZ = tz.gettz('Europe/London')

# --- Entry Point: store_weather_data ---
def store_weather_data(request: Request):
    """
    Cloud Function to receive and store weather data from Raspberry Pi Pico.
    It adds London-time based 'day', 'week', 'month', 'year' fields.
    
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
        "timestamp": "2024-01-01T12:00:00Z" // UTC timestamp from Pico
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
        
        # 3. Validate required fields (adjust based on your actual sensor setup)
        # These are fields expected directly from the Pico/client
        required_client_fields = ['temperature', 'humidity', 'pressure', 'timestamp'] 
        missing_fields = [field for field in required_client_fields if field not in weather_data]
        
        if missing_fields:
            logging.warning(f"Missing required fields: {', '.join(missing_fields)}")
            return ({'error': f'Missing required fields: {", ".join(missing_fields)}'}, 400, headers)
            
        # 4. Add server processing timestamp (Firestore specific timestamp)
        weather_data['processed_at'] = firestore.SERVER_TIMESTAMP
        
        # 5. Derive London-time based day, week, month, year from client's UTC timestamp
        client_utc_timestamp_str = weather_data.get('timestamp')

        if client_utc_timestamp_str and LONDON_TZ:
            try:
                utc_time_dt = datetime.strptime(client_utc_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                utc_time_dt = utc_time_dt.replace(tzinfo=tz.tzutc()) # Ensure it's UTC aware

                local_time_dt = utc_time_dt.astimezone(LONDON_TZ)

                weather_data["day"] = local_time_dt.strftime("%j")      # Day of year (e.g., "001")
                weather_data["week"] = local_time_dt.strftime("%W")     # Week # (Mon first, e.g., "00")
                weather_data["month"] = local_time_dt.strftime("%m")    # Month # (e.g., "01")
                weather_data["year"] = local_time_dt.strftime("%Y")     # Year (e.g., "2024")
            except ValueError as e:
                logging.warning(f"Could not parse client timestamp '{client_utc_timestamp_str}' to derive date parts. Error: {e}. These fields might be missing.")
            except Exception as e:
                logging.error(f"Unexpected error deriving date parts from client timestamp '{client_utc_timestamp_str}': {e}")
        elif not client_utc_timestamp_str:
            logging.warning("Client 'timestamp' field is missing. Cannot derive custom date parts from it.")
        elif not LONDON_TZ: # Should not happen if tz.gettz is correct
             logging.error("LONDON_TZ is not defined. Cannot derive custom date parts.")


        # 6. Validate and convert numeric fields (optional, but good practice)
        numeric_fields = ['temperature', 'humidity', 'pressure', 'rain', 
                          'rain_rate', 'luminance', 'wind_speed', 'wind_direction']
        for field in numeric_fields:
            if field in weather_data:
                try:
                    weather_data[field] = float(weather_data[field])
                except (ValueError, TypeError):
                    logging.warning(f"Field '{field}' with value '{weather_data[field]}' is not a valid number.")
                    return ({'error': f'Field {field} must be a number. Received: {weather_data[field]}'}, 400, headers)
        
        # 7. Store in Firestore
        # Create a document ID from the original timestamp to help ensure uniqueness if desired
        # and make it somewhat human-readable/sortable if Browse raw IDs.
        doc_id_timestamp_part = weather_data['timestamp'].replace(':', '-').replace('.', '-').replace('Z','')
        doc_id = f"reading_{doc_id_timestamp_part}"
        
        collection_ref = db.collection('weather-readings')
        doc_ref = collection_ref.document(doc_id)
        doc_ref.set(weather_data) # weather_data now includes the derived day, week, month, year
        
        logging.info(f"Successfully stored weather data with ID: {doc_id} and data: {weather_data}")
        
        return ({
            'status': 'success',
            'message': 'Weather data stored successfully',
            'document_id': doc_id,
            'timestamp': weather_data['timestamp'] # Return the original timestamp
        }, 200, headers)
        
    except GoogleCloudError as e:
        logging.error(f"Firestore error: {str(e)}")
        return ({'error': 'Database error occurred', 'details': str(e)}, 500, headers)
    except Exception as e:
        logging.error(f"Unexpected error in store_weather_data: {str(e)}", exc_info=True) # exc_info=True for stack trace
        return ({'error': 'Internal server error', 'details': str(e)}, 500, headers)

# --- Optional Entry Point: get_recent_weather (if deployed from same main.py) ---
def get_recent_weather(request: Request):
    """
    Cloud Function to retrieve recent weather readings.
    Query parameter: limit (default: 10, max: 100)
    """
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*', # Adjust in production
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {'Access-Control-Allow-Origin': '*'} # Adjust in production

    if db is None:
        logging.error("Firestore client is not initialized. Cannot process get_recent_weather request.")
        return ({'error': 'Internal server configuration error: Firestore client unavailable'}, 500, headers)
    
    if request.method != 'GET':
        logging.warning(f"Invalid request method for get_recent_weather: {request.method}")
        return ({'error': 'Only GET method allowed for get_recent_weather'}, 405, headers)

    try:
        limit_str = request.args.get('limit', '10')
        try:
            limit = int(limit_str)
            limit = min(max(limit, 1), 100)  # Cap between 1 and 100 records
        except ValueError:
            logging.warning(f"Invalid limit parameter: {limit_str}. Using default 10.")
            limit = 10
            
        collection_ref = db.collection('weather-readings')
        # Order by the original client timestamp in descending order to get the latest
        query = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        
        docs = query.stream()
        readings = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id # Optionally include the document ID
            readings.append(data)
            
        return ({
            'status': 'success',
            'count': len(readings),
            'limit_applied': limit,
            'readings': readings
        }, 200, headers)
        
    except GoogleCloudError as e:
        logging.error(f"Firestore error in get_recent_weather: {str(e)}")
        return ({'error': 'Database error occurred while retrieving data', 'details': str(e)}, 500, headers)
    except Exception as e:
        logging.error(f"Unexpected error in get_recent_weather: {str(e)}", exc_info=True)
        return ({'error': 'Internal server error while retrieving data', 'details': str(e)}, 500, headers)