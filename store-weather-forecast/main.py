
import os
import logging
from datetime import datetime, timezone, timedelta
from dateutil import tz
import requests
from google.cloud import firestore
from flask import Request, jsonify

# --- Configuration ---
# Initialize Firestore client globally
FIRESTORE_DATABASE_ID = "weatherdata"
try:
    db = firestore.Client(database=FIRESTORE_DATABASE_ID)
except Exception as e:
    logging.critical(f"Failed to initialize Firestore client with database '{FIRESTORE_DATABASE_ID}': {e}")
    db = None

# Met Office API Configuration from Environment Variables
MET_OFFICE_API_KEY = os.environ.get('MET_OFFICE_API_KEY')
LATITUDE = os.environ.get('LATITUDE')
LONGITUDE = os.environ.get('LONGITUDE')

# API Endpoint
API_URL = f"https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point/hourly?dataSource=BD1&latitude={LATITUDE}&longitude={LONGITUDE}"

# --- Entry Point: get_and_store_forecast ---
def store_weather_forecast(request: Request):
    """
    Cloud Function triggered by Cloud Scheduler to fetch the next 24-hour
    weather forecast and store it in Firestore.
    """
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    # --- Validate Configuration ---
    if db is None:
        logging.error("Firestore client is not initialized.")
        return jsonify({'error': 'Internal server configuration error: Firestore client unavailable'}), 500, headers

    if not all([MET_OFFICE_API_KEY, LATITUDE, LONGITUDE]):
        logging.error("Missing required environment variables: MET_OFFICE_API_KEY, LATITUDE, or LONGITUDE.")
        return jsonify({'error': 'Internal server configuration error: Missing API credentials or location'}), 500, headers

    # --- API Call ---
    api_headers = {
        'apikey': MET_OFFICE_API_KEY,
        'Accept': 'application/json'
    }

    try:
        response = requests.get(API_URL, headers=api_headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        api_data = response.json()
        logging.info("Successfully fetched data from Met Office API.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Met Office API: {e}")
        return jsonify({'error': 'Failed to fetch data from external API', 'details': str(e)}), 502, headers
    except ValueError as e: # Catches JSON decoding errors
        logging.error(f"Error decoding JSON from Met Office API: {e}")
        return jsonify({'error': 'Invalid JSON response from external API', 'details': str(e)}), 502, headers


    # --- Data Processing ---
    try:
        time_series = api_data['features'][0]['properties']['timeSeries']
        processed_forecasts = []

        fetch_time = datetime.now(timezone.utc)
        limit_time = fetch_time + timedelta(hours=24)

        for entry in time_series:
            # Convert time string to datetime object
            forecast_time = datetime.fromisoformat(entry['time'].replace('Z', '+00:00'))

            # Stop if the forecast time is beyond our 24-hour window
            if forecast_time > limit_time:
                break

            # Convert pressure from Pascals to hPa/millibars
            pressure_hpa = entry['mslp'] / 100.0 if 'mslp' in entry else None

            processed_forecasts.append({
                'time': forecast_time,
                'temperature': entry.get('screenTemperature'),
                'humidity': entry.get('screenRelativeHumidity'),
                'pressure': pressure_hpa,
                'rain_total': entry.get('totalPrecipAmount'),
                'prob_of_rain': entry.get('probOfPrecipitation'),
                'wind_speed': entry.get('windSpeed10m')
            })

        # --- Store individual hourly forecasts in the 'weather-forecasts' collection ---
        collection_ref = db.collection('weather-forecasts')
        LONDON_TZ = tz.gettz('Europe/London') # Define London timezone

        for hourly_forecast in processed_forecasts:
            # The 'time' field from the forecast is already a UTC datetime object
            utc_time = hourly_forecast['time']

            # Convert to London time to create a UK-friendly document ID
            local_time = utc_time.astimezone(LONDON_TZ)
            doc_id = local_time.strftime('%Y-%m-%d-%H')

            # Store the document using the localized ID
            # The data within the document still contains the original UTC timestamp
            collection_ref.document(doc_id).set(hourly_forecast)

        logging.info(f"Successfully stored {len(processed_forecasts)} individual hourly forecasts in 'weather-forecasts'.")

        return jsonify({
            'status': 'success',
            'message': f"Stored {len(processed_forecasts)} hourly forecasts successfully.",
            'hourly_forecasts_stored': len(processed_forecasts)
        }), 200, headers

    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"Failed to parse API data structure. Error: {e}", exc_info=True)
        return jsonify({'error': 'Invalid or unexpected API data structure', 'details': str(e)}), 500, headers
    except Exception as e:
        logging.error(f"An unexpected error occurred during data processing or Firestore upload: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500, headers
