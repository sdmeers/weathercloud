
import os
import logging
from datetime import datetime, timezone, timedelta
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
                'prob_of_rain': entry.get('probOfPrecipitation')
            })

        # Prepare document for Firestore
        doc_id = f"forecast_{fetch_time.strftime('%Y-%m-%d')}"

        # Prepare hourly forecasts as fields for the main document
        hourly_fields = {}
        for hourly_forecast in processed_forecasts:
            field_name = hourly_forecast['time'].strftime('forecast-%Y-%m-%d-%H')
            hourly_fields[field_name] = hourly_forecast

        # Prepare the main document for Firestore
        forecast_document = {
            'fetch_timestamp_utc': fetch_time,
            'latitude': float(LATITUDE),
            'longitude': float(LONGITUDE),
            **hourly_fields # Unpack the hourly forecasts as fields
        }

        # --- Store in Firestore ---
        collection_ref = db.collection('weather-forecast')
        doc_ref = collection_ref.document(doc_id) # doc_id is already 'forecast_YYYY-MM-DD'
        doc_ref.set(forecast_document)

        logging.info(f"Successfully stored weather forecast for {doc_id} with {len(processed_forecasts)} hourly fields.")

        return jsonify({
            'status': 'success',
            'message': f"Weather forecast stored successfully for {doc_id} with {len(processed_forecasts)} hourly fields",
            'document_id': doc_id,
            'hourly_fields_stored': len(processed_forecasts)
        }), 200, headers

    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"Failed to parse API data structure. Error: {e}", exc_info=True)
        return jsonify({'error': 'Invalid or unexpected API data structure', 'details': str(e)}), 500, headers
    except Exception as e:
        logging.error(f"An unexpected error occurred during data processing or Firestore upload: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500, headers
