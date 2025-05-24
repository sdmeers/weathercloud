import json
import logging
from datetime import datetime
from flask import Request
from google.cloud import firestore
from google.cloud.exceptions import GoogleCloudError

# Initialize Firestore client
db = firestore.Client(database="weatherdata")

def store_weather_data(request: Request):
    """
    Cloud Function to receive and store weather data from Raspberry Pi Pico
    
    Expected JSON format:
    {
        "temperature": 22.5,
        "humidity": 65.2,
        "pressure": 1013.25,
        "timestamp": "2024-01-01T12:00:00Z"
        // ... other sensor readings
    }
    """
    
    # Set CORS headers for preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for main request
    headers = {'Access-Control-Allow-Origin': '*'}
    
    try:
        # Validate request method
        if request.method != 'POST':
            return ({'error': 'Only POST method allowed'}, 405, headers)
        
        # Get JSON data from request
        if not request.is_json:
            return ({'error': 'Request must be JSON'}, 400, headers)
        
        weather_data = request.get_json()
        
        if not weather_data:
            return ({'error': 'No data provided'}, 400, headers)
        
        # Validate required fields (adjust based on your sensor setup)
        required_fields = ['temperature', 'humidity', 'pressure']
        missing_fields = [field for field in required_fields if field not in weather_data]
        
        if missing_fields:
            return ({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }, 400, headers)
        
        # Add server timestamp if not provided
        if 'timestamp' not in weather_data:
            weather_data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add processing timestamp
        weather_data['processed_at'] = firestore.SERVER_TIMESTAMP
        
        # Validate numeric fields
        numeric_fields = ['temperature', 'humidity', 'pressure']
        for field in numeric_fields:
            if field in weather_data:
                try:
                    weather_data[field] = float(weather_data[field])
                except (ValueError, TypeError):
                    return ({
                        'error': f'Field {field} must be a number'
                    }, 400, headers)
        
        # Store in Firestore
        # Using timestamp as document ID for easy querying and uniqueness
        timestamp_str = weather_data['timestamp'].replace(':', '-').replace('.', '-')
        doc_id = f"reading_{timestamp_str}"
        
        # Reference to the weather-readings collection
        collection_ref = db.collection('weather-readings')
        
        # Add the document
        doc_ref = collection_ref.document(doc_id)
        doc_ref.set(weather_data)
        
        logging.info(f"Successfully stored weather data with ID: {doc_id}")
        
        return ({
            'status': 'success',
            'message': 'Weather data stored successfully',
            'document_id': doc_id,
            'timestamp': weather_data['timestamp']
        }, 200, headers)
        
    except GoogleCloudError as e:
        logging.error(f"Firestore error: {str(e)}")
        return ({
            'error': 'Database error occurred',
            'details': str(e)
        }, 500, headers)
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return ({
            'error': 'Internal server error',
            'details': str(e)
        }, 500, headers)

# Optional: Function to retrieve recent weather data
def get_recent_weather(request: Request):
    """
    Cloud Function to retrieve recent weather readings
    Query parameter: limit (default: 10)
    """
    
    headers = {'Access-Control-Allow-Origin': '*'}
    
    if request.method == 'OPTIONS':
        headers.update({
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        })
        return ('', 204, headers)
    
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 10))
        limit = min(limit, 100)  # Cap at 100 records
        
        # Query Firestore for recent readings
        collection_ref = db.collection('weather-readings')
        query = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        
        docs = query.stream()
        readings = []
        
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            readings.append(data)
        
        return ({
            'status': 'success',
            'count': len(readings),
            'readings': readings
        }, 200, headers)
        
    except Exception as e:
        logging.error(f"Error retrieving weather data: {str(e)}")
        return ({
            'error': 'Failed to retrieve weather data',
            'details': str(e)
        }, 500, headers)