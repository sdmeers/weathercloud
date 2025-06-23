import functions_framework
from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Image as VertexImage
import base64
import json
from datetime import datetime
import os

# Initialize Vertex AI
PROJECT_ID = os.environ.get('PROJECT_ID')
LOCATION = os.environ.get('LOCATION', 'europe-west2')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

vertexai.init(project=PROJECT_ID, location=LOCATION)

@functions_framework.http
def weather_image_classifier(request):
    """HTTP Cloud Function that handles image upload, classification, and web display."""
    
    # Enable CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    try:
        if request.method == 'POST':
            # Handle image upload and classification
            return handle_image_upload(request, headers)
        elif request.method == 'GET':
            # Display the webpage
            return display_webpage(headers)
        else:
            return ('Method not allowed', 405, headers)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return (f'Error: {str(e)}', 500, headers)

def handle_image_upload(request, headers):
    """Handle image upload, classify it, and store results."""
    
    # Get image from request
    if 'image' not in request.files:
        return ('No image provided', 400, headers)
    
    image_file = request.files['image']
    if image_file.filename == '':
        return ('No image selected', 400, headers)

    # Read image data
    image_data = image_file.read()
    
    # Classify the weather
    classification = classify_weather_image(image_data)
    
    # Store image and classification in Cloud Storage
    store_results(image_data, classification)
    
    response_data = {
        'classification': classification,
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }
    
    return (json.dumps(response_data), 200, headers)

def classify_weather_image(image_data):
    """Classify weather conditions in the image using Vertex AI."""
    
    try:
        # Convert image data to base64 for Vertex AI
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create Vertex AI image object
        vertex_image = VertexImage.from_bytes(image_data)
        
        # Initialize Gemini model
        model = GenerativeModel("gemini-1.5-flash")
        
        prompt = """Analyze this outdoor image and classify the weather conditions. 
        Look at the sky, lighting, and any visible precipitation or weather phenomena.
        Respond with exactly ONE of these weather labels:
        - sunny
        - partly_cloudy  
        - overcast
        - raining
        - snowing
        - foggy
        - night
        - dawn
        - dusk
        
        Only respond with the single weather label, nothing else."""
        
        # Generate classification
        response = model.generate_content([prompt, vertex_image])
        classification = response.text.strip().lower()
        
        # Validate classification
        valid_labels = ['sunny', 'partly_cloudy', 'overcast', 'raining', 'snowing', 'foggy', 'night', 'dawn', 'dusk']
        if classification not in valid_labels:
            classification = 'unknown'
        
        return classification
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return 'error'

def store_results(image_data, classification):
    """Store the latest image and classification in Cloud Storage."""
    
    try:
        # Initialize Cloud Storage client
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        
        # Store image (always overwrite the same file)
        image_blob = bucket.blob('latest_weather_image.jpg')
        image_blob.upload_from_string(image_data, content_type='image/jpeg')
        
        # Store classification metadata
        metadata = {
            'classification': classification,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_blob = bucket.blob('latest_weather_data.json')
        metadata_blob.upload_from_string(
            json.dumps(metadata), 
            content_type='application/json'
        )
        
        print(f"Stored image with classification: {classification}")
        
    except Exception as e:
        print(f"Storage error: {str(e)}")
        raise

def display_webpage(headers):
    """Display a simple webpage with the latest image and classification."""
    
    try:
        # Get the latest image and classification from Cloud Storage
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        
        # Get classification data
        try:
            metadata_blob = bucket.blob('latest_weather_data.json')
            metadata_json = metadata_blob.download_as_text()
            metadata = json.loads(metadata_json)
            classification = metadata.get('classification', 'No data available')
            timestamp = metadata.get('timestamp', 'Unknown')
        except:
            classification = 'No data available'
            timestamp = 'Unknown'
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Weather Station</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f0f8ff;
                    text-align: center;
                }}
                .container {{
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    margin-bottom: 30px;
                }}
                .weather-image {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    margin-bottom: 20px;
                }}
                .classification {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #34495e;
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                    text-transform: capitalize;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-size: 14px;
                    margin-top: 10px;
                }}
                .upload-section {{
                    margin-top: 30px;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .upload-form {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 10px;
                }}
                input[type="file"] {{
                    padding: 10px;
                    border: 2px dashed #3498db;
                    border-radius: 5px;
                    background-color: white;
                }}
                button {{
                    background-color: #3498db;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }}
                button:hover {{
                    background-color: #2980b9;
                }}
                .refresh-btn {{
                    margin-top: 20px;
                    background-color: #27ae60;
                }}
                .refresh-btn:hover {{
                    background-color: #229954;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌤️ Weather Station</h1>
                
                <img src="https://storage.googleapis.com/{BUCKET_NAME}/latest_weather_image.jpg?t={datetime.now().timestamp()}" 
                     alt="Latest Weather Image" 
                     class="weather-image"
                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlIEF2YWlsYWJsZTwvdGV4dD48L3N2Zz4='; this.alt='No image available';">
                
                <div class="classification">
                    Weather Condition: {classification.replace('_', ' ')}
                </div>
                
                <div class="timestamp">
                    Last Updated: {timestamp}
                </div>
                
                <button class="refresh-btn" onclick="location.reload()">
                    🔄 Refresh
                </button>
                
                <div class="upload-section">
                    <h3>Upload New Weather Image</h3>
                    <form class="upload-form" enctype="multipart/form-data" method="post">
                        <input type="file" name="image" accept="image/*" required>
                        <button type="submit">📸 Upload & Analyze</button>
                    </form>
                </div>
            </div>
            
            <script>
                // Auto-refresh every 5 minutes
                setTimeout(() => location.reload(), 300000);
            </script>
        </body>
        </html>
        """
        
        headers['Content-Type'] = 'text/html'
        return (html_content, 200, headers)
        
    except Exception as e:
        print(f"Webpage error: {str(e)}")
        error_html = f"""
        <html><body>
        <h1>Weather Station</h1>
        <p>Error loading weather data: {str(e)}</p>
        </body></html>
        """
        headers['Content-Type'] = 'text/html'
        return (error_html, 500, headers)