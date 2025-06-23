import functions_framework
from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Image as VertexImage
import base64
import json
from datetime import datetime
import os

# Initialize Vertex AI - USE US-CENTRAL1 for better model availability
PROJECT_ID = os.environ.get('PROJECT_ID')
LOCATION = os.environ.get('LOCATION', 'us-central1')  # Changed default to us-central1
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
    
    # Log image size for debugging
    print(f"Image size: {len(image_data)} bytes")
    print(f"Image filename: {image_file.filename}")
    
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
        # Validate image data
        if not image_data or len(image_data) == 0:
            print("Error: Empty image data")
            return 'error'
        
        print(f"Processing image of size: {len(image_data)} bytes")
        print(f"Using Vertex AI location: {LOCATION}")
        
        # Create Vertex AI image object directly from bytes
        vertex_image = VertexImage.from_bytes(image_data)
        
        # Updated model names - prioritize models most likely to be available
        model_names = [
            # Gemini 2.0 models (newer, more likely to be available)
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            # Gemini 1.5 models (older but stable)
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-001", 
            "gemini-1.5-flash",
            "gemini-1.5-pro-002",
            "gemini-1.5-pro-001",
            "gemini-1.5-pro",
            # Legacy models as final fallback
            "gemini-1.0-pro-vision-001",
            "gemini-1.0-pro-vision",
            "gemini-pro-vision"
        ]
        
        model = None
        selected_model = None
        last_error = None
        
        for model_name in model_names:
            try:
                print(f"Attempting to initialize model: {model_name}")
                model = GenerativeModel(model_name)
                selected_model = model_name
                print(f"Successfully initialized model: {model_name}")
                break
            except Exception as model_error:
                print(f"Failed to initialize {model_name}: {str(model_error)}")
                last_error = model_error
                continue
        
        if model is None:
            print(f"Failed to initialize any model. Last error: {str(last_error)}")
            print(f"Current location: {LOCATION}")
            print("Consider switching to us-central1 region for better model availability")
            return 'error'
        
        # Improved prompt with more detailed instructions
        prompt = """You are a weather classification expert. Analyze this image carefully and classify the weather conditions based on what you observe.

Look specifically for:
- Sky conditions (clear, cloudy, overcast)
- Lighting conditions (bright sun, dim light, artificial light)
- Precipitation (rain, snow, visible water drops)
- Atmospheric conditions (fog, mist, haze)
- Time of day indicators

Classify the weather as exactly ONE of these labels:
- sunny (bright, clear skies, strong sunlight)
- partly_cloudy (some clouds but still bright)
- overcast (gray, cloudy skies covering most of the sky)
- raining (visible rain, wet surfaces, rain drops)
- snowing (visible snow, snowy conditions)
- foggy (reduced visibility, fog or mist)
- night (dark, nighttime conditions)
- dawn (early morning light, sunrise)
- dusk (evening light, sunset)

Important: Respond with ONLY the weather label. Do not include any explanation or additional text."""
        
        # Generate classification with better error handling
        try:
            print(f"Generating content with model: {selected_model}")
            response = model.generate_content(
                [prompt, vertex_image],
                generation_config={
                    "max_output_tokens": 50,
                    "temperature": 0.1,
                    "top_p": 0.8,
                }
            )
            
            if not response or not response.text:
                print("Empty response from model")
                return 'error'
                
        except Exception as generation_error:
            print(f"Generation failed with {selected_model}: {str(generation_error)}")
            # If generation fails, try with a different model
            if "gemini-1.5" in selected_model or "gemini-1.0" in selected_model:
                print("Trying fallback generation approach...")
                try:
                    response = model.generate_content([prompt, vertex_image])
                except Exception as fallback_error:
                    print(f"Fallback generation also failed: {str(fallback_error)}")
                    return 'error'
            else:
                return 'error'
        
        # Extract and clean the response
        classification = response.text.strip().lower()
        print(f"Raw model response: '{response.text}'")
        print(f"Cleaned classification: '{classification}'")
        
        # Validate classification against allowed labels
        valid_labels = ['sunny', 'partly_cloudy', 'overcast', 'raining', 'snowing', 'foggy', 'night', 'dawn', 'dusk']
        
        # Handle variations in response
        classification_mapping = {
            'clear': 'sunny',
            'cloudy': 'partly_cloudy',
            'partially_cloudy': 'partly_cloudy',
            'partial_cloudy': 'partly_cloudy',
            'rain': 'raining',
            'snow': 'snowing',
            'mist': 'foggy',
            'misty': 'foggy',
            'morning': 'dawn',
            'evening': 'dusk',
            'sunset': 'dusk',
            'sunrise': 'dawn'
        }
        
        # Check if classification needs mapping
        if classification in classification_mapping:
            classification = classification_mapping[classification]
        
        # Final validation
        if classification not in valid_labels:
            print(f"Invalid classification received: '{classification}'. Defaulting to 'unknown'")
            classification = 'unknown'
        
        print(f"Final classification: {classification} using model: {selected_model}")
        
        # Store the model used for later reference
        classify_weather_image.last_model = selected_model
        
        return classification
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
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
            'timestamp': datetime.now().isoformat(),
            'image_size': len(image_data),
            'model_used': getattr(classify_weather_image, 'last_model', 'unknown'),
            'vertex_location': LOCATION
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
            image_size = metadata.get('image_size', 'Unknown')
            model_used = metadata.get('model_used', 'Unknown')
            vertex_location = metadata.get('vertex_location', 'Unknown')
        except:
            classification = 'No data available'
            timestamp = 'Unknown'
            image_size = 'Unknown'
            model_used = 'Unknown'
            vertex_location = 'Unknown'
        
        # Generate HTML with debug info
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
                .debug-info {{
                    font-size: 12px;
                    color: #7f8c8d;
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                    text-align: left;
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
                .status-indicator {{
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 5px;
                }}
                .status-success {{
                    background-color: #27ae60;
                }}
                .status-error {{
                    background-color: #e74c3c;
                }}
                .status-unknown {{
                    background-color: #f39c12;
                }}
                .warning {{
                    background-color: #fff3cd;
                    color: #856404;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                    border-left: 4px solid #ffc107;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌤️ Weather Station</h1>
                
                {"<div class='warning'>⚠️ Note: For best results, ensure your Cloud Function is deployed in us-central1 region for optimal model availability.</div>" if vertex_location != 'us-central1' else ""}
                
                <img src="https://storage.googleapis.com/{BUCKET_NAME}/latest_weather_image.jpg?t={datetime.now().timestamp()}" 
                     alt="Latest Weather Image" 
                     class="weather-image"
                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlIEF2YWlsYWJsZTwvdGV4dD48L3N2Zz4='; this.alt='No image available';">
                
                <div class="classification">
                    <span class="status-indicator status-{('success' if classification not in ['error', 'unknown', 'No data available'] else 'error' if classification == 'error' else 'unknown')}"></span>
                    Weather Condition: {classification.replace('_', ' ')}
                </div>
                
                <div class="debug-info">
                    <strong>Debug Information:</strong><br>
                    Image Size: {image_size} bytes<br>
                    Classification: {classification}<br>
                    Model Used: {model_used}<br>
                    Vertex Location: {vertex_location}<br>
                    Timestamp: {timestamp}
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
                    
                    <div style="margin-top: 15px; font-size: 12px; color: #666;">
                        <strong>Tips for better results:</strong><br>
                        • Use outdoor images with visible sky<br>
                        • Ensure good image quality and lighting<br>
                        • Avoid images that are too dark or blurry<br>
                        • Deploy Cloud Function in us-central1 for best model availability
                    </div>
                </div>
            </div>
            
            <script>
                // Auto-refresh every 5 minutes
                setTimeout(() => location.reload(), 300000);
                
                // Show upload progress
                document.querySelector('form').addEventListener('submit', function(e) {{
                    const button = this.querySelector('button');
                    button.textContent = '⏳ Analyzing...';
                    button.disabled = true;
                }});
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