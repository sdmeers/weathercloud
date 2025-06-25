import functions_framework
from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Image as VertexImage
import base64
import json
from datetime import datetime
import os
import hashlib
import uuid

# Initialize Vertex AI - USE EUROPE-WEST2 (London) for gemini-2.0-flash-lite
PROJECT_ID = os.environ.get('PROJECT_ID')
LOCATION = os.environ.get('LOCATION', 'europe-west1')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Simple in-memory cache for classifications
classification_cache = {}

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
    
    # Generate unique image hash BEFORE classification
    image_hash = hashlib.md5(image_data).hexdigest()
          
    # Store image and classification in Cloud Storage
    classification = classify_weather_image(image_data, image_hash)
    store_results(image_data, classification, image_hash)
    
    response_data = {
        'classification': classification,
        'timestamp': datetime.now().isoformat(),
        'status': 'success',
        'image_hash': image_hash,
    }
    
    return (json.dumps(response_data), 200, headers)

def classify_weather_image(image_data, image_hash):
    """Classify weather conditions in the image using Vertex AI."""
    
    try:
        # Validate image data
        if not image_data or len(image_data) == 0:
            print("Error: Empty image data")
            return 'error'
        
        # Check in-memory cache first (for same session)
        if image_hash in classification_cache:
            print(f"Using in-memory cached classification for image hash: {image_hash}")
            return classification_cache[image_hash]
        
        print(f"Processing image of size: {len(image_data)} bytes")
        print(f"Image hash: {image_hash}")
        print(f"Using Vertex AI location: {LOCATION}")
        
        # Create Vertex AI image object directly from bytes
        vertex_image = VertexImage.from_bytes(image_data)
        
        # Use gemini-2.0-flash-lite - available in europe-west2, cost-effective for image classification
        model_name = "gemini-2.0-flash-lite-001"
        
        try:
            print(f"Initializing model: {model_name}")
            model = GenerativeModel(model_name)
            print(f"Successfully initialized model: {model_name}")
        except Exception as model_error:
            print(f"Failed to initialize {model_name}: {str(model_error)}")
            print("Make sure you're using LOCATION='europe-west2' where gemini-2.0-flash-lite is available")
            return 'error'
        
        # Deterministic prompt with explicit constraints
        prompt = """Analyze this weather image and classify it with exactly ONE label.

CLASSIFICATION RULES:
1. Look at the overall lighting and sky conditions
2. Prioritize the most dominant weather feature
3. Use ONLY these exact labels (respond with just the label, nothing else):

sunny - bright sunlight, clear blue sky, strong shadows
partly_cloudy - mixed sun and clouds, still bright overall  
overcast - gray cloudy sky, no direct sunlight
raining - visible rain, wet surfaces, or heavy rain clouds
snowing - visible snow falling or snow-covered scene
foggy - reduced visibility from fog, mist, or haze
night - dark conditions, artificial lighting dominant
dawn - early morning light, sunrise colors
dusk - evening light, sunset colors

IMPORTANT: Respond with ONLY the single most appropriate label. No explanation."""
        
        # Use completely deterministic settings
        try:
            print(f"Generating content with model: {model_name}")
            response = model.generate_content(
                [prompt, vertex_image],
                generation_config={
                    "max_output_tokens": 10,
                    "temperature": 0,         # Completely deterministic
                    "top_p": 1,              # No nucleus sampling randomness
                    "top_k": 1,              # Always pick the top choice
                }
            )
            
            if not response or not response.text:
                print("Empty response from model")
                return 'error'
                
        except Exception as generation_error:
            print(f"Generation failed with {model_name}: {str(generation_error)}")
            return 'error'
        
        # Extract and clean the response
        classification = response.text.strip().lower()
        print(f"Raw model response: '{response.text}'")
        print(f"Cleaned classification: '{classification}'")
        
        # Validate and map classification
        classification = validate_and_map_classification(classification)
        
        print(f"Final classification: {classification} using model: {model_name}")
        
        # Cache the result in memory for this session
        classification_cache[image_hash] = classification
        
        # Store the model used for later reference
        classify_weather_image.last_model = model_name
        
        return classification
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return 'error'

def validate_and_map_classification(classification):
    """Validate and map classification to allowed labels."""
    
    valid_labels = ['sunny', 'partly_cloudy', 'overcast', 'raining', 'snowing', 'foggy', 'night', 'dawn', 'dusk']
    
    # Comprehensive mapping for edge cases
    classification_mapping = {
        'clear': 'sunny',
        'bright': 'sunny',
        'sunshine': 'sunny',
        'cloudy': 'partly_cloudy',
        'partially_cloudy': 'partly_cloudy',
        'partial_cloudy': 'partly_cloudy',
        'mixed': 'partly_cloudy',
        'scattered_clouds': 'partly_cloudy',
        'gray': 'overcast',
        'grey': 'overcast',
        'gloomy': 'overcast',
        'rain': 'raining',
        'wet': 'raining',
        'rainy': 'raining',
        'drizzle': 'raining',
        'snow': 'snowing',
        'snowy': 'snowing',
        'blizzard': 'snowing',
        'mist': 'foggy',
        'misty': 'foggy',
        'hazy': 'foggy',
        'fog': 'foggy',
        'morning': 'dawn',
        'evening': 'dusk',
        'sunset': 'dusk',
        'sunrise': 'dawn',
        'dark': 'night',
        'nighttime': 'night',
        'twilight': 'dusk'
    }
    
    # Check if classification needs mapping
    if classification in classification_mapping:
        classification = classification_mapping[classification]
    
    # Handle multi-word responses by taking first valid word
    words = classification.split()
    for word in words:
        if word in valid_labels:
            classification = word
            break
    
    # Final validation
    if classification not in valid_labels:
        print(f"Invalid classification received: '{classification}'. Defaulting to 'unknown'")
        classification = 'unknown'
    
    return classification

def store_results(image_data, classification, image_hash):
    """
    1. Write <hash>.jpg   (e.g. 79939b898ac5….jpg)
    2. Write <hash>.json
    3. Overwrite latest_pointer.json  → {"latest": "<hash>"}
    4. Sweep the bucket and delete *every* object that does not belong
       to <hash> and is not latest_pointer.json

    After a successful upload the bucket will always contain exactly:
        <hash>.jpg   <hash>.json   latest_pointer.json
    """
    try:
        client  = storage.Client()
        bucket  = client.bucket(BUCKET_NAME)

        # ---------------- 1) upload image ----------------------------------
        img_name  = f"{image_hash}.jpg"
        bucket.blob(img_name).upload_from_string(
            image_data, content_type="image/jpeg"
        )

        # ---------------- 2) upload metadata -------------------------------
        meta_name = f"{image_hash}.json"
        meta      = {
            "classification":  classification,
            "timestamp":       datetime.now().isoformat(),
            "image_size":      len(image_data),
            "image_hash_full": image_hash,
            "image_filename":  img_name,
            "model_used":      getattr(classify_weather_image, "last_model", "unknown"),
            "vertex_location": LOCATION,
        }
        bucket.blob(meta_name).upload_from_string(
            json.dumps(meta), content_type="application/json"
        )

        # ---------------- 3) switch the pointer ----------------------------
        bucket.blob("latest_pointer.json").upload_from_string(
            json.dumps({"latest": image_hash}), content_type="application/json"
        )

        # ---------------- 4) sweep & delete stale objects ------------------
        try:
            keep_prefix  = image_hash                  # current hash
            keep_pointer = "latest_pointer.json"
            stale_blobs  = []

            for blob in bucket.list_blobs():
                name = blob.name
                if name == keep_pointer:            # keep pointer file
                    continue
                if name.startswith(keep_prefix):    # keep latest <hash>.jpg/.json
                    continue
                stale_blobs.append(blob)

            if stale_blobs:
                bucket.delete_blobs(stale_blobs)
                print(
                    f"Deleted {len(stale_blobs)} obsolete file(s): "
                    + ", ".join(b.name for b in stale_blobs)
                )
        except Exception as cleanup_err:
            print(f"Warning: bucket sweep failed — {cleanup_err}")

        print(f"Stored new pair ({img_name}, {meta_name}) with label '{classification}'")

    except Exception as e:
        print(f"Storage error: {e}")
        raise

def display_webpage(headers):
    """Display a simple webpage with the latest image and classification."""
    try:
        # 1) Set up Cloud Storage client ------------------------------------
        client  = storage.Client()
        bucket  = client.bucket(BUCKET_NAME)

        # 2) Read the tiny pointer file to find the current hash ------------
        try:
            pointer_blob = bucket.blob("latest_pointer.json")
            pointer      = json.loads(pointer_blob.download_as_text())
            latest_hash  = pointer.get("latest")          # e.g. "c7779055384ee0..."

            if not latest_hash:
                raise ValueError("Pointer file missing 'latest' field")

            # 3) Load the matching metadata file ----------------------------
            meta_blob  = bucket.blob(f"{latest_hash}.json")
            metadata   = json.loads(meta_blob.download_as_text())

            classification   = metadata.get("classification", "No data available")
            timestamp        = metadata.get("timestamp",       "Unknown")
            image_size       = metadata.get("image_size",      "Unknown")
            image_hash       = metadata.get("image_hash_full", latest_hash)
            image_filename   = f"{latest_hash}.jpg"
            model_used       = metadata.get("model_used",      "Unknown")
            vertex_location  = metadata.get("vertex_location", "Unknown")

            # 4) Public URL for the JPEG (no cache-buster needed; filename is unique)
            image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{image_filename}"

        except Exception as e:
            # If the pointer or metadata doesn't exist yet, fall back to placeholders
            print(f"display_webpage: could not load latest data — {e}")

            classification   = "No data available"
            timestamp        = "Unknown"
            image_size       = "Unknown"
            image_hash       = "Unknown"
            image_filename   = ""
            model_used       = "Unknown"
            vertex_location  = "Unknown"
            image_url        = ""  # no image to display

        # Generate HTML with improved image loading
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Weather Station</title>
            <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
            <meta http-equiv="Pragma" content="no-cache">
            <meta http-equiv="Expires" content="0">
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
                .improvement-note {{
                    background-color: #fff3cd;
                    color: #856404;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 15px 0;
                    border-left: 4px solid #ffc107;
                    text-align: left;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌤️ Weather Station</h1>
                
                <div class="improvement-note">
                    <strong>🎯 Simplified Model Strategy:</strong><br>
                    • Now uses only gemini-2.0-flash-lite (cost-effective, reliable)<br>
                    • Deployed in europe-west2 (London) region for model availability<br>
                    • No more complex fallback logic - single, known-working model<br>
                    • Should consistently classify images without errors
                </div>
                
                <img src="{image_url}" 
                     alt="Latest Weather Image" 
                     class="weather-image"
                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlIEF2YWlsYWJsZTwvdGV4dD48L3N2Zz4='; this.alt='No image available';">
                
                <div class="classification">
                    <span class="status-indicator status-{('success' if classification not in ['error', 'unknown', 'No data available'] else 'error' if classification == 'error' else 'unknown')}"></span>
                    Weather Condition: {classification.replace('_', ' ')}
                </div>
                
                <div class="debug-info">
                    <strong>Debug Information:</strong><br>
                    Image Hash: {image_hash}<br>
                    Image Size: {image_size} bytes<br>
                    Classification: {classification}<br>
                    Model Used: {model_used}<br>
                    Vertex Location: {vertex_location}<br>
                    Image URL: {image_filename}<br>
                    Timestamp: {timestamp}
                </div>
                
                <div class="timestamp">
                    Last Updated: {timestamp}
                </div>
                
                <button class="refresh-btn" onclick="window.location.reload(true)">
                    🔄 Hard Refresh (Force reload)
                </button>
                
                <div class="upload-section">
                    <h3>Upload New Weather Image</h3>
                    <form class="upload-form" enctype="multipart/form-data" method="post">
                        <input type="file" name="image" accept="image/*" required>
                        <button type="submit">📸 Upload & Analyze</button>
                    </form>
                    
                    <div style="margin-top: 15px; font-size: 12px; color: #666;">
                        <strong>Expected Behavior:</strong><br>
                        • Uses gemini-2.0-flash-lite model (cost-effective and reliable)<br>
                        • Deployed in europe-west2 (London) region<br>
                        • Should consistently classify images without model errors<br>
                        • Same image uploaded again will use stored result
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        headers['Content-Type'] = 'text/html'
        headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        headers['Pragma'] = 'no-cache'
        headers['Expires'] = '0'
        
        return (html_content, 200, headers)
        
    except Exception as e:
        print(f"Webpage error: {str(e)}")
        error_html = f"""
        <html><body>
        <h1>Weather Station</h1>
        <p>Error loading weather data: {str(e)}</p>
        <p>Please check your Cloud Storage bucket and try again.</p>
        </body></html>
        """
        headers['Content-Type'] = 'text/html'
        return (error_html, 500, headers)