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
                <title>Weather Image Classifier</title>

                <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap">
                <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;700&display=swap">
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                <script src="https://kit.fontawesome.com/e1d7788428.js" crossorigin="anonymous"></script>

                <style>
                    /* ========= page-specific tweaks ========= */
                    html, body {{
                        margin: 0;
                        padding: 0;
                        background: #ffffff;
                        font-family: "Raleway", sans-serif;
                    }}

                    /* keep content from sliding under the fixed nav */
                    main {{
                        max-width: 800px;
                        margin: 80px auto 40px;   /* 44 px nav + extra */
                        padding: 0 16px;
                        text-align: center;
                    }}

                    .weather-image {{
                        max-width: 100%;
                        height: auto;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
                    }}

                    .classification-result {{
                        margin-top: 16px;
                        font-size: 20px;
                        font-weight: 500;
                        color: #333;
                        font-family: "Raleway", sans-serif;
                    }}

                    .timestamp {{
                        margin-top: 6px;
                        font-size: 13px;
                        color: #666;
                    }}

                    .upload-section {{
                        margin-top: 40px;
                        padding: 24px;
                        border: 1px dashed #bbb;
                        border-radius: 8px;
                    }}
                    .upload-section input[type=file] {{
                        margin-bottom: 12px;
                    }}
                    .upload-section button {{
                        background:#007bff;
                        color:#fff;
                        border:none;
                        padding:10px 22px;
                        border-radius:5px;
                        cursor:pointer;
                    }}
                    .upload-section button:hover {{ background:#0069d9; }}

                    .common_navbar {{
                        width: 100%;
                        background-color: black;
                        color: white;
                        padding: 0 16px;
                        z-index: 4;
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        height: 44px;
                        font-family: 'Roboto', sans-serif;
                        box-sizing: border-box;
                    }}
                    .common_navbar a {{
                        text-decoration: none;
                        color: white !important;
                        font-family: 'Roboto', sans-serif;
                        font-size: 18px;
                        font-weight: 400;
                        display: flex;
                        align-items: center;
                        white-space: nowrap;
                        line-height: 1;
                    }}
                    .common_navbar a:hover {{
                        color: #ccc !important;
                    }}
                    .common_navbar i {{
                        margin-right: 5px;
                        font-size: 18px;
                        line-height: 1;
                    }}
                </style>
            </head>
            <body class="w3-white">

                <!-- =========  shared navigation bar  ========= -->
                <div class="common_navbar">
                    <a href="https://interactive-dashboard-728445650450.europe-west2.run.app/">
                        <i class="fa-solid fa-magnifying-glass-chart"></i>&nbsp;Interactive&nbsp;Dashboard
                    </a>
                    <a href="https://weather-chat-728445650450.europe-west2.run.app/">
                        <i class="fa-solid fa-comments"></i>&nbsp;Chatbot
                    </a>
                </div>

                <!-- =========  main content  ========= -->
                <main>
                    <h1 style="font-weight:600; font-size:32px;">Latest Weather Image</h1>

                    <img src="{image_url}" alt="Latest Weather Image" class="weather-image"
                        onerror="this.style.display='none';" />

                    <div class="classification-result">
                        Classification: {classification.replace('_',' ').title()}
                    </div>

                    <div class="timestamp">Last updated: {timestamp}</div>

                    <!-- =====  upload form  ===== -->
                    <div class="upload-section">
                        <h3 style="margin-top:0;">Upload a new photo</h3>
                        <form enctype="multipart/form-data" method="post">
                            <input type="file" name="image" accept="image/*" required>
                            <br>
                            <button type="submit"><i class="fa fa-upload"></i>&nbsp;Upload&nbsp;&amp;&nbsp;Analyze</button>
                        </form>
                    </div>
                </main>
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