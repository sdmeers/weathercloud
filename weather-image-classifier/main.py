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
import time

# Initialize Vertex AI - USE EUROPE-WEST2 (London) for gemini-2.0-flash-lite
PROJECT_ID = os.environ.get('PROJECT_ID')
LOCATION = os.environ.get('LOCATION', 'europe-west1')
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
    
    # Generate unique image hash BEFORE classification
    # Add timestamp to ensure uniqueness even for identical images
    timestamp_ms = int(time.time() * 1000)
    combined_data = image_data + str(timestamp_ms).encode()
    image_hash = hashlib.md5(combined_data).hexdigest()
          
    # Classify and get both classification and model name
    result = classify_weather_image(image_data, image_hash)
    
    # Handle the returned tuple or single value for backward compatibility
    if isinstance(result, tuple):
        classification, model_name = result
    else:
        classification = result
        model_name = "unknown"
    
    # Store image and classification in Cloud Storage
    store_results(image_data, classification, image_hash, model_name, timestamp_ms)

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
        
        # Return both classification and model name for storage
        return classification, model_name
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return 'error', 'unknown'
    
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

def store_results(image_data, classification, image_hash, model_name, timestamp_ms):
    """
    Improved atomic storage operations with better error handling and cleanup.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Generate unique filenames with timestamp
        img_name = f"{image_hash}_{timestamp_ms}.jpg"
        meta_name = f"{image_hash}_{timestamp_ms}.json"
        
        # Prepare metadata with additional cache-busting info
        meta = {
            "classification": classification,
            "timestamp": datetime.now().isoformat(),
            "timestamp_ms": timestamp_ms,
            "image_size": len(image_data),
            "image_hash_full": image_hash,
            "image_filename": img_name,
            "model_used": model_name,
            "vertex_location": LOCATION,
            "cache_buster": str(uuid.uuid4())  # Additional uniqueness
        }

        # Get list of ALL existing files before uploading new ones
        existing_blobs = []
        try:
            existing_blobs = list(bucket.list_blobs())
            print(f"Found {len(existing_blobs)} existing files in bucket")
        except Exception as list_error:
            print(f"Warning: Could not list existing blobs: {list_error}")

        # Upload new files first
        print(f"Uploading new files: {img_name}, {meta_name}")
        
        # Upload with explicit cache control headers
        img_blob = bucket.blob(img_name)
        img_blob.cache_control = "no-cache, no-store, must-revalidate"
        img_blob.upload_from_string(image_data, content_type="image/jpeg")
        
        meta_blob = bucket.blob(meta_name)
        meta_blob.cache_control = "no-cache, no-store, must-revalidate"
        meta_blob.upload_from_string(json.dumps(meta), content_type="application/json")

        # Update pointer atomically
        pointer_data = {
            "latest": image_hash,
            "timestamp_ms": timestamp_ms,
            "img_filename": img_name,
            "meta_filename": meta_name,
            "updated_at": datetime.now().isoformat()
        }
        
        pointer_blob = bucket.blob("latest_pointer.json")
        pointer_blob.cache_control = "no-cache, no-store, must-revalidate"
        pointer_blob.upload_from_string(json.dumps(pointer_data), content_type="application/json")
        
        print(f"Updated pointer to: {image_hash}")

        # Clean up old files (keep only the latest)
        files_to_delete = []
        for blob in existing_blobs:
            # Don't delete the pointer file or the files we just uploaded
            if (blob.name != "latest_pointer.json" and 
                blob.name != img_name and 
                blob.name != meta_name):
                files_to_delete.append(blob)

        if files_to_delete:
            try:
                bucket.delete_blobs(files_to_delete)
                print(f"Cleaned up {len(files_to_delete)} old files")
            except Exception as cleanup_error:
                print(f"Warning: Cleanup failed: {cleanup_error}")

        print(f"Successfully stored new image with classification: {classification}")

    except Exception as e:
        print(f"Storage error: {e}")
        import traceback
        print(f"Storage traceback: {traceback.format_exc()}")
        raise

def display_webpage(headers):
    """Display webpage with improved cache handling and error recovery."""
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Check if bucket is empty or has any files
        try:
            all_blobs = list(bucket.list_blobs(max_results=1))
            if not all_blobs:
                print("Bucket is completely empty")
                return display_empty_bucket_page(headers)
        except Exception as e:
            print(f"Could not check bucket contents: {e}")
            return display_empty_bucket_page(headers)

        # Try to read pointer file
        try:
            pointer_blob = bucket.blob("latest_pointer.json")
            if not pointer_blob.exists():
                print("Pointer file does not exist")
                return display_empty_bucket_page(headers)
                
            pointer = json.loads(pointer_blob.download_as_text())
            latest_hash = pointer.get("latest")
            timestamp_ms = pointer.get("timestamp_ms", "")
            img_filename = pointer.get("img_filename")
            meta_filename = pointer.get("meta_filename")

            if not latest_hash or not img_filename or not meta_filename:
                print("Pointer file is incomplete")
                return display_empty_bucket_page(headers)

            # Check if referenced files actually exist
            img_blob = bucket.blob(img_filename)
            meta_blob = bucket.blob(meta_filename)
            
            if not img_blob.exists() or not meta_blob.exists():
                print(f"Referenced files don't exist: {img_filename}, {meta_filename}")
                return display_empty_bucket_page(headers)

            # Load metadata
            metadata = json.loads(meta_blob.download_as_text())
            
            classification = metadata.get("classification", "Unknown")
            timestamp = metadata.get("timestamp", "Unknown")
            image_size = metadata.get("image_size", "Unknown")
            model_used = metadata.get("model_used", "Unknown")
            vertex_location = metadata.get("vertex_location", "Unknown")

            # Create cache-busted image URL
            cache_buster = int(time.time())
            image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{img_filename}?cb={cache_buster}"

        except Exception as e:
            print(f"Error loading current data: {e}")
            return display_empty_bucket_page(headers)

        # Generate HTML with strong cache prevention
        html_content = generate_main_html(
            image_url, classification, timestamp, 
            image_size, model_used, vertex_location, latest_hash
        )
        
        # Strong cache prevention headers
        headers.update({
            'Content-Type': 'text/html',
            'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Last-Modified': datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT'),
            'ETag': f'"{latest_hash}-{int(time.time())}"'
        })
        
        return (html_content, 200, headers)
        
    except Exception as e:
        print(f"Webpage error: {str(e)}")
        return display_error_page(str(e), headers)

def display_empty_bucket_page(headers):
    """Display page when bucket is empty or has no valid data."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Weather Image Classifier</title>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap">
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;700&display=swap">
        <script src="https://kit.fontawesome.com/e1d7788428.js" crossorigin="anonymous"></script>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: #ffffff;
                font-family: "Raleway", sans-serif;
            }}
            .w3-main {{
                padding-top: 44px;
            }}
            main {{
                max-width: 800px;
                margin: 40px auto 40px;
                padding: 0 16px;
                text-align: center;
            }}
            .empty-state {{ padding: 40px; background: #f8f9fa; border-radius: 8px; margin: 20px 0; }}
            .upload-section {{ margin-top: 40px; padding: 24px; border: 1px dashed #bbb; border-radius: 8px; }}
            .upload-section input[type=file] {{ margin-bottom: 12px; }}
            .upload-section button {{ background:#007bff; color:#fff; border:none; padding:10px 22px; border-radius:5px; cursor:pointer; }}
            .upload-section button:hover {{ background:#0069d9; }}
            
            /* --- Start of new navbar CSS --- */
            .common_navbar {{
                width: 100%;
                background-color: black;
                color: white;
                padding: 8px 16px;
                z-index: 1001;
                position: fixed;
                top: 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
                height: 44px;
                font-family: 'Roboto', sans-serif;
                box-sizing: border-box;
            }}
            .hamburger-button {{ background: none; border: none; color: white; font-size: 22px; cursor: pointer; }}
            .navbar-title {{ font-size: 18px; font-weight: 400; }}
            .navbar-title i {{ margin-right: 8px; }}
            .sidenav {{
                height: 100%;
                width: 200px;
                position: fixed;
                z-index: 1002;
                top: 0;
                left: 0;
                background-color: #111;
                overflow-x: hidden;
                transform: translateX(-100%);
                transition: transform 0.3s ease-in-out;
            }}
            @media screen and (min-width: 600px) {{
                .sidenav {{
                    width: 280px;
                }}
            }}
            .sidenav-open {{ transform: translateX(0); }}
            .sidenav-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 20px;
                border-bottom: 1px solid #444;
                min-height: 44px;
            }}
            .sidenav-title {{ color: white; font-size: 20px; margin: 0; font-weight: 500; }}
            .close-btn {{ background: none; border: none; color: #818181; font-size: 22px; cursor: pointer; }}
            .close-btn:hover {{ color: #f1f1f1; }}
            .sidenav a {{
                padding: 10px 15px 10px 20px;
                text-decoration: none;
                font-size: 18px;
                color: #818181;
                display: block;
                transition: 0.3s;
                text-align: left;
            }}
            .sidenav a:hover {{ color: #f1f1f1; }}
            .sidenav .fa-fw {{ margin-right: 8px; }}
            /* --- End of new navbar CSS --- */
        </style>
    </head>
    <body>
        <!-- Sidenav/menu -->
        <nav class="sidenav" id="mySidenav">
          <div class="sidenav-header">
            <h4 class="sidenav-title">Navigation</h4>
            <button class="close-btn" id="close-sidenav-btn"><i class="fa-solid fa-xmark"></i></button>
          </div>
          <a href="https://weather-dashboard-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-dashboard fa-fw"></i>  Summary</a>
          <a href="https://interactive-dashboard-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-magnifying-glass-chart fa-fw"></i>  Dashboard</a>
          <a href="https://weather-chat-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-comments fa-fw"></i>  Chatbot</a>
          <a href="https://display-weather-data-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-database fa-fw"></i>  Data</a>
          <a href="https://europe-west1-weathercloud-460719.cloudfunctions.net/weather-image-classifier" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-camera-retro fa-fw"></i>  Image Classifier</a>
        </nav>

        <!-- Top container -->
        <div class="common_navbar">
          <button class="hamburger-button" id="hamburger-btn">
            <i class="fa fa-bars"></i>
          </button>
          <span class="navbar-title"><i class="fa-solid fa-camera-retro fa-fw"></i> Image Classifier</span>
        </div>

        <div class="w3-main">
            <main>
                <h1 style="font-weight:600; font-size:32px;">Weather Image Classifier</h1>
                
                <div class="empty-state">
                    <i class="fa-solid fa-cloud" style="font-size: 48px; color: #6c757d; margin-bottom: 20px;"></i>
                    <h3>No weather images yet</h3>
                    <p>Upload your first weather photo to get started!</p>
                </div>

                <div class="upload-section">
                    <h3 style="margin-top:0;">Upload a weather photo</h3>
                    <form enctype="multipart/form-data" method="post">
                        <input type="file" name="image" accept="image/*" required>
                        <br>
                        <button type="submit"><i class="fa fa-upload"></i>&nbsp;Upload&nbsp;&amp;&nbsp;Analyze</button>
                    </form>
                </div>
            </main>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const hamburgerBtn = document.getElementById('hamburger-btn');
                const sidenav = document.getElementById('mySidenav');
                const closeSidenavBtn = document.getElementById('close-sidenav-btn');
                hamburgerBtn.addEventListener('click', function() {{ sidenav.classList.add('sidenav-open'); }});
                closeSidenavBtn.addEventListener('click', function() {{ sidenav.classList.remove('sidenav-open'); }});
                document.addEventListener('click', function(event) {{
                    if (!sidenav.contains(event.target) && !hamburgerBtn.contains(event.target)) {{
                        sidenav.classList.remove('sidenav-open');
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    headers.update({
        'Content-Type': 'text/html',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
    
    return (html_content, 200, headers)

def display_error_page(error_message, headers):
    """Display error page."""
    html_content = f"""
    <html><body style="font-family: Arial, sans-serif; text-align: center; margin-top: 100px;">
    <h1>Weather Image Classifier</h1>
    <p style="color: red;">Error loading weather data: {error_message}</p>
    <p>Please check your Cloud Storage bucket and try again.</p>
    <a href="javascript:location.reload()">Refresh Page</a>
    </body></html>
    """
    
    headers.update({
        'Content-Type': 'text/html',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
    
    return (html_content, 500, headers)

def generate_main_html(image_url, classification, timestamp, image_size, model_used, vertex_location, image_hash):
    """Generate the main HTML content with cache-busting."""
    cache_buster = int(time.time())
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Weather Image Classifier</title>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">

        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap">
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;700&display=swap">
        <script src="https://kit.fontawesome.com/e1d7788428.js" crossorigin="anonymous"></script>

        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: #ffffff;
                font-family: "Raleway", sans-serif;
            }}
            .w3-main {{
                padding-top: 44px;
            }}
            main {{
                max-width: 800px;
                margin: 40px auto 40px;
                padding: 0 16px;
                text-align: center;
            }}
            .weather-image {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.15); }}
            .classification-result {{ margin-top: 16px; font-size: 20px; font-weight: 500; color: #333; }}
            .timestamp {{ margin-top: 6px; font-size: 13px; color: #666; }}
            .upload-section {{ margin-top: 40px; padding: 24px; border: 1px dashed #bbb; border-radius: 8px; }}
            .upload-section input[type=file] {{ margin-bottom: 12px; }}
            .upload-section button {{ background:#007bff; color:#fff; border:none; padding:10px 22px; border-radius:5px; cursor:pointer; }}
            .upload-section button:hover {{ background:#0069d9; }}
            
            /* --- Start of new navbar CSS --- */
            .common_navbar {{
                width: 100%;
                background-color: black;
                color: white;
                padding: 8px 16px;
                z-index: 1001;
                position: fixed;
                top: 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
                height: 44px;
                font-family: 'Roboto', sans-serif;
                box-sizing: border-box;
            }}
            .hamburger-button {{ background: none; border: none; color: white; font-size: 22px; cursor: pointer; }}
            .navbar-title {{ font-size: 18px; font-weight: 400; }}
            .navbar-title i {{ margin-right: 8px; }}
            .sidenav {{
                height: 100%;
                width: 200px;
                position: fixed;
                z-index: 1002;
                top: 0;
                left: 0;
                background-color: #111;
                overflow-x: hidden;
                transform: translateX(-100%);
                transition: transform 0.3s ease-in-out;
            }}
            @media screen and (min-width: 600px) {{
                .sidenav {{
                    width: 280px;
                }}
            }}
            .sidenav-open {{ transform: translateX(0); }}
            .sidenav-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 20px;
                border-bottom: 1px solid #444;
                min-height: 44px;
            }}
            .sidenav-title {{ color: white; font-size: 20px; margin: 0; font-weight: 500; }}
            .close-btn {{ background: none; border: none; color: #818181; font-size: 22px; cursor: pointer; }}
            .close-btn:hover {{ color: #f1f1f1; }}
            .sidenav a {{
                padding: 10px 15px 10px 20px;
                text-decoration: none;
                font-size: 18px;
                color: #818181;
                display: block;
                transition: 0.3s;
                text-align: left;
            }}
            .sidenav a:hover {{ color: #f1f1f1; }}
            .sidenav .fa-fw {{ margin-right: 8px; }}
            /* --- End of new navbar CSS --- */
        </style>
    </head>
    <body>
        <!-- Sidenav/menu -->
        <nav class="sidenav" id="mySidenav">
          <div class="sidenav-header">
            <h4 class="sidenav-title">Navigation</h4>
            <button class="close-btn" id="close-sidenav-btn"><i class="fa-solid fa-xmark"></i></button>
          </div>
          <a href="https://weather-dashboard-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-dashboard fa-fw"></i>  Summary</a>
          <a href="https://interactive-dashboard-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-magnifying-glass-chart fa-fw"></i>  Dashboard</a>
          <a href="https://weather-chat-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-comments fa-fw"></i>  Chatbot</a>
          <a href="https://display-weather-data-728445650450.europe-west2.run.app/" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-database fa-fw"></i>  Data</a>
          <a href="https://europe-west1-weathercloud-460719.cloudfunctions.net/weather-image-classifier" class="w3-bar-item w3-button w3-padding"><i class="fa-solid fa-camera-retro fa-fw"></i>  Image Classifier</a>
        </nav>

        <!-- Top container -->
        <div class="common_navbar">
          <button class="hamburger-button" id="hamburger-btn">
            <i class="fa fa-bars"></i>
          </button>
          <span class="navbar-title"><i class="fa-solid fa-camera-retro fa-fw"></i> Image Classifier</span>
        </div>

        <div class="w3-main">
            <main>
                <h1 style="font-weight:600; font-size:32px;">Latest Weather Image</h1>

                <img src="{image_url}" alt="Latest Weather Image" class="weather-image"
                    onerror="this.style.display='none';" />

                <div class="classification-result">
                    Classification: {classification.replace('_',' ').title()}
                </div>

                <div class="timestamp" id="timestamp-display">Last updated: Unknown</div>

                <div class="upload-section">
                    <h3 style="margin-top:0;">Upload a new photo</h3>
                    <form enctype="multipart/form-data" method="post">
                        <input type="file" name="image" accept="image/*" required>
                        <br>
                        <button type="submit"><i class="fa fa-upload"></i>&nbsp;Upload&nbsp;&amp;&nbsp;Analyze</button>
                    </form>
                </div>
            </main>
        </div>

        <script>
            // Hamburger menu script
            document.addEventListener('DOMContentLoaded', function() {{
                const hamburgerBtn = document.getElementById('hamburger-btn');
                const sidenav = document.getElementById('mySidenav');
                const closeSidenavBtn = document.getElementById('close-sidenav-btn');
                hamburgerBtn.addEventListener('click', function() {{ sidenav.classList.add('sidenav-open'); }});
                closeSidenavBtn.addEventListener('click', function() {{ sidenav.classList.remove('sidenav-open'); }});
                document.addEventListener('click', function(event) {{
                    if (!sidenav.contains(event.target) && !hamburgerBtn.contains(event.target)) {{
                        sidenav.classList.remove('sidenav-open');
                    }}
                }});
            }});

            // Original page script
            const timestamp = "{timestamp}";
            if (timestamp !== "Unknown") {{
                try {{
                    const date = new Date(timestamp);
                    const formatted = date.toLocaleString('en-GB', {{
                        hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit', year: 'numeric', hour12: false
                    }}).replace(',', ',');
                    document.getElementById('timestamp-display').textContent = `Last updated: ${{formatted}}`;
                }} catch (e) {{
                    console.error('Error formatting timestamp:', e);
                    document.getElementById('timestamp-display').textContent = 'Last updated: Unknown';
                }}
            }}
            const img = document.querySelector('.weather-image');
            if (img) {{
                img.onerror = function() {{
                    console.log('Image failed to load, trying refresh...');
                    setTimeout(() => {{
                        this.src = this.src.split('?')[0] + '?cb=' + new Date().getTime();
                    }}, 1000);
                }};
            }}
        </script>
    </body>
    </html>
    """