#!/usr/bin/env python3
"""
Simple test script to upload an image to the weather classification Cloud Function.
Usage: python test_upload.py path/to/your/image.jpg
"""

import requests
import sys
import os
import json

# Configuration - Update this with your Cloud Function URL after deployment
#CLOUD_FUNCTION_URL = "https://europe-west2-weathercloud-460719.cloudfunctions.net/weather-image-classifier"
CLOUD_FUNCTION_URL = "https://europe-west1-weathercloud-460719.cloudfunctions.net/weather-image-classifier"
#CLOUD_FUNCTION_URL = "https://us-central1-weathercloud-460719.cloudfunctions.net/weather-image-classifier"

def test_image_upload(image_path):
    """Upload a test image to the Cloud Function."""
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    
    # Check if it's likely an image file
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"⚠️  Warning: File doesn't appear to be an image: {image_path}")
    
    try:
        print(f"📤 Uploading image: {image_path}")
        print(f"🌐 Cloud Function URL: {CLOUD_FUNCTION_URL}")
        
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            
            print("⏳ Sending request...")
            response = requests.post(
                CLOUD_FUNCTION_URL, 
                files=files,
                timeout=120  # Give it 2 minutes for processing
            )
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                classification = result.get('classification', 'unknown')
                timestamp = result.get('timestamp', 'unknown')
                
                print("✅ SUCCESS!")
                print(f"🌤️  Weather Classification: {classification}")
                print(f"⏰ Timestamp: {timestamp}")
                print(f"🌐 View results at: {CLOUD_FUNCTION_URL}")
                
                return True
                
            except json.JSONDecodeError:
                print("✅ Upload successful, but response wasn't JSON:")
                print(response.text)
                return True
                
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The function might still be processing.")
        print("Check the Cloud Function logs in GCP Console.")
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Check your Cloud Function URL and internet connection.")
        return False
        
    except Exception as e:
        print(f"❌ Error uploading image: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments."""
    
    print("🌤️  Weather Classification Test Script")
    print("=" * 50)
    
    # Check if image path was provided
    if len(sys.argv) != 2:
        print("Usage: python test_upload.py <image_path>")
        print("\nExample:")
        print("  python test_upload.py sunny_day.jpg")
        print("  python test_upload.py /path/to/cloudy_sky.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if Cloud Function URL is configured
    if "your-region-your-project" in CLOUD_FUNCTION_URL:
        print("⚠️  WARNING: Please update CLOUD_FUNCTION_URL in this script!")
        print("   Get your URL from: gcloud functions describe weather-classifier --region=your-region")
        print("   Or check the GCP Console > Cloud Functions")
        print()
    
    # Upload and test
    success = test_image_upload(image_path)
    
    if success:
        print("\n🎉 Test completed successfully!")
        print(f"🌐 Open {CLOUD_FUNCTION_URL} in your browser to see the web interface")
    else:
        print("\n💔 Test failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure your Cloud Function is deployed")
        print("2. Check the Cloud Function URL is correct")
        print("3. Verify the image file exists and is readable")
        print("4. Check GCP Console > Cloud Functions > Logs for details")

if __name__ == "__main__":
    main()