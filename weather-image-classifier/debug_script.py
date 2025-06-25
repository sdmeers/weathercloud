#!/usr/bin/env python3
"""
Debug script to check the current state of your Cloud Storage bucket
and see what's actually stored vs what the webpage is showing.
"""

import requests
import json
from google.cloud import storage
import os

# Configuration
BUCKET_NAME = "weathercloud-460719-weather-images"  # Update with your bucket name
CLOUD_FUNCTION_URL = "https://europe-west1-weathercloud-460719.cloudfunctions.net/weather-image-classifier"

def check_bucket_contents():
    """List all files in the bucket with their details."""
    print("🪣 BUCKET CONTENTS:")
    print("=" * 50)
    
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        
        all_blobs = list(bucket.list_blobs())
        
        if not all_blobs:
            print("❌ Bucket is empty!")
            return
        
        print(f"📁 Found {len(all_blobs)} files in bucket:")
        
        for blob in all_blobs:
            print(f"  📄 {blob.name}")
            print(f"     Size: {blob.size} bytes")
            print(f"     Updated: {blob.updated}")
            print(f"     Content-Type: {blob.content_type}")
            print()
            
    except Exception as e:
        print(f"❌ Error accessing bucket: {e}")
        return

def check_pointer_file():
    """Check what the pointer file contains."""
    print("🔍 POINTER FILE ANALYSIS:")
    print("=" * 50)
    
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        
        pointer_blob = bucket.blob("latest_pointer.json")
        if not pointer_blob.exists():
            print("❌ latest_pointer.json does not exist!")
            return
            
        pointer_content = pointer_blob.download_as_text()
        print(f"📝 latest_pointer.json contents:")
        print(pointer_content)
        
        pointer_data = json.loads(pointer_content)
        latest_hash = pointer_data.get("latest")
        
        if latest_hash:
            print(f"👉 Points to hash: {latest_hash}")
            
            # Check if the corresponding files exist
            img_blob = bucket.blob(f"{latest_hash}.jpg")
            meta_blob = bucket.blob(f"{latest_hash}.json")
            
            print(f"📸 {latest_hash}.jpg exists: {img_blob.exists()}")
            print(f"📋 {latest_hash}.json exists: {meta_blob.exists()}")
            
            if meta_blob.exists():
                meta_content = meta_blob.download_as_text()
                meta_data = json.loads(meta_content)
                print(f"📊 Metadata:")
                print(f"   Classification: {meta_data.get('classification')}")
                print(f"   Timestamp: {meta_data.get('timestamp')}")
                print(f"   Image Size: {meta_data.get('image_size')}")
        else:
            print("❌ Pointer file has no 'latest' field!")
            
    except Exception as e:
        print(f"❌ Error reading pointer file: {e}")

def test_webpage_response():
    """Test what the webpage is actually returning."""
    print("🌐 WEBPAGE RESPONSE TEST:")
    print("=" * 50)
    
    try:
        response = requests.get(CLOUD_FUNCTION_URL, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Webpage returned status {response.status_code}")
            print(response.text[:500])
            return
            
        html_content = response.text
        
        # Extract the image URL from the HTML
        import re
        img_url_match = re.search(r'<img src="([^"]+)"', html_content)
        if img_url_match:
            img_url = img_url_match.group(1)
            print(f"🖼️  Image URL in webpage: {img_url}")
            
            # Extract the filename from the URL
            if "/weathercloud-460719-weather-images/" in img_url:
                filename = img_url.split("/weathercloud-460719-weather-images/")[1]
                if "?" in filename:
                    filename = filename.split("?")[0]
                print(f"📄 Filename: {filename}")
                
                if filename.endswith(".jpg"):
                    hash_from_url = filename.replace(".jpg", "")
                    print(f"🔑 Hash from URL: {hash_from_url}")
            
        # Look for classification in HTML
        class_match = re.search(r'Classification: ([^<]+)', html_content)
        if class_match:
            classification = class_match.group(1).strip()
            print(f"🏷️  Classification shown: {classification}")
            
    except Exception as e:
        print(f"❌ Error testing webpage: {e}")

def main():
    print("🔧 WEATHER CLASSIFIER DEBUG TOOL")
    print("=" * 50)
    print()
    
    # Check what's actually in the bucket
    check_bucket_contents()
    print()
    
    # Check the pointer file
    check_pointer_file()
    print()
    
    # Check what the webpage is serving
    test_webpage_response()
    print()
    
    print("💡 DEBUGGING TIPS:")
    print("1. The bucket should only contain exactly 3 files")
    print("2. The pointer should point to the same hash as your latest upload")
    print("3. The webpage should show the same hash as the pointer")
    print("4. If any of these don't match, there's your problem!")

if __name__ == "__main__":
    main()