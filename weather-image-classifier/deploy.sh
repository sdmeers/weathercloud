#!/bin/bash

# Configuration - Update these values
PROJECT_ID="weathercloud-460719"
FUNCTION_NAME="weather-image-classifier"
REGION="europe-west2"
BUCKET_NAME="weathercloud-460719-weather-images"

# Create Cloud Storage bucket if it doesn't exist
echo "Creating Cloud Storage bucket..."
gsutil mb gs://$BUCKET_NAME 2>/dev/null || echo "Bucket already exists"

# Make bucket publicly readable for images (skip if permission error)
echo "Making bucket publicly readable..."
gsutil iam ch allUsers:objectViewer gs://$BUCKET_NAME || echo "Warning: Could not make bucket public. You may need to do this manually."

# Deploy the Cloud Function
echo "Deploying Cloud Function..."
gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --runtime=python311 \
    --region=$REGION \
    --source=. \
    --entry-point=weather_image_classifier \
    --trigger-http \
    --allow-unauthenticated \
    --memory=1Gi \
    --timeout=300s \
    --set-env-vars PROJECT_ID=$PROJECT_ID,LOCATION=$REGION,BUCKET_NAME=$BUCKET_NAME

echo "Deployment complete!"
echo "Function URL:"
gcloud functions describe $FUNCTION_NAME --region=$REGION --format="value(serviceConfig.uri)"