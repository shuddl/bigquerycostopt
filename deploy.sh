#!/bin/bash

# Deployment script for BigQuery Cost Intelligence Engine

set -e

# Configuration
PROJECT_ID=${1:-""}
REGION=${2:-"us-central1"}
SERVICE_NAME="bqcost-engine"

# Check if project ID is provided
if [ -z "$PROJECT_ID" ]; then
    echo "Error: GCP project ID is required"
    echo "Usage: ./deploy.sh PROJECT_ID [REGION]"
    exit 1
fi

echo "Deploying BigQuery Cost Intelligence Engine to project: $PROJECT_ID in region: $REGION"

# Ensure gcloud is configured with the correct project
gcloud config set project $PROJECT_ID

# Create Pub/Sub topics
echo "Creating Pub/Sub topics..."
gcloud pubsub topics create analysis-requests --project=$PROJECT_ID || echo "Topic already exists"
gcloud pubsub topics create analysis-results --project=$PROJECT_ID || echo "Topic already exists"

# Create service account for the application
echo "Creating service account..."
SERVICE_ACCOUNT="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts create $SERVICE_NAME \
    --display-name="BigQuery Cost Intelligence Engine Service Account" \
    --project=$PROJECT_ID || echo "Service account already exists"

# Grant necessary permissions
echo "Granting IAM permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/pubsub.publisher"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/pubsub.subscriber"

# Create BigQuery dataset for storing recommendations
echo "Creating BigQuery dataset for recommendations..."
bq --location=$REGION mk \
    --dataset \
    --description="BigQuery Cost Intelligence Engine Recommendations" \
    ${PROJECT_ID}:bqcost_recommendations || echo "Dataset already exists"

# Deploy Cloud Run service
echo "Building and deploying API service to Cloud Run..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --service-account $SERVICE_ACCOUNT \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},ANALYSIS_REQUEST_TOPIC=analysis-requests" \
    --allow-unauthenticated

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo "\nDeployment completed successfully!"
echo "API Service URL: $SERVICE_URL"
echo "\nTo test the API:"
echo "curl -X POST \\
  -H 'Content-Type: application/json' \\
  -H 'X-API-Key: your-api-key-here' \\
  -d '{\"project_id\":\"$PROJECT_ID\", \"dataset_id\":\"your_dataset_id\"}' \\
  $SERVICE_URL/api/v1/analyze"
