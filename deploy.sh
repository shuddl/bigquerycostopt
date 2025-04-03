#!/bin/bash

# Deployment script for BigQuery Cost Intelligence Engine using Terraform

set -e

# Configuration
ENVIRONMENT=${1:-"dev"}
PROJECT_ID=${2:-""}
REGION=${3:-"us-central1"}
API_VERSION=${4:-"latest"}
API_SERVER_TYPE=${5:-"fastapi"}  # Options: flask, fastapi

# Check if project ID is provided
if [ -z "$PROJECT_ID" ]; then
    echo "Error: GCP project ID is required"
    echo "Usage: ./deploy.sh [ENVIRONMENT] PROJECT_ID [REGION] [API_VERSION] [API_SERVER_TYPE]"
    echo "  ENVIRONMENT: dev, staging, prod (default: dev)"
    echo "  PROJECT_ID: GCP project ID (required)"
    echo "  REGION: GCP region (default: us-central1)"
    echo "  API_VERSION: API version to deploy (default: latest)"
    echo "  API_SERVER_TYPE: flask or fastapi (default: fastapi)"
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    echo "Error: ENVIRONMENT must be one of: dev, staging, prod"
    exit 1
fi

echo "Deploying BigQuery Cost Intelligence Engine ($ENVIRONMENT environment)"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "API Version: $API_VERSION"
echo "API Server Type: $API_SERVER_TYPE"

# Verify Terraform installation
if ! command -v terraform &> /dev/null; then
    echo "Error: terraform is not installed or not in PATH"
    exit 1
fi

# Ensure gcloud is configured with the correct project
gcloud config set project $PROJECT_ID

# Build API Docker image
echo "Building API Docker image..."
docker build -t gcr.io/$PROJECT_ID/bqcostopt-api:$API_VERSION .

echo "Pushing Docker image to Container Registry..."
docker push gcr.io/$PROJECT_ID/bqcostopt-api:$API_VERSION

# Create function_source.zip for Cloud Functions
echo "Creating function source archive..."
(cd function_source/analysis_worker && zip -r ../../../function_source.zip .)

# Navigate to Terraform directory for the specified environment
TERRAFORM_DIR="infra/terraform/environments/$ENVIRONMENT"
cd $TERRAFORM_DIR

# Generate API key if not provided
if [ -z "$API_KEY" ]; then
    API_KEY=$(openssl rand -base64 32)
    echo "Generated API key: $API_KEY"
    echo "WARNING: Store this API key securely. It will not be shown again."
fi

# Initialize Terraform
echo "Initializing Terraform..."
terraform init

# Create Terraform variables file
cat > terraform.tfvars << EOF
project_id = "$PROJECT_ID"
region = "$REGION"
api_version = "$API_VERSION"
api_key = "$API_KEY"
api_server_type = "$API_SERVER_TYPE"
alert_email = "admin@example.com"
enable_cost_dashboard = "true"
EOF

if [ "$ENVIRONMENT" = "prod" ]; then
    echo "key_dataset_id = \"your_key_dataset\"" >> terraform.tfvars
    echo "pagerduty_service_key = \"your_pagerduty_key\"" >> terraform.tfvars
fi

# Plan Terraform changes
echo "Planning Terraform changes..."
terraform plan -var-file=terraform.tfvars -out=tfplan

# Apply Terraform changes with confirmation
echo "Applying Terraform changes..."
read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    terraform apply tfplan
    
    # Get outputs
    API_URL=$(terraform output -raw api_url)
    
    echo "Deployment completed successfully!"
    echo "API Service URL: $API_URL"
    echo
    echo "To test the API:"
    echo "curl -X GET \\"
    echo "  -H 'X-API-Key: $API_KEY' \\"
    echo "  $API_URL/api/v1/health"
    echo
    echo "To analyze a dataset:"
    echo "curl -X POST \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -H 'X-API-Key: $API_KEY' \\"
    echo "  -d '{\"project_id\":\"$PROJECT_ID\", \"dataset_id\":\"your_dataset_id\"}' \\"
    echo "  $API_URL/api/v1/analyze"
    echo
    echo "To use the Cost Attribution Dashboard:"
    echo "curl -X GET \\"
    echo "  -H 'X-API-Key: $API_KEY' \\"
    echo "  $API_URL/api/v1/cost-dashboard/summary?project_id=$PROJECT_ID&days=30"
else
    echo "Deployment cancelled."
fi
