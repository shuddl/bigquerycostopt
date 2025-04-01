# BigQuery Cost Intelligence Engine - Operations Runbook

This runbook provides guidance for common operational tasks and troubleshooting scenarios for the BigQuery Cost Intelligence Engine.

## Table of Contents

1. [Environment Access](#environment-access)
2. [Deployment](#deployment)
3. [Monitoring](#monitoring)
4. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
5. [Backup & Recovery](#backup--recovery)
6. [Security Operations](#security-operations)
7. [Maintenance Tasks](#maintenance-tasks)

## Environment Access

### GCP Console Access

Access to the GCP console is required for many administrative tasks:

```bash
# Open GCP console for a specific project
gcloud auth login
gcloud config set project PROJECT_ID
```

### Cloud Run Services

```bash
# List Cloud Run services
gcloud run services list --platform managed --region REGION

# View service logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bqcostopt-api" --limit 20
```

### Cloud Functions

```bash
# List Cloud Functions
gcloud functions list

# View function logs
gcloud logging read "resource.type=cloud_function AND resource.labels.function_name=bqcostopt-analysis-worker" --limit 20
```

### BigQuery Access

```bash
# List datasets
bq ls PROJECT_ID:

# Query recommendations table
bq query --use_legacy_sql=false 'SELECT * FROM PROJECT_ID.bqcostopt.recommendations LIMIT 10'
```

## Deployment

### CI/CD Pipeline

The system is deployed using GitHub Actions. The main workflows are:

- `ci.yml` - Continuous integration (runs on all branches and PRs)
- `cd.yml` - Continuous deployment (runs on main branch and tags)

Deployment to production should be done by creating a release tag:

```bash
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
```

### Manual Deployment

In case manual deployment is needed:

#### Deploy API Service

```bash
# Build and deploy API service
cd /path/to/bigquerycostopt
docker build -t gcr.io/PROJECT_ID/bqcostopt-api:latest .
docker push gcr.io/PROJECT_ID/bqcostopt-api:latest

gcloud run deploy bqcostopt-api \
  --image=gcr.io/PROJECT_ID/bqcostopt-api:latest \
  --platform=managed \
  --region=REGION \
  --project=PROJECT_ID
```

#### Deploy Cloud Function

```bash
# Deploy analysis worker function
cd /path/to/bigquerycostopt/function_source/analysis_worker
gcloud functions deploy bqcostopt-analysis-worker \
  --runtime=python39 \
  --entry-point=process_analysis_request \
  --trigger-topic=bqcostopt-analysis-requests \
  --region=REGION \
  --project=PROJECT_ID \
  --memory=2048MB \
  --timeout=540s
```

### Infrastructure Updates

Infrastructure is managed using Terraform:

```bash
# Apply infrastructure changes
cd /path/to/bigquerycostopt/infra/terraform/environments/ENVIRONMENT
terraform init
terraform plan -var="project_id=PROJECT_ID" -var="api_key=API_KEY" -var="alert_email=ALERT_EMAIL"
terraform apply
```

## Monitoring

### Dashboards

Access monitoring dashboards:

1. Go to GCP Console > Monitoring > Dashboards
2. Select "BigQuery Cost Intelligence Engine Overview"

### Alerts

View active alerts:

1. Go to GCP Console > Monitoring > Alerting
2. Check for any firing alerts

### Health Checks

Verify API health:

```bash
curl -H "X-API-Key: YOUR_API_KEY" https://bqcostopt-api-XXXX-uc.a.run.app/api/v1/health
```

Expected response:
```json
{
  "status": "OK",
  "version": "1.2.3",
  "timestamp": "2023-07-31T12:34:56Z"
}
```

### Logs

View logs for components:

1. Go to GCP Console > Logging > Logs Explorer
2. Use the following queries:

```
resource.type="cloud_run_revision" AND resource.labels.service_name="bqcostopt-api"
```

```
resource.type="cloud_function" AND resource.labels.function_name="bqcostopt-analysis-worker"
```

## Common Issues & Troubleshooting

### API Service Errors

#### 500 Internal Server Errors

1. Check API logs for error details
2. Verify environment variables are correctly set
3. Check if BigQuery/Pub/Sub permissions are correct

#### Authentication Failures

1. Verify API key is correctly set
2. Check if the API key has been rotated

### Analysis Worker Failures

#### Function Timeouts

1. Check if analysis is processing very large datasets
2. Increase function timeout limit
3. Consider breaking analysis into smaller chunks

#### Permission Errors

1. Verify service account permissions
2. Check if IAM roles have been changed

### Recommendation Quality Issues

1. Verify that ML models are up to date
2. Check if data collection is operating correctly
3. Review feedback data for patterns

## Backup & Recovery

### BigQuery Data Backup

Backup important tables:

```bash
# Export table to GCS
bq extract --destination_format=NEWLINE_DELIMITED_JSON \
  PROJECT_ID:bqcostopt.recommendations \
  gs://BACKUP_BUCKET/backups/recommendations_$(date +%Y%m%d).json
```

### Restore From Backup

```bash
# Import data from GCS backup
bq load --source_format=NEWLINE_DELIMITED_JSON \
  PROJECT_ID:bqcostopt.recommendations \
  gs://BACKUP_BUCKET/backups/recommendations_YYYYMMDD.json \
  /path/to/schema.json
```

### Disaster Recovery

In case of complete environment failure:

1. Create a new project with same project ID (if needed)
2. Deploy infrastructure using Terraform
3. Restore data from backups
4. Verify system functionality

## Security Operations

### API Key Rotation

```bash
# Generate new API key
NEW_API_KEY=$(openssl rand -base64 32)

# Update Secret Manager
gcloud secrets versions add api-key --data-file=<(echo -n "$NEW_API_KEY")

# Test new key before full deployment
curl -H "X-API-Key: $NEW_API_KEY" https://bqcostopt-api-XXXX-uc.a.run.app/api/v1/health
```

### Access Reviews

Perform regular access reviews:

```bash
# List IAM bindings for project
gcloud projects get-iam-policy PROJECT_ID --format=json > iam_policy.json

# Review service account permissions
gcloud iam service-accounts list --project=PROJECT_ID
```

### Security Scanning

1. Run regular vulnerability scans on container images
2. Ensure dependencies are up to date
3. Review Cloud Security Command Center findings

## Maintenance Tasks

### ML Model Retraining

Retrain ML models periodically:

```bash
# Trigger model retraining
curl -X POST -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"full_retrain": true}' \
  https://bqcostopt-api-XXXX-uc.a.run.app/api/v1/ml/train
```

### Database Maintenance

```bash
# Run query to cleanup old analysis results
bq query --use_legacy_sql=false '
DELETE FROM `PROJECT_ID.bqcostopt.analysis_results` 
WHERE analysis_date < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
AND status != "active"
'
```

### Resource Cleanup

```bash
# Delete old GCS objects
gsutil rm gs://PROJECT_ID-analysis-results/analysis_results/$(date --date="90 days ago" +%Y%m%d)*
```

### Performance Optimization

1. Review Cloud Monitoring metrics for performance bottlenecks
2. Adjust resource allocations as needed
3. Optimize queries and processing logic