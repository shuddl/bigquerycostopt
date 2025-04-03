# BigQuery Cost Intelligence Engine API Reference

**API Version: 1.0.0**

This reference provides detailed technical information about the BigQuery Cost Intelligence Engine API endpoints, parameters, request/response formats, and authentication.

## Base URL

All API requests should be made to the following base URL:

```
https://your-api-server.com
```

Replace `your-api-server.com` with your actual API server domain.

## Authentication

All API endpoints require authentication. Two authentication methods are supported:

### API Key Authentication

Include your API key in the Authorization header:

```
Authorization: Bearer your-api-key-here
```

### Service Account Authentication

For server-to-server requests, you can use Google Cloud Service Account authentication:

1. Create a signed JWT using your service account credentials
2. Include the JWT in the Authorization header:

```
Authorization: Bearer your-signed-jwt-token
```

## API Endpoints

### Health Check

Check the health status of the API server.

```
GET /api/health
```

#### Response

Status: 200 OK

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-04-02T14:32:15.123456"
}
```

### Cost Dashboard API

#### Get Cost Summary

Get a summary of BigQuery costs for a project.

```
GET /api/v1/cost-dashboard/summary
```

##### Query Parameters

| Parameter   | Type   | Required | Description                                     |
|-------------|--------|----------|-------------------------------------------------|
| project_id  | string | Yes      | GCP project ID                                  |
| days        | int    | No       | Number of days to analyze (default: 30)         |
| anonymize   | bool   | No       | Anonymize user identifiers (default: false)     |

##### Response

Status: 200 OK

```json
{
  "total_cost": 1250.45,
  "query_count": 15240,
  "total_bytes_processed": 12405894651,
  "avg_cost_per_query": 0.082,
  "top_users": [
    {"user": "john@example.com", "cost": 320.15},
    {"user": "jane@example.com", "cost": 215.32}
  ],
  "top_datasets": [
    {"dataset": "production_data", "cost": 645.21},
    {"dataset": "analytics", "cost": 320.45}
  ],
  "generated_at": "2025-04-02T14:32:15"
}
```

#### Get Cost Trends

Get historical cost trends for a project.

```
GET /api/v1/cost-dashboard/trends
```

##### Query Parameters

| Parameter   | Type   | Required | Description                                           |
|-------------|--------|----------|-------------------------------------------------------|
| project_id  | string | Yes      | GCP project ID                                        |
| days        | int    | No       | Number of days to analyze (default: 90)               |
| granularity | string | No       | Aggregation level (day, week, month) (default: day)   |

##### Response

Status: 200 OK

```json
{
  "trends": [
    {"period": "2025-03-01", "cost": 45.23, "bytes_processed": 4523156444},
    {"period": "2025-03-02", "cost": 52.15, "bytes_processed": 5215689542}
  ],
  "query_params": {
    "project_id": "your-project-id",
    "days": 90,
    "granularity": "day"
  },
  "generated_at": "2025-04-02T14:32:15"
}
```

#### Get Cost Anomalies

Detect cost anomalies for a project.

```
GET /api/v1/cost-dashboard/anomalies
```

##### Query Parameters

| Parameter        | Type   | Required | Description                                                |
|------------------|--------|----------|------------------------------------------------------------|
| project_id       | string | Yes      | GCP project ID                                             |
| days             | int    | No       | Number of days to analyze (default: 30)                    |
| detection_method | string | No       | Method to use (statistical, ml) (default: ml)              |
| sensitivity      | float  | No       | Anomaly sensitivity (0.0-1.0) (default: 0.7)              |

##### Response

Status: 200 OK

```json
{
  "anomalies": [
    {
      "date": "2025-03-15",
      "expected_cost": 45.32,
      "actual_cost": 123.45,
      "deviation_percent": 172.3,
      "severity": "high",
      "likely_cause": "Large export query from user john@example.com",
      "query_id": "abcdef123456"
    }
  ],
  "detection_method": "ml",
  "anomaly_count": 1,
  "generated_at": "2025-04-02T14:32:15"
}
```

#### Compare Cost Periods

Compare costs between two time periods.

```
GET /api/v1/cost-dashboard/compare-periods
```

##### Query Parameters

| Parameter     | Type   | Required | Description                                         |
|---------------|--------|----------|-----------------------------------------------------|
| project_id    | string | Yes      | GCP project ID                                      |
| current_days  | int    | No       | Number of days in current period (default: 30)      |
| previous_days | int    | No       | Number of days in previous period (default: 30)     |

##### Response

Status: 200 OK

```json
{
  "current_period": {
    "start_date": "2025-03-03",
    "end_date": "2025-04-02",
    "total_cost": 1250.45,
    "query_count": 15240
  },
  "previous_period": {
    "start_date": "2025-02-01",
    "end_date": "2025-03-02",
    "total_cost": 1100.23,
    "query_count": 14120
  },
  "change": {
    "cost_change_percent": 13.7,
    "query_count_change_percent": 7.9
  },
  "generated_at": "2025-04-02T14:32:15"
}
```

#### Get User Cost Attribution

Get cost attribution by user.

```
GET /api/v1/cost-dashboard/user-attribution
```

##### Query Parameters

| Parameter  | Type   | Required | Description                                    |
|------------|--------|----------|------------------------------------------------|
| project_id | string | Yes      | GCP project ID                                 |
| days       | int    | No       | Number of days to analyze (default: 30)        |
| limit      | int    | No       | Maximum number of users to return (default: 20)|
| anonymize  | bool   | No       | Anonymize user identifiers (default: false)    |

##### Response

Status: 200 OK

```json
{
  "users": [
    {
      "user": "john@example.com",
      "total_cost": 320.15,
      "query_count": 1530,
      "avg_cost_per_query": 0.21,
      "top_query_pattern": "SELECT * FROM `dataset.large_table`"
    }
  ],
  "total_users": 45,
  "generated_at": "2025-04-02T14:32:15"
}
```

#### Get User Clusters

Group users into clusters based on query patterns and costs.

```
GET /api/v1/cost-dashboard/user-clusters
```

##### Query Parameters

| Parameter    | Type   | Required | Description                                    |
|--------------|--------|----------|------------------------------------------------|
| project_id   | string | Yes      | GCP project ID                                 |
| days         | int    | No       | Number of days to analyze (default: 30)        |
| num_clusters | int    | No       | Number of clusters to generate (default: 3)    |
| anonymize    | bool   | No       | Anonymize user identifiers (default: false)    |

##### Response

Status: 200 OK

```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "user_count": 12,
      "avg_cost": 235.45,
      "description": "High-volume data scientists",
      "typical_query_patterns": ["SELECT * FROM", "UNNEST array_column"]
    }
  ],
  "generated_at": "2025-04-02T14:32:15"
}
```

### Analysis API

#### Trigger Dataset Analysis

Trigger an analysis of a BigQuery dataset.

```
POST /api/v1/analyze
```

##### Request Body

```json
{
  "project_id": "your-project-id",
  "dataset_id": "your_dataset",
  "callback_url": "https://your-webhook.com/callback",
  "options": {
    "analyze_query_patterns": true,
    "analyze_schema_optimization": true,
    "analyze_storage_optimization": true
  }
}
```

##### Response

Status: 202 Accepted

```json
{
  "analysis_id": "abc123def456",
  "status": "queued",
  "estimated_completion_time": "2025-04-02T15:32:15"
}
```

#### Get Analysis Status

Check the status of an analysis job.

```
GET /api/v1/analyze/status/{analysis_id}
```

##### Path Parameters

| Parameter   | Type   | Description             |
|-------------|--------|-------------------------|
| analysis_id | string | Unique analysis job ID  |

##### Response

Status: 200 OK

```json
{
  "analysis_id": "abc123def456",
  "status": "running",
  "progress": 0.45,
  "estimated_completion_time": "2025-04-02T15:32:15"
}
```

#### Get Analysis Results

Get the results of a completed analysis job.

```
GET /api/v1/analyze/results/{analysis_id}
```

##### Path Parameters

| Parameter   | Type   | Description             |
|-------------|--------|-------------------------|
| analysis_id | string | Unique analysis job ID  |

##### Response

Status: 200 OK

```json
{
  "analysis_id": "abc123def456",
  "project_id": "your-project-id",
  "dataset_id": "your_dataset",
  "completion_time": "2025-04-02T15:32:15",
  "summary": {
    "total_tables": 45,
    "total_size_gb": 1250.4,
    "total_rows": 3452345234
  },
  "recommendations_count": 12
}
```

### Recommendations API

#### Get Recommendations

Get optimization recommendations for a project or dataset.

```
GET /api/v1/recommendations
```

##### Query Parameters

| Parameter  | Type   | Required | Description                                               |
|------------|--------|----------|-----------------------------------------------------------|
| project_id | string | Yes      | GCP project ID                                            |
| dataset_id | string | No       | Specific dataset to get recommendations for               |
| category   | string | No       | Filter by category (storage, schema, query)               |
| priority   | string | No       | Filter by priority (high, medium, low)                    |
| limit      | int    | No       | Maximum number of recommendations to return (default: 50) |

##### Response

Status: 200 OK

```json
{
  "recommendations": [
    {
      "recommendation_id": "STORAGE_001",
      "category": "storage",
      "type": "partitioning_add",
      "table_id": "large_events_table",
      "dataset_id": "analytics",
      "project_id": "your-project-id",
      "description": "Add partitioning by event_date",
      "recommendation": "Partition table by event_date field",
      "rationale": "Table is frequently queried by date range",
      "annual_savings_usd": 1250.45,
      "implementation_cost_usd": 240.0,
      "roi": 5.21,
      "priority": "high"
    }
  ],
  "total_count": 12,
  "total_annual_savings_usd": 4530.23,
  "generated_at": "2025-04-02T14:32:15"
}
```

#### Get Recommendation Details

Get detailed information about a specific recommendation.

```
GET /api/v1/recommendations/{recommendation_id}
```

##### Path Parameters

| Parameter        | Type   | Description                       |
|------------------|--------|-----------------------------------|
| recommendation_id| string | Unique recommendation identifier  |

##### Response

Status: 200 OK

```json
{
  "recommendation_id": "STORAGE_001",
  "category": "storage",
  "type": "partitioning_add",
  "table_id": "large_events_table",
  "dataset_id": "analytics",
  "project_id": "your-project-id",
  "description": "Add partitioning by event_date",
  "recommendation": "Partition table by event_date field",
  "rationale": "Table is frequently queried by date range",
  "annual_savings_usd": 1250.45,
  "implementation_cost_usd": 240.0,
  "roi": 5.21,
  "priority": "high",
  "current_state": {
    "partitioning": "none",
    "size_gb": 500.23,
    "row_count": 2500000000,
    "average_queries_per_day": 120
  },
  "ml_insights": {
    "confidence_score": 0.92,
    "similar_cases": 15,
    "pattern_matches": ["frequent_date_filtering"]
  }
}
```

#### Get Implementation Plan

Get an implementation plan for a recommendation.

```
GET /api/v1/recommendations/{recommendation_id}/implementation
```

##### Path Parameters

| Parameter        | Type   | Description                       |
|------------------|--------|-----------------------------------|
| recommendation_id| string | Unique recommendation identifier  |

##### Response

Status: 200 OK

```json
{
  "recommendation_id": "STORAGE_001",
  "implementation_steps": [
    {
      "order": 1,
      "description": "Create backup of current table",
      "sql": "CREATE OR REPLACE TABLE `project.dataset.table_backup` AS SELECT * FROM `project.dataset.table`",
      "estimated_time_minutes": 30
    },
    {
      "order": 2,
      "description": "Create partitioned table",
      "sql": "CREATE OR REPLACE TABLE `project.dataset.table_new` (...)  PARTITION BY DATE(event_date)",
      "estimated_time_minutes": 45
    }
  ],
  "verification_steps": [
    {
      "description": "Verify row counts match",
      "sql": "SELECT COUNT(*) FROM `project.dataset.table_backup` a, COUNT(*) FROM `project.dataset.table_new` b"
    }
  ],
  "rollback_procedure": {
    "description": "Rename tables to revert to original state",
    "sql": "DROP TABLE IF EXISTS `project.dataset.table`; ALTER TABLE `project.dataset.table_backup` RENAME TO `project.dataset.table`"
  }
}
```

### Batch API

#### Execute Batch Requests

Execute multiple API calls in a single request.

```
POST /api/v1/batch
```

##### Request Body

```json
{
  "requests": [
    {
      "path": "/api/v1/cost-dashboard/summary",
      "method": "GET",
      "params": {
        "project_id": "your-project-id",
        "days": 30
      }
    },
    {
      "path": "/api/v1/cost-dashboard/trends",
      "method": "GET",
      "params": {
        "project_id": "your-project-id",
        "days": 90
      }
    }
  ]
}
```

##### Response

Status: 200 OK

```json
{
  "results": [
    {
      "status": 200,
      "path": "/api/v1/cost-dashboard/summary",
      "data": {
        "total_cost": 1250.45,
        "query_count": 15240
      }
    },
    {
      "status": 200,
      "path": "/api/v1/cost-dashboard/trends",
      "data": {
        "trends": [...]
      }
    }
  ],
  "generated_at": "2025-04-02T14:32:15"
}
```

## HTTP Response Codes

| Status Code | Description                                           |
|-------------|-------------------------------------------------------|
| 200         | OK - Request succeeded                                |
| 202         | Accepted - Request accepted for processing            |
| 400         | Bad Request - Invalid parameters or request data      |
| 401         | Unauthorized - Authentication failed                  |
| 403         | Forbidden - Insufficient permissions                  |
| 404         | Not Found - Resource not found                        |
| 429         | Too Many Requests - Rate limit exceeded               |
| 500         | Internal Server Error - Server error                  |

## Rate Limits

The API enforces the following rate limits:

- 60 requests per minute per API key
- 10,000 requests per day per API key

When a rate limit is exceeded, the API returns a 429 status code with a Retry-After header indicating the number of seconds to wait before retrying.

## Pagination

For endpoints that return large result sets, the API supports pagination using the following parameters:

- `page`: Page number (1-based)
- `page_size`: Number of items per page (default: 20, max: 100)

Response format for paginated endpoints:

```json
{
  "items": [...],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 145,
    "total_pages": 8
  }
}
```

## Error Responses

When an error occurs, the API returns a standard error format:

```json
{
  "error": "Invalid parameter",
  "message": "The 'granularity' parameter must be one of: day, week, month",
  "code": "INVALID_PARAMETER",
  "request_id": "abc123def456"
}
```

## Client Libraries

Official client libraries are available for:

- Python: `bigquerycostopt.client`
- JavaScript: `@bigquerycostopt/client`

## Changelog

### Version 1.0.0 (April 2, 2025)

- Initial public release
- Complete cost dashboard API
- ML-powered anomaly detection
- Comprehensive recommendation engine