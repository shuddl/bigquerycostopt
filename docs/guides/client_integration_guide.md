# BigQuery Cost Intelligence Engine - Client Integration Guide

This comprehensive guide provides all the information needed to integrate with the BigQuery Cost Intelligence Engine API, including authentication, endpoint usage, client libraries, and best practices.

## Table of Contents

- [Overview](#overview)
- [API Server Options](#api-server-options)
- [Authentication](#authentication)
- [API Reference](#api-reference)
  - [Health Check](#health-check)
  - [Cost Dashboard API](#cost-dashboard-api)
    - [Cost Summary](#cost-summary)
    - [Cost Attribution](#cost-attribution)
    - [Cost Trends](#cost-trends)
    - [Period Comparison](#period-comparison)
    - [Expensive Queries](#expensive-queries)
    - [Cost Anomalies](#cost-anomalies)
    - [Cost Forecast](#cost-forecast)
    - [User Clusters](#user-clusters)
    - [Team Mapping](#team-mapping)
    - [Cost Alerts](#cost-alerts)
  - [Analysis API](#analysis-api)
    - [Trigger Analysis](#trigger-analysis)
    - [Analysis Status](#analysis-status)
- [Client Libraries](#client-libraries)
  - [Python Client](#python-client)
  - [JavaScript Client](#javascript-client)
  - [API Rate Limits](#api-rate-limits)
  - [Response Caching](#response-caching)
- [Retool Integration](#retool-integration)
  - [Setting Up Retool](#setting-up-retool)
  - [Example Dashboard Components](#example-dashboard-components)
  - [JavaScript Transformations](#javascript-transformations)
- [Advanced Usage](#advanced-usage)
  - [Handling Errors](#handling-errors)
  - [Multi-Project Analysis](#multi-project-analysis)
  - [Custom Anomaly Detection](#custom-anomaly-detection)
  - [Webhook Integration](#webhook-integration)
- [Security Considerations](#security-considerations)
  - [API Key Management](#api-key-management)
  - [Network Security](#network-security)
  - [Data Privacy](#data-privacy)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Debugging Tools](#debugging-tools)
  - [Logging](#logging)
- [Support](#support)

## Overview

The BigQuery Cost Intelligence Engine provides a comprehensive API for analyzing, optimizing, and reporting on BigQuery costs. The API enables:

* Cost attribution by user, team, query pattern, and dataset
* Anomaly detection to identify unusual spending patterns
* Cost forecasting based on historical usage
* User behavior clustering to identify patterns
* Cost-saving recommendations with implementation plans

This guide covers everything you need to integrate your applications with the API.

## API Server Options

The BigQuery Cost Intelligence Engine supports two API server implementations:

1. **FastAPI** (Recommended): Modern, high-performance API framework with automatic OpenAPI documentation
2. **Flask**: Lightweight API framework

Both implementations provide identical functionality and endpoints. The FastAPI implementation includes additional features like:

* Interactive API documentation (Swagger UI at `/api/docs`)
* Request validation with Pydantic
* Better performance under load
* Enhanced security headers

To specify the server type when deploying:

```bash
# Run with FastAPI (recommended)
python -m bigquerycostopt.src.api.server --server-type fastapi

# Run with Flask
python -m bigquerycostopt.src.api.server --server-type flask
```

## Authentication

All API endpoints require authentication using an API key. Provide the key in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

To generate an API key, use the provided key management tool:

```bash
python -m bigquerycostopt.src.utils.security generate_api_key --name "Client Name"
```

API keys can be revoked or rotated using:

```bash
# Revoke a key
python -m bigquerycostopt.src.utils.security revoke_api_key --key KEY_ID

# Rotate a key
python -m bigquerycostopt.src.utils.security rotate_api_key --key KEY_ID
```

API keys are subject to rate limiting to prevent abuse. See the [API Rate Limits](#api-rate-limits) section for details.

## API Reference

All endpoints return JSON responses with consistent structures. Common HTTP status codes:

* `200 OK`: Request successful
* `400 Bad Request`: Invalid parameters
* `401 Unauthorized`: Authentication failed
* `429 Too Many Requests`: Rate limit exceeded
* `500 Internal Server Error`: Server-side error

### Health Check

```
GET /api/health
```

Returns the health status of the API server.

**Response Example:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-04-02T15:30:45.123456"
}
```

### Cost Dashboard API

#### Cost Summary

```
GET /api/v1/cost-dashboard/summary
```

Get a summary of BigQuery costs for the specified period.

**Query Parameters:**
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 30)

**Response Example:**

```json
{
  "summary": {
    "total_cost_usd": 1256.78,
    "daily_avg_cost_usd": 41.89,
    "total_queries": 15432,
    "avg_query_cost_usd": 0.08,
    "cost_by_query_type": {
      "SELECT": 1022.45,
      "INSERT": 154.33,
      "MERGE": 80.0
    }
  },
  "query_params": {
    "project_id": "my-project",
    "days": 30
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

#### Cost Attribution

```
GET /api/v1/cost-dashboard/attribution
```

Get detailed cost attribution data by various dimensions.

**Query Parameters:**
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 30)
- `dimensions` (optional): Comma-separated list of attribution dimensions (default: user,team,pattern,day,table)

**Response Example:**

```json
{
  "attribution": {
    "cost_by_user": [
      {
        "user_email": "user1@example.com",
        "team": "Data Science",
        "estimated_cost_usd": 523.45,
        "query_count": 856
      },
      {
        "user_email": "user2@example.com",
        "team": "Engineering",
        "estimated_cost_usd": 321.67,
        "query_count": 542
      }
    ],
    "cost_by_team": [
      {
        "team": "Data Science",
        "estimated_cost_usd": 678.23,
        "user_count": 12,
        "query_count": 1245
      },
      {
        "team": "Engineering",
        "estimated_cost_usd": 456.78,
        "user_count": 8,
        "query_count": 876
      }
    ]
  },
  "query_params": {
    "project_id": "my-project",
    "days": 30,
    "dimensions": "user,team"
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

#### Cost Trends

```
GET /api/v1/cost-dashboard/trends
```

Get cost trends over time with optional granularity.

**Query Parameters:**
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 90)
- `granularity` (optional): Time granularity (day, week, month) (default: day)

**Response Example:**

```json
{
  "trends": [
    {
      "period": "2025-03-01",
      "total_cost_usd": 42.56,
      "query_count": 567,
      "cost_ma": 40.23
    },
    {
      "period": "2025-03-02",
      "total_cost_usd": 38.91,
      "query_count": 498,
      "cost_ma": 40.12
    }
  ],
  "query_params": {
    "project_id": "my-project",
    "days": 90,
    "granularity": "day"
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

#### Period Comparison

```
GET /api/v1/cost-dashboard/compare-periods
```

Compare costs between two time periods.

**Query Parameters:**
- `project_id` (required): GCP project ID
- `current_days` (optional): Number of days in current period (default: 30)
- `previous_days` (optional): Number of days in previous period (default: 30)

**Response Example:**

```json
{
  "comparison": {
    "current_period": {
      "start_date": "2025-03-03",
      "end_date": "2025-04-02",
      "total_cost_usd": 1256.78,
      "daily_avg_cost_usd": 41.89
    },
    "previous_period": {
      "start_date": "2025-02-01",
      "end_date": "2025-03-02",
      "total_cost_usd": 1156.45,
      "daily_avg_cost_usd": 38.55
    },
    "change": {
      "absolute_usd": 100.33,
      "percent": 8.68,
      "daily_avg_absolute_usd": 3.34,
      "daily_avg_percent": 8.68
    }
  },
  "query_params": {
    "project_id": "my-project",
    "current_days": 30,
    "previous_days": 30
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

#### Expensive Queries

```
GET /api/v1/cost-dashboard/expensive-queries
```

Get the most expensive queries.

**Query Parameters:**
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 30)
- `limit` (optional): Maximum number of queries to return (default: 100)

**Response Example:**

```json
{
  "queries": [
    {
      "query_id": "2c4a5d8e7f",
      "user_email": "user1@example.com",
      "estimated_cost_usd": 25.67,
      "bytes_processed": 1256789012,
      "creation_time": "2025-03-15T12:34:56.789012",
      "query_text": "SELECT * FROM `project.dataset.large_table` WHERE ...",
      "optimization_potential": "high"
    }
  ],
  "query_params": {
    "project_id": "my-project",
    "days": 30,
    "limit": 100
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

#### Cost Anomalies

```
GET /api/v1/cost-dashboard/anomalies
```

Detect cost anomalies using statistical or ML-based methods.

**Query Parameters:**
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 30)
- `anomaly_types` (optional): Comma-separated list of anomaly types (default: daily,user,team,pattern)
- `use_ml` (optional): Whether to use ML-enhanced anomaly detection (default: false)

**Response Example (Statistical Anomalies):**

```json
{
  "anomalies": {
    "daily_anomalies": [
      {
        "date": "2025-03-15",
        "total_cost_usd": 125.45,
        "expected_cost_usd": 42.56,
        "z_score": 3.86,
        "percent_change": 194.76
      }
    ],
    "user_anomalies": [
      {
        "user_email": "user1@example.com",
        "team": "Data Science",
        "estimated_cost_usd_current": 125.45,
        "estimated_cost_usd_previous": 42.56,
        "percent_change": 194.76
      }
    ],
    "anomaly_counts": {
      "daily": 1,
      "user": 1,
      "team": 0,
      "pattern": 0
    }
  },
  "query_params": {
    "project_id": "my-project",
    "days": 30,
    "anomaly_types": "daily,user,team,pattern",
    "use_ml": false
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

**Response Example (ML Anomalies):**

```json
{
  "anomalies": {
    "daily_anomalies": {
      "dates": ["2025-03-01", "2025-03-02", "2025-03-03"],
      "costs": [42.56, 38.91, 125.45],
      "is_anomaly": [false, false, true],
      "anomaly_score": [0.12, 0.08, 0.95]
    },
    "model_info": {
      "model_type": "isolation_forest",
      "training_data_size": 90,
      "threshold": 0.8
    }
  },
  "query_params": {
    "project_id": "my-project",
    "days": 30,
    "anomaly_types": "daily",
    "use_ml": true
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

#### Cost Forecast

```
GET /api/v1/cost-dashboard/forecast
```

Get cost forecast for future periods.

**Query Parameters:**
- `project_id` (required): GCP project ID
- `training_days` (optional): Number of days to use for training (default: 90)
- `forecast_days` (optional): Number of days to forecast (default: 7)

**Response Example:**

```json
{
  "forecast": {
    "historical_data": [
      {
        "date": "2025-03-01",
        "total_cost_usd": 42.56
      },
      {
        "date": "2025-03-02",
        "total_cost_usd": 38.91
      }
    ],
    "forecast": [
      {
        "date": "2025-04-03",
        "forecasted_cost_usd": 43.21,
        "lower_bound": 39.45,
        "upper_bound": 47.32
      },
      {
        "date": "2025-04-04",
        "forecasted_cost_usd": 44.56,
        "lower_bound": 40.12,
        "upper_bound": 48.95
      }
    ],
    "model_info": {
      "model_type": "prophet",
      "training_data_size": 90,
      "mape": 8.45
    }
  },
  "query_params": {
    "project_id": "my-project",
    "training_days": 90,
    "forecast_days": 7
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

#### User Clusters

```
GET /api/v1/cost-dashboard/user-clusters
```

Get user behavior clusters to identify patterns.

**Query Parameters:**
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 90)
- `n_clusters` (optional): Number of clusters to generate (default: 5)

**Response Example:**

```json
{
  "clusters": {
    "cluster_assignments": [
      {
        "user_email": "user1@example.com",
        "cluster": 0,
        "distance_to_centroid": 0.45
      },
      {
        "user_email": "user2@example.com",
        "cluster": 1,
        "distance_to_centroid": 0.32
      }
    ],
    "cluster_profiles": [
      {
        "cluster_id": 0,
        "size": 12,
        "avg_cost_usd": 56.78,
        "avg_queries_per_day": 8.4,
        "common_patterns": ["SELECT with JOIN", "Large TABLE SCAN"]
      },
      {
        "cluster_id": 1,
        "size": 8,
        "avg_cost_usd": 21.34,
        "avg_queries_per_day": 12.6,
        "common_patterns": ["Light transformation", "Small lookups"]
      }
    ]
  },
  "metrics": {
    "silhouette_score": 0.68,
    "calinski_harabasz_score": 245.67
  },
  "query_params": {
    "project_id": "my-project",
    "days": 90,
    "n_clusters": 5
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

#### Team Mapping

```
POST /api/v1/cost-dashboard/team-mapping
```

Update the mapping of users to teams for cost attribution.

**Request Body:**

```json
{
  "project_id": "my-project",
  "mapping": {
    "user1@example.com": "Data Science",
    "user2@example.com": "Engineering",
    "*@data-team.example.com": "Data Science",
    "*@eng.example.com": "Engineering"
  }
}
```

**Response Example:**

```json
{
  "status": "success",
  "message": "Team mapping updated with 4 entries",
  "updated_at": "2025-04-02T15:30:45.123456"
}
```

#### Cost Alerts

```
GET /api/v1/cost-dashboard/alerts
```

Get cost alerts for significant cost increases.

**Query Parameters:**
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to look back (default: 7)
- `min_cost_increase_usd` (optional): Minimum cost increase to trigger alert (default: 100.0)

**Response Example:**

```json
{
  "alerts": [
    {
      "id": "alert-2c4a5d8e7f",
      "type": "daily_cost_increase",
      "severity": "high",
      "timestamp": "2025-03-15T12:00:00Z",
      "message": "Daily cost increased by $82.89 (194.8%) on 2025-03-15",
      "details": {
        "date": "2025-03-15",
        "actual_cost_usd": 125.45,
        "expected_cost_usd": 42.56,
        "absolute_increase_usd": 82.89,
        "percent_increase": 194.8
      },
      "recommendations": [
        "Review queries run by user1@example.com on this date",
        "Check for new scheduled queries that may have started"
      ]
    }
  ],
  "alert_count": 1,
  "query_params": {
    "project_id": "my-project",
    "days": 7,
    "min_cost_increase_usd": 100.0
  },
  "generated_at": "2025-04-02T15:30:45.123456"
}
```

### Analysis API

#### Trigger Analysis

```
POST /api/v1/analyze
```

Trigger an analysis of a BigQuery dataset.

**Request Body:**

```json
{
  "project_id": "my-project",
  "dataset_id": "my_dataset",
  "callback_url": "https://my-service.example.com/callback"
}
```

**Response Example:**

```json
{
  "analysis_id": "2c4a5d8e-7f9a-4b3c-8d2e-1a5b6c7d8e9f",
  "status": "pending",
  "message": "Analysis request submitted successfully"
}
```

#### Analysis Status

```
GET /api/v1/analysis/{analysis_id}
```

Get the status of a previously submitted analysis.

**Path Parameters:**
- `analysis_id` (required): The ID of the analysis to check

**Response Example:**

```json
{
  "analysis_id": "2c4a5d8e-7f9a-4b3c-8d2e-1a5b6c7d8e9f",
  "status": "in_progress",
  "progress": 45,
  "message": "Analyzing query patterns"
}
```

## Client Libraries

### Python Client

A Python client is available to simplify API interaction. Install it with:

```bash
pip install bigquerycostopt[client]
```

Or use the example client directly:

```python
from bigquerycostopt.examples.cost_dashboard_client import CostDashboardClient

# Initialize client
client = CostDashboardClient(base_url="http://localhost:8080", api_key="your-api-key")

# Get cost summary for the last 30 days
summary = client.get_cost_summary("your-project-id", days=30)

# Get cost attribution data
attribution = client.get_cost_attribution("your-project-id", days=30)

# Get cost anomalies with ML enhancement
anomalies = client.get_anomalies("your-project-id", days=30, use_ml=True)

# Get cost forecast for the next 14 days
forecast = client.get_forecast("your-project-id", training_days=90, forecast_days=14)
```

You can also run the client as a standalone script:

```bash
python -m bigquerycostopt.examples.cost_dashboard_client \
  --project-id=your-project-id \
  --days=30 \
  --use-ml \
  --visualize
```

### JavaScript Client

A JavaScript client is available for browser and Node.js applications:

```javascript
// Import the client
import { CostDashboardClient } from 'bigquerycostopt-client';

// Initialize client
const client = new CostDashboardClient({
  baseUrl: 'http://your-api-server.example.com',
  apiKey: 'your-api-key'
});

// Get cost summary
client.getCostSummary({
  projectId: 'your-project-id',
  days: 30
})
.then(summary => {
  console.log('Total cost:', summary.summary.total_cost_usd);
})
.catch(error => {
  console.error('Error:', error);
});

// Using async/await
async function getCostData() {
  try {
    const summary = await client.getCostSummary({
      projectId: 'your-project-id',
      days: 30
    });
    
    const attribution = await client.getCostAttribution({
      projectId: 'your-project-id',
      days: 30,
      dimensions: 'user,team'
    });
    
    return {
      totalCost: summary.summary.total_cost_usd,
      userCosts: attribution.attribution.cost_by_user
    };
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### API Rate Limits

The API implements rate limiting to prevent abuse:

* Default: 60 requests per minute per API key
* Health check endpoint: 120 requests per minute per IP address

When rate limits are exceeded, the API returns a `429 Too Many Requests` response with retry information:

```json
{
  "error": "Rate limit exceeded",
  "message": "Maximum 60 requests per minute allowed",
  "retry_after": 45
}
```

Clients should implement exponential backoff for retries when encountering rate limit errors.

### Response Caching

The API implements server-side caching to improve performance:

* Most endpoints cache responses for 5 minutes
* Alert endpoints use a shorter 1-minute cache
* Each combination of parameters has a separate cache entry

Clients can also implement local caching for frequently accessed data.

## Retool Integration

The BigQuery Cost Intelligence Engine is designed to integrate seamlessly with Retool dashboards.

### Setting Up Retool

1. Create a new Retool application
2. Add a REST API resource pointing to your API server
3. Configure authentication with your API key
4. Create queries for each endpoint

Example REST API resource configuration:

```
Name: BigQueryCostAPI
Base URL: https://your-api-server.example.com
Default Headers:
  Authorization: Bearer your-api-key
  Content-Type: application/json
```

### Example Dashboard Components

#### Cost Summary Component

```javascript
// Query: costSummary
{
  method: 'GET',
  url: `/api/v1/cost-dashboard/summary`,
  params: {
    project_id: {{ projectId.value }},
    days: {{ daysSlider.value }}
  }
}

// Display component
<Container>
  <Heading>Cost Summary</Heading>
  <Statistic 
    value={{ formatCurrency(costSummary.data.summary.total_cost_usd) }} 
    label="Total Cost" 
  />
  <Statistic 
    value={{ formatCurrency(costSummary.data.summary.daily_avg_cost_usd) }} 
    label="Daily Average" 
  />
  <Statistic 
    value={{ costSummary.data.summary.total_queries }} 
    label="Total Queries" 
  />
</Container>
```

#### Cost Trends Chart

```javascript
// Query: costTrends
{
  method: 'GET',
  url: `/api/v1/cost-dashboard/trends`,
  params: {
    project_id: {{ projectId.value }},
    days: {{ daysSlider.value }},
    granularity: {{ granularitySelect.value }}
  }
}

// Display component
<Container>
  <Heading>Cost Trends</Heading>
  <LineChart 
    data={{ costTrends.data.trends }}
    x="period"
    y="total_cost_usd"
    yAxisTitle="Cost (USD)"
  />
</Container>
```

#### Anomaly Detection Timeline

```javascript
// Query: costAnomalies
{
  method: 'GET',
  url: `/api/v1/cost-dashboard/anomalies`,
  params: {
    project_id: {{ projectId.value }},
    days: {{ daysSlider.value }},
    use_ml: {{ useMLSwitch.value }}
  }
}

// Display component
<Container>
  <Heading>Cost Anomalies</Heading>
  <Table 
    data={{ costAnomalies.data.anomalies.daily_anomalies }}
    columns={[
      { header: 'Date', accessorKey: 'date' },
      { header: 'Actual Cost', accessorKey: 'total_cost_usd', formatter: 'currency' },
      { header: 'Expected Cost', accessorKey: 'expected_cost_usd', formatter: 'currency' },
      { header: 'Change', accessorKey: 'percent_change', formatter: 'percent' },
      { header: 'Z-Score', accessorKey: 'z_score' }
    ]}
  />
</Container>
```

### JavaScript Transformations

Useful transformations for Retool dashboards:

```javascript
// Format currency values
function formatCurrency(value) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(value);
}

// Calculate month-over-month change
function calculateMoMChange(comparisonData) {
  if (!comparisonData || !comparisonData.comparison) return null;
  
  const current = comparisonData.comparison.current_period.total_cost_usd;
  const previous = comparisonData.comparison.previous_period.total_cost_usd;
  const percentChange = ((current - previous) / previous) * 100;
  
  return {
    value: percentChange,
    formatted: percentChange.toFixed(2) + '%',
    positive: percentChange > 0
  };
}

// Process anomaly data for timeline visualization
function processAnomaliesForTimeline(anomalies) {
  if (!anomalies || !anomalies.anomalies || !anomalies.anomalies.daily_anomalies) {
    return [];
  }
  
  return anomalies.anomalies.daily_anomalies.map(anomaly => ({
    date: new Date(anomaly.date),
    cost: anomaly.total_cost_usd,
    expected: anomaly.expected_cost_usd,
    severity: anomaly.z_score > 3 ? 'high' : 
              anomaly.z_score > 2 ? 'medium' : 'low',
    tooltip: `Cost: ${formatCurrency(anomaly.total_cost_usd)}<br>` +
             `Expected: ${formatCurrency(anomaly.expected_cost_usd)}<br>` +
             `Change: ${anomaly.percent_change.toFixed(2)}%`
  }));
}
```

## Advanced Usage

### Handling Errors

Implement robust error handling in your client code:

```javascript
async function fetchWithErrorHandling(endpoint, params) {
  try {
    const response = await fetch(endpoint + '?' + new URLSearchParams(params), {
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Accept': 'application/json'
      }
    });
    
    if (!response.ok) {
      // Handle specific error codes
      if (response.status === 401) {
        throw new Error('Authentication failed. Check your API key.');
      } else if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After') || 60;
        throw new Error(`Rate limit exceeded. Retry after ${retryAfter} seconds.`);
      } else {
        const errorData = await response.json();
        throw new Error(`API error: ${errorData.error || response.statusText}`);
      }
    }
    
    return await response.json();
  } catch (error) {
    console.error('Request failed:', error);
    throw error;
  }
}
```

### Multi-Project Analysis

To analyze multiple GCP projects:

```python
from bigquerycostopt.examples.cost_dashboard_client import CostDashboardClient

# Initialize client
client = CostDashboardClient(base_url="http://localhost:8080", api_key="your-api-key")

# List of projects to analyze
projects = ["project-1", "project-2", "project-3"]

# Analyze each project
aggregated_results = {
    "total_cost_usd": 0,
    "cost_by_project": {},
    "cost_by_user_across_projects": {}
}

user_costs = {}

for project_id in projects:
    # Get cost summary
    summary = client.get_cost_summary(project_id, days=30)
    project_cost = summary["summary"]["total_cost_usd"]
    
    # Update aggregated results
    aggregated_results["total_cost_usd"] += project_cost
    aggregated_results["cost_by_project"][project_id] = project_cost
    
    # Get cost attribution
    attribution = client.get_cost_attribution(project_id, days=30, dimensions="user")
    
    # Aggregate user costs across projects
    for user_cost in attribution["attribution"]["cost_by_user"]:
        user_email = user_cost["user_email"]
        cost = user_cost["estimated_cost_usd"]
        
        if user_email in user_costs:
            user_costs[user_email] += cost
        else:
            user_costs[user_email] = cost

# Convert to list format
aggregated_results["cost_by_user_across_projects"] = [
    {"user_email": email, "estimated_cost_usd": cost}
    for email, cost in user_costs.items()
]

# Sort by cost (highest first)
aggregated_results["cost_by_user_across_projects"].sort(
    key=lambda x: x["estimated_cost_usd"], 
    reverse=True
)
```

### Custom Anomaly Detection

You can implement custom anomaly detection thresholds:

```python
from bigquerycostopt.examples.cost_dashboard_client import CostDashboardClient

# Initialize client
client = CostDashboardClient(base_url="http://localhost:8080", api_key="your-api-key")

# Get anomalies
anomalies = client.get_anomalies("your-project-id", days=30, use_ml=False)

# Custom filtering with different threshold
custom_threshold = 2.0  # Lower than the default threshold
custom_anomalies = []

for anomaly in anomalies["anomalies"]["daily_anomalies"]:
    if abs(anomaly["z_score"]) > custom_threshold:
        custom_anomalies.append(anomaly)

print(f"Found {len(custom_anomalies)} anomalies with custom threshold {custom_threshold}")
```

### Webhook Integration

Set up webhooks to receive alerts automatically:

1. Create an endpoint in your application to receive alert data
2. Register the webhook URL with the API server
3. Configure alert thresholds

Example webhook registration:

```python
import requests

# Register webhook
response = requests.post(
    "http://localhost:8080/api/v1/webhooks/register",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "project_id": "your-project-id",
        "webhook_url": "https://your-app.example.com/webhooks/cost-alerts",
        "triggers": ["daily_anomaly", "user_anomaly"],
        "min_cost_increase_usd": 50.0,
        "min_z_score": 2.5
    }
)

print(response.json())
```

Example webhook handler (Flask):

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhooks/cost-alerts', methods=['POST'])
def handle_cost_alert():
    data = request.json
    
    # Verify webhook signature
    signature = request.headers.get('X-Webhook-Signature')
    # Implement signature verification logic
    
    # Process alert
    alert_type = data.get('type')
    severity = data.get('severity')
    message = data.get('message')
    
    # Log or store the alert
    print(f"Received {severity} alert: {message}")
    
    # You could send the alert to Slack, email, etc.
    
    return jsonify({"status": "received"})

if __name__ == '__main__':
    app.run(port=5000)
```

## Security Considerations

### API Key Management

Best practices for API key management:

1. **Use unique keys** for each integration or client
2. **Rotate keys** regularly (at least every 90 days)
3. **Limit access** based on the principle of least privilege
4. **Monitor usage** to detect unusual patterns
5. **Store keys securely** in environment variables or secrets management services
6. **Never expose keys** in client-side code or repositories

### Network Security

The API server implements several security measures:

1. **HTTPS enforcement** with strict transport security
2. **CORS restrictions** with configurable allowed origins
3. **Security headers** like Content-Security-Policy and X-XSS-Protection
4. **Rate limiting** to prevent abuse
5. **Request validation** to prevent malformed inputs

For additional security in production, deploy the API server behind a reverse proxy or API gateway.

### Data Privacy

The API handles sensitive cost data. Consider these privacy measures:

1. **Data minimization**: Only collect and expose necessary data
2. **Aggregation**: Use team-level aggregation instead of individual attribution where appropriate
3. **Access controls**: Restrict access to cost data based on roles
4. **Audit logging**: Enable audit logging to track API usage
5. **Data retention**: Implement data retention policies

## Troubleshooting

### Common Issues

1. **Authentication errors**
   - Verify that your API key is valid and properly formatted
   - Check that the Authorization header is correctly set
   - Ensure the API key has not been revoked or expired

2. **Rate limit errors**
   - Implement backoff and retry logic
   - Consider batching requests or caching responses
   - Contact support to request higher rate limits for legitimate use cases

3. **Missing or incomplete data**
   - Verify that BigQuery usage logs are enabled
   - Check that the analysis period includes sufficient data
   - Ensure users have run queries during the analysis period

4. **Slow responses**
   - For larger datasets, increase request timeouts
   - Use more specific date ranges to reduce data volume
   - Implement client-side caching for frequently accessed data

### Debugging Tools

1. **API Logs**
   - Enable debug logging on the API server
   - Check server logs for detailed error information

2. **Request Inspection**
   - Use tools like Postman or cURL to test API endpoints directly
   - Examine request and response headers for debugging information

3. **Client Debugging**
   - Enable verbose logging in client libraries
   - Add timeout and retry monitoring

### Logging

The API server logs request information including:

- Request path and method
- Response status code
- Processing time
- Client IP address or API key ID (anonymized)
- Error details (for failed requests)

To enable debug logging on the server:

```bash
export API_DEBUG=true
python -m bigquerycostopt.src.api.server --server-type fastapi
```

To enable debug logging in the Python client:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

client = CostDashboardClient(base_url="http://localhost:8080", api_key="your-api-key")
```

## Support

For issues or questions about the API:

1. Check the documentation in the `/docs` directory
2. Review the API specification at `/api/docs` (FastAPI) or `/api/swagger.json` (Flask)
3. Contact support at support@example.com or submit issues via GitHub

When reporting issues, please include:

- API endpoint and request parameters
- Error message and status code
- Client library version (if applicable)
- Steps to reproduce the issue