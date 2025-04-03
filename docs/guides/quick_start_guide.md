# BigQuery Cost Intelligence Engine - Quick Start Guide

This quick start guide will help you get up and running with the BigQuery Cost Intelligence Engine, a tool for analyzing and optimizing your BigQuery costs.

## Prerequisites

Before you begin, make sure you have:

- Python 3.9 or higher installed
- A Google Cloud project with BigQuery enabled
- Sufficient permissions to access BigQuery usage data (typically `roles/bigquery.admin` or `roles/bigquery.resourceViewer`)
- Service account credentials with appropriate permissions (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd bigquerycostopt
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the Package

For basic functionality:

```bash
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev,test]"
```

For ML-enhanced features:

```bash
pip install -e ".[ml]"
```

### 4. Set Up Authentication

Set up credentials for Google Cloud:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

On Windows:

```bash
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account-key.json
```

## Basic Usage

### 1. Run a Simple Analysis

Analyze a dataset to get cost optimization recommendations:

```python
from bigquerycostopt.src.analysis.query_optimizer import QueryOptimizer
from bigquerycostopt.src.analysis.schema_optimizer import SchemaOptimizer
from bigquerycostopt.src.analysis.storage_optimizer import StorageOptimizer

# Initialize optimizers
project_id = "your-project-id"
dataset_id = "your_dataset"

query_optimizer = QueryOptimizer(project_id)
schema_optimizer = SchemaOptimizer(project_id)
storage_optimizer = StorageOptimizer(project_id)

# Get recommendations
query_recommendations = query_optimizer.analyze_dataset_queries(dataset_id)
schema_recommendations = schema_optimizer.analyze_dataset_schemas(dataset_id)
storage_recommendations = storage_optimizer.analyze_dataset(dataset_id)

# Print recommendations count
print(f"Found {len(query_recommendations)} query optimizations")
print(f"Found {len(schema_recommendations)} schema optimizations")
print(f"Found {len(storage_recommendations)} storage optimizations")
```

### 2. Start the API Server

Launch the API server for integrating with dashboards:

```bash
python -m bigquerycostopt.src.api.server --server-type fastapi
```

The server will start at http://0.0.0.0:8080 by default.

## Using the Cost Dashboard

### 1. Access the Dashboard API

Once the server is running, you can access the API endpoints:

```bash
# Get a summary of costs
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:8080/api/v1/cost-dashboard/summary?project_id=your-project-id"

# Get cost anomalies with ML enhancement
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:8080/api/v1/cost-dashboard/anomalies?project_id=your-project-id&use_ml=true"
```

### 2. Use the Python Client

You can also use the included Python client:

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

## Common Operations

### Analyzing a New Dataset

```python
from bigquerycostopt.src.analysis.query_optimizer import QueryOptimizer

optimizer = QueryOptimizer("your-project-id")
recommendations = optimizer.analyze_dataset_queries("your_dataset")

# Generate implementation SQL
for rec in recommendations:
    implementation_sql = optimizer.generate_implementation_sql(rec)
    print(f"Implementation SQL for {rec['recommendation_id']}:\n{implementation_sql}")
```

### Generating a Cost Report

```python
from bigquerycostopt.src.analysis.cost_attribution import CostAttributionAnalyzer

analyzer = CostAttributionAnalyzer("your-project-id")
cost_summary = analyzer.get_cost_summary(days_back=30)
attribution = analyzer.attribute_costs(days_back=30)

# Print cost by team
for team, cost in attribution["cost_by_team"].items():
    print(f"Team: {team}, Cost: ${cost:.2f}")
```

### Getting Cost Forecasts with ML

```python
from bigquerycostopt.src.analysis.cost_attribution import CostAttributionAnalyzer
from bigquerycostopt.src.ml.cost_anomaly_detection import TimeSeriesForecaster

analyzer = CostAttributionAnalyzer("your-project-id")
forecaster = TimeSeriesForecaster(analyzer)

# Generate a forecast for the next 7 days
forecast = forecaster.forecast_daily_costs(training_days=90, forecast_days=7)
print(f"Forecast for next 7 days: ${forecast['forecast_total']:.2f}")
```

## Next Steps

After getting started, you might want to:

1. Explore the [Cost Dashboard Guide](cost_dashboard_guide.md) for dashboard integration
2. Check the [User Guide](user_guide.md) for more detailed usage information
3. Set up automated analysis jobs using the API
4. Integrate with your existing monitoring systems

## Troubleshooting

### Common Issues

#### Authentication Errors

```
google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials
```

Solution: Make sure you've set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable correctly.

#### Missing Dependencies

```
ImportError: cannot import name 'storage' from 'google.cloud'
```

Solution: Install the Google Cloud Storage dependency with `pip install google-cloud-storage`.

#### API Connection Issues

```
requests.exceptions.ConnectionError: Failed to establish a connection
```

Solution: Make sure the API server is running and the URL is correct.

### Getting Help

If you encounter issues or have questions:

1. Check the documentation in the `/docs` directory
2. Submit issues via GitHub
3. Run diagnostics with `python -m bigquerycostopt.verify_installation`