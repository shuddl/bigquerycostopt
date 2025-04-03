# BigQuery Cost Attribution Dashboard Guide

This guide explains how to use the BigQuery Cost Attribution Dashboard with Anomaly Detection to track and optimize your BigQuery costs.

## Overview

The Cost Attribution Dashboard provides comprehensive visibility into your BigQuery spending with the following key features:

- **Cost Attribution**: Track costs by user, team, query pattern, and dataset
- **Anomaly Detection**: Identify unusual spending patterns using statistical and ML-based approaches
- **Cost Forecasting**: Predict future costs based on historical usage patterns
- **User Behavior Analysis**: Cluster users based on their query patterns and costs
- **Cost Alerts**: Get notified about significant cost increases

## Getting Started

### Installation

To install and set up the BigQuery Cost Attribution Dashboard:

1. Install the package with ML dependencies:
   ```
   pip install -e ".[ml]"
   ```

2. Set up your authentication:
   ```
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```
   
   The service account must have the following permissions:
   - `bigquery.jobs.list`
   - `bigquery.tables.get`
   - `bigquery.tables.list`

3. Start the API server:
   ```
   python -m bigquerycostopt.src.api.server --server-type fastapi
   ```

### Configuration

Configure the dashboard by setting the following environment variables:

- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `API_PORT`: Port for the API server (default: 8080)
- `API_HOST`: Host for the API server (default: 0.0.0.0)
- `API_SERVER_TYPE`: Server implementation to use (flask or fastapi)
- `API_DEBUG`: Enable debug mode (true or false)

## Dashboard Components

### Cost Explorer

The Cost Explorer provides hierarchical drill-down into your BigQuery costs:

1. **Project-level view**: Overall cost summary with trends
2. **Team breakdown**: Costs attributed to different teams
3. **User breakdown**: Individual user costs within teams
4. **Query pattern analysis**: Costs by query type and pattern
5. **Dataset/table usage**: Costs by dataset and table

### Anomaly Detection Timeline

The Anomaly Detection Timeline helps you spot unusual spending patterns:

1. **Daily cost anomalies**: Unusually high or low daily spending
2. **User-based anomalies**: Users with unexpected spending patterns
3. **Team-based anomalies**: Teams with unusual cost trends
4. **Pattern-based anomalies**: Unusual query pattern distributions

### Recommendation Action Center

The Recommendation Action Center provides actionable cost-saving recommendations:

1. **Query optimizations**: Suggestions for improving expensive queries
2. **Schema optimizations**: Recommendations for table structure improvements
3. **Storage optimizations**: Ideas for reducing storage costs
4. **Implementation planning**: Generate implementation scripts for recommendations

## API Reference

The Cost Attribution Dashboard is accessible through a RESTful API that supports both Flask and FastAPI implementations.

### Authentication

All API endpoints require authentication. Provide an API key in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

### Endpoints

#### Cost Summary

```
GET /api/v1/cost-dashboard/summary
```

Query parameters:
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 30)

#### Cost Attribution

```
GET /api/v1/cost-dashboard/attribution
```

Query parameters:
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 30)
- `dimensions` (optional): Comma-separated list of attribution dimensions (default: user,team,pattern,day,table)

#### Cost Trends

```
GET /api/v1/cost-dashboard/trends
```

Query parameters:
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 90)
- `granularity` (optional): Time granularity (day, week, month) (default: day)

#### Period Comparison

```
GET /api/v1/cost-dashboard/compare-periods
```

Query parameters:
- `project_id` (required): GCP project ID
- `current_days` (optional): Number of days in current period (default: 30)
- `previous_days` (optional): Number of days in previous period (default: 30)

#### Expensive Queries

```
GET /api/v1/cost-dashboard/expensive-queries
```

Query parameters:
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 30)
- `limit` (optional): Maximum number of queries to return (default: 100)

#### Cost Anomalies

```
GET /api/v1/cost-dashboard/anomalies
```

Query parameters:
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 30)
- `anomaly_types` (optional): Comma-separated list of anomaly types (default: daily,user,team,pattern)
- `use_ml` (optional): Whether to use ML-enhanced anomaly detection (default: false)

#### Cost Forecast

```
GET /api/v1/cost-dashboard/forecast
```

Query parameters:
- `project_id` (required): GCP project ID
- `training_days` (optional): Number of days to use for training (default: 90)
- `forecast_days` (optional): Number of days to forecast (default: 7)

#### User Clusters

```
GET /api/v1/cost-dashboard/user-clusters
```

Query parameters:
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to analyze (default: 90)
- `n_clusters` (optional): Number of clusters to generate (default: 5)

#### Team Mapping

```
POST /api/v1/cost-dashboard/team-mapping
```

Request body:
```json
{
  "project_id": "your-project-id",
  "mapping": {
    "user1@example.com": "Team A",
    "user2@example.com": "Team B",
    "*@engineering.example.com": "Engineering"
  }
}
```

#### Cost Alerts

```
GET /api/v1/cost-dashboard/alerts
```

Query parameters:
- `project_id` (required): GCP project ID
- `days` (optional): Number of days to look back (default: 7)
- `min_cost_increase_usd` (optional): Minimum cost increase to trigger alert (default: 100.0)

## Retool Integration

The Cost Attribution Dashboard is designed to integrate seamlessly with Retool dashboards.

### Setting Up Retool

1. Create a new Retool application
2. Add a REST API resource pointing to your API server
3. Configure authentication with your API key
4. Create the following queries:

### Example Retool Components

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

## Client Usage

You can use the included Python client to interact with the API programmatically:

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

```
python -m bigquerycostopt.examples.cost_dashboard_client --project-id=your-project-id --days=30 --use-ml --visualize
```

## Best Practices

### Cost Attribution

1. **Set up team mapping**: Use the team mapping API to correctly attribute costs to teams
2. **Use consistent project structure**: Organize datasets by team or function for better attribution
3. **Add user labels**: Add labels to your queries for more detailed attribution

### Anomaly Detection

1. **Start with statistical anomalies**: Begin with basic statistical anomaly detection before using ML
2. **Adjust thresholds**: Fine-tune the anomaly detection thresholds based on your usage patterns
3. **Combine with alerting**: Set up alerts for significant anomalies to be notified in real-time

### Cost Optimization

1. **Focus on high-impact areas**: Prioritize optimizations for your most expensive query patterns
2. **Regular reviews**: Schedule regular cost review sessions with stakeholders
3. **Implement and verify**: Track the impact of implemented optimizations over time

## Advanced Topics

### Custom Machine Learning Models

You can extend the ML-based anomaly detection with custom models:

1. Create a subclass of `MLCostAnomalyDetector`
2. Override the `train` and `predict` methods
3. Register your model in the `detect_anomalies_with_ml` function

### Integrating with Monitoring Systems

The Cost Attribution Dashboard can be integrated with monitoring systems:

1. Use the alerts API to fetch cost alerts
2. Push alerts to your monitoring system (e.g., PagerDuty, Slack)
3. Set up automated responses to cost anomalies

### Multi-Project Aggregation

For organizations with multiple projects:

1. Deploy the API server with access to all projects
2. Create a metadata table to store project information
3. Implement custom aggregation logic to combine costs across projects

## Troubleshooting

### Common Issues

1. **Authentication errors**: Ensure your service account has the necessary permissions
2. **Missing data**: Check that your BigQuery usage logs are enabled
3. **API timeouts**: For large datasets, increase the API request timeout

### Debugging

1. Enable debug mode with `--debug` or `API_DEBUG=true`
2. Check the API server logs for detailed error information
3. Validate your API requests with a tool like Postman

## Getting Help

If you encounter issues or have questions:

1. Check the documentation in the `/docs` directory
2. Submit issues via GitHub
3. Contact the support team at support@example.com