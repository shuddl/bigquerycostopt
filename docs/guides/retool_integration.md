# Retool Integration Guide

This guide provides specific instructions for integrating the BigQuery Cost Intelligence Engine with Retool dashboards.

## Overview

Retool is a powerful platform for building internal tools and dashboards. The BigQuery Cost Intelligence Engine's API is fully compatible with Retool, allowing you to create custom dashboards for cost optimization, attribution, and anomaly detection.

## Setting Up the Retool Resource

1. In Retool, create a new Resource:
   - Go to **Resources** > **Create new** > **REST API**
   - Name: `BigQuery Cost Intelligence API`
   - Base URL: `https://your-api-server.com` (replace with your actual API endpoint)
   - Authentication: Select **Bearer token**
   - Token: Enter your API key
   - Headers: Add `Content-Type: application/json`

2. Test the connection by clicking **Test** and using a simple endpoint like `/api/health`

## Example Retool Dashboard Components

### 1. Project Selector

Create a dropdown to select from available GCP projects:

```json
{
  "name": "projectSelector",
  "type": "select",
  "options": "{{self.data.map(p => ({label: p.name, value: p.id}))}}"
}
```

Create a query to load projects:

```javascript
{
  "method": "GET",
  "url": "{{resourceUrl}}/api/v1/cost-dashboard/projects",
  "headers": {}
}
```

### 2. Cost Summary Component

Create a query:

```javascript
{
  "method": "GET",
  "url": "{{resourceUrl}}/api/v1/cost-dashboard/summary",
  "params": {
    "project_id": "{{projectSelector.value}}",
    "days": "{{daysSelector.value || 30}}"
  }
}
```

Create a container with summary cards:

```json
{
  "type": "container",
  "items": [
    {
      "type": "statistic",
      "label": "Total Cost",
      "value": "{{formatCurrency(costSummary.data.total_cost)}}"
    },
    {
      "type": "statistic",
      "label": "Queries Run",
      "value": "{{formatNumber(costSummary.data.query_count)}}"
    },
    {
      "type": "statistic",
      "label": "Cost Per Query",
      "value": "{{formatCurrency(costSummary.data.avg_cost_per_query)}}"
    }
  ]
}
```

### 3. Cost Trends Chart

Create a query:

```javascript
{
  "method": "GET",
  "url": "{{resourceUrl}}/api/v1/cost-dashboard/trends",
  "params": {
    "project_id": "{{projectSelector.value}}",
    "days": "{{daysSelector.value || 90}}",
    "granularity": "{{granularitySelector.value || 'day'}}"
  }
}
```

Create a chart component:

```json
{
  "type": "chart",
  "data": "{{costTrends.data.trends}}",
  "chartType": "line",
  "xAxis": "period",
  "series": [
    {
      "dataKey": "cost",
      "name": "Cost (USD)",
      "color": "#4CAF50"
    }
  ],
  "height": 300
}
```

### 4. Top Users Table

Create a query:

```javascript
{
  "method": "GET",
  "url": "{{resourceUrl}}/api/v1/cost-dashboard/user-attribution",
  "params": {
    "project_id": "{{projectSelector.value}}",
    "days": "{{daysSelector.value || 30}}",
    "limit": 10
  }
}
```

Create a table component:

```json
{
  "type": "table",
  "data": "{{userAttribution.data.users}}",
  "columns": [
    {
      "id": "user",
      "title": "User",
      "type": "text"
    },
    {
      "id": "total_cost",
      "title": "Total Cost",
      "type": "currency"
    },
    {
      "id": "query_count",
      "title": "Queries",
      "type": "number"
    },
    {
      "id": "avg_cost_per_query",
      "title": "Avg. Cost/Query",
      "type": "currency"
    }
  ]
}
```

### 5. Anomalies Section

Create a query:

```javascript
{
  "method": "GET",
  "url": "{{resourceUrl}}/api/v1/cost-dashboard/anomalies",
  "params": {
    "project_id": "{{projectSelector.value}}",
    "days": "{{daysSelector.value || 30}}",
    "detection_method": "{{detectionMethodSelector.value || 'ml'}}",
    "sensitivity": 0.7
  }
}
```

Create an anomalies list component:

```json
{
  "type": "list",
  "data": "{{anomalies.data.anomalies}}",
  "itemView": {
    "type": "container",
    "items": [
      {
        "type": "text",
        "value": "{{item.date}}: {{item.severity}} anomaly",
        "style": {
          "fontWeight": "bold",
          "color": "{{item.severity === 'high' ? 'red' : (item.severity === 'medium' ? 'orange' : 'yellow')}}"
        }
      },
      {
        "type": "text",
        "value": "Expected: {{formatCurrency(item.expected_cost)}} | Actual: {{formatCurrency(item.actual_cost)}} ({{item.deviation_percent}}% deviation)"
      },
      {
        "type": "text",
        "value": "Likely cause: {{item.likely_cause}}"
      }
    ]
  }
}
```

### 6. Recommendations Table

Create a query:

```javascript
{
  "method": "GET",
  "url": "{{resourceUrl}}/api/v1/recommendations",
  "params": {
    "project_id": "{{projectSelector.value}}",
    "priority": "{{prioritySelector.value || 'high'}}",
    "limit": 10
  }
}
```

Create a recommendations table component:

```json
{
  "type": "table",
  "data": "{{recommendations.data.recommendations}}",
  "columns": [
    {
      "id": "description",
      "title": "Recommendation",
      "type": "text"
    },
    {
      "id": "annual_savings_usd",
      "title": "Annual Savings",
      "type": "currency"
    },
    {
      "id": "roi",
      "title": "ROI",
      "type": "number",
      "format": "{{value.toFixed(1)}}x"
    },
    {
      "id": "priority",
      "title": "Priority",
      "type": "badge",
      "valueMapping": {
        "high": { "color": "red" },
        "medium": { "color": "orange" },
        "low": { "color": "blue" }
      }
    }
  ]
}
```

## Complete Dashboard Example

A complete Retool JSON configuration is available in our examples folder at `/examples/retool_dashboard.json`. You can import this directly into Retool to get started quickly.

## Best Practices

1. **Implement Caching**: Use Retool's caching for infrequently changing data to reduce API load

2. **Add Refreshers**: Add refresh buttons or automatic refreshing for critical metrics

3. **Use Filters Effectively**: Implement date range filters, project selectors, and other filters to allow users to drill down into the data

4. **Set Up Permissions**: Configure Retool permissions to ensure users can only see data for projects they should have access to

5. **Create Multiple Views**: Consider creating separate dashboard tabs for different audiences:
   - Executive summary for management
   - Detailed cost attribution for finance
   - Technical recommendations for engineers
   - Anomaly detection for operations

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Check that your API key is valid and properly formatted in the Retool resource configuration

2. **Missing Data**: Ensure that the project ID is being correctly passed from the project selector to your API queries

3. **Chart Formatting Issues**: If charts aren't displaying correctly, check the data format being returned from the API and ensure it matches what the chart component expects

4. **Performance Problems**: For slow-loading dashboards, implement pagination and filtering to reduce the amount of data being loaded at once

## Example JavaScript Transforms

Here are some useful JavaScript transforms for formatting data in your Retool dashboard:

### Format Bytes

```javascript
function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
```

### Calculate Percent Change

```javascript
function percentChange(current, previous) {
  if (previous === 0) return 'N/A';
  return ((current - previous) / previous * 100).toFixed(1) + '%';
}
```

### Format Duration

```javascript
function formatDuration(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  
  return `${h > 0 ? h + 'h ' : ''}${m > 0 ? m + 'm ' : ''}${s}s`;
}
```

## Custom Visualizations

For advanced visualizations beyond Retool's built-in components, consider embedding custom visualizations:

1. Create a custom HTML component
2. Use libraries like D3.js or Chart.js
3. Pass data from your API queries using template variables

Example D3.js integration:

```html
<div id="customVisualization"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
  const data = {{JSON.stringify(costTrends.data.trends)}};
  
  // Use D3 to create a custom visualization here
  // Example: cost heatmap by user and time
</script>
```

## Next Steps

1. **Set Up Alerts**: Configure Retool alerts for cost anomalies or threshold breaches
2. **Create Custom Reports**: Use Retool's PDF export to generate scheduled reports
3. **Integrate with Slack**: Set up Slack notifications for critical alerts
4. **Implement User Feedback**: Add feedback buttons to get user input on recommendations