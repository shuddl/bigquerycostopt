# BigQuery Cost Intelligence Engine - User Guide

The BigQuery Cost Intelligence Engine is a powerful tool for analyzing your BigQuery datasets and identifying cost optimization opportunities. This guide provides instructions on how to use the system effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Analyzing Datasets](#analyzing-datasets)
3. [Viewing Recommendations](#viewing-recommendations)
4. [Implementing Recommendations](#implementing-recommendations)
5. [Providing Feedback](#providing-feedback)
6. [Tracking Savings](#tracking-savings)
7. [Dashboard Guide](#dashboard-guide)
8. [API Integration](#api-integration)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

## Getting Started

### System Access

To access the BigQuery Cost Intelligence Engine, you'll need:

1. **Dashboard Access**: URL provided by your administrator
2. **API Key**: For programmatic access (if needed)
3. **GCP Permissions**: BigQuery dataset access with at least READ permissions

### Initial Setup

1. Log in to the dashboard
2. Set up your notification preferences (email, Slack, etc.)
3. Connect your GCP projects for analysis

## Analyzing Datasets

### Starting an Analysis

1. From the dashboard, select "Analyze Dataset"
2. Choose the GCP project and dataset to analyze
3. Click "Start Analysis"

The system will analyze:
- Table structure and schema
- Storage patterns and usage
- Query patterns and costs
- Optimization opportunities

Analysis usually takes 1-10 minutes depending on dataset size.

### Scheduled Analysis

You can schedule regular analyses:

1. Go to "Settings" > "Scheduled Analysis"
2. Click "Add Schedule"
3. Select projects/datasets to analyze
4. Choose frequency (daily, weekly, monthly)
5. Set notification preferences

## Viewing Recommendations

### Recommendation Dashboard

The main dashboard shows all recommendations:

- **Priority Score**: Importance ranking (0-10)
- **Estimated Savings**: Monthly cost reduction
- **Implementation Complexity**: Difficulty rating (1-5)
- **Type**: Category of recommendation

### Filtering Recommendations

Use filters to focus on specific recommendations:

- Filter by project or dataset
- Filter by recommendation type
- Sort by priority, savings, or complexity
- Show only active/implemented recommendations

### Recommendation Details

Click on any recommendation to see details:

- Detailed description and justification
- Step-by-step implementation instructions
- SQL statements for implementation
- ROI calculation with assumptions
- ML-enhanced insights (if available)

## Implementing Recommendations

### Implementation Options

For each recommendation, you have several options:

1. **Manual Implementation**: Follow provided steps and SQL
2. **Guided Implementation**: Use our step-by-step wizard
3. **Automated Implementation**: One-click implementation (where available)

### Implementation Process

For guided implementation:

1. Click "Implement" on the recommendation card
2. Review implementation plan
3. Choose to run a dry-run first (recommended)
4. View implementation progress
5. Confirm successful implementation

### Implementation Risks

Each recommendation has a risk level:

- **Low Risk**: Minor table changes, typically safe
- **Medium Risk**: Schema changes, query pattern changes
- **High Risk**: Major structural changes, requires careful testing

Always test high-risk recommendations in development before production.

## Providing Feedback

### Recommendation Feedback

After implementing a recommendation:

1. Go to "Implemented Recommendations"
2. Click "Provide Feedback"
3. Rate the recommendation (1-5 stars)
4. Enter actual savings (if known)
5. Add comments about your experience

Your feedback helps improve future recommendations through our ML system.

### Issue Reporting

For system issues:

1. Go to "Settings" > "Report Issue"
2. Describe the problem in detail
3. Attach screenshots if helpful
4. Submit the report

## Tracking Savings

### Savings Dashboard

The Savings Dashboard provides:

- Total estimated monthly savings
- Implemented vs. potential savings
- Savings by recommendation type
- Historical savings trends

### ROI Reports

Generate detailed ROI reports:

1. Go to "Reports" > "ROI Analysis"
2. Select date range and projects
3. Choose report format (PDF, Excel, etc.)
4. Click "Generate Report"

## Dashboard Guide

### Navigation

The dashboard has several main sections:

- **Overview**: Summary metrics and high-level insights
- **Recommendations**: List of all optimization recommendations
- **Projects**: Project-level analysis and recommendations
- **History**: Record of past analyses and implementations
- **Reports**: Custom reports and savings tracking
- **Settings**: User preferences and configuration

### Visualization

Key visualizations include:

- **Savings Potential**: Bar chart of savings by category
- **Implementation Status**: Pie chart of recommendation status
- **Priority Distribution**: Histogram of priority scores
- **Cost Trend**: Line chart of BigQuery costs over time

## API Integration

### API Access

The system provides a RESTful API for integration:

1. Go to "Settings" > "API Keys"
2. Generate a new API key
3. Use the key in your API requests

### Example API Usage

Trigger a dataset analysis programmatically:

```bash
curl -X POST https://api.bqcostopt.example.com/api/v1/analyze \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "your-project-id",
    "dataset_id": "your_dataset",
    "callback_url": "https://your-callback-url.example.com/webhook"
  }'
```

See the [API Documentation](../api/api_specification.yaml) for complete details.

## Troubleshooting

### Common Issues

#### Analysis Takes Too Long

- Try analyzing a smaller dataset first
- Check if there are any access permission issues
- Verify BigQuery is not experiencing delays

#### Recommendations Not Appearing

- Ensure analysis completed successfully
- Check if filters are hiding recommendations
- Verify you have the correct permissions

#### Implementation Errors

- Review error messages in the implementation log
- Check if table structures have changed since analysis
- Verify you have the necessary permissions for BigQuery

### Getting Help

For additional assistance:

1. Check the FAQ section below
2. Contact support at support@example.com
3. Reach out to your account representative

## FAQ

### General Questions

**Q: How accurate are the savings estimates?**
A: Savings estimates are based on current usage patterns and pricing. They're typically within 15% of actual savings but may vary based on future usage changes.

**Q: How often should I run analysis?**
A: We recommend monthly analysis for most environments, or after significant changes to your BigQuery usage patterns.

**Q: Can I export recommendations to share with my team?**
A: Yes, use the Export function on the Recommendations page to download CSV, Excel, or PDF reports.

### Technical Questions

**Q: Does implementing recommendations require downtime?**
A: Most recommendations don't require downtime. Schema changes may temporarily lock tables during alteration, but this is typically brief.

**Q: How does the ML enhancement work?**
A: Our ML system analyzes patterns across your BigQuery usage, learns from feedback, and enhances recommendations for better relevance and higher ROI.

**Q: Can I customize the priority calculation?**
A: Yes, go to Settings > Recommendation Preferences to adjust weights for different factors in the priority score calculation.