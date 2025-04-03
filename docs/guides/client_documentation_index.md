# BigQuery Cost Intelligence Engine - Client Documentation Index

This document serves as the primary index for all client-related documentation for the BigQuery Cost Intelligence Engine. Use this as your starting point to find specific documentation about API usage, integration guides, and code examples.

## Getting Started

- [Client Integration Guide](client_integration_guide.md) - Comprehensive documentation covering all aspects of API integration
- [Cost Dashboard Guide](cost_dashboard_guide.md) - Guide for using the BigQuery Cost Attribution Dashboard with Anomaly Detection
- [User Guide](user_guide.md) - General user guide for the BigQuery Cost Intelligence Engine

## API Documentation

- [API Reference](../api/api_specification.yaml) - OpenAPI specification for all API endpoints
- [API Server Options](client_integration_guide.md#api-server-options) - Information about the FastAPI and Flask server implementations
- [Authentication Guide](client_integration_guide.md#authentication) - Details on API key management and authentication

## Integration Guides

- [Retool Integration Guide](client_integration_guide.md#retool-integration) - How to integrate with Retool dashboards
- [Python Client Usage](client_integration_guide.md#python-client) - Documentation for the Python client library
- [JavaScript Client Usage](client_integration_guide.md#javascript-client) - Documentation for the JavaScript client library
- [Webhook Integration](client_integration_guide.md#webhook-integration) - How to set up webhooks for cost alerts

## Features

- [Cost Attribution](cost_dashboard_guide.md#overview) - Understanding cost attribution by user, team, and query pattern
- [Anomaly Detection](client_integration_guide.md#cost-anomalies) - Statistical and ML-based methods for detecting cost anomalies
- [Cost Forecasting](client_integration_guide.md#cost-forecast) - Predicting future costs based on historical usage
- [User Behavior Analysis](client_integration_guide.md#user-clusters) - Clustering users based on their query patterns and costs

## Advanced Topics

- [Security Considerations](client_integration_guide.md#security-considerations) - Best practices for secure API usage
- [Multi-Project Analysis](client_integration_guide.md#multi-project-analysis) - Analyzing costs across multiple GCP projects
- [Custom Anomaly Detection](client_integration_guide.md#custom-anomaly-detection) - Implementing custom anomaly detection thresholds
- [Response Caching](client_integration_guide.md#response-caching) - How to use and configure API response caching

## Code Examples

- [Python Client Examples](../examples/cost_dashboard_client.py) - Complete Python client implementation with examples
- [JavaScript Transformations](client_integration_guide.md#javascript-transformations) - Useful JavaScript functions for data processing
- [Error Handling](client_integration_guide.md#handling-errors) - Examples of robust error handling for API requests
- [Retool Component Examples](client_integration_guide.md#example-dashboard-components) - Sample Retool dashboard components

## Troubleshooting

- [Common Issues](client_integration_guide.md#common-issues) - Solutions for frequently encountered problems
- [Debugging Tools](client_integration_guide.md#debugging-tools) - Tools and techniques for debugging API integration
- [Logging](client_integration_guide.md#logging) - How to enable and use logging for troubleshooting

## Implementation Status

- [Implementation Status](IMPLEMENTATION_STATUS.md) - Current status of implemented features
- [System Architecture](../architecture/system_architecture.md) - Overview of the system architecture
- [Production Readiness](../operations/production_readiness.md) - Guidelines for production deployment

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