# BigQuery Cost Intelligence Engine - Client Documentation

## Documentation Overview

This document provides an overview of all available client documentation for the BigQuery Cost Intelligence Engine. All the documentation files referenced below are included in this package.

### 1. Main Integration Guide

The **[Client Integration Guide](./guides/client_integration_guide.md)** is your primary reference document. It covers everything from installation to advanced usage, including:

- API Overview
- Installation and Setup
- Authentication
- API Reference
- Client Libraries
- Error Handling
- Performance Considerations
- Security Guidelines
- Troubleshooting
- Advanced Usage

### 2. Retool Integration

The **[Retool Integration Guide](./guides/retool_integration.md)** provides detailed instructions specifically for integrating with Retool dashboards:

- Setting up Retool resources
- Dashboard component examples
- JavaScript transforms
- Best practices for Retool dashboards
- Troubleshooting Retool-specific issues

### 3. API Reference

The **[API Reference](./api/api_reference.md)** provides technical details about all available API endpoints:

- Complete endpoint documentation
- Request/response formats
- Authentication methods
- Rate limits
- Error codes
- Pagination

### 4. Cost Dashboard Guide

The **[Cost Dashboard Guide](./guides/cost_dashboard_guide.md)** explains how to use the cost attribution dashboard:

- Dashboard features
- Configuration options
- Interpreting cost data
- Setting up alerts

### 5. Example Client Code

Several example clients are available in the `examples` directory:

- `cost_dashboard_client.py`: Python client for the cost dashboard API
- `analyze_cost_attribution.py`: Example code for cost attribution analysis
- `analyze_cost_anomalies_ml.py`: Example code for ML-based anomaly detection
- `generate_recommendations.py`: Example code for generating optimization recommendations

## Getting Started

For new users, we recommend the following steps:

1. Read the **[Client Integration Guide](./guides/client_integration_guide.md)** for a comprehensive overview
2. Set up your environment following the installation instructions
3. Try the example clients to verify your connection
4. Integrate with Retool using the **[Retool Integration Guide](./guides/retool_integration.md)**
5. Refer to the **[API Reference](./api/api_reference.md)** for detailed endpoint information

## Support

If you need assistance, please contact:

- Support Email: support@example.com
- GitHub Issues: https://github.com/example/bigquerycostopt/issues

## Deployment Documentation

For deployment and operations documentation, refer to:

- [Implementation Summary](./operations/implementation_summary.md)
- [Production Readiness](./operations/production_readiness.md)
- [Operations Runbook](./operations/runbook.md)