# BigQuery Cost Intelligence Engine (BCIE)

A serverless application that analyzes BigQuery datasets and provides actionable cost optimization recommendations with ROI estimates.

## Features

- Analyzes complete BigQuery datasets (100K to 50M+ records)
- Identifies storage optimization opportunities (partitioning, clustering)
- Detects inefficient query patterns
- Recommends schema optimizations
- Generates implementation plans with SQL scripts
- Calculates ROI and effort estimates
- Integrates with Retool dashboards

## Architecture

BCIE uses a serverless, event-driven architecture on Google Cloud Platform:

- Cloud Run for API endpoints
- Cloud Functions for analysis modules
- Cloud Pub/Sub for event handling
- BigQuery for data storage and analysis
- Cloud Storage for processing artifacts

See `architecture.md` for the complete system design.

## Getting Started

```bash
# Install dependencies
pip install -e .

# Run tests
pytest

# Deploy to GCP
./deploy.sh
```

## Documentation

- [Architecture Document](architecture.md)
- [API Documentation](docs/api.md)
- [Module Reference](docs/modules.md)

## License

MIT