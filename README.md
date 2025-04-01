# BigQuery Cost Intelligence Engine (BCIE)

A comprehensive solution that analyzes BigQuery datasets and provides actionable cost optimization recommendations with ROI estimates, implementation plans, and ML-enhanced insights.

## Features

- Analyzes complete BigQuery datasets (100K to 50M+ records)
- Identifies storage optimization opportunities (partitioning, clustering)
- Detects inefficient query patterns and suggests query optimizations
- Recommends schema optimizations for cost reduction
- Provides ML-enhanced recommendations with business context
- Generates implementation plans with SQL scripts and step-by-step instructions
- Calculates detailed ROI and effort estimates
- Integrates with Retool dashboards for visualization
- Tracks implementation success and actual savings

## Architecture

BCIE uses a serverless, event-driven architecture on Google Cloud Platform with infrastructure as code:

- Cloud Run for API endpoints (autoscaling, high availability)
- Cloud Functions for analysis workers
- Cloud Pub/Sub for asynchronous event handling
- BigQuery for data storage and analysis
- Cloud Monitoring for observability and alerting
- Secret Manager for secure configuration
- Infrastructure defined with Terraform

See [System Architecture](docs/architecture/system_architecture.md) for the complete system design.

## Production Infrastructure

The production infrastructure includes:

- **Infrastructure as Code**: Complete Terraform modules for all GCP resources
- **Security Controls**: Least-privilege IAM roles, Secret Manager, VPC Service Controls
- **Monitoring & Observability**: Custom dashboards, metrics, and alerting
- **CI/CD Pipeline**: GitHub Actions for testing and deployment
- **Multi-Environment**: Development, staging, and production environments

## Getting Started

### Development Setup

```bash
# Set up development environment
./setup_dev.sh

# Activate virtual environment
source venv/bin/activate

# Run tests
./run_tests.sh
```

### Deployment

```bash
# Deploy to GCP (development environment)
./deploy.sh dev PROJECT_ID

# Deploy to production
./deploy.sh prod PROJECT_ID
```

## Documentation

### Architecture & Design
- [System Architecture](docs/architecture/system_architecture.md)
- [API Specification](docs/api/api_specification.yaml)

### User Guides
- [User Guide](docs/guides/user_guide.md)

### Operations
- [Operations Runbook](docs/operations/runbook.md)
- [Production Readiness Checklist](docs/operations/production_readiness.md)

## Repository Structure

- `src/` - Source code for all modules
  - `analysis/` - Analysis and optimizer modules
  - `api/` - API service endpoints
  - `connectors/` - Data source connectors
  - `implementation/` - Implementation planning
  - `ml/` - Machine learning enhancement
  - `recommender/` - Recommendation engine
  - `utils/` - Utility functions

- `infra/` - Infrastructure as code
  - `terraform/` - Terraform modules and environments
  
- `function_source/` - Cloud Functions source code

- `tests/` - Test suites
  - `unit/` - Unit tests
  - `integration/` - Integration tests
  - `performance/` - Performance tests

- `docs/` - Documentation

## License

MIT