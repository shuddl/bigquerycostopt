# Claude Coding Assistant Guidelines

## Commands

- Build: `pip install -e .` or `poetry install`
- Lint: `flake8 bigquerycostopt tests`
- Type check: `mypy bigquerycostopt`
- Test all: `pytest`
- Test single: `pytest tests/path_to_test.py::TestClass::test_name`

## Code Style

- Follow PEP 8 for formatting
- Use type hints for all function parameters and return values
- Import order: standard library, third-party, local packages
- Name classes in CamelCase, functions/variables in snake_case
- Use descriptive variable names - avoid abbreviations
- Handle errors explicitly with try/except - log all exceptions
- Use f-strings for string formatting
- Document all public functions and classes with docstrings
- Maximum line length: 88 characters (Black formatter default)
- Use dataclasses for data containers

## Project Overview

The BigQuery Cost Intelligence Engine (BCIE) is designed to analyze large BigQuery datasets and provide actionable cost-saving recommendations. It integrates with Retool dashboards, stores recommendations in BigQuery, and leverages machine learning for enhanced insights.

## Architecture Guidelines

- **Serverless Focus:**
  - Utilize serverless GCP components (Cloud Run, Cloud Functions, Pub/Sub)
- **Asynchronous Processing:**
  - Employ Pub/Sub for asynchronous messaging and processing of large datasets
- **BigQuery Storage:**
  - Store all recommendations and status information in BigQuery tables
- **API Design:**
  - Design RESTful APIs for integration with Retool dashboards
  - Ensure APIs are secure and authenticated
- **Modular Design:**
  - Maintain a modular design with clear separation of concerns

## Module-Specific Guidelines

- **Data Connector:**
  - Use appropriate authentication methods for BigQuery API access
  - Implement efficient, paginated queries for large result sets
- **Optimizer Modules:**
  - Develop specific analysis algorithms for each optimization area
  - Calculate potential cost savings and ROI for recommendations
- **Recommendation Engine:**
  - Standardize recommendation data structures
  - Implement prioritization logic based on impact and effort
- **Implementation Plan Generator:**
  - Generate executable SQL scripts and verification queries
  - Provide rollback procedures for safety
- **Retool Integration Layer:**
  - Handle webhook calls and asynchronous processing
  - Provide status updates and store results in BigQuery
- **Machine Learning Enhancement Module:**
  - Implement feature engineering and model training pipelines
  - Enhance recommendations with ML-derived insights

## Testing Guidelines

- **Unit Tests:** Write unit tests for all modules with high coverage
- **Integration Tests:** Create tests to verify interactions between modules
- **End-to-End Tests:** Implement tests to validate complete workflows
- **Performance Tests:** Run tests with varying dataset sizes
