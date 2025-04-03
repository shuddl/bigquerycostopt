# BigQuery Cost Intelligence Engine - Troubleshooting Guide

This guide provides solutions for common issues you may encounter when using the BigQuery Cost Intelligence Engine.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Authentication Issues](#authentication-issues)
- [API Server Issues](#api-server-issues)
- [Performance Issues](#performance-issues)
- [Data Access Issues](#data-access-issues)
- [ML Component Issues](#ml-component-issues)
- [Dashboard Integration Issues](#dashboard-integration-issues)
- [Diagnostic Tools](#diagnostic-tools)

## Installation Issues

### Missing Dependencies

**Problem:** Installation fails with dependency errors.

**Solution:**

1. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
2. For ML features, install ML dependencies:
   ```bash
   pip install -r requirements_ml.txt
   ```

3. Install specific missing packages:
   ```bash
   pip install google-cloud-bigquery google-cloud-storage
   ```

### Python Version Compatibility

**Problem:** Errors about incompatible Python version.

**Solution:** Ensure you're using Python 3.9 or higher:

```bash
python --version
# If older than 3.9, upgrade Python or create a new environment
python3.9 -m venv venv
source venv/bin/activate
```

### Installation in Development Mode

**Problem:** Changes to code are not reflected when running the application.

**Solution:** Install in development mode:

```bash
pip install -e .
```

## Authentication Issues

### Missing Credentials

**Problem:** Authentication errors when connecting to Google Cloud.

**Solution:**

1. Set up service account credentials:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

2. Verify credentials are working:
   ```python
   from google.cloud import bigquery
   client = bigquery.Client()
   # If no error, authentication is working
   ```

### Insufficient Permissions

**Problem:** Access denied errors when accessing BigQuery.

**Solution:**

1. Ensure your service account has the necessary roles:
   - `roles/bigquery.admin` or `roles/bigquery.resourceViewer` for reading usage data
   - `roles/bigquery.jobUser` for running queries
   - `roles/bigquery.dataViewer` for reading table data

2. Check project-level permissions in Google Cloud Console.

3. Verify dataset access:
   ```bash
   bq ls --project_id=your-project-id your_dataset
   ```

## API Server Issues

### Server Won't Start

**Problem:** API server fails to start with errors.

**Solution:**

1. Check for port conflicts:
   ```bash
   lsof -i :8080
   # If port is in use, change the port
   python -m bigquerycostopt.src.api.server --port=8081
   ```

2. Check for FastAPI installation:
   ```bash
   pip install fastapi uvicorn
   ```

3. Look for detailed error messages in the console output.

### API Authentication Errors

**Problem:** Unauthorized errors when accessing API endpoints.

**Solution:**

1. Verify API key is set correctly in request headers:
   ```bash
   curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8080/api/v1/health
   ```

2. Check API key configuration in your server settings.

### API Timeouts

**Problem:** API requests timeout for large datasets.

**Solution:**

1. Increase timeout settings:
   ```bash
   # In client code
   requests.get(url, timeout=300)  # 5 minutes
   ```

2. Use pagination for large result sets.

3. Consider caching frequently accessed data.

## Performance Issues

### Slow Dataset Analysis

**Problem:** Dataset analysis takes longer than the 4-minute requirement.

**Solution:**

1. Use caching for repeated analyses:
   ```python
   from bigquerycostopt.src.utils.cache import cache_result
   
   @cache_result('dataset_analysis', ttl_seconds=3600)  # 1 hour cache
   def analyze_dataset(dataset_id):
       # Analysis code
   ```

2. Limit the scope of analysis:
   ```python
   # Analyze only the largest tables
   optimizer.analyze_dataset_queries(dataset_id, table_size_threshold_gb=1.0)
   ```

3. Use parallel processing for independent components:
   ```python
   import concurrent.futures
   
   with concurrent.futures.ThreadPoolExecutor() as executor:
       query_future = executor.submit(query_optimizer.analyze_dataset_queries, dataset_id)
       schema_future = executor.submit(schema_optimizer.analyze_dataset_schemas, dataset_id)
       storage_future = executor.submit(storage_optimizer.analyze_dataset, dataset_id)
       
       query_recommendations = query_future.result()
       schema_recommendations = schema_future.result()
       storage_recommendations = storage_future.result()
   ```

### Memory Issues

**Problem:** Out of memory errors when processing large datasets.

**Solution:**

1. Process data in chunks:
   ```python
   # Process tables in batches
   for table_batch in optimizer.get_tables_in_batches(dataset_id, batch_size=10):
       optimizer.analyze_tables(dataset_id, table_batch)
   ```

2. Limit the number of recommendations:
   ```python
   # Get only top recommendations
   optimizer.analyze_dataset_queries(dataset_id, max_recommendations=20)
   ```

3. Filter tables by size:
   ```python
   # Only process tables larger than 1GB
   optimizer.analyze_dataset(dataset_id, min_table_size_gb=1.0)
   ```

## Data Access Issues

### Missing Tables or Datasets

**Problem:** Tables or datasets not found when running analysis.

**Solution:**

1. Verify dataset existence:
   ```python
   from google.cloud import bigquery
   
   client = bigquery.Client()
   try:
       client.get_dataset(f"{project_id}.{dataset_id}")
       print("Dataset exists")
   except Exception as e:
       print(f"Dataset error: {e}")
   ```

2. Check for misspelled dataset or table names.

3. Ensure cross-project access is set up if accessing datasets in other projects.

### Access to Usage Data

**Problem:** Cannot access BigQuery usage and billing data.

**Solution:**

1. Enable BigQuery usage export:
   - Go to the Google Cloud Console
   - Navigate to BigQuery > Settings > Data Usage Export
   - Enable export to a dataset

2. Set up proper access to exported data:
   ```bash
   bq add-iam-policy-binding --member=serviceAccount:your-service-account@your-project.iam.gserviceaccount.com --role=roles/bigquery.dataViewer your-project:your_billing_export_dataset
   ```

## ML Component Issues

### Missing ML Dependencies

**Problem:** ML-enhanced features don't work due to missing dependencies.

**Solution:**

1. Install ML dependencies:
   ```bash
   pip install -e ".[ml]"
   ```

2. Handle missing dependencies gracefully:
   ```python
   try:
       from google.cloud import storage
       _has_storage = True
   except ImportError:
       _has_storage = False
       storage = None
   
   # Then check before using
   if _has_storage:
       # Use storage
   else:
       # Fallback behavior
   ```

### Prophet Installation Issues

**Problem:** Prophet fails to install with errors about pystan or Cython.

**Solution:**

1. Install dependencies first:
   ```bash
   pip install numpy cython
   pip install pystan==2.19.1.1
   pip install prophet
   ```

2. On macOS, you may need additional compiler tools:
   ```bash
   xcode-select --install
   ```

3. As a fallback, use non-Prophet forecasting:
   ```python
   # Use statsmodels for forecasting instead
   from bigquerycostopt.src.ml.cost_anomaly_detection import TimeSeriesForecaster
   
   forecaster = TimeSeriesForecaster(analyzer, use_prophet=False)
   ```

## Dashboard Integration Issues

### Retool Integration Problems

**Problem:** Retool dashboard isn't displaying data correctly.

**Solution:**

1. Verify API endpoints using curl:
   ```bash
   curl -H "Authorization: Bearer YOUR_API_KEY" "http://your-api-host:8080/api/v1/cost-dashboard/summary?project_id=your-project-id"
   ```

2. Check CORS configuration if accessing from a browser:
   ```python
   # In server.py or fastapi_server.py
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Restrict in production
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

3. Verify the Retool resource configuration:
   - Check base URL
   - Ensure authentication headers are set
   - Test the resource connection

### Data Format Issues

**Problem:** Dashboard receives data but displays errors or blank components.

**Solution:**

1. Check the API response format:
   ```bash
   curl -H "Authorization: Bearer YOUR_API_KEY" "http://your-api-host:8080/api/v1/cost-dashboard/summary?project_id=your-project-id" | jq
   ```

2. Verify match between API response and Retool component expectations:
   - Check for null values
   - Verify date format consistency
   - Ensure numeric values are the expected type

3. Add debug logging to trace the issue:
   ```javascript
   // In Retool query
   console.log(JSON.stringify(responseData, null, 2));
   ```

## Diagnostic Tools

### Verify Installation

Run the installation verification script:

```bash
python -m bigquerycostopt.verify_installation
```

This checks:
- Python version
- Installed dependencies
- Google Cloud authentication
- Access to required APIs

### Run Tests

Execute the test suite to validate functionality:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run a specific test
pytest tests/unit/test_query_optimizer.py::TestQueryOptimizer::test_analyze_dataset_queries
```

### Performance Testing

Measure dataset processing performance:

```bash
# Run with default settings
python -m bigquerycostopt.tests.performance.dataset_processing_test

# Run with specific project
python -m bigquerycostopt.tests.performance.dataset_processing_test --project-id=your-project-id
```

This will generate a performance report showing:
- Processing times for datasets of various sizes
- Whether each dataset meets the 4-minute requirement
- Component-level timing breakdown
- Recommendations for optimization

### Logs and Debugging

Enable debug logging for detailed information:

```bash
# Set environment variable
export BIGQUERYCOSTOPT_LOG_LEVEL=DEBUG

# Or in Python
import logging
logging.getLogger('bigquerycostopt').setLevel(logging.DEBUG)
```

Check logs for specific components:

```bash
# Check API server logs
tail -f logs/api_server.log

# Check analysis logs
tail -f logs/analysis.log
```

## Getting Additional Help

If you continue to experience issues:

1. Check the [GitHub repository issues](https://github.com/yourusername/bigquerycostopt/issues) for similar problems and solutions.

2. Run the diagnostic script and share the output:
   ```bash
   python -m bigquerycostopt.diagnostics --full > diagnostics_output.txt
   ```

3. Submit a detailed issue with:
   - Steps to reproduce the problem
   - Full error messages and stack traces
   - Version information from `pip freeze`
   - Environment details (OS, Python version)