# Optimizing Large Datasets with BigQuery Cost Intelligence Engine

This guide provides detailed strategies and code examples for optimizing large BigQuery datasets (100,000+ records) using the BigQuery Cost Intelligence Engine.

## Table of Contents

- [Understanding Performance Challenges](#understanding-performance-challenges)
- [Optimization Techniques](#optimization-techniques)
- [Example 1: Batch Processing](#example-1-batch-processing)
- [Example 2: Parallel Analysis](#example-2-parallel-analysis)
- [Example 3: Incremental Analysis](#example-3-incremental-analysis)
- [Monitoring and Performance Tuning](#monitoring-and-performance-tuning)
- [Best Practices](#best-practices)

## Understanding Performance Challenges

When analyzing very large BigQuery datasets, you may encounter several performance challenges:

1. **Query Processing Time**: Extracting metadata and usage information from BigQuery can be time-consuming for large datasets.
2. **Memory Constraints**: Loading large datasets into memory can cause resource issues.
3. **Complex Calculations**: Schema and query analysis algorithms have higher complexity with larger datasets.
4. **BigQuery API Limits**: You may hit API quotas or rate limits when making multiple requests.

The BigQuery Cost Intelligence Engine is designed to handle datasets of 100,000+ records within 4 minutes by using the following optimization techniques.

## Optimization Techniques

### 1. Caching

The system implements caching to avoid repeated processing:

```python
from datetime import datetime, timedelta

# Simple cache implementation
_cache = {}
_cache_ttl = {}

def get_from_cache(key, ttl_seconds=300):
    """Get data from cache if not expired."""
    if key in _cache and key in _cache_ttl:
        if datetime.now().timestamp() < _cache_ttl[key]:
            return _cache[key]
    return None

def set_in_cache(key, data, ttl_seconds=300):
    """Store data in cache with TTL."""
    _cache[key] = data
    _cache_ttl[key] = datetime.now().timestamp() + ttl_seconds
```

### 2. Batch Processing

Process large datasets in smaller batches:

```python
def analyze_dataset_in_batches(dataset_id, batch_size=10):
    """Analyze dataset in batches of tables."""
    from src.analysis.metadata import MetadataExtractor
    
    # Extract dataset metadata
    extractor = MetadataExtractor(project_id="your-project-id")
    dataset_metadata = extractor.extract_dataset_metadata(dataset_id)
    
    # Get tables sorted by size (largest first for early results)
    tables = sorted(
        dataset_metadata.get("tables", []),
        key=lambda t: t.get("size_bytes", 0),
        reverse=True
    )
    
    all_recommendations = []
    
    # Process in batches
    for i in range(0, len(tables), batch_size):
        batch = tables[i:i+batch_size]
        batch_tables = [table["table_id"] for table in batch]
        
        print(f"Processing batch {i//batch_size + 1} of {(len(tables) + batch_size - 1)//batch_size}")
        
        # Process this batch
        batch_recommendations = process_table_batch(dataset_id, batch_tables)
        all_recommendations.extend(batch_recommendations)
    
    return all_recommendations
```

### 3. Parallel Processing

Use parallel processing for independent components:

```python
import concurrent.futures

def analyze_dataset_parallel(dataset_id):
    """Analyze dataset with parallel processing."""
    from src.analysis.query_optimizer import QueryOptimizer
    from src.analysis.schema_optimizer import SchemaOptimizer 
    from src.analysis.storage_optimizer import StorageOptimizer
    
    results = {}
    
    # Run optimizers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks
        query_future = executor.submit(
            lambda: QueryOptimizer("your-project-id").analyze_dataset_queries(dataset_id)
        )
        
        schema_future = executor.submit(
            lambda: SchemaOptimizer("your-project-id").analyze_dataset_schemas(dataset_id)
        )
        
        storage_future = executor.submit(
            lambda: StorageOptimizer("your-project-id").analyze_dataset(dataset_id)
        )
        
        # Collect results
        results["query_recommendations"] = query_future.result()
        results["schema_recommendations"] = schema_future.result()
        results["storage_recommendations"] = storage_future.result()
    
    # Combine recommendations
    all_recommendations = []
    all_recommendations.extend(results["query_recommendations"])
    all_recommendations.extend(results["schema_recommendations"])
    all_recommendations.extend(results["storage_recommendations"])
    
    return all_recommendations
```

### 4. Statistical Sampling

For very large datasets, use statistical sampling:

```python
def analyze_with_sampling(dataset_id, sampling_ratio=0.1, min_sample_size=100):
    """Analyze dataset with statistical sampling for large tables."""
    from src.analysis.metadata import MetadataExtractor
    from src.analysis.query_optimizer import QueryOptimizer
    
    # Extract dataset metadata
    extractor = MetadataExtractor(project_id="your-project-id")
    dataset_metadata = extractor.extract_dataset_metadata(dataset_id)
    
    # Initialize optimizer
    optimizer = QueryOptimizer("your-project-id")
    
    all_recommendations = []
    
    # Process each table with sampling for large ones
    for table in dataset_metadata.get("tables", []):
        table_id = table["table_id"]
        row_count = table.get("row_count", 0)
        
        # Determine if sampling is needed
        if row_count > 10000:
            # Calculate sample size (at least min_sample_size)
            sample_size = max(int(row_count * sampling_ratio), min_sample_size)
            
            # Analyze with sampling
            recommendations = optimizer.analyze_table_queries_with_sampling(
                dataset_id, 
                table_id, 
                sample_size=sample_size
            )
        else:
            # Small table, no sampling needed
            recommendations = optimizer.analyze_table_queries(dataset_id, table_id)
        
        all_recommendations.extend(recommendations)
    
    return all_recommendations
```

### 5. Query Optimization

Optimize BigQuery queries to reduce processing time:

```python
def get_table_usage_optimized(client, project_id, dataset_id, table_id, days_back=30):
    """Get table usage statistics with optimized query."""
    # Use a more efficient query that:
    # 1. Restricts data to specific table
    # 2. Uses partitioning if available
    # 3. Selects only needed columns
    
    query = f"""
    SELECT
      COUNT(*) AS query_count,
      SUM(total_bytes_processed) AS total_bytes_processed,
      SUM(total_slot_ms) AS total_slot_ms,
      AVG(total_bytes_processed) AS avg_bytes_processed_per_query
    FROM `{project_id}.region-us.INFORMATION_SCHEMA.JOBS`
    WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
      AND state = 'DONE'
      AND error_result IS NULL
      AND statement_type = 'SELECT'
      AND REGEXP_CONTAINS(query, r'`{project_id}.{dataset_id}.{table_id}`')
    """
    
    query_job = client.query(query)
    df = query_job.to_dataframe()
    
    if df.empty or df.iloc[0]['query_count'] == 0:
        return {
            "query_count_30d": 0,
            "total_bytes_processed_30d": 0,
            "total_slot_ms_30d": 0,
            "avg_bytes_processed_per_query": 0
        }
    
    return {
        "query_count_30d": int(df.iloc[0]['query_count']),
        "total_bytes_processed_30d": int(df.iloc[0]['total_bytes_processed'] or 0),
        "total_slot_ms_30d": int(df.iloc[0]['total_slot_ms'] or 0),
        "avg_bytes_processed_per_query": int(df.iloc[0]['avg_bytes_processed_per_query'] or 0)
    }
```

## Example 1: Batch Processing

This example demonstrates how to analyze a large dataset by processing tables in batches:

```python
import time
from src.analysis.metadata import MetadataExtractor
from src.analysis.query_optimizer import QueryOptimizer
from src.analysis.schema_optimizer import SchemaOptimizer
from src.analysis.storage_optimizer import StorageOptimizer

def process_large_dataset_in_batches(project_id, dataset_id, batch_size=10):
    """Process a large dataset in batches of tables."""
    start_time = time.time()
    print(f"Starting batch processing of dataset {dataset_id}")
    
    # Extract metadata
    extractor = MetadataExtractor(project_id)
    try:
        dataset_metadata = extractor.extract_dataset_metadata(dataset_id)
    except Exception as e:
        print(f"Error extracting dataset metadata: {e}")
        return []
    
    # Get tables sorted by size (largest first)
    tables = dataset_metadata.get("tables", [])
    if not tables:
        print(f"No tables found in dataset {dataset_id}")
        return []
    
    tables.sort(key=lambda t: t.get("size_bytes", 0), reverse=True)
    print(f"Found {len(tables)} tables in dataset {dataset_id}")
    
    # Create optimizers
    query_optimizer = QueryOptimizer(project_id)
    schema_optimizer = SchemaOptimizer(project_id)
    storage_optimizer = StorageOptimizer(project_id)
    
    # Process in batches
    all_recommendations = []
    batch_count = (len(tables) + batch_size - 1) // batch_size
    
    for i in range(0, len(tables), batch_size):
        batch_start_time = time.time()
        batch_tables = tables[i:i+batch_size]
        batch_table_ids = [t.get("table_id") for t in batch_tables]
        
        print(f"Processing batch {i//batch_size + 1}/{batch_count} with {len(batch_table_ids)} tables")
        
        # Process batch
        batch_recommendations = []
        
        # Query recommendations
        query_recs = query_optimizer.analyze_tables(dataset_id, batch_table_ids)
        batch_recommendations.extend(query_recs)
        
        # Schema recommendations
        schema_recs = schema_optimizer.analyze_tables(dataset_id, batch_table_ids)
        batch_recommendations.extend(schema_recs)
        
        # Storage recommendations
        storage_recs = storage_optimizer.analyze_tables(dataset_id, batch_table_ids)
        batch_recommendations.extend(storage_recs)
        
        all_recommendations.extend(batch_recommendations)
        
        batch_time = time.time() - batch_start_time
        print(f"Batch {i//batch_size + 1} completed in {batch_time:.2f} seconds")
        print(f"Found {len(batch_recommendations)} recommendations in this batch")
        
        # Optional: Add a small delay between batches to reduce API load
        if i + batch_size < len(tables):
            time.sleep(0.5)
    
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time:.2f} seconds")
    print(f"Found {len(all_recommendations)} total recommendations")
    
    return all_recommendations

# Usage
project_id = "your-project-id"
dataset_id = "your_large_dataset"
recommendations = process_large_dataset_in_batches(project_id, dataset_id, batch_size=5)
```

## Example 2: Parallel Analysis

This example shows how to analyze components in parallel:

```python
import time
import concurrent.futures
from src.analysis.metadata import MetadataExtractor
from src.analysis.query_optimizer import QueryOptimizer
from src.analysis.schema_optimizer import SchemaOptimizer
from src.analysis.storage_optimizer import StorageOptimizer

def process_large_dataset_parallel(project_id, dataset_id):
    """Process a large dataset with parallel analysis."""
    start_time = time.time()
    print(f"Starting parallel processing of dataset {dataset_id}")
    
    # Initialize optimizers
    query_optimizer = QueryOptimizer(project_id)
    schema_optimizer = SchemaOptimizer(project_id)
    storage_optimizer = StorageOptimizer(project_id)
    
    # Run optimizers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks
        future_to_optimizer = {
            executor.submit(query_optimizer.analyze_dataset_queries, dataset_id): "query",
            executor.submit(schema_optimizer.analyze_dataset_schemas, dataset_id): "schema",
            executor.submit(storage_optimizer.analyze_dataset, dataset_id): "storage"
        }
        
        # Collect results as they complete
        recommendations = {
            "query": [],
            "schema": [],
            "storage": []
        }
        
        for future in concurrent.futures.as_completed(future_to_optimizer):
            optimizer_type = future_to_optimizer[future]
            try:
                optimizer_recommendations = future.result()
                recommendations[optimizer_type] = optimizer_recommendations
                print(f"{optimizer_type.capitalize()} optimizer completed, found {len(optimizer_recommendations)} recommendations")
            except Exception as e:
                print(f"Error in {optimizer_type} optimizer: {e}")
    
    # Combine all recommendations
    all_recommendations = []
    for optimizer_type, optimizer_recommendations in recommendations.items():
        all_recommendations.extend(optimizer_recommendations)
    
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time:.2f} seconds")
    print(f"Found {len(all_recommendations)} total recommendations")
    
    return all_recommendations

# Usage
project_id = "your-project-id"
dataset_id = "your_large_dataset"
recommendations = process_large_dataset_parallel(project_id, dataset_id)
```

## Example 3: Incremental Analysis

This example demonstrates incremental analysis of a dataset, focusing on changes since the last analysis:

```python
import time
import datetime
import json
from pathlib import Path
from src.analysis.metadata import MetadataExtractor
from src.analysis.query_optimizer import QueryOptimizer

def incremental_analysis(project_id, dataset_id):
    """Perform incremental analysis of a dataset."""
    start_time = time.time()
    print(f"Starting incremental analysis of dataset {dataset_id}")
    
    # Get last analysis timestamp
    history_file = Path(f"analysis_history/{project_id}_{dataset_id}.json")
    
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
            last_analysis = datetime.datetime.fromisoformat(history.get("last_analysis", "2000-01-01T00:00:00"))
    else:
        # No previous analysis, default to 30 days ago
        last_analysis = datetime.datetime.now() - datetime.timedelta(days=30)
        history = {"recommendations": []}
    
    print(f"Last analysis was at {last_analysis.isoformat()}")
    
    # Extract metadata
    extractor = MetadataExtractor(project_id)
    dataset_metadata = extractor.extract_dataset_metadata(dataset_id)
    
    # Find modified tables
    modified_tables = []
    for table in dataset_metadata.get("tables", []):
        table_modified = datetime.datetime.fromisoformat(table.get("last_modified", "2000-01-01T00:00:00"))
        
        if table_modified > last_analysis:
            modified_tables.append(table)
    
    print(f"Found {len(modified_tables)} modified tables since last analysis")
    
    # Process only modified tables
    if not modified_tables:
        print("No modified tables to analyze")
        return history.get("recommendations", [])
    
    # Analyze modified tables
    query_optimizer = QueryOptimizer(project_id)
    new_recommendations = []
    
    for table in modified_tables:
        table_id = table.get("table_id")
        print(f"Analyzing modified table: {table_id}")
        
        try:
            table_recommendations = query_optimizer.analyze_table_queries(dataset_id, table_id)
            new_recommendations.extend(table_recommendations)
        except Exception as e:
            print(f"Error analyzing table {table_id}: {e}")
    
    # Combine with existing recommendations
    existing_ids = {rec.get("recommendation_id") for rec in history.get("recommendations", [])}
    
    # Add only new recommendations
    for rec in new_recommendations:
        if rec.get("recommendation_id") not in existing_ids:
            history["recommendations"].append(rec)
    
    # Update last analysis timestamp
    history["last_analysis"] = datetime.datetime.now().isoformat()
    
    # Save updated history
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    total_time = time.time() - start_time
    print(f"Incremental analysis completed in {total_time:.2f} seconds")
    print(f"Found {len(new_recommendations)} new recommendations")
    print(f"Total recommendations: {len(history['recommendations'])}")
    
    return history.get("recommendations", [])

# Usage
project_id = "your-project-id"
dataset_id = "your_large_dataset"
recommendations = incremental_analysis(project_id, dataset_id)
```

## Monitoring and Performance Tuning

To ensure optimal performance with large datasets, monitor these key metrics:

1. **Processing Time**: Track the time taken for each component of analysis
2. **Memory Usage**: Monitor memory consumption during analysis
3. **API Rate Limits**: Watch for BigQuery API quotas and rate limits
4. **Cache Hit Rate**: Monitor cache effectiveness

Implement a monitoring solution:

```python
import time
import psutil
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        logger.info(f"Starting {func.__name__} with {len(args)} args and {len(kwargs)} kwargs")
        logger.info(f"Initial memory usage: {start_memory:.2f} MB")
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            execution_time = end_time - start_time
            memory_diff = end_memory - start_memory
            
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            logger.info(f"Memory change: {memory_diff:.2f} MB (final: {end_memory:.2f} MB)")
            
            # Log warning if execution time is high
            if execution_time > 60:  # 1 minute
                logger.warning(f"Long execution time for {func.__name__}: {execution_time:.2f} seconds")
            
            return result
        
        except Exception as e:
            end_time = time.time()
            logger.error(f"Error in {func.__name__} after {end_time - start_time:.2f} seconds: {e}")
            raise
    
    return wrapper

# Usage example
@monitor_performance
def analyze_large_dataset(dataset_id):
    # Analysis code here
    pass
```

## Best Practices

When working with large datasets (100,000+ records), follow these best practices:

1. **Prioritize Large Tables**: Process the largest tables first to identify high-impact recommendations quickly

2. **Use Asynchronous Processing**: For web applications, perform analysis asynchronously:

   ```python
   # In your API endpoint
   @app.post("/api/v1/analyze")
   async def trigger_analysis(analysis_request: AnalysisRequest):
       # Submit analysis job to background worker
       from src.utils.background_tasks import submit_task
       
       task_id = submit_task(
           "analyze_dataset",
           project_id=analysis_request.project_id,
           dataset_id=analysis_request.dataset_id
       )
       
       return {"task_id": task_id, "status": "submitted"}
   ```

3. **Implement Timeouts**: Add timeouts to prevent indefinite processing:

   ```python
   import signal
   
   class TimeoutError(Exception):
       pass
   
   def timeout_handler(signum, frame):
       raise TimeoutError("Processing timed out")
   
   def process_with_timeout(func, args=None, kwargs=None, timeout_sec=240):
       """Run a function with a timeout."""
       if args is None:
           args = ()
       if kwargs is None:
           kwargs = {}
       
       # Set the timeout handler
       signal.signal(signal.SIGALRM, timeout_handler)
       signal.alarm(timeout_sec)
       
       try:
           result = func(*args, **kwargs)
           signal.alarm(0)  # Disable the alarm
           return result
       except TimeoutError:
           print(f"Processing timed out after {timeout_sec} seconds")
           return None
   ```

4. **Segment Analysis**: Break analysis into tiers based on dataset size:

   - Small (< 10K records): Full analysis with all optimizations
   - Medium (10K - 100K records): Full analysis with parallel processing
   - Large (100K - 500K records): Batch processing with parallel components
   - Very Large (> 500K records): Sampling-based analysis or incremental updates

5. **Optimize Storage**: Compress large intermediate results:

   ```python
   import gzip
   import json
   
   def save_compressed_results(data, filename):
       """Save results as compressed JSON."""
       with gzip.open(filename, 'wt', encoding='UTF-8') as f:
           json.dump(data, f)
   
   def load_compressed_results(filename):
       """Load compressed JSON results."""
       with gzip.open(filename, 'rt', encoding='UTF-8') as f:
           return json.load(f)
   ```

By implementing these strategies, you can effectively analyze even the largest BigQuery datasets within the 4-minute completion requirement, ensuring optimal performance and resource utilization.