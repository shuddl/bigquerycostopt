#!/usr/bin/env python
"""
Performance testing script for BigQuery Cost Intelligence Engine dataset processing.

This script measures processing time for datasets of various sizes to validate
the 4-minute completion requirement for datasets with 100K+ records.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import datetime
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import directly from the modules
from src.analysis.metadata import extract_dataset_metadata
from src.analysis.query_optimizer import QueryOptimizer
from src.analysis.schema_optimizer import SchemaOptimizer
from src.analysis.storage_optimizer import StorageOptimizer
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class PerformanceTester:
    """Performance tester for BigQuery Cost Intelligence Engine."""

    def __init__(self, project_id: str, dataset_sizes: List[int], iterations: int = 3):
        """Initialize the performance tester.
        
        Args:
            project_id: Google Cloud project ID
            dataset_sizes: List of dataset sizes to test (number of records)
            iterations: Number of iterations to run for each test
        """
        self.project_id = project_id
        self.dataset_sizes = dataset_sizes
        self.iterations = iterations
        self.results = {
            'datasets': {},
            'summary': {},
            'recommendations': {}
        }
        
        # Set up mocks for BigQuery client
        self.mock_client = None
        self.setup_mocks()
        
    def setup_mocks(self):
        """Set up mock objects for testing without a real BigQuery connection."""
        from unittest.mock import MagicMock
        
        self.mock_client = MagicMock()
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.location = "US"
        mock_dataset.created = datetime.datetime(2023, 1, 1)
        mock_dataset.modified = datetime.datetime(2023, 1, 2)
        mock_dataset.default_partition_expiration_ms = None
        mock_dataset.default_table_expiration_ms = None
        
        self.mock_client.dataset.return_value = MagicMock()
        self.mock_client.get_dataset.return_value = mock_dataset
    
    def generate_mock_tables(self, dataset_id: str, num_tables: int, rows_per_table: int) -> List[Dict[str, Any]]:
        """Generate mock tables for testing.
        
        Args:
            dataset_id: Dataset ID
            num_tables: Number of tables to generate
            rows_per_table: Number of rows per table
            
        Returns:
            List of mock table metadata
        """
        tables = []
        total_rows = 0
        
        for i in range(num_tables):
            # Create table
            table_id = f"table_{i+1}"
            table_rows = rows_per_table
            total_rows += table_rows
            
            # Generate random schema (between 5 and 20 columns)
            num_columns = np.random.randint(5, 21)
            columns = []
            
            for j in range(num_columns):
                # Select random column type
                col_type = np.random.choice(['INTEGER', 'STRING', 'FLOAT', 'BOOLEAN', 'TIMESTAMP', 'DATE'])
                col_name = f"col_{j+1}"
                
                # Add some nullable columns
                nullable = np.random.choice([True, False], p=[0.8, 0.2])
                mode = "NULLABLE" if nullable else "REQUIRED"
                
                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "mode": mode
                })
            
            # Calculate average row size (rough estimate)
            # INTEGER: ~8 bytes, STRING: ~50 bytes, FLOAT: ~8 bytes, BOOLEAN: ~1 byte,
            # TIMESTAMP: ~8 bytes, DATE: ~4 bytes
            size_map = {
                'INTEGER': 8,
                'STRING': 50,
                'FLOAT': 8,
                'BOOLEAN': 1,
                'TIMESTAMP': 8,
                'DATE': 4
            }
            
            avg_row_size = sum(size_map[col['type']] for col in columns)
            size_bytes = table_rows * avg_row_size
            
            # Add partitioning for some tables
            partitioned = np.random.choice([True, False], p=[0.3, 0.7])
            partitioning = None
            if partitioned:
                # Find a TIMESTAMP or DATE column for partitioning
                partition_candidates = [col for col in columns if col['type'] in ['TIMESTAMP', 'DATE']]
                if partition_candidates:
                    partition_column = np.random.choice(partition_candidates)
                    partitioning = {
                        "type": "DAY",
                        "field": partition_column['name']
                    }
            
            # Add clustering for some tables
            clustered = np.random.choice([True, False], p=[0.2, 0.8])
            clustering = None
            if clustered:
                # Select 1-3 columns for clustering
                num_clustering_cols = np.random.randint(1, min(4, num_columns))
                clustering_columns = np.random.choice([col['name'] for col in columns], size=num_clustering_cols, replace=False)
                clustering = {
                    "fields": list(clustering_columns)
                }
            
            # Create table metadata
            table_metadata = {
                "table_id": table_id,
                "dataset_id": dataset_id,
                "project_id": self.project_id,
                "row_count": table_rows,
                "size_bytes": size_bytes,
                "size_gb": size_bytes / (1024 ** 3),
                "schema": columns,
                "partitioning": partitioning or {},
                "clustering": clustering or {"fields": []},
                "created": datetime.datetime(2023, 1, 1).isoformat(),
                "last_modified": datetime.datetime(2023, 1, 2).isoformat(),
                "description": f"Mock table {table_id} with {table_rows} rows"
            }
            
            tables.append(table_metadata)
        
        logger.info(f"Generated {num_tables} mock tables with {total_rows} total rows")
        return tables
    
    def generate_mock_queries(self, dataset_id: str, tables: List[Dict[str, Any]], num_queries: int) -> List[Dict[str, Any]]:
        """Generate mock queries for testing.
        
        Args:
            dataset_id: Dataset ID
            tables: List of table metadata
            num_queries: Number of queries to generate
            
        Returns:
            List of mock query metadata
        """
        queries = []
        
        for i in range(num_queries):
            # Select random tables (1-3)
            num_tables_in_query = np.random.randint(1, min(4, len(tables) + 1))
            selected_tables = np.random.choice(tables, size=num_tables_in_query, replace=False)
            
            # Create query text
            table_refs = [f"`{self.project_id}.{dataset_id}.{table['table_id']}`" for table in selected_tables]
            
            # Generate different query patterns
            query_type = np.random.choice(['simple', 'join', 'aggregate', 'window', 'subquery'], 
                                          p=[0.3, 0.3, 0.2, 0.1, 0.1])
            
            if query_type == 'simple':
                # Simple SELECT query
                query_text = f"SELECT * FROM {table_refs[0]} LIMIT 1000"
            elif query_type == 'join' and len(table_refs) > 1:
                # JOIN query
                cols1 = [col['name'] for col in selected_tables[0]['schema']]
                cols2 = [col['name'] for col in selected_tables[1]['schema']]
                
                query_text = f"""
                SELECT t1.*, t2.*
                FROM {table_refs[0]} t1
                JOIN {table_refs[1]} t2
                ON t1.{cols1[0]} = t2.{cols2[0]}
                LIMIT 1000
                """
            elif query_type == 'aggregate':
                # Aggregation query
                numeric_cols = [col['name'] for col in selected_tables[0]['schema'] 
                               if col['type'] in ['INTEGER', 'FLOAT']]
                
                if numeric_cols:
                    agg_col = np.random.choice(numeric_cols)
                    query_text = f"""
                    SELECT 
                        {selected_tables[0]['schema'][0]['name']},
                        SUM({agg_col}) as total,
                        AVG({agg_col}) as average
                    FROM {table_refs[0]}
                    GROUP BY {selected_tables[0]['schema'][0]['name']}
                    """
                else:
                    query_text = f"SELECT COUNT(*) FROM {table_refs[0]}"
            elif query_type == 'window':
                # Window function query
                numeric_cols = [col['name'] for col in selected_tables[0]['schema'] 
                               if col['type'] in ['INTEGER', 'FLOAT']]
                
                if numeric_cols:
                    agg_col = np.random.choice(numeric_cols)
                    partition_col = selected_tables[0]['schema'][0]['name']
                    query_text = f"""
                    SELECT 
                        *,
                        SUM({agg_col}) OVER (PARTITION BY {partition_col}) as partition_sum,
                        ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {agg_col}) as row_num
                    FROM {table_refs[0]}
                    """
                else:
                    query_text = f"SELECT * FROM {table_refs[0]} LIMIT 1000"
            else:
                # Subquery
                query_text = f"""
                SELECT * FROM (
                    SELECT * FROM {table_refs[0]} LIMIT 10000
                ) subq
                WHERE subq.{selected_tables[0]['schema'][0]['name']} IS NOT NULL
                LIMIT 1000
                """
            
            # Random processing time and cost
            slot_ms = np.random.randint(1000, 100000)
            bytes_processed = np.random.randint(10**6, 10**9)
            cost = bytes_processed * 5 / (10**12)  # $5 per TB
            
            # Create query metadata
            query_metadata = {
                "query_id": f"query_{i+1}",
                "project_id": self.project_id,
                "user_email": f"user{np.random.randint(1, 10)}@example.com",
                "query_text": query_text,
                "creation_time": datetime.datetime(2023, 1, 1, 12, 0, 0) + datetime.timedelta(hours=i),
                "slot_ms": slot_ms,
                "bytes_processed": bytes_processed,
                "estimated_cost_usd": cost,
                "referenced_tables": [{"project_id": self.project_id, "dataset_id": dataset_id, "table_id": table['table_id']} 
                                     for table in selected_tables]
            }
            
            queries.append(query_metadata)
        
        logger.info(f"Generated {num_queries} mock queries")
        return queries
    
    def run_tests(self):
        """Run performance tests for all dataset sizes."""
        for size in self.dataset_sizes:
            logger.info(f"Testing dataset with {size} records")
            self.test_dataset(f"test_dataset_{size}", size)
        
        self.calculate_summary()
        self.save_results()
        
    def test_dataset(self, dataset_id: str, size: int):
        """Run performance test for a single dataset.
        
        Args:
            dataset_id: Dataset ID
            size: Dataset size (number of records)
        """
        # Determine number of tables and queries based on dataset size
        if size <= 10000:
            num_tables = 5
            num_queries = 20
        elif size <= 50000:
            num_tables = 10
            num_queries = 50
        elif size <= 100000:
            num_tables = 20
            num_queries = 100
        else:
            num_tables = 30
            num_queries = 200
        
        rows_per_table = size // num_tables
        
        # Generate mock data
        tables = self.generate_mock_tables(dataset_id, num_tables, rows_per_table)
        queries = self.generate_mock_queries(dataset_id, tables, num_queries)
        
        # Initialize results
        self.results['datasets'][dataset_id] = {
            'size': size,
            'num_tables': num_tables,
            'num_queries': num_queries,
            'iterations': []
        }
        
        # Mock the dataset metadata function
        def mock_extract_dataset_metadata(project_id, dataset_id, client=None):
            return {
                "project_id": project_id,
                "dataset_id": dataset_id,
                "location": "US",
                "table_count": num_tables,
                "total_size_bytes": sum(table['size_bytes'] for table in tables),
                "total_size_gb": sum(table['size_bytes'] for table in tables) / (1024 ** 3),
                "tables": tables
            }
        
        # Mock query results
        mock_query_df = pd.DataFrame(queries)
        mock_query_job = type('MockQueryJob', (), {'to_dataframe': lambda: mock_query_df})
        self.mock_client.query.return_value = mock_query_job
        
        # Run test iterations
        for iteration in range(self.iterations):
            logger.info(f"Running iteration {iteration+1}/{self.iterations} for dataset {dataset_id}")
            
            iteration_results = {
                'query_optimizer_time': 0,
                'schema_optimizer_time': 0,
                'storage_optimizer_time': 0,
                'total_time': 0,
                'recommendations': {
                    'query': 0,
                    'schema': 0,
                    'storage': 0
                }
            }
            
            # Test QueryOptimizer
            start_time = time.time()
            query_optimizer = QueryOptimizer(self.project_id)
            query_optimizer.extract_metadata = mock_extract_dataset_metadata  # Replace with mock
            query_recommendations = query_optimizer.analyze_dataset_queries(dataset_id, client=self.mock_client)
            query_time = time.time() - start_time
            
            # Test SchemaOptimizer
            start_time = time.time()
            schema_optimizer = SchemaOptimizer(self.project_id)
            schema_optimizer.extract_metadata = mock_extract_dataset_metadata  # Replace with mock
            schema_recommendations = schema_optimizer.analyze_dataset_schemas(dataset_id, client=self.mock_client)
            schema_time = time.time() - start_time
            
            # Test StorageOptimizer
            start_time = time.time()
            storage_optimizer = StorageOptimizer(self.project_id)
            storage_optimizer.extract_metadata = mock_extract_dataset_metadata  # Replace with mock
            storage_recommendations = storage_optimizer.analyze_dataset(dataset_id, client=self.mock_client)
            storage_time = time.time() - start_time
            
            # Record results
            total_time = query_time + schema_time + storage_time
            iteration_results['query_optimizer_time'] = query_time
            iteration_results['schema_optimizer_time'] = schema_time
            iteration_results['storage_optimizer_time'] = storage_time
            iteration_results['total_time'] = total_time
            
            # Count recommendations
            iteration_results['recommendations']['query'] = len(query_recommendations)
            iteration_results['recommendations']['schema'] = len(schema_recommendations)
            iteration_results['recommendations']['storage'] = len(storage_recommendations)
            
            # Add iteration results
            self.results['datasets'][dataset_id]['iterations'].append(iteration_results)
            
            logger.info(f"Iteration {iteration+1} completed in {total_time:.2f} seconds")
            logger.info(f"  Query Optimizer: {query_time:.2f}s, Schema Optimizer: {schema_time:.2f}s, Storage Optimizer: {storage_time:.2f}s")
    
    def calculate_summary(self):
        """Calculate summary statistics for all datasets."""
        summary = {}
        
        for dataset_id, dataset_results in self.results['datasets'].items():
            size = dataset_results['size']
            
            # Calculate average times
            query_times = [iteration['query_optimizer_time'] for iteration in dataset_results['iterations']]
            schema_times = [iteration['schema_optimizer_time'] for iteration in dataset_results['iterations']]
            storage_times = [iteration['storage_optimizer_time'] for iteration in dataset_results['iterations']]
            total_times = [iteration['total_time'] for iteration in dataset_results['iterations']]
            
            # Calculate recommendation counts
            query_recs = [iteration['recommendations']['query'] for iteration in dataset_results['iterations']]
            schema_recs = [iteration['recommendations']['schema'] for iteration in dataset_results['iterations']]
            storage_recs = [iteration['recommendations']['storage'] for iteration in dataset_results['iterations']]
            
            # Record summary
            summary[dataset_id] = {
                'size': size,
                'average_times': {
                    'query_optimizer': sum(query_times) / len(query_times),
                    'schema_optimizer': sum(schema_times) / len(schema_times),
                    'storage_optimizer': sum(storage_times) / len(storage_times),
                    'total': sum(total_times) / len(total_times)
                },
                'min_times': {
                    'query_optimizer': min(query_times),
                    'schema_optimizer': min(schema_times),
                    'storage_optimizer': min(storage_times),
                    'total': min(total_times)
                },
                'max_times': {
                    'query_optimizer': max(query_times),
                    'schema_optimizer': max(schema_times),
                    'storage_optimizer': max(storage_times),
                    'total': max(total_times)
                },
                'average_recommendations': {
                    'query': sum(query_recs) / len(query_recs),
                    'schema': sum(schema_recs) / len(schema_recs),
                    'storage': sum(storage_recs) / len(storage_recs),
                    'total': (sum(query_recs) + sum(schema_recs) + sum(storage_recs)) / len(query_recs)
                }
            }
            
            # Check if meets 4-minute requirement
            meets_requirement = summary[dataset_id]['average_times']['total'] < 240
            summary[dataset_id]['meets_4m_requirement'] = meets_requirement
            
            logger.info(f"Dataset {dataset_id} (size: {size}) performance summary:")
            logger.info(f"  Average total time: {summary[dataset_id]['average_times']['total']:.2f} seconds")
            logger.info(f"  Meets 4-minute requirement: {meets_requirement}")
        
        self.results['summary'] = summary
    
    def save_results(self, output_file: str = "dataset_performance_results.json"):
        """Save results to a JSON file.
        
        Args:
            output_file: Output file path
        """
        # Convert datetime objects to strings
        results_serializable = json.loads(
            json.dumps(self.results, default=lambda obj: obj.isoformat() if isinstance(obj, datetime.datetime) else str(obj))
        )
        
        # Add metadata
        results_serializable['metadata'] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'platform': sys.platform,
            'python_version': sys.version
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def print_report(self):
        """Print a performance report."""
        print("\n===== BigQuery Cost Intelligence Engine Performance Report =====\n")
        print("Performance testing for dataset processing time requirements\n")
        
        # Table headers
        print(f"{'Dataset Size':15} | {'Total Time (s)':15} | {'Meets 4m Req':15} | {'Query Opt (s)':15} | {'Schema Opt (s)':15} | {'Storage Opt (s)':15}")
        print("-" * 100)
        
        # Sort datasets by size
        sorted_datasets = sorted(self.results['summary'].items(), key=lambda x: x[1]['size'])
        
        for dataset_id, summary in sorted_datasets:
            size = summary['size']
            total_time = summary['average_times']['total']
            meets_req = "✓" if summary['meets_4m_requirement'] else "✗"
            query_time = summary['average_times']['query_optimizer']
            schema_time = summary['average_times']['schema_optimizer']
            storage_time = summary['average_times']['storage_optimizer']
            
            print(f"{size:15,d} | {total_time:15.2f} | {meets_req:15} | {query_time:15.2f} | {schema_time:15.2f} | {storage_time:15.2f}")
        
        print("\n")
        
        # Check if any dataset with 100K+ records doesn't meet requirement
        large_datasets = [summary for dataset_id, summary in self.results['summary'].items() if summary['size'] >= 100000]
        failing_datasets = [summary for summary in large_datasets if not summary['meets_4m_requirement']]
        
        if failing_datasets:
            print("⚠️ PERFORMANCE CONCERN: Some large datasets (100K+ records) do not meet the 4-minute completion requirement.")
            print("\nRecommendations for optimization:")
            print("1. Implement query result caching to reduce repeated processing")
            print("2. Parallelize analysis of individual tables within datasets")
            print("3. Consider batching recommendations to process critical ones first")
            print("4. Optimize SQL queries used in analysis to reduce BigQuery processing time")
            print("5. Implement timeout mechanisms to prevent excessive processing time")
        else:
            print("✓ Performance meets requirements: All datasets process within the 4-minute threshold.")
            
            # Check if there's room for improvement
            largest_dataset = max(large_datasets, key=lambda x: x['size'])
            headroom = 240 - largest_dataset['average_times']['total']
            
            if headroom > 60:
                print(f"\nPerformance headroom: {headroom:.2f} seconds under the 4-minute requirement for largest dataset.")
                print("Consider adding additional analysis features to leverage available processing time.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BigQuery Cost Intelligence Engine Performance Testing")
    parser.add_argument("--project-id", default="test-project", help="Google Cloud project ID")
    parser.add_argument("--output", default="dataset_performance_results.json", help="Output file path")
    args = parser.parse_args()
    
    # Define dataset sizes to test (number of records)
    dataset_sizes = [5000, 20000, 50000, 100000, 250000, 500000]
    
    # Initialize and run performance tester
    tester = PerformanceTester(args.project_id, dataset_sizes)
    tester.run_tests()
    tester.print_report()
    tester.save_results(args.output)

if __name__ == "__main__":
    main()