"""BigQuery mock objects for testing.

This module provides comprehensive mock objects for BigQuery testing,
allowing tests to run without actual BigQuery connections.
"""

import pandas as pd
import datetime
from unittest.mock import MagicMock, PropertyMock
from typing import Dict, List, Any, Optional, Tuple, Union

class MockBigQueryClient:
    """Comprehensive mock for BigQuery client with realistic behavior."""
    
    def __init__(self, project_id="test-project"):
        """Initialize with optional project ID."""
        self.project_id = project_id
        self._tables = {}  # Store mock tables
        self._datasets = {}  # Store mock datasets
        self._query_results = {}  # Store prepared query results
        
    def dataset(self, dataset_id):
        """Get a dataset reference."""
        dataset_ref = MagicMock()
        dataset_ref.dataset_id = dataset_id
        return dataset_ref
        
    def get_dataset(self, dataset_ref):
        """Get a dataset."""
        if isinstance(dataset_ref, str):
            dataset_id = dataset_ref.split('.')[-1]
        else:
            dataset_id = dataset_ref.dataset_id
            
        if dataset_id in self._datasets:
            return self._datasets[dataset_id]
            
        # Create a new mock dataset
        dataset = MagicMock()
        dataset.dataset_id = dataset_id
        dataset.location = "US"
        dataset.created = datetime.datetime(2023, 1, 1)
        dataset.modified = datetime.datetime(2023, 1, 2)
        
        self._datasets[dataset_id] = dataset
        return dataset
        
    def list_tables(self, dataset):
        """List tables in a dataset."""
        dataset_id = dataset.dataset_id if hasattr(dataset, 'dataset_id') else dataset
        
        if dataset_id not in self._tables:
            return []
            
        return [self.get_table(f"{dataset_id}.{table_id}") for table_id in self._tables[dataset_id]]
        
    def get_table(self, table_ref):
        """Get a table."""
        if isinstance(table_ref, str):
            parts = table_ref.split('.')
            dataset_id = parts[-2] if len(parts) >= 2 else "default"
            table_id = parts[-1]
        else:
            dataset_id = table_ref.dataset_id
            table_id = table_ref.table_id
            
        table_key = f"{dataset_id}.{table_id}"
        
        if table_key in self._tables:
            return self._tables[table_key]
            
        # Create a new mock table
        table = MagicMock()
        table.table_id = table_id
        table.dataset_id = dataset_id
        table.project = self.project_id
        table.created = datetime.datetime(2023, 1, 1)
        table.modified = datetime.datetime(2023, 1, 2)
        table.num_rows = 1000
        table.num_bytes = 1000000
        
        # Add schema
        schema_field = MagicMock()
        schema_field.name = "id"
        schema_field.field_type = "INTEGER"
        schema_field.mode = "REQUIRED"
        table.schema = [schema_field]
        
        self._tables[table_key] = table
        return table
        
    def query(self, query):
        """Execute a query."""
        # Check if we have a prepared result for this query
        for query_pattern, result in self._query_results.items():
            if query_pattern in query:
                return MockQueryJob(result)
                
        # Default empty result
        return MockQueryJob(pd.DataFrame())
        
    def set_query_result(self, query_pattern, result_df):
        """Set a prepared result for a query pattern."""
        self._query_results[query_pattern] = result_df
        
    def add_mock_table(self, dataset_id, table_id, schema=None, num_rows=1000, num_bytes=1000000):
        """Add a mock table with specified properties."""
        table = MagicMock()
        table.table_id = table_id
        table.dataset_id = dataset_id
        table.project = self.project_id
        table.created = datetime.datetime(2023, 1, 1)
        table.modified = datetime.datetime(2023, 1, 2)
        table.num_rows = num_rows
        table.num_bytes = num_bytes
        
        # Add schema
        if schema:
            table.schema = []
            for field_spec in schema:
                field = MagicMock()
                field.name = field_spec["name"]
                field.field_type = field_spec["type"]
                field.mode = field_spec.get("mode", "NULLABLE")
                table.schema.append(field)
        else:
            schema_field = MagicMock()
            schema_field.name = "id"
            schema_field.field_type = "INTEGER"
            schema_field.mode = "REQUIRED"
            table.schema = [schema_field]
        
        # Store the table
        if dataset_id not in self._tables:
            self._tables[dataset_id] = []
        
        table_key = f"{dataset_id}.{table_id}"
        self._tables[table_key] = table
        return table


class MockQueryJob:
    """Mock for BigQuery query job with result handling."""
    
    def __init__(self, result_df):
        """Initialize with a result DataFrame."""
        self.result_df = result_df
        
    def to_dataframe(self):
        """Convert result to DataFrame."""
        return self.result_df
        
    def result(self):
        """Get query result."""
        return [tuple(row) for row in self.result_df.itertuples(index=False)]