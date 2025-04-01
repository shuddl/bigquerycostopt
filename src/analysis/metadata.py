"""Metadata extraction functions for BigQuery datasets."""

from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from collections import defaultdict
import json

from ..utils.logging import setup_logger
from ..connectors.bigquery import BigQueryConnector

logger = setup_logger(__name__)

# Constants
MAX_RESULTS_PER_PAGE = 1000
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
DEFAULT_ANALYSIS_PERIOD_DAYS = 30


class MetadataExtractor:
    """Class for extracting comprehensive metadata from BigQuery."""
    
    def __init__(self, connector: Optional[BigQueryConnector] = None,
                project_id: Optional[str] = None, 
                credentials_path: Optional[str] = None):
        """Initialize the metadata extractor.
        
        Args:
            connector: Optional existing BigQueryConnector instance
            project_id: GCP project ID if connector not provided
            credentials_path: Path to service account credentials if connector not provided
        """
        if connector:
            self.connector = connector
            self.client = connector.client
            self.project_id = connector.project_id
        else:
            if not project_id:
                raise ValueError("Either connector or project_id must be provided")
            self.connector = BigQueryConnector(project_id, credentials_path)
            self.client = self.connector.client
            self.project_id = project_id
        
        logger.info(f"Initialized MetadataExtractor for project {self.project_id}")
        
        # Cache for frequently accessed data
        self._cache = {}
    
    def extract_dataset_metadata(self, dataset_id: str, include_tables: bool = True,
                               include_usage_stats: bool = True,
                               include_column_stats: bool = True) -> Dict[str, Any]:
        """Extract comprehensive metadata for a BigQuery dataset.
        
        Args:
            dataset_id: The BigQuery dataset ID to analyze
            include_tables: Whether to include detailed table metadata
            include_usage_stats: Whether to include usage statistics
            include_column_stats: Whether to include column statistics
            
        Returns:
            Dict containing detailed dataset metadata
        """
        start_time = time.time()
        dataset_key = f"{self.project_id}.{dataset_id}"
        logger.info(f"Starting metadata extraction for dataset {dataset_key}")
        
        try:
            # Get dataset metadata from API
            dataset_ref = self.client.dataset(dataset_id)
            dataset = self.client.get_dataset(dataset_ref)
            
            # Get INFORMATION_SCHEMA metadata which has more details
            is_metadata = self._get_dataset_schema_metadata(dataset_id)
            
            # Basic dataset info
            metadata = {
                "dataset_id": dataset_id,
                "project_id": self.project_id,
                "full_name": dataset_key,
                "location": dataset.location,
                "created": dataset.created.isoformat(),
                "last_modified": dataset.modified.isoformat(),
                "default_partition_expiration_ms": dataset.default_partition_expiration_ms,
                "default_table_expiration_ms": dataset.default_table_expiration_ms,
                "access_entries": self._extract_access_entries(dataset),
                "labels": dataset.labels or {},
                "tables": []
            }
            
            # Add information schema metadata
            metadata.update(is_metadata)
            
            # Get list of tables (paginated for large datasets)
            tables_list = []
            page_token = None
            
            while True:
                tables_page, page_token = self.client.list_tables(
                    dataset_ref, max_results=MAX_RESULTS_PER_PAGE, page_token=page_token
                )
                tables_list.extend(list(tables_page))
                if not page_token:
                    break
            
            metadata["table_count"] = len(tables_list)
            logger.info(f"Found {len(tables_list)} tables in dataset {dataset_key}")
            
            # Process tables if requested
            if include_tables:
                total_size_bytes = 0
                for i, table_ref in enumerate(tables_list):
                    try:
                        logger.debug(f"Extracting metadata for table {i+1}/{len(tables_list)}: {table_ref.table_id}")
                        table_metadata = self.extract_table_metadata(
                            dataset_id, 
                            table_ref.table_id,
                            include_usage_stats=include_usage_stats,
                            include_column_stats=include_column_stats
                        )
                        metadata["tables"].append(table_metadata)
                        total_size_bytes += table_metadata.get("size_bytes", 0)
                    except Exception as e:
                        logger.warning(f"Error extracting metadata for table {table_ref.table_id}: {e}")
                        # Add minimal information for the table
                        metadata["tables"].append({
                            "table_id": table_ref.table_id,
                            "error": str(e)
                        })
                
                # Add aggregated dataset statistics
                metadata["total_size_bytes"] = total_size_bytes
                metadata["total_size_gb"] = total_size_bytes / (1024**3)
                metadata["total_size_tb"] = total_size_bytes / (1024**4)
                
                # Calculate monthly storage cost (using $0.02 per GB per month for active storage)
                metadata["estimated_monthly_storage_cost"] = metadata["total_size_gb"] * 0.02
            
            # Include usage statistics for the dataset as a whole if requested
            if include_usage_stats:
                try:
                    usage_stats = self.get_dataset_usage_stats(dataset_id)
                    metadata.update(usage_stats)
                except Exception as e:
                    logger.warning(f"Could not retrieve dataset usage stats: {e}")
            
            end_time = time.time()
            metadata["extraction_time_seconds"] = end_time - start_time
            logger.info(f"Completed metadata extraction for dataset {dataset_key} in {metadata['extraction_time_seconds']:.2f} seconds")
            
            return metadata
            
        except Exception as e:
            logger.exception(f"Error extracting dataset metadata for {dataset_id}: {e}")
            raise
    
    def extract_table_metadata(self, dataset_id: str, table_id: str, 
                             include_usage_stats: bool = True,
                             include_column_stats: bool = True) -> Dict[str, Any]:
        """Extract detailed metadata for a single BigQuery table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            include_usage_stats: Whether to include usage statistics
            include_column_stats: Whether to include column statistics
            
        Returns:
            Dict containing detailed table metadata
        """
        table_key = f"{self.project_id}.{dataset_id}.{table_id}"
        logger.debug(f"Extracting metadata for table {table_key}")
        
        try:
            # Get table metadata from API
            table_ref = self.client.dataset(dataset_id).table(table_id)
            table = self.client.get_table(table_ref)
            
            # Get additional metadata from INFORMATION_SCHEMA
            is_metadata = self._get_table_schema_metadata(dataset_id, table_id)
            
            # Process schema information
            schema_fields = []
            for field in table.schema:
                field_info = {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or ""
                }
                
                # Extract nested fields if any
                if field.fields:
                    field_info["fields"] = self._extract_nested_fields(field.fields)
                
                schema_fields.append(field_info)
            
            # Extract partitioning information
            partitioning_info = None
            if table.time_partitioning:
                partitioning_info = {
                    "type": table.time_partitioning.type_,
                    "field": table.time_partitioning.field,
                    "expiration_ms": table.time_partitioning.expiration_ms,
                    "require_partition_filter": table.time_partitioning.require_partition_filter
                }
            elif table.range_partitioning:
                partitioning_info = {
                    "type": "RANGE",
                    "field": table.range_partitioning.field,
                    "range": {
                        "start": table.range_partitioning.range_.start,
                        "end": table.range_partitioning.range_.end,
                        "interval": table.range_partitioning.range_.interval
                    }
                }
            
            # Extract clustering information
            clustering_info = None
            if table.clustering_fields:
                clustering_info = {
                    "fields": table.clustering_fields
                }
            
            # Build metadata object
            metadata = {
                "table_id": table.table_id,
                "dataset_id": dataset_id,
                "project_id": self.project_id,
                "full_name": table_key,
                "type": table.table_type,
                "created": table.created.isoformat(),
                "last_modified": table.modified.isoformat(),
                "expires": table.expires.isoformat() if table.expires else None,
                "num_rows": int(table.num_rows) if table.num_rows is not None else None,
                "num_bytes": int(table.num_bytes) if table.num_bytes is not None else None,
                "size_bytes": int(table.num_bytes) if table.num_bytes is not None else 0,
                "size_gb": float(table.num_bytes) / (1024**3) if table.num_bytes is not None else 0,
                "schema": schema_fields,
                "partitioning": partitioning_info,
                "clustering": clustering_info,
                "description": table.description,
                "labels": table.labels or {},
                "field_count": len(schema_fields)
            }
            
            # Add information schema metadata
            metadata.update(is_metadata)
            
            # Add column statistics if requested
            if include_column_stats:
                try:
                    column_stats = self.get_column_stats(dataset_id, table_id)
                    metadata["column_stats"] = column_stats
                except Exception as e:
                    logger.warning(f"Could not retrieve column stats for {table_key}: {e}")
            
            # Add usage statistics if requested
            if include_usage_stats:
                try:
                    usage_stats = self.get_table_usage_stats(dataset_id, table_id)
                    metadata.update(usage_stats)
                except Exception as e:
                    logger.warning(f"Could not retrieve usage stats for {table_key}: {e}")
            
            # Calculate monthly storage cost (using $0.02 per GB per month for active storage)
            metadata["estimated_monthly_storage_cost"] = metadata["size_gb"] * 0.02
            
            # Calculate avg row size
            if metadata["num_rows"] and metadata["num_rows"] > 0:
                metadata["avg_row_bytes"] = metadata["size_bytes"] / metadata["num_rows"]
            else:
                metadata["avg_row_bytes"] = None
            
            return metadata
            
        except Exception as e:
            logger.exception(f"Error extracting table metadata for {table_key}: {e}")
            raise
    
    def get_table_usage_stats(self, dataset_id: str, table_id: str, days: int = DEFAULT_ANALYSIS_PERIOD_DAYS) -> Dict[str, Any]:
        """Get comprehensive usage statistics for a table from INFORMATION_SCHEMA.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            days: Number of days to analyze
            
        Returns:
            Dict containing detailed usage statistics
        """
        table_key = f"{self.project_id}.{dataset_id}.{table_id}"
        logger.debug(f"Getting usage stats for table {table_key} over {days} days")
        
        try:
            # Query for basic usage stats
            usage_query = f"""
            SELECT
              COUNT(*) AS query_count,
              SUM(total_bytes_processed) AS total_bytes_processed,
              SUM(total_slot_ms) AS total_slot_ms,
              AVG(total_bytes_processed) AS avg_bytes_processed_per_query,
              MAX(total_bytes_processed) AS max_bytes_processed,
              SUM(total_bytes_billed) AS total_bytes_billed,
              COUNT(DISTINCT user_email) AS distinct_users
            FROM
              `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS
            WHERE
              creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
              AND job_type = 'QUERY'
              AND state = 'DONE'
              AND REGEXP_CONTAINS(query, r'[^\\w]`?{self.project_id}\\.{dataset_id}\\.{table_id}`?[^\\w]')
            """
            
            basic_stats_df = self.connector.query_to_dataframe(usage_query)
            
            # Initialize results with default values
            usage_stats = {
                "query_count_{}d".format(days): 0,
                "total_bytes_processed_{}d".format(days): 0,
                "total_slot_ms_{}d".format(days): 0,
                "avg_bytes_processed_per_query": 0,
                "max_bytes_processed": 0,
                "total_bytes_billed_{}d".format(days): 0,
                "distinct_users_{}d".format(days): 0,
                "estimated_query_cost_{}d".format(days): 0,
                "frequently_joined_tables": [],
                "frequent_filters": [],
                "cached_queries_pct": 0
            }
            
            if not basic_stats_df.empty:
                row = basic_stats_df.iloc[0]
                
                usage_stats.update({
                    "query_count_{}d".format(days): int(row["query_count"]),
                    "total_bytes_processed_{}d".format(days): int(row["total_bytes_processed"]) if row["total_bytes_processed"] else 0,
                    "total_slot_ms_{}d".format(days): int(row["total_slot_ms"]) if row["total_slot_ms"] else 0,
                    "avg_bytes_processed_per_query": int(row["avg_bytes_processed_per_query"]) if row["avg_bytes_processed_per_query"] else 0,
                    "max_bytes_processed": int(row["max_bytes_processed"]) if row["max_bytes_processed"] else 0,
                    "total_bytes_billed_{}d".format(days): int(row["total_bytes_billed"]) if row["total_bytes_billed"] else 0,
                    "distinct_users_{}d".format(days): int(row["distinct_users"]) if row["distinct_users"] else 0
                })
                
                # Calculate estimated query cost ($5 per TB)
                tb_processed = usage_stats[f"total_bytes_processed_{days}d"] / (1024**4)
                usage_stats[f"estimated_query_cost_{days}d"] = tb_processed * 5.0
                
                # Further analysis only if we have queries to analyze
                if usage_stats[f"query_count_{days}d"] > 0:
                    # Get caching statistics
                    cache_query = f"""
                    SELECT
                      COUNT(*) AS total_queries,
                      COUNTIF(cache_hit) AS cached_queries
                    FROM
                      `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS
                    WHERE
                      creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
                      AND job_type = 'QUERY'
                      AND state = 'DONE'
                      AND REGEXP_CONTAINS(query, r'[^\\w]`?{self.project_id}\\.{dataset_id}\\.{table_id}`?[^\\w]')
                    """
                    
                    cache_df = self.connector.query_to_dataframe(cache_query)
                    if not cache_df.empty:
                        row = cache_df.iloc[0]
                        if row["total_queries"] > 0:
                            usage_stats["cached_queries_pct"] = 100.0 * row["cached_queries"] / row["total_queries"]
                    
                    # Analyze query patterns - get frequently joined tables
                    join_query = f"""
                    WITH extracted_tables AS (
                      SELECT
                        query_text,
                        REGEXP_EXTRACT_ALL(LOWER(query_text), r'join\\s+`?([a-zA-Z0-9-_]+\\.{dataset_id}\\.[a-zA-Z0-9-_]+)`?') AS joined_tables
                      FROM
                        `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS
                      WHERE
                        creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
                        AND job_type = 'QUERY'
                        AND state = 'DONE'
                        AND REGEXP_CONTAINS(LOWER(query_text), r'join')
                        AND REGEXP_CONTAINS(query_text, r'[^\\w]`?{self.project_id}\\.{dataset_id}\\.{table_id}`?[^\\w]')
                    ),
                    flattened_tables AS (
                      SELECT
                        joined_table
                      FROM
                        extracted_tables,
                        UNNEST(joined_tables) AS joined_table
                      WHERE
                        joined_table != '{self.project_id}.{dataset_id}.{table_id}'
                    )
                    SELECT
                      joined_table,
                      COUNT(*) AS join_count
                    FROM
                      flattened_tables
                    GROUP BY
                      joined_table
                    ORDER BY
                      join_count DESC
                    LIMIT 10
                    """
                    
                    join_df = self.connector.query_to_dataframe(join_query)
                    if not join_df.empty:
                        usage_stats["frequently_joined_tables"] = [
                            {"table": row["joined_table"], "count": int(row["join_count"])}
                            for _, row in join_df.iterrows()
                        ]
                    
                    # Analyze query patterns - get frequent filters
                    filter_query = f"""
                    WITH extracted_filters AS (
                      SELECT
                        query_text,
                        REGEXP_EXTRACT_ALL(LOWER(query_text), r'where.*?([a-zA-Z0-9_]+)\\s*(=|>|<|>=|<=|like|in|between)') AS filters
                      FROM
                        `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS
                      WHERE
                        creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
                        AND job_type = 'QUERY'
                        AND state = 'DONE'
                        AND REGEXP_CONTAINS(LOWER(query_text), r'where')
                        AND REGEXP_CONTAINS(query_text, r'[^\\w]`?{self.project_id}\\.{dataset_id}\\.{table_id}`?[^\\w]')
                    ),
                    flattened_filters AS (
                      SELECT
                        filter_column
                      FROM
                        extracted_filters,
                        UNNEST(filters) AS filter_column
                    )
                    SELECT
                      filter_column,
                      COUNT(*) AS filter_count
                    FROM
                      flattened_filters
                    GROUP BY
                      filter_column
                    ORDER BY
                      filter_count DESC
                    LIMIT 10
                    """
                    
                    try:
                        filter_df = self.connector.query_to_dataframe(filter_query)
                        if not filter_df.empty:
                            usage_stats["frequent_filters"] = [
                                {"column": row["filter_column"], "count": int(row["filter_count"])}
                                for _, row in filter_df.iterrows()
                            ]
                    except Exception as e:
                        logger.warning(f"Error analyzing filter patterns: {e}")
            
            return usage_stats
            
        except Exception as e:
            logger.warning(f"Error getting usage stats for {table_key}: {e}")
            return {
                "query_count_{}d".format(days): None,
                "total_bytes_processed_{}d".format(days): None,
                "total_slot_ms_{}d".format(days): None,
                "avg_bytes_processed_per_query": None
            }
    
    def get_dataset_usage_stats(self, dataset_id: str, days: int = DEFAULT_ANALYSIS_PERIOD_DAYS) -> Dict[str, Any]:
        """Get usage statistics for an entire dataset.
        
        Args:
            dataset_id: BigQuery dataset ID
            days: Number of days to analyze
            
        Returns:
            Dict containing dataset usage statistics
        """
        dataset_key = f"{self.project_id}.{dataset_id}"
        logger.debug(f"Getting usage stats for dataset {dataset_key} over {days} days")
        
        try:
            usage_query = f"""
            SELECT
              COUNT(*) AS query_count,
              SUM(total_bytes_processed) AS total_bytes_processed,
              SUM(total_slot_ms) AS total_slot_ms,
              SUM(total_bytes_billed) AS total_bytes_billed,
              COUNT(DISTINCT user_email) AS distinct_users
            FROM
              `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS
            WHERE
              creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
              AND job_type = 'QUERY'
              AND state = 'DONE'
              AND REGEXP_CONTAINS(query, r'[^\\w]`?{self.project_id}\\.{dataset_id}\\.')
            """
            
            df = self.connector.query_to_dataframe(usage_query)
            
            if not df.empty:
                row = df.iloc[0]
                stats = {
                    "dataset_query_count_{}d".format(days): int(row["query_count"]),
                    "dataset_bytes_processed_{}d".format(days): int(row["total_bytes_processed"]) if row["total_bytes_processed"] else 0,
                    "dataset_slot_ms_{}d".format(days): int(row["total_slot_ms"]) if row["total_slot_ms"] else 0,
                    "dataset_bytes_billed_{}d".format(days): int(row["total_bytes_billed"]) if row["total_bytes_billed"] else 0,
                    "dataset_distinct_users_{}d".format(days): int(row["distinct_users"]) if row["distinct_users"] else 0
                }
                
                # Calculate estimated cost
                tb_processed = stats[f"dataset_bytes_processed_{days}d"] / (1024**4)
                stats[f"dataset_estimated_cost_{days}d"] = tb_processed * 5.0
                
                # Get daily usage trends
                trends_query = f"""
                SELECT
                  DATE(creation_time) AS query_date,
                  COUNT(*) AS query_count,
                  SUM(total_bytes_processed) AS bytes_processed
                FROM
                  `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS
                WHERE
                  creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
                  AND job_type = 'QUERY'
                  AND state = 'DONE'
                  AND REGEXP_CONTAINS(query, r'[^\\w]`?{self.project_id}\\.{dataset_id}\\.')
                GROUP BY
                  query_date
                ORDER BY
                  query_date
                """
                
                trends_df = self.connector.query_to_dataframe(trends_query)
                if not trends_df.empty:
                    stats["daily_usage_trends"] = [
                        {
                            "date": row["query_date"].strftime("%Y-%m-%d"),
                            "query_count": int(row["query_count"]),
                            "bytes_processed": int(row["bytes_processed"]) if row["bytes_processed"] else 0
                        }
                        for _, row in trends_df.iterrows()
                    ]
                
                return stats
            
            return {
                "dataset_query_count_{}d".format(days): 0,
                "dataset_bytes_processed_{}d".format(days): 0,
                "dataset_slot_ms_{}d".format(days): 0,
                "dataset_bytes_billed_{}d".format(days): 0,
                "dataset_distinct_users_{}d".format(days): 0,
                "dataset_estimated_cost_{}d".format(days): 0
            }
            
        except Exception as e:
            logger.warning(f"Error getting dataset usage stats for {dataset_key}: {e}")
            return {
                "dataset_query_count_{}d".format(days): None,
                "dataset_bytes_processed_{}d".format(days): None,
                "dataset_slot_ms_{}d".format(days): None
            }
    
    def get_column_stats(self, dataset_id: str, table_id: str) -> List[Dict[str, Any]]:
        """Get statistics for columns in a table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            List of column statistics including data distribution info
        """
        table_key = f"{self.project_id}.{dataset_id}.{table_id}"
        logger.debug(f"Getting column stats for table {table_key}")
        
        try:
            # Get column statistics from INFORMATION_SCHEMA
            query = f"""
            SELECT
              column_name,
              data_type,
              is_nullable,
              IF(table_catalog IS NULL, false, true) AS is_partitioning_column,
              SAFE_CAST(numeric_precision AS INT64) AS numeric_precision,
              SAFE_CAST(numeric_scale AS INT64) AS numeric_scale,
              CAST(NULL AS INT64) AS distinct_values_count,
              CAST(NULL AS FLOAT64) AS null_percentage
            FROM
              `{self.project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE
              table_name = '{table_id}'
            ORDER BY
              ordinal_position
            """
            
            df = self.connector.query_to_dataframe(query)
            
            if df.empty:
                return []
            
            # Try to get additional statistics if possible
            # Note: This might not work for very large tables
            try:
                # Check if table size is reasonable to compute statistics
                table_metadata = self.client.get_table(f"{self.project_id}.{dataset_id}.{table_id}")
                # Only compute statistics for tables under 10GB
                if table_metadata.num_bytes and table_metadata.num_bytes < 10 * (1024**3):
                    # For each column, compute additional statistics
                    for i, row in df.iterrows():
                        column_name = row["column_name"]
                        data_type = row["data_type"]
                        
                        # Skip complex types
                        if data_type in ('STRUCT', 'ARRAY', 'GEOGRAPHY', 'JSON'):
                            continue
                        
                        # Compute distinct count and null percentage
                        stats_query = f"""
                        SELECT
                          COUNT(DISTINCT {column_name}) AS distinct_count,
                          COUNTIF({column_name} IS NULL) / COUNT(*) * 100 AS null_percentage
                        FROM
                          `{self.project_id}.{dataset_id}.{table_id}`
                        """
                        
                        try:
                            stats_df = self.connector.query_to_dataframe(stats_query)
                            if not stats_df.empty:
                                df.at[i, "distinct_values_count"] = int(stats_df.iloc[0]["distinct_count"])
                                df.at[i, "null_percentage"] = float(stats_df.iloc[0]["null_percentage"])
                        except Exception as e:
                            logger.debug(f"Could not compute advanced stats for column {column_name}: {e}")
            except Exception as e:
                logger.debug(f"Skipping advanced column statistics for {table_key}: {e}")
            
            # Convert dataframe to list of dicts
            columns_stats = []
            for _, row in df.iterrows():
                column_stats = {
                    "name": row["column_name"],
                    "type": row["data_type"],
                    "nullable": row["is_nullable"] == "YES",
                    "is_partitioning_column": bool(row["is_partitioning_column"])
                }
                
                # Add numeric precision and scale if available
                if row["numeric_precision"] is not None:
                    column_stats["numeric_precision"] = int(row["numeric_precision"])
                if row["numeric_scale"] is not None:
                    column_stats["numeric_scale"] = int(row["numeric_scale"])
                
                # Add advanced statistics if available
                if row["distinct_values_count"] is not None:
                    column_stats["distinct_values_count"] = int(row["distinct_values_count"])
                if row["null_percentage"] is not None:
                    column_stats["null_percentage"] = float(row["null_percentage"])
                
                columns_stats.append(column_stats)
            
            return columns_stats
            
        except Exception as e:
            logger.warning(f"Error getting column stats for {table_key}: {e}")
            return []

    def get_query_patterns(self, dataset_id: str, table_id: Optional[str] = None, 
                         days: int = DEFAULT_ANALYSIS_PERIOD_DAYS) -> Dict[str, Any]:
        """Analyze query patterns for a dataset or table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: Optional BigQuery table ID (if None, analyzes whole dataset)
            days: Number of days to analyze
            
        Returns:
            Dict containing query pattern analysis
        """
        if table_id:
            entity_key = f"{self.project_id}.{dataset_id}.{table_id}"
            table_filter = f"AND REGEXP_CONTAINS(query, r'[^\\w]`?{self.project_id}\\.{dataset_id}\\.{table_id}`?[^\\w]')"
        else:
            entity_key = f"{self.project_id}.{dataset_id}"
            table_filter = f"AND REGEXP_CONTAINS(query, r'[^\\w]`?{self.project_id}\\.{dataset_id}\\.')"
        
        logger.debug(f"Analyzing query patterns for {entity_key} over {days} days")
        
        try:
            # Get query history
            query = f"""
            SELECT
              query_text,
              total_bytes_processed,
              total_slot_ms,
              creation_time,
              user_email,
              error_result,
              cache_hit,
              total_bytes_billed
            FROM
              `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS
            WHERE
              creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
              AND job_type = 'QUERY'
              {table_filter}
            ORDER BY
              total_bytes_processed DESC
            LIMIT 500
            """
            
            df = self.connector.query_to_dataframe(query)
            
            if df.empty:
                return {"query_count": 0, "patterns": {}}
            
            # Initialize patterns dictionary
            patterns = {
                "query_count": len(df),
                "select_star_count": 0,
                "missing_where_count": 0,
                "no_partition_filter_count": 0,
                "cache_hit_count": 0,
                "full_table_scan_count": 0,
                "high_cost_queries": [],
                "common_filter_columns": [],
                "common_group_by_columns": [],
                "common_join_tables": [],
                "error_types": {},
                "user_distribution": {}
            }
            
            # Analyze queries
            for _, row in df.iterrows():
                query_text = row["query_text"].lower() if row["query_text"] else ""
                
                # Check for SELECT *
                if re.search(r"select\s+\*", query_text):
                    patterns["select_star_count"] += 1
                
                # Check for missing WHERE clause
                if not re.search(r"\swhere\s", query_text):
                    patterns["missing_where_count"] += 1
                
                # Check for cache hits
                if row["cache_hit"]:
                    patterns["cache_hit_count"] += 1
                
                # Track high cost queries
                if row["total_bytes_processed"] and row["total_bytes_processed"] > 1024**3:  # > 1GB
                    patterns["high_cost_queries"].append({
                        "bytes_processed": int(row["total_bytes_processed"]),
                        "slot_ms": int(row["total_slot_ms"]) if row["total_slot_ms"] else 0,
                        "creation_time": row["creation_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "bytes_billed": int(row["total_bytes_billed"]) if row["total_bytes_billed"] else 0,
                        "estimated_cost": (row["total_bytes_billed"] / (1024**4)) * 5.0 if row["total_bytes_billed"] else 0,
                        "user": row["user_email"]
                    })
                
                # Count errors by type
                if row["error_result"]:
                    error_msg = str(row["error_result"])
                    error_type = self._categorize_error(error_msg)
                    if error_type not in patterns["error_types"]:
                        patterns["error_types"][error_type] = 0
                    patterns["error_types"][error_type] += 1
                
                # Count queries by user
                user = row["user_email"] or "unknown"
                if user not in patterns["user_distribution"]:
                    patterns["user_distribution"][user] = 0
                patterns["user_distribution"][user] += 1
            
            # Get common filter columns
            filter_query = f"""
            WITH extracted_filters AS (
              SELECT
                REGEXP_EXTRACT_ALL(LOWER(query_text), r'where.*?([a-zA-Z0-9_]+)\\s*(=|>|<|>=|<=|like|in|between)') AS filters
              FROM
                `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS
              WHERE
                creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
                AND job_type = 'QUERY'
                AND state = 'DONE'
                AND REGEXP_CONTAINS(LOWER(query_text), r'where')
                {table_filter}
            ),
            flattened_filters AS (
              SELECT
                filter_column
              FROM
                extracted_filters,
                UNNEST(filters) AS filter_column
            )
            SELECT
              filter_column,
              COUNT(*) AS filter_count
            FROM
              flattened_filters
            GROUP BY
              filter_column
            ORDER BY
              filter_count DESC
            LIMIT 10
            """
            
            try:
                filter_df = self.connector.query_to_dataframe(filter_query)
                if not filter_df.empty:
                    patterns["common_filter_columns"] = [
                        {"column": row["filter_column"], "count": int(row["filter_count"])}
                        for _, row in filter_df.iterrows()
                    ]
            except Exception as e:
                logger.debug(f"Error extracting filter patterns: {e}")
            
            # Calculate metrics
            patterns["select_star_pct"] = 100.0 * patterns["select_star_count"] / patterns["query_count"]
            patterns["missing_where_pct"] = 100.0 * patterns["missing_where_count"] / patterns["query_count"]
            patterns["cache_hit_pct"] = 100.0 * patterns["cache_hit_count"] / patterns["query_count"]
            
            # Sort users by query count
            patterns["user_distribution"] = dict(sorted(
                patterns["user_distribution"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])  # Top 10 users
            
            # Limit high cost queries to top 10
            patterns["high_cost_queries"] = sorted(
                patterns["high_cost_queries"],
                key=lambda x: x["bytes_processed"],
                reverse=True
            )[:10]
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error analyzing query patterns for {entity_key}: {e}")
            return {"query_count": 0, "error": str(e)}
    
    def analyze_partition_usage(self, dataset_id: str, table_id: str, 
                              days: int = DEFAULT_ANALYSIS_PERIOD_DAYS) -> Dict[str, Any]:
        """Analyze how partitioning is being used for a table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            days: Number of days to analyze
            
        Returns:
            Dict with partition usage analysis
        """
        table_key = f"{self.project_id}.{dataset_id}.{table_id}"
        logger.debug(f"Analyzing partition usage for {table_key}")
        
        try:
            # First check if table is partitioned
            table = self.client.get_table(f"{self.project_id}.{dataset_id}.{table_id}")
            
            if not table.time_partitioning and not table.range_partitioning:
                return {"is_partitioned": False, "message": "Table is not partitioned"}
            
            # Determine partition type and field
            if table.time_partitioning:
                partition_type = table.time_partitioning.type_
                partition_field = table.time_partitioning.field
            else:
                partition_type = "RANGE"
                partition_field = table.range_partitioning.field
            
            # Query to analyze partition usage
            query = f"""
            WITH query_analysis AS (
              SELECT
                query_text,
                REGEXP_CONTAINS(LOWER(query_text), r'where.*{partition_field}') AS has_partition_filter,
                total_bytes_processed,
                total_slot_ms
              FROM
                `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS
              WHERE
                creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
                AND job_type = 'QUERY'
                AND state = 'DONE'
                AND REGEXP_CONTAINS(query, r'[^\\w]`?{self.project_id}\\.{dataset_id}\\.{table_id}`?[^\\w]')
            )
            SELECT
              COUNT(*) AS total_queries,
              COUNTIF(has_partition_filter) AS queries_with_partition_filter,
              AVG(IF(has_partition_filter, total_bytes_processed, NULL)) AS avg_bytes_with_filter,
              AVG(IF(NOT has_partition_filter, total_bytes_processed, NULL)) AS avg_bytes_without_filter,
              SUM(IF(has_partition_filter, total_bytes_processed, 0)) AS total_bytes_with_filter,
              SUM(IF(NOT has_partition_filter, total_bytes_processed, 0)) AS total_bytes_without_filter
            FROM
              query_analysis
            """
            
            df = self.connector.query_to_dataframe(query)
            
            if df.empty:
                return {
                    "is_partitioned": True,
                    "partition_type": partition_type,
                    "partition_field": partition_field,
                    "no_queries_found": True
                }
            
            row = df.iloc[0]
            total_queries = int(row["total_queries"])
            queries_with_filter = int(row["queries_with_partition_filter"])
            
            if total_queries == 0:
                return {
                    "is_partitioned": True,
                    "partition_type": partition_type,
                    "partition_field": partition_field,
                    "no_queries_found": True
                }
            
            filter_usage_pct = 100.0 * queries_with_filter / total_queries
            
            # Get partition statistics
            partition_stats = None
            if partition_type != "RANGE":
                try:
                    # For time-based partitioning
                    partition_query = f"""
                    SELECT
                      COUNT(*) AS partition_count,
                      MIN(partition_id) AS oldest_partition,
                      MAX(partition_id) AS newest_partition,
                      AVG(total_rows) AS avg_rows_per_partition,
                      AVG(total_logical_bytes) / POWER(1024, 3) AS avg_gb_per_partition,
                      MIN(total_rows) AS min_rows,
                      MAX(total_rows) AS max_rows,
                      STDDEV(total_rows) AS stddev_rows
                    FROM
                      `{self.project_id}.{dataset_id}.INFORMATION_SCHEMA.PARTITIONS`
                    WHERE
                      table_name = '{table_id}'
                    """
                    
                    partition_df = self.connector.query_to_dataframe(partition_query)
                    
                    if not partition_df.empty:
                        p_row = partition_df.iloc[0]
                        partition_stats = {
                            "partition_count": int(p_row["partition_count"]),
                            "oldest_partition": p_row["oldest_partition"],
                            "newest_partition": p_row["newest_partition"],
                            "avg_rows_per_partition": float(p_row["avg_rows_per_partition"]) if p_row["avg_rows_per_partition"] else 0,
                            "avg_gb_per_partition": float(p_row["avg_gb_per_partition"]) if p_row["avg_gb_per_partition"] else 0,
                            "min_rows": int(p_row["min_rows"]) if p_row["min_rows"] else 0,
                            "max_rows": int(p_row["max_rows"]) if p_row["max_rows"] else 0,
                            "stddev_rows": float(p_row["stddev_rows"]) if p_row["stddev_rows"] else 0
                        }
                except Exception as e:
                    logger.debug(f"Error getting partition statistics: {e}")
            
            # Build result
            result = {
                "is_partitioned": True,
                "partition_type": partition_type,
                "partition_field": partition_field,
                "total_queries": total_queries,
                "queries_with_partition_filter": queries_with_filter,
                "filter_usage_percentage": filter_usage_pct,
                "avg_bytes_with_filter": float(row["avg_bytes_with_filter"]) if row["avg_bytes_with_filter"] else 0,
                "avg_bytes_without_filter": float(row["avg_bytes_without_filter"]) if row["avg_bytes_without_filter"] else 0,
                "total_bytes_with_filter": float(row["total_bytes_with_filter"]) if row["total_bytes_with_filter"] else 0,
                "total_bytes_without_filter": float(row["total_bytes_without_filter"]) if row["total_bytes_without_filter"] else 0,
                "partition_statistics": partition_stats
            }
            
            # Calculate potential savings if all queries used partition filter
            if queries_with_filter > 0 and total_queries > queries_with_filter:
                # Average improvement ratio when using partition filters
                if result["avg_bytes_without_filter"] > 0 and result["avg_bytes_with_filter"] > 0:
                    improvement_ratio = result["avg_bytes_with_filter"] / result["avg_bytes_without_filter"]
                    
                    # Estimate potential bytes saved if all queries used filters
                    potential_bytes_saved = result["total_bytes_without_filter"] * (1 - improvement_ratio)
                    result["potential_bytes_saved"] = potential_bytes_saved
                    result["potential_cost_savings"] = (potential_bytes_saved / (1024**4)) * 5.0  # $5 per TB
            
            return result
            
        except Exception as e:
            logger.warning(f"Error analyzing partition usage for {table_key}: {e}")
            return {"error": str(e)}
    
    def _get_dataset_schema_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a dataset from INFORMATION_SCHEMA views.
        
        Args:
            dataset_id: BigQuery dataset ID
            
        Returns:
            Dict with dataset metadata from INFORMATION_SCHEMA
        """
        try:
            # Query INFORMATION_SCHEMA.SCHEMATA
            query = f"""
            SELECT
              catalog_name,
              schema_name,
              location AS schema_location,
              creation_time,
              last_modified_time
            FROM
              `{self.project_id}.{dataset_id}.INFORMATION_SCHEMA.SCHEMATA`
            WHERE
              schema_name = '{dataset_id}'
            """
            
            df = self.connector.query_to_dataframe(query)
            
            if df.empty:
                return {}
            
            row = df.iloc[0]
            
            return {
                "catalog_name": row["catalog_name"],
                "schema_name": row["schema_name"],
                "schema_location": row["schema_location"],
                "is_metadata_creation_time": row["creation_time"].isoformat() if "creation_time" in df.columns and row["creation_time"] else None,
                "is_metadata_last_modified_time": row["last_modified_time"].isoformat() if "last_modified_time" in df.columns and row["last_modified_time"] else None,
            }
            
        except Exception as e:
            logger.debug(f"Error getting dataset schema metadata: {e}")
            return {}
    
    def _get_table_schema_metadata(self, dataset_id: str, table_id: str) -> Dict[str, Any]:
        """Get metadata for a table from INFORMATION_SCHEMA views.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            Dict with table metadata from INFORMATION_SCHEMA
        """
        try:
            # Query INFORMATION_SCHEMA.TABLES
            query = f"""
            SELECT
              table_catalog,
              table_schema,
              table_name,
              table_type,
              is_insertable_into,
              is_typed,
              creation_time,
              ddl
            FROM
              `{self.project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
            WHERE
              table_name = '{table_id}'
            """
            
            df = self.connector.query_to_dataframe(query)
            
            if df.empty:
                return {}
            
            row = df.iloc[0]
            
            return {
                "table_catalog": row["table_catalog"],
                "table_schema": row["table_schema"],
                "is_metadata_table_type": row["table_type"],
                "is_insertable_into": row["is_insertable_into"] == "YES",
                "is_typed": row["is_typed"] == "YES",
                "is_metadata_creation_time": row["creation_time"].isoformat() if "creation_time" in df.columns and row["creation_time"] else None,
                "ddl": row["ddl"] if "ddl" in df.columns else None
            }
            
        except Exception as e:
            logger.debug(f"Error getting table schema metadata: {e}")
            return {}
    
    def _extract_access_entries(self, dataset) -> List[Dict[str, Any]]:
        """Extract access control entries from a dataset.
        
        Args:
            dataset: BigQuery dataset object
            
        Returns:
            List of access control entries
        """
        entries = []
        
        for entry in dataset.access_entries:
            access_info = {"access_type": None}
            
            if entry.role:
                access_info["access_type"] = "role"
                access_info["role"] = entry.role
            elif entry.entity_type:
                access_info["access_type"] = entry.entity_type
                
                if entry.entity_id:
                    access_info["entity_id"] = entry.entity_id
                
            entries.append(access_info)
        
        return entries
    
    def _extract_nested_fields(self, fields) -> List[Dict[str, Any]]:
        """Extract nested field information recursively.
        
        Args:
            fields: List of schema field objects
            
        Returns:
            List of nested field definitions
        """
        nested_fields = []
        
        for field in fields:
            field_info = {
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
                "description": field.description or ""
            }
            
            if field.fields:
                field_info["fields"] = self._extract_nested_fields(field.fields)
                
            nested_fields.append(field_info)
            
        return nested_fields
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize a query error message.
        
        Args:
            error_message: Error message string
            
        Returns:
            Error category
        """
        if "access denied" in error_message.lower() or "permission" in error_message.lower():
            return "Permission Error"
        elif "not found" in error_message.lower():
            return "Not Found Error"
        elif "syntax" in error_message.lower():
            return "Syntax Error"
        elif "timeout" in error_message.lower() or "deadline exceeded" in error_message.lower():
            return "Timeout Error"
        elif "quota" in error_message.lower():
            return "Quota Error"
        elif "duplicate" in error_message.lower():
            return "Duplicate Error"
        elif "resources exceeded" in error_message.lower():
            return "Resources Exceeded"
        else:
            return "Other Error"


def extract_dataset_metadata(project_id: str, dataset_id: str, client: Optional[bigquery.Client] = None) -> Dict[str, Any]:
    """Legacy function to extract metadata for an entire BigQuery dataset.
    
    This function maintains backward compatibility with the original implementation.
    For new code, it's recommended to use the MetadataExtractor class directly.
    
    Args:
        project_id: The GCP project ID containing the dataset
        dataset_id: The BigQuery dataset ID to analyze
        client: Optional BigQuery client instance
        
    Returns:
        Dict containing dataset metadata including size, table count, etc.
    """
    if client is None:
        client = bigquery.Client(project=project_id)
    
    extractor = MetadataExtractor(project_id=project_id)
    return extractor.extract_dataset_metadata(dataset_id)


def extract_table_metadata(client: bigquery.Client, table_ref: bigquery.TableReference) -> Dict[str, Any]:
    """Legacy function to extract metadata for a single BigQuery table.
    
    This function maintains backward compatibility with the original implementation.
    For new code, it's recommended to use the MetadataExtractor class directly.
    
    Args:
        client: BigQuery client instance
        table_ref: Reference to the table
        
    Returns:
        Dict containing table metadata
    """
    table = client.get_table(table_ref)
    
    # Get schema information
    schema_fields = [{
        "name": field.name,
        "type": field.field_type,
        "mode": field.mode,
        "description": field.description
    } for field in table.schema]
    
    # Extract partitioning information
    partitioning_info = None
    if table.time_partitioning:
        partitioning_info = {
            "type": table.time_partitioning.type_,
            "field": table.time_partitioning.field,
            "expiration_ms": table.time_partitioning.expiration_ms
        }
    
    # Extract clustering information
    clustering_info = None
    if table.clustering_fields:
        clustering_info = {
            "fields": table.clustering_fields
        }
    
    # Build metadata object
    metadata = {
        "table_id": table.table_id,
        "created": table.created.isoformat(),
        "last_modified": table.modified.isoformat(),
        "row_count": table.num_rows,
        "size_bytes": table.num_bytes,
        "size_gb": table.num_bytes / (1024**3),
        "schema": schema_fields,
        "partitioning": partitioning_info,
        "clustering": clustering_info,
        "description": table.description,
        "labels": table.labels,
        "field_count": len(schema_fields)
    }
    
    # Get table usage statistics from INFORMATION_SCHEMA
    try:
        usage_stats = get_table_usage_stats(client, table_ref.project, table_ref.dataset_id, table.table_id)
        metadata.update(usage_stats)
    except Exception as e:
        logger.warning(f"Could not retrieve usage stats for {table.full_table_id}: {e}")
    
    return metadata


def get_table_usage_stats(client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Dict[str, Any]:
    """Legacy function to get usage statistics for a table from INFORMATION_SCHEMA.
    
    This function maintains backward compatibility with the original implementation.
    For new code, it's recommended to use the MetadataExtractor class directly.
    
    Args:
        client: BigQuery client instance
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        
    Returns:
        Dict containing usage statistics
    """
    # Query to get table statistics for the last 30 days
    query = f"""
    SELECT
      COUNT(*) AS query_count,
      SUM(total_bytes_processed) AS total_bytes_processed,
      SUM(total_slot_ms) AS total_slot_ms,
      AVG(total_bytes_processed) AS avg_bytes_processed_per_query
    FROM
      `{project_id}.region-us`.INFORMATION_SCHEMA.JOBS
    WHERE
      creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY) AND CURRENT_TIMESTAMP()
      AND job_type = 'QUERY'
      AND state = 'DONE'
      AND REGEXP_CONTAINS(query, r'[^\\w]`?{project_id}\\.{dataset_id}\\.{table_id}`?[^\\w]')
    """
    
    try:
        df = client.query(query).to_dataframe()
        if not df.empty:
            row = df.iloc[0]
            return {
                "query_count_30d": int(row["query_count"]),
                "total_bytes_processed_30d": int(row["total_bytes_processed"]) if row["total_bytes_processed"] else 0,
                "total_slot_ms_30d": int(row["total_slot_ms"]) if row["total_slot_ms"] else 0,
                "avg_bytes_processed_per_query": int(row["avg_bytes_processed_per_query"]) if row["avg_bytes_processed_per_query"] else 0
            }
        return {
            "query_count_30d": 0,
            "total_bytes_processed_30d": 0,
            "total_slot_ms_30d": 0,
            "avg_bytes_processed_per_query": 0
        }
    except Exception as e:
        logger.warning(f"Error getting usage stats: {e}")
        return {
            "query_count_30d": None,
            "total_bytes_processed_30d": None,
            "total_slot_ms_30d": None,
            "avg_bytes_processed_per_query": None
        }