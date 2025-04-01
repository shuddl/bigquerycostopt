"""Schema optimization analyzer for BigQuery datasets."""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import copy
import hashlib

from ..utils.logging import setup_logger
from .metadata import MetadataExtractor

logger = setup_logger(__name__)

# Constants
DEFAULT_ANALYSIS_PERIOD_DAYS = 30
STORAGE_COST_PER_GB_PER_MONTH = 0.02  # $0.02 per GB per month for active storage

# Data type optimization mapping
# Maps from less efficient to more efficient data types, with conditions
DATA_TYPE_OPTIMIZATIONS = {
    # Format: "original_type": {"target_type": "...", "conditions": [...], "savings_pct": 0.0}
    "STRING": [
        {"target_type": "BYTES", "conditions": ["binary_data"], "savings_pct": 20.0},
        {"target_type": "BOOL", "conditions": ["boolean_values"], "savings_pct": 95.0},
        {"target_type": "INT64", "conditions": ["numeric_values"], "savings_pct": 50.0},
        {"target_type": "FLOAT64", "conditions": ["decimal_values"], "savings_pct": 30.0},
        {"target_type": "DATE", "conditions": ["date_values"], "savings_pct": 60.0},
        {"target_type": "TIMESTAMP", "conditions": ["timestamp_values"], "savings_pct": 40.0},
    ],
    "FLOAT64": [
        {"target_type": "INT64", "conditions": ["integer_values"], "savings_pct": 20.0},
        {"target_type": "NUMERIC", "conditions": ["financial_data", "precise_decimals"], "savings_pct": 30.0},
    ],
    "TIMESTAMP": [
        {"target_type": "DATE", "conditions": ["date_only_values"], "savings_pct": 50.0},
    ],
    "INTEGER": [
        {"target_type": "BOOL", "conditions": ["boolean_values"], "savings_pct": 90.0}
    ]
}


class SchemaOptimizer:
    """Analyzer for BigQuery schema optimization opportunities."""
    
    def __init__(self, metadata_extractor: Optional[MetadataExtractor] = None,
                project_id: Optional[str] = None,
                credentials_path: Optional[str] = None):
        """Initialize the schema optimizer.
        
        Args:
            metadata_extractor: Optional existing MetadataExtractor instance
            project_id: GCP project ID if metadata_extractor not provided
            credentials_path: Path to service account credentials if metadata_extractor not provided
        """
        if metadata_extractor:
            self.metadata_extractor = metadata_extractor
            self.project_id = metadata_extractor.project_id
        else:
            if not project_id:
                raise ValueError("Either metadata_extractor or project_id must be provided")
            self.metadata_extractor = MetadataExtractor(project_id=project_id, credentials_path=credentials_path)
            self.project_id = project_id
        
        self.client = self.metadata_extractor.client
        self.connector = self.metadata_extractor.connector
        logger.info(f"Initialized SchemaOptimizer for project {self.project_id}")
        
        # Cache for schema information
        self._schema_cache = {}
    
    def analyze_dataset_schemas(self, dataset_id: str, min_table_size_gb: float = 1.0) -> Dict[str, Any]:
        """Analyze schemas for all tables in a dataset and identify optimization opportunities.
        
        Args:
            dataset_id: BigQuery dataset ID
            min_table_size_gb: Minimum table size in GB to analyze (skip smaller tables)
            
        Returns:
            Dict containing schema optimization recommendations
        """
        logger.info(f"Analyzing schema optimizations for dataset {self.project_id}.{dataset_id}")
        
        # Fetch dataset metadata
        dataset_metadata = self.metadata_extractor.extract_dataset_metadata(dataset_id)
        
        # Skip analysis if no tables found
        if not dataset_metadata or "tables" not in dataset_metadata or not dataset_metadata["tables"]:
            logger.warning(f"No tables found in dataset {dataset_id}")
            return {
                "dataset_id": dataset_id,
                "project_id": self.project_id,
                "total_tables": 0,
                "tables_analyzed": 0,
                "total_size_gb": 0,
                "recommendations": [],
                "summary": {
                    "total_recommendations": 0,
                    "estimated_storage_savings_gb": 0,
                    "estimated_storage_savings_percentage": 0,
                    "estimated_monthly_cost_savings": 0,
                    "estimated_annual_cost_savings": 0
                }
            }
        
        # Process the dataset
        total_tables = len(dataset_metadata["tables"])
        total_size_gb = dataset_metadata.get("total_size_gb", 0)
        
        logger.info(f"Found {total_tables} tables in dataset {dataset_id} with total size {total_size_gb:.2f} GB")
        
        # Filter tables by minimum size
        tables_to_analyze = [table for table in dataset_metadata["tables"] 
                            if table.get("size_gb", 0) >= min_table_size_gb]
        
        tables_analyzed = len(tables_to_analyze)
        logger.info(f"Analyzing {tables_analyzed} tables with size >= {min_table_size_gb} GB")
        
        # Initialize results
        all_recommendations = []
        
        # Process each table in parallel (using the BatchTool from the tools array would be ideal here)
        for table in tables_to_analyze:
            table_id = table.get("table_id")
            
            # Skip tables with errors
            if "error" in table:
                logger.warning(f"Skipping table {table_id} due to error: {table['error']}")
                continue
                
            try:
                # Analyze the table schema
                table_recommendations = self.analyze_table_schema(dataset_id, table_id, table)
                all_recommendations.extend(table_recommendations)
            except Exception as e:
                logger.warning(f"Error analyzing schema for table {table_id}: {e}")
        
        # Calculate total savings
        total_storage_savings_gb = sum(rec.get("estimated_storage_savings_gb", 0) for rec in all_recommendations)
        storage_savings_pct = (total_storage_savings_gb / total_size_gb * 100) if total_size_gb > 0 else 0
        
        # Calculate cost savings
        monthly_cost_savings = total_storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH
        annual_cost_savings = monthly_cost_savings * 12
        
        # Group and prioritize recommendations
        prioritized_recs = self._prioritize_recommendations(all_recommendations)
        
        # Build result
        result = {
            "dataset_id": dataset_id,
            "project_id": self.project_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_tables": total_tables,
            "tables_analyzed": tables_analyzed,
            "total_size_gb": total_size_gb,
            "recommendations": prioritized_recs,
            "summary": {
                "total_recommendations": len(prioritized_recs),
                "total_patterns_detected": len(all_recommendations),
                "estimated_storage_savings_gb": total_storage_savings_gb,
                "estimated_storage_savings_percentage": storage_savings_pct,
                "estimated_monthly_cost_savings": monthly_cost_savings,
                "estimated_annual_cost_savings": annual_cost_savings
            }
        }
        
        return result
    
    def analyze_table_schema(self, dataset_id: str, table_id: str, 
                           table_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Analyze schema for a specific table and identify optimization opportunities.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            table_metadata: Optional pre-fetched table metadata
            
        Returns:
            List of schema optimization recommendations
        """
        logger.info(f"Analyzing schema for table {self.project_id}.{dataset_id}.{table_id}")
        
        # Fetch table metadata if not provided
        if not table_metadata:
            table_metadata = self.metadata_extractor.extract_table_metadata(dataset_id, table_id)
        
        # Collect all recommendations
        recommendations = []
        
        # Run different schema analysis algorithms
        data_type_recs = self._analyze_data_types(table_metadata)
        recommendations.extend(data_type_recs)
        
        column_usage_recs = self._analyze_column_usage(table_metadata)
        recommendations.extend(column_usage_recs)
        
        repeated_field_recs = self._analyze_repeated_fields(table_metadata)
        recommendations.extend(repeated_field_recs)
        
        denorm_recs = self._analyze_denormalization(table_metadata)
        recommendations.extend(denorm_recs)
        
        # Format each recommendation with table information
        for rec in recommendations:
            rec.update({
                "table_id": table_id,
                "dataset_id": dataset_id,
                "project_id": self.project_id,
                "full_table_id": f"{self.project_id}.{dataset_id}.{table_id}"
            })
        
        return recommendations
    
    def _analyze_data_types(self, table_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze and recommend data type optimizations for a table.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of data type optimization recommendations
        """
        recommendations = []
        
        # Extract schema and size information
        schema = table_metadata.get("schema", [])
        table_size_gb = table_metadata.get("size_gb", 0)
        table_bytes = table_metadata.get("size_bytes", 0)
        row_count = table_metadata.get("num_rows", 0)
        
        # Get column statistics if available
        column_stats = table_metadata.get("column_stats", [])
        column_stats_dict = {stat["name"]: stat for stat in column_stats if "name" in stat}
        
        # Check each column for potential type optimizations
        for column in schema:
            column_name = column.get("name", "")
            current_type = column.get("type", "")
            column_mode = column.get("mode", "NULLABLE")
            
            # Skip complex types for this analysis
            if current_type in ("STRUCT", "ARRAY", "RECORD"):
                continue
                
            # Analyze string columns more intensively
            if current_type == "STRING":
                string_optimizations = self._analyze_string_column(
                    column_name, column, table_metadata, column_stats_dict
                )
                recommendations.extend(string_optimizations)
                continue
                
            # Check if column is using a less efficient numeric type
            if current_type == "FLOAT64":
                # Check if column actually contains all integers
                if column_name in column_stats_dict:
                    col_stats = column_stats_dict[column_name]
                    
                    # If we have statistics indicating it's all integers
                    if "all_integers" in col_stats and col_stats["all_integers"]:
                        # Estimate the column's size contribution
                        column_size_estimate = self._estimate_column_size_contribution(
                            column_name, current_type, table_metadata
                        )
                        
                        # Calculate potential savings (INT64 is typically 40% smaller than FLOAT64)
                        savings_pct = 40.0  # Conservative estimate
                        storage_savings_gb = (column_size_estimate * savings_pct / 100)
                        
                        recommendations.append({
                            "type": "datatype_float_to_int",
                            "column_name": column_name,
                            "current_type": current_type,
                            "recommended_type": "INT64",
                            "description": f"Convert column '{column_name}' from FLOAT64 to INT64",
                            "rationale": f"The column contains only integer values but is stored as FLOAT64, which uses more storage than necessary.",
                            "estimated_storage_savings_pct": savings_pct,
                            "estimated_storage_savings_gb": storage_savings_gb,
                            "estimated_monthly_cost_savings": storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH,
                            "implementation_complexity": "low",
                            "backward_compatibility_risk": "low",
                            "implementation_sql": self._generate_type_change_sql(
                                table_metadata, column_name, "INT64"
                            ),
                            "priority_score": 70
                        })
                
            # Check if TIMESTAMP column actually contains only dates
            if current_type == "TIMESTAMP":
                # Determine from column name or statistics if it's likely a date-only column
                is_date_only = False
                
                # Check column name patterns
                column_name_lower = column_name.lower()
                if ("date" in column_name_lower and "time" not in column_name_lower) or \
                   column_name_lower.endswith("_dt") or \
                   column_name_lower.startswith("dt_"):
                    is_date_only = True
                
                # Check column statistics if available
                if column_name in column_stats_dict:
                    col_stats = column_stats_dict[column_name]
                    if "is_date_only" in col_stats and col_stats["is_date_only"]:
                        is_date_only = True
                
                if is_date_only:
                    # Estimate the column's size contribution
                    column_size_estimate = self._estimate_column_size_contribution(
                        column_name, current_type, table_metadata
                    )
                    
                    # Calculate potential savings (DATE is typically 50% smaller than TIMESTAMP)
                    savings_pct = 50.0
                    storage_savings_gb = (column_size_estimate * savings_pct / 100)
                    
                    recommendations.append({
                        "type": "datatype_timestamp_to_date",
                        "column_name": column_name,
                        "current_type": current_type,
                        "recommended_type": "DATE",
                        "description": f"Convert column '{column_name}' from TIMESTAMP to DATE",
                        "rationale": f"The column appears to contain only date values (no time component) but is stored as TIMESTAMP.",
                        "estimated_storage_savings_pct": savings_pct,
                        "estimated_storage_savings_gb": storage_savings_gb,
                        "estimated_monthly_cost_savings": storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH,
                        "implementation_complexity": "low",
                        "backward_compatibility_risk": "medium",
                        "implementation_sql": self._generate_type_change_sql(
                            table_metadata, column_name, "DATE"
                        ),
                        "priority_score": 65
                    })
        
        return recommendations
    
    def _analyze_string_column(self, column_name: str, column: Dict[str, Any], 
                             table_metadata: Dict[str, Any], 
                             column_stats_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze a STRING column for potential optimizations.
        
        Args:
            column_name: Name of the column
            column: Column schema information
            table_metadata: Table metadata
            column_stats_dict: Dictionary of column statistics
            
        Returns:
            List of string column optimization recommendations
        """
        recommendations = []
        column_name_lower = column_name.lower()
        current_type = column.get("type", "STRING")
        
        # Skip if not a STRING column
        if current_type != "STRING":
            return recommendations
            
        # Estimate the column's size contribution
        column_size_estimate = self._estimate_column_size_contribution(
            column_name, current_type, table_metadata
        )
        
        # Check if column has statistics
        if column_name in column_stats_dict:
            col_stats = column_stats_dict[column_name]
            
            # Check for low cardinality STRING columns (good candidates for ENUM or INT64 mapping)
            if "distinct_values_count" in col_stats:
                distinct_count = col_stats["distinct_values_count"]
                row_count = table_metadata.get("num_rows", 0)
                
                # If the column has low cardinality (few unique values compared to row count)
                if distinct_count > 0 and row_count > 0:
                    cardinality_ratio = distinct_count / row_count
                    
                    if cardinality_ratio < 0.01 and distinct_count < 100:
                        # This is a very good ENUM candidate (< 1% unique values, < 100 distinct values)
                        
                        # Calculate potential savings
                        # ENUMs can typically save 90% for very low cardinality columns
                        savings_pct = 90.0
                        storage_savings_gb = (column_size_estimate * savings_pct / 100)
                        
                        recommendations.append({
                            "type": "datatype_string_to_enum",
                            "column_name": column_name,
                            "current_type": current_type,
                            "recommended_type": "ENUM (or INT64 mapping)",
                            "description": f"Convert low-cardinality STRING column '{column_name}' to ENUM",
                            "rationale": f"The column contains only {distinct_count} unique values out of {row_count} rows ({cardinality_ratio:.2%} cardinality), making it ideal for an ENUM type.",
                            "distinct_values_count": distinct_count,
                            "cardinality_ratio": cardinality_ratio,
                            "estimated_storage_savings_pct": savings_pct,
                            "estimated_storage_savings_gb": storage_savings_gb,
                            "estimated_monthly_cost_savings": storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH,
                            "implementation_complexity": "medium",
                            "backward_compatibility_risk": "medium",
                            "implementation_sql": self._generate_enum_conversion_sql(
                                table_metadata, column_name
                            ),
                            "priority_score": 80
                        })
        
        # Check for binary data in STRING columns (good candidates for BYTES)
        if ("hash" in column_name_lower or "binary" in column_name_lower or 
            "image" in column_name_lower or "blob" in column_name_lower):
            
            # Calculate potential savings (BYTES is typically 20% smaller than STRING for binary data)
            savings_pct = 20.0
            storage_savings_gb = (column_size_estimate * savings_pct / 100)
            
            recommendations.append({
                "type": "datatype_string_to_bytes",
                "column_name": column_name,
                "current_type": current_type,
                "recommended_type": "BYTES",
                "description": f"Convert column '{column_name}' from STRING to BYTES",
                "rationale": f"The column name suggests it contains binary data, which is more efficiently stored as BYTES.",
                "estimated_storage_savings_pct": savings_pct,
                "estimated_storage_savings_gb": storage_savings_gb,
                "estimated_monthly_cost_savings": storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH,
                "implementation_complexity": "low",
                "backward_compatibility_risk": "medium",
                "implementation_sql": self._generate_type_change_sql(
                    table_metadata, column_name, "BYTES"
                ),
                "priority_score": 60
            })
        
        # Check for likely boolean values in STRING columns
        if (column_name_lower in ("active", "enabled", "is_active", "is_enabled", "valid", "flag") or
            column_name_lower.startswith("is_") or column_name_lower.startswith("has_")):
            
            # Calculate potential savings (BOOL is typically 90% smaller than STRING for boolean values)
            savings_pct = 90.0
            storage_savings_gb = (column_size_estimate * savings_pct / 100)
            
            recommendations.append({
                "type": "datatype_string_to_bool",
                "column_name": column_name,
                "current_type": current_type,
                "recommended_type": "BOOL",
                "description": f"Convert column '{column_name}' from STRING to BOOL",
                "rationale": f"The column name suggests it contains boolean values (true/false), which are more efficiently stored as BOOL.",
                "estimated_storage_savings_pct": savings_pct,
                "estimated_storage_savings_gb": storage_savings_gb,
                "estimated_monthly_cost_savings": storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH,
                "implementation_complexity": "low",
                "backward_compatibility_risk": "medium",
                "implementation_sql": self._generate_type_change_sql(
                    table_metadata, column_name, "BOOL",
                    is_boolean_string=True
                ),
                "priority_score": 75
            })
        
        # Check for date/time strings
        if ("date" in column_name_lower or "time" in column_name_lower or "created" in column_name_lower or 
            "updated" in column_name_lower or "timestamp" in column_name_lower):
            
            recommended_type = "TIMESTAMP"
            if "date" in column_name_lower and "time" not in column_name_lower:
                recommended_type = "DATE"
                
            # Calculate potential savings
            savings_pct = 60.0 if recommended_type == "DATE" else 40.0
            storage_savings_gb = (column_size_estimate * savings_pct / 100)
            
            recommendations.append({
                "type": f"datatype_string_to_{recommended_type.lower()}",
                "column_name": column_name,
                "current_type": current_type,
                "recommended_type": recommended_type,
                "description": f"Convert column '{column_name}' from STRING to {recommended_type}",
                "rationale": f"The column name suggests it contains {recommended_type.lower()} values stored as strings.",
                "estimated_storage_savings_pct": savings_pct,
                "estimated_storage_savings_gb": storage_savings_gb,
                "estimated_monthly_cost_savings": storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH,
                "implementation_complexity": "medium",
                "backward_compatibility_risk": "medium",
                "implementation_sql": self._generate_type_change_sql(
                    table_metadata, column_name, recommended_type
                ),
                "priority_score": 70
            })
        
        return recommendations
    
    def _analyze_column_usage(self, table_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze and recommend removing or archiving unused columns.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of unused column recommendations
        """
        recommendations = []
        
        # Extract schema and size information
        schema = table_metadata.get("schema", [])
        table_size_gb = table_metadata.get("size_gb", 0)
        table_bytes = table_metadata.get("size_bytes", 0)
        
        # Get column statistics and query patterns
        column_stats = table_metadata.get("column_stats", [])
        query_patterns = table_metadata.get("query_patterns", {})
        
        # Get columns referenced in queries
        referenced_columns = set()
        if "frequent_filters" in table_metadata:
            for filter_info in table_metadata.get("frequent_filters", []):
                if "column" in filter_info:
                    referenced_columns.add(filter_info["column"])
        
        # Identify candidate unused or rarely used columns
        unused_candidates = []
        nullable_no_description_candidates = []
        
        for column in schema:
            column_name = column.get("name", "")
            column_mode = column.get("mode", "NULLABLE")
            column_description = column.get("description", "")
            
            # Check for likely legacy or temporary columns by name
            is_likely_unused = False
            column_name_lower = column_name.lower()
            
            if any(pattern in column_name_lower for pattern in 
                  ["deprecated", "temp", "old", "legacy", "_v1", "_bak", "obsolete"]):
                is_likely_unused = True
                
            # Check for nullable columns with no description
            if column_mode == "NULLABLE" and not column_description:
                nullable_no_description_candidates.append(column_name)
                
            # Check if column is not referenced in queries
            not_referenced = column_name not in referenced_columns
                
            if is_likely_unused or not_referenced:
                unused_candidates.append((column_name, "likely_unused" if is_likely_unused else "not_referenced"))
        
        # If we have candidates, create recommendations
        if unused_candidates:
            # Group by reason
            likely_unused = [col for col, reason in unused_candidates if reason == "likely_unused"]
            not_referenced = [col for col, reason in unused_candidates if reason == "not_referenced"]
            
            # Pick a suitable set to recommend removing
            columns_to_remove = likely_unused or not_referenced[:5]  # Prefer likely unused, limit to 5 max
            
            if columns_to_remove:
                # Estimate storage savings
                # Assume each column contributes roughly equally to table size, adjusted by a conservative factor
                if len(schema) > 0:
                    # Conservatively estimate column space (assuming some columns use more space than others)
                    avg_col_size_pct = 0.7 / len(schema)
                    estimated_pct = len(columns_to_remove) * avg_col_size_pct * 100
                    # Cap at reasonable value and ensure it's positive
                    savings_pct = min(max(estimated_pct, 1.0), 50.0)
                    storage_savings_gb = table_size_gb * (savings_pct / 100)
                    
                    recommendations.append({
                        "type": "remove_unused_columns",
                        "columns": columns_to_remove,
                        "description": f"Remove potentially unused columns: {', '.join(columns_to_remove)}",
                        "rationale": "These columns appear to be unused or deprecated based on naming conventions and query patterns.",
                        "estimated_storage_savings_pct": savings_pct,
                        "estimated_storage_savings_gb": storage_savings_gb,
                        "estimated_monthly_cost_savings": storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH,
                        "implementation_complexity": "medium",
                        "backward_compatibility_risk": "high",
                        "implementation_sql": self._generate_column_removal_sql(
                            table_metadata, columns_to_remove
                        ),
                        "priority_score": 65
                    })
        
        return recommendations
    
    def _analyze_repeated_fields(self, table_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze and recommend optimizations for repeated fields.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of repeated field optimization recommendations
        """
        recommendations = []
        
        # Extract schema and size information
        schema = table_metadata.get("schema", [])
        table_size_gb = table_metadata.get("size_gb", 0)
        table_bytes = table_metadata.get("size_bytes", 0)
        
        # Find repeated and nested fields
        repeated_fields = []
        nested_fields = []
        
        for column in schema:
            column_name = column.get("name", "")
            column_type = column.get("type", "")
            column_mode = column.get("mode", "")
            
            if column_mode == "REPEATED":
                repeated_fields.append(column_name)
                
            if column_type in ("STRUCT", "RECORD"):
                nested_fields.append(column_name)
        
        # Check for potential denormalization issues with too many repeated fields
        if len(repeated_fields) >= 3 and table_size_gb > 10:
            # Heavy use of repeated fields can cause data explosion and performance issues
            # Recommend considering normalization for large tables with many repeated fields
            
            # Conservative estimate: denormalization can save 20-30% in extreme cases
            savings_pct = 25.0
            storage_savings_gb = table_size_gb * (savings_pct / 100)
            
            recommendations.append({
                "type": "denormalize_repeated_fields",
                "repeated_fields": repeated_fields,
                "description": f"Consider normalizing table with multiple repeated fields",
                "rationale": f"The table has {len(repeated_fields)} repeated fields which can cause data explosion and inefficient storage. Consider using multiple normalized tables instead.",
                "estimated_storage_savings_pct": savings_pct,
                "estimated_storage_savings_gb": storage_savings_gb,
                "estimated_monthly_cost_savings": storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH,
                "implementation_complexity": "high",
                "backward_compatibility_risk": "high",
                "implementation_sql": self._generate_normalization_example(
                    table_metadata, repeated_fields[0]
                ),
                "priority_score": 60
            })
            
        # Check for individual large repeated fields that could be normalized
        for field in repeated_fields:
            # Estimate the field's size contribution
            field_size_estimate = self._estimate_column_size_contribution(
                field, "REPEATED", table_metadata
            )
            
            # Only recommend for fields that are significant
            if field_size_estimate > 1.0:  # More than 1 GB
                # Conservative estimate: normalizing can save 20-30% for a repeated field
                savings_pct = 20.0
                storage_savings_gb = field_size_estimate * (savings_pct / 100)
                
                recommendations.append({
                    "type": "normalize_single_repeated_field",
                    "column_name": field,
                    "description": f"Consider normalizing repeated field '{field}'",
                    "rationale": f"The repeated field '{field}' may be contributing significantly to table size. Consider normalizing to a separate table.",
                    "estimated_storage_savings_pct": savings_pct,
                    "estimated_storage_savings_gb": storage_savings_gb,
                    "estimated_monthly_cost_savings": storage_savings_gb * STORAGE_COST_PER_GB_PER_MONTH,
                    "implementation_complexity": "high",
                    "backward_compatibility_risk": "high",
                    "implementation_sql": self._generate_normalization_example(
                        table_metadata, field
                    ),
                    "priority_score": 50
                })
        
        return recommendations
    
    def _analyze_denormalization(self, table_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze and recommend denormalization opportunities based on join patterns.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of denormalization recommendations
        """
        recommendations = []
        
        # Extract join patterns if available
        frequently_joined_tables = table_metadata.get("frequently_joined_tables", [])
        
        # No recommendations if no join information
        if not frequently_joined_tables:
            return recommendations
            
        # Find tables that are frequently joined with this one
        # Focus on the most frequently joined table
        if len(frequently_joined_tables) > 0:
            top_joined = frequently_joined_tables[0]
            
            # Only recommend for tables with high join count
            if "count" in top_joined and top_joined["count"] >= 10:
                joined_table = top_joined.get("table", "unknown")
                
                # Estimate potential savings
                # Denormalization can improve query performance significantly
                # but may increase storage slightly due to redundancy
                
                # Conservative estimate: 5% storage increase, but 30-50% query performance improvement
                storage_impact_gb = -0.05 * table_metadata.get("size_gb", 0)  # Negative savings (cost increase)
                query_improvement_pct = 40.0
                
                recommendations.append({
                    "type": "consider_denormalization",
                    "joined_table": joined_table,
                    "join_count": top_joined.get("count", 0),
                    "description": f"Consider denormalizing frequently joined table '{joined_table}'",
                    "rationale": f"The table is frequently joined with '{joined_table}' ({top_joined.get('count', 0)} times). Denormalizing might improve query performance.",
                    "storage_impact_description": "May increase storage usage but improve query performance",
                    "estimated_storage_impact_gb": storage_impact_gb,
                    "estimated_query_improvement_pct": query_improvement_pct,
                    "implementation_complexity": "high",
                    "backward_compatibility_risk": "medium",
                    "implementation_sql": self._generate_denormalization_example(
                        table_metadata, joined_table
                    ),
                    "priority_score": 40  # Lower priority as it increases storage
                })
        
        return recommendations
    
    def _estimate_column_size_contribution(self, column_name: str, column_type: str, 
                                         table_metadata: Dict[str, Any]) -> float:
        """Estimate the storage size contribution of a column.
        
        Args:
            column_name: Name of the column
            column_type: Data type of the column
            table_metadata: Table metadata
            
        Returns:
            Estimated size in GB
        """
        table_size_gb = table_metadata.get("size_gb", 0)
        schema = table_metadata.get("schema", [])
        
        # If we don't have enough information, use a simple heuristic
        if not schema or len(schema) == 0:
            return table_size_gb * 0.1  # Assume 10% of table size
            
        # Get column statistics if available
        column_stats = table_metadata.get("column_stats", [])
        column_stats_dict = {stat["name"]: stat for stat in column_stats if "name" in stat}
        
        # Default type weights based on typical sizes
        type_weight = {
            "BOOL": 0.2,
            "INT64": 1.0,
            "FLOAT64": 1.2,
            "NUMERIC": 1.5,
            "STRING": 3.0,
            "BYTES": 2.5,
            "DATE": 1.0,
            "TIMESTAMP": 1.5,
            "STRUCT": 2.0,
            "ARRAY": 2.0,
            "REPEATED": 3.0,
            "RECORD": 2.0
        }
        
        # Adjust for column mode
        mode_multiplier = 1.0
        col_info = next((col for col in schema if col.get("name") == column_name), None)
        if col_info:
            if col_info.get("mode") == "REPEATED":
                mode_multiplier = 2.0
            elif col_info.get("mode") == "REQUIRED":
                mode_multiplier = 0.9  # Slightly smaller due to no nullability overhead
        
        # Calculate total weight
        total_weight = sum(type_weight.get(col.get("type", "STRING"), 1.0) for col in schema)
        
        # Calculate this column's weight
        column_weight = type_weight.get(column_type, 1.0) * mode_multiplier
        
        # Estimate column size
        column_size_ratio = column_weight / total_weight
        column_size_gb = table_size_gb * column_size_ratio
        
        # Apply additional adjustments if we have column statistics
        if column_name in column_stats_dict:
            col_stats = column_stats_dict[column_name]
            
            # Adjust for null percentage
            if "null_percentage" in col_stats:
                null_pct = col_stats["null_percentage"]
                column_size_gb *= (1 - (null_pct / 100) * 0.8)  # Nulls still take some space
        
        return max(column_size_gb, 0.01)  # Ensure minimum size
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize and group recommendations.
        
        Args:
            recommendations: List of schema optimization recommendations
            
        Returns:
            Prioritized list of recommendations
        """
        if not recommendations:
            return []
            
        # Group by table
        table_groups = {}
        for rec in recommendations:
            table_id = f"{rec.get('project_id')}.{rec.get('dataset_id')}.{rec.get('table_id')}"
            
            if table_id not in table_groups:
                table_groups[table_id] = []
            table_groups[table_id].append(rec)
        
        # Prioritize recommendations within each table
        merged_recs = []
        for table_id, group in table_groups.items():
            # Sort by priority score and savings
            table_recs = sorted(
                group,
                key=lambda r: (r.get("priority_score", 0), 
                              r.get("estimated_storage_savings_gb", 0)),
                reverse=True
            )
            
            # Add to merged recommendations
            merged_recs.extend(table_recs)
        
        # Final sort by overall priority and savings
        sorted_recs = sorted(
            merged_recs,
            key=lambda r: (r.get("priority_score", 0), 
                          r.get("estimated_storage_savings_gb", 0)),
            reverse=True
        )
        
        return sorted_recs
    
    def _generate_type_change_sql(self, table_metadata: Dict[str, Any], column_name: str, 
                                new_type: str, is_boolean_string: bool = False) -> str:
        """Generate SQL for changing a column's data type.
        
        Args:
            table_metadata: Table metadata
            column_name: Name of the column to change
            new_type: New data type
            is_boolean_string: Whether this is converting a string to boolean
            
        Returns:
            SQL statement for type conversion
        """
        project_id = table_metadata.get("project_id", self.project_id)
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        # Get the column's mode
        column_mode = "NULLABLE"  # Default
        for column in table_metadata.get("schema", []):
            if column.get("name") == column_name:
                column_mode = column.get("mode", "NULLABLE")
                break
        
        # Generate the cast expression based on the new type
        cast_expr = f"CAST({column_name} AS {new_type})"
        
        if is_boolean_string:
            # For string to boolean conversion, use a CASE statement
            cast_expr = f"""CASE 
    WHEN LOWER({column_name}) IN ('true', 't', 'yes', 'y', '1') THEN TRUE
    WHEN LOWER({column_name}) IN ('false', 'f', 'no', 'n', '0') THEN FALSE
    ELSE NULL
END"""
        elif new_type == "DATE" and "TIMESTAMP" in table_metadata.get("schema", {}).get(column_name, {}).get("type", ""):
            # For timestamp to date conversion
            cast_expr = f"DATE({column_name})"
        elif new_type == "TIMESTAMP" and table_metadata.get("schema", {}).get(column_name, {}).get("type", "") == "STRING":
            # For string to timestamp conversion
            cast_expr = f"PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', {column_name})"
        elif new_type == "DATE" and table_metadata.get("schema", {}).get(column_name, {}).get("type", "") == "STRING":
            # For string to date conversion
            cast_expr = f"PARSE_DATE('%Y-%m-%d', {column_name})"
            
        # Generate CREATE OR REPLACE TABLE statement
        sql = f"""-- Step 1: Create backup table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 2: Create new table with modified column type
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_new` AS
SELECT
"""
        
        # Append the column list with the appropriate cast for the target column
        column_list = []
        for column in table_metadata.get("schema", []):
            column_expr = column.get("name")
            if column_expr == column_name:
                column_expr = f"{cast_expr} AS {column_name}"
            column_list.append(f"  {column_expr}")
        
        sql += ",\n".join(column_list)
        sql += f"\nFROM `{project_id}.{dataset_id}.{table_id}`;\n\n"
        
        # Add verification steps
        sql += f"""-- Step 3: Verify row counts match
SELECT
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_new`) AS new_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_new`) AS counts_match;

-- Step 4: Swap tables (run these commands one at a time after verification)
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_new` RENAME TO `{project_id}.{dataset_id}.{table_id}`;

-- Step 5: Drop old table after confirming everything works
-- DROP TABLE `{project_id}.{dataset_id}.{table_id}_old`;
"""
        
        return sql
    
    def _generate_enum_conversion_sql(self, table_metadata: Dict[str, Any], column_name: str) -> str:
        """Generate SQL for converting a string column to an ENUM type.
        
        Args:
            table_metadata: Table metadata
            column_name: Name of the column to convert
            
        Returns:
            SQL statement for enum conversion
        """
        project_id = table_metadata.get("project_id", self.project_id)
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        sql = f"""-- Step 1: Create backup table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 2: Analyze the distinct values for the column
SELECT
  {column_name},
  COUNT(*) as frequency
FROM `{project_id}.{dataset_id}.{table_id}`
GROUP BY {column_name}
ORDER BY frequency DESC;

-- Step 3: Create an ENUM type (example, adjust based on actual values)
CREATE TYPE `{project_id}.{dataset_id}.{column_name}_enum` AS ENUM (
  -- Add your values here based on the results above
  'value1', 'value2', 'value3'
);

-- Step 4: Create a new table with the ENUM type
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_new` AS
SELECT
"""
        
        # Append the column list with ENUM cast for the target column
        column_list = []
        for column in table_metadata.get("schema", []):
            column_expr = column.get("name")
            if column_expr == column_name:
                column_expr = f"CAST({column_name} AS `{project_id}.{dataset_id}.{column_name}_enum`) AS {column_name}"
            column_list.append(f"  {column_expr}")
        
        sql += ",\n".join(column_list)
        sql += f"\nFROM `{project_id}.{dataset_id}.{table_id}`;\n\n"
        
        # Add verification steps
        sql += f"""-- Step 5: Verify row counts match
SELECT
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_new`) AS new_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_new`) AS counts_match;

-- Step 6: Swap tables (run these commands one at a time after verification)
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_new` RENAME TO `{project_id}.{dataset_id}.{table_id}`;

-- Step 7: Drop old table after confirming everything works
-- DROP TABLE `{project_id}.{dataset_id}.{table_id}_old`;
"""
        
        # Add alternative approach using an INT mapping
        sql += f"""
-- ALTERNATIVE APPROACH: Integer mapping instead of ENUM
-- This can be more efficient especially for very large tables

-- Step 1: Create a mapping table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{column_name}_mapping` AS
SELECT
  ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS id,
  {column_name} AS value,
  COUNT(*) as frequency
FROM `{project_id}.{dataset_id}.{table_id}`
GROUP BY {column_name}
ORDER BY frequency DESC;

-- Step 2: Create a new table with the integer mapping
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_int_mapped` AS
SELECT
  t.*,
  m.id AS {column_name}_id
FROM
  `{project_id}.{dataset_id}.{table_id}` t
JOIN
  `{project_id}.{dataset_id}.{column_name}_mapping` m
ON t.{column_name} = m.value;

-- Step 3: Create a final table dropping the original string column
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_new` AS
SELECT
"""
        
        # Generate column list for the integer mapping approach
        int_mapping_columns = []
        for column in table_metadata.get("schema", []):
            if column.get("name") != column_name:
                int_mapping_columns.append(f"  {column.get('name')}")
        int_mapping_columns.append(f"  {column_name}_id AS {column_name}")
        
        sql += ",\n".join(int_mapping_columns)
        sql += f"\nFROM `{project_id}.{dataset_id}.{table_id}_int_mapped`;\n"
        
        return sql
    
    def _generate_column_removal_sql(self, table_metadata: Dict[str, Any], columns_to_remove: List[str]) -> str:
        """Generate SQL for removing unused columns.
        
        Args:
            table_metadata: Table metadata
            columns_to_remove: List of columns to remove
            
        Returns:
            SQL statement for column removal
        """
        project_id = table_metadata.get("project_id", self.project_id)
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        sql = f"""-- Step 1: Create backup table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 2: Create new table without the unused columns
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_new` AS
SELECT
"""
        
        # Generate column list excluding the columns to remove
        kept_columns = []
        for column in table_metadata.get("schema", []):
            if column.get("name") not in columns_to_remove:
                kept_columns.append(f"  {column.get('name')}")
        
        sql += ",\n".join(kept_columns)
        sql += f"\nFROM `{project_id}.{dataset_id}.{table_id}`;\n\n"
        
        # Add verification steps
        sql += f"""-- Step 3: Verify row counts match
SELECT
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_new`) AS new_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_new`) AS counts_match;

-- Step 4: Swap tables (run these commands one at a time after verification)
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_new` RENAME TO `{project_id}.{dataset_id}.{table_id}`;

-- Step 5: Drop old table after confirming everything works
-- DROP TABLE `{project_id}.{dataset_id}.{table_id}_old`;
"""
        
        # Add a warning about potential impact
        sql += f"""
-- WARNING: Removing columns may break existing queries and applications!
-- Before implementing this change, make sure to:
-- 1. Identify all queries that reference these columns
-- 2. Test the change in a development environment
-- 3. Notify users of the schema change
-- 4. Consider creating views to maintain backward compatibility
"""
        
        return sql
    
    def _generate_normalization_example(self, table_metadata: Dict[str, Any], repeated_field: str) -> str:
        """Generate example SQL for normalizing a repeated field.
        
        Args:
            table_metadata: Table metadata
            repeated_field: Repeated field to normalize
            
        Returns:
            SQL example for normalization
        """
        project_id = table_metadata.get("project_id", self.project_id)
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        # Choose a field to use as the ID
        id_field = "id"  # Default
        for column in table_metadata.get("schema", []):
            if "id" in column.get("name", "").lower() and column.get("type") in ("INTEGER", "INT64"):
                id_field = column.get("name")
                break
        
        sql = f"""-- Example normalization approach for repeated field '{repeated_field}'

-- Step 1: Create a backup of the original table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 2: Create a normalized child table for the repeated field
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_{repeated_field}` AS
SELECT
  parent.{id_field} AS parent_id,
  child AS {repeated_field}_value,
  ROW_NUMBER() OVER(PARTITION BY parent.{id_field}) AS {repeated_field}_order
FROM
  `{project_id}.{dataset_id}.{table_id}` parent,
  UNNEST(parent.{repeated_field}) AS child;

-- Step 3: Create a new parent table without the repeated field
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_new` AS
SELECT
"""
        
        # Generate column list excluding the repeated field
        parent_columns = []
        for column in table_metadata.get("schema", []):
            if column.get("name") != repeated_field:
                parent_columns.append(f"  {column.get('name')}")
        
        sql += ",\n".join(parent_columns)
        sql += f"\nFROM `{project_id}.{dataset_id}.{table_id}`;\n\n"
        
        # Add example query to join the tables
        sql += f"""-- Example query to reconstruct the original data:
SELECT
  parent.*,
  ARRAY_AGG(child.{repeated_field}_value ORDER BY child.{repeated_field}_order) AS {repeated_field}
FROM
  `{project_id}.{dataset_id}.{table_id}_new` parent
LEFT JOIN
  `{project_id}.{dataset_id}.{table_id}_{repeated_field}` child
ON parent.{id_field} = child.parent_id
GROUP BY
  {', '.join(parent_columns)}
"""
        
        return sql
    
    def _generate_denormalization_example(self, table_metadata: Dict[str, Any], joined_table: str) -> str:
        """Generate example SQL for denormalizing with a frequently joined table.
        
        Args:
            table_metadata: Table metadata
            joined_table: Name of the table that is frequently joined
            
        Returns:
            SQL example for denormalization
        """
        project_id = table_metadata.get("project_id", self.project_id)
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        # Extract table name if it's a fully qualified name
        joined_table_short = joined_table.split(".")[-1] if "." in joined_table else joined_table
        
        # Choose a field to use as the ID
        id_field = "id"  # Default
        join_field = f"{joined_table_short}_id"  # Guess at the foreign key
        
        sql = f"""-- Example denormalization approach for frequently joined table '{joined_table}'

-- Step 1: Create a backup of the original table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 2: Create a denormalized table with joined data
-- Note: This is an example that needs to be adjusted based on actual schema
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_denormalized` AS
SELECT
  t.*,
  -- Include relevant columns from the joined table
  j.column1 AS {joined_table_short}_column1,
  j.column2 AS {joined_table_short}_column2
  -- Add more columns as needed
FROM
  `{project_id}.{dataset_id}.{table_id}` t
LEFT JOIN
  `{project_id}.{dataset_id}.{joined_table_short}` j
ON t.{join_field} = j.{id_field};

-- Verify the new table
SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_denormalized`;
"""
        
        # Add considerations
        sql += """
-- Important considerations for denormalization:
-- 1. Storage usage will increase due to data duplication
-- 2. Updates to the joined table won't automatically propagate to the denormalized table
-- 3. Data consistency must be maintained through ETL processes
-- 4. Query performance will typically improve due to elimination of joins
-- 5. Consider using a view instead if data changes frequently
"""
        
        return sql
    
    def generate_recommendations_report(self, recommendations: Dict[str, Any], format: str = 'md') -> str:
        """Generate a formatted report of schema optimization recommendations.
        
        Args:
            recommendations: Recommendations from analyze_dataset_schemas
            format: Output format ('md' for Markdown, 'html', or 'text')
            
        Returns:
            Formatted report as a string
        """
        dataset_id = recommendations.get("dataset_id", "unknown")
        total_recs = recommendations.get("summary", {}).get("total_recommendations", 0)
        monthly_savings = recommendations.get("summary", {}).get("estimated_monthly_cost_savings", 0)
        annual_savings = recommendations.get("summary", {}).get("estimated_annual_cost_savings", 0)
        storage_savings_gb = recommendations.get("summary", {}).get("estimated_storage_savings_gb", 0)
        savings_pct = recommendations.get("summary", {}).get("estimated_storage_savings_percentage", 0)
        
        if format == 'md':
            # Markdown format
            report = [
                f"# Schema Optimization Recommendations: {dataset_id}",
                "",
                "## Summary",
                "",
                f"- **Total Recommendations:** {total_recs}",
                f"- **Estimated Monthly Savings:** ${monthly_savings:.2f}",
                f"- **Estimated Annual Savings:** ${annual_savings:.2f}",
                f"- **Storage Reduction:** {storage_savings_gb:.2f} GB ({savings_pct:.1f}%)",
                f"- **Total Tables Analyzed:** {recommendations.get('tables_analyzed', 0)} of {recommendations.get('total_tables', 0)}",
                "",
                "## Top Recommendations",
                ""
            ]
            
            # Add each recommendation
            for i, rec in enumerate(recommendations.get("recommendations", [])[:10], 1):
                rec_type = rec.get("type", "unknown").replace("_", " ").title()
                description = rec.get("description", "")
                savings = rec.get("estimated_monthly_cost_savings", 0)
                table_id = rec.get("table_id", "")
                
                report.append(f"### {i}. {rec_type}: {description}")
                report.append("")
                report.append(f"**Table:** `{table_id}`")
                report.append(f"**Estimated Monthly Savings:** ${savings:.2f}")
                report.append("")
                report.append(f"**Rationale:** {rec.get('rationale', '')}")
                report.append("")
                report.append("**Implementation Complexity:** " + rec.get('implementation_complexity', 'unknown').title())
                report.append("**Backward Compatibility Risk:** " + rec.get('backward_compatibility_risk', 'unknown').title())
                report.append("")
                
                if "implementation_sql" in rec:
                    report.append("```sql")
                    report.append("-- Implementation SQL")
                    report.append(rec.get("implementation_sql", "").split("\n\n")[0])  # First block only for brevity
                    report.append("```")
                    report.append("")
                
                if "columns" in rec:
                    report.append(f"**Affected Columns:** {', '.join(rec['columns'])}")
                    report.append("")
                    
                if "column_name" in rec:
                    current_type = rec.get("current_type", "")
                    recommended_type = rec.get("recommended_type", "")
                    if current_type and recommended_type:
                        report.append(f"**Type Change:** `{current_type}` → `{recommended_type}`")
                        report.append("")
                
            return "\n".join(report)
            
        elif format == 'html':
            # HTML format (simplified example)
            report = [
                f"<h1>Schema Optimization Recommendations: {dataset_id}</h1>",
                "<h2>Summary</h2>",
                "<ul>",
                f"<li><strong>Total Recommendations:</strong> {total_recs}</li>",
                f"<li><strong>Estimated Monthly Savings:</strong> ${monthly_savings:.2f}</li>",
                f"<li><strong>Estimated Annual Savings:</strong> ${annual_savings:.2f}</li>",
                f"<li><strong>Storage Reduction:</strong> {storage_savings_gb:.2f} GB ({savings_pct:.1f}%)</li>",
                "</ul>",
                "<h2>Top Recommendations</h2>"
            ]
            
            # Add each recommendation
            for i, rec in enumerate(recommendations.get("recommendations", [])[:10], 1):
                rec_type = rec.get("type", "unknown").replace("_", " ").title()
                description = rec.get("description", "")
                savings = rec.get("estimated_monthly_cost_savings", 0)
                table_id = rec.get("table_id", "")
                
                report.append(f"<h3>{i}. {rec_type}: {description}</h3>")
                report.append(f"<p><strong>Table:</strong> <code>{table_id}</code></p>")
                report.append(f"<p><strong>Estimated Monthly Savings:</strong> ${savings:.2f}</p>")
                report.append(f"<p><strong>Rationale:</strong> {rec.get('rationale', '')}</p>")
                
                if "implementation_sql" in rec:
                    report.append("<pre><code>")
                    report.append("-- Implementation SQL")
                    report.append(rec.get("implementation_sql", "").split("\n\n")[0])  # First block only for brevity
                    report.append("</code></pre>")
                
            return "\n".join(report)
            
        else:
            # Plain text format
            report = [
                f"SCHEMA OPTIMIZATION RECOMMENDATIONS: {dataset_id}",
                "=" * 50,
                "",
                "SUMMARY:",
                f"- Total Recommendations: {total_recs}",
                f"- Estimated Monthly Savings: ${monthly_savings:.2f}",
                f"- Estimated Annual Savings: ${annual_savings:.2f}",
                f"- Storage Reduction: {storage_savings_gb:.2f} GB ({savings_pct:.1f}%)",
                "",
                "TOP RECOMMENDATIONS:",
                ""
            ]
            
            # Add each recommendation
            for i, rec in enumerate(recommendations.get("recommendations", [])[:10], 1):
                rec_type = rec.get("type", "unknown").replace("_", " ").title()
                description = rec.get("description", "")
                savings = rec.get("estimated_monthly_cost_savings", 0)
                table_id = rec.get("table_id", "")
                
                report.append(f"{i}. {rec_type}: {description}")
                report.append(f"   Table: {table_id}")
                report.append(f"   Estimated Monthly Savings: ${savings:.2f}")
                report.append(f"   Rationale: {rec.get('rationale', '')}")
                report.append(f"   Implementation Complexity: {rec.get('implementation_complexity', 'unknown').title()}")
                report.append(f"   Backward Compatibility Risk: {rec.get('backward_compatibility_risk', 'unknown').title()}")
                report.append("")
                
            return "\n".join(report)