"""Storage optimization analyzer for BigQuery datasets."""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import re

from ..utils.logging import setup_logger
from .metadata import MetadataExtractor

logger = setup_logger(__name__)

# Constants
DEFAULT_ANALYSIS_PERIOD_DAYS = 30
STORAGE_COST_PER_GB_PER_MONTH = 0.02  # $0.02 per GB per month for active storage
LTS_STORAGE_COST_PER_GB_PER_MONTH = 0.01  # $0.01 per GB per month for long-term storage


class StorageOptimizer:
    """Analyzes BigQuery tables for storage optimization opportunities."""
    
    def __init__(self, metadata_extractor: Optional[MetadataExtractor] = None,
                project_id: Optional[str] = None,
                credentials_path: Optional[str] = None):
        """Initialize the storage optimizer.
        
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
        logger.info(f"Initialized StorageOptimizer for project {self.project_id}")
    
    def analyze_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Analyze a dataset for storage optimization opportunities.
        
        Args:
            dataset_id: BigQuery dataset ID
            
        Returns:
            Dict containing optimization recommendations and potential savings
        """
        logger.info(f"Analyzing storage optimizations for dataset {self.project_id}.{dataset_id}")
        
        # Extract dataset metadata
        dataset_metadata = self.metadata_extractor.extract_dataset_metadata(dataset_id)
        
        # Initialize results
        results = {
            "dataset_id": dataset_id,
            "project_id": self.project_id,
            "total_size_gb": dataset_metadata.get("total_size_gb", 0),
            "table_count": dataset_metadata.get("table_count", 0),
            "current_monthly_storage_cost": dataset_metadata.get("total_size_gb", 0) * STORAGE_COST_PER_GB_PER_MONTH,
            "analysis_timestamp": datetime.now().isoformat(),
            "optimization_summary": {
                "total_recommendations": 0,
                "estimated_monthly_savings": 0,
                "estimated_annual_savings": 0,
                "estimated_size_reduction_gb": 0,
                "estimated_size_reduction_percentage": 0
            },
            "recommendations": []
        }
        
        # Process each table
        for table in dataset_metadata.get("tables", []):
            # Skip tables with errors
            if "error" in table:
                continue
                
            # Analyze partitioning opportunities
            partitioning_recs = self._analyze_partitioning(table)
            results["recommendations"].extend(partitioning_recs)
            
            # Analyze clustering opportunities
            clustering_recs = self._analyze_clustering(table)
            results["recommendations"].extend(clustering_recs)
            
            # Analyze long-term storage opportunities
            lts_recs = self._analyze_long_term_storage(table)
            results["recommendations"].extend(lts_recs)
        
        # Calculate summary statistics
        total_recommendations = len(results["recommendations"])
        total_monthly_savings = sum(rec.get("estimated_monthly_savings", 0) for rec in results["recommendations"])
        total_size_reduction_gb = sum(rec.get("estimated_size_reduction_gb", 0) for rec in results["recommendations"])
        
        # Update summary
        results["optimization_summary"]["total_recommendations"] = total_recommendations
        results["optimization_summary"]["estimated_monthly_savings"] = total_monthly_savings
        results["optimization_summary"]["estimated_annual_savings"] = total_monthly_savings * 12
        results["optimization_summary"]["estimated_size_reduction_gb"] = total_size_reduction_gb
        
        if dataset_metadata.get("total_size_gb", 0) > 0:
            reduction_pct = 100.0 * total_size_reduction_gb / dataset_metadata["total_size_gb"]
            results["optimization_summary"]["estimated_size_reduction_percentage"] = round(reduction_pct, 2)
        
        # Sort recommendations by estimated savings (highest first)
        results["recommendations"].sort(key=lambda x: x.get("estimated_monthly_savings", 0), reverse=True)
        
        return results
    
    def _analyze_partitioning(self, table_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze partitioning optimization opportunities for a table.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of partitioning recommendations
        """
        recommendations = []
        
        # Skip small tables (< 1 GB)
        if table_metadata.get("size_gb", 0) < 1.0:
            return recommendations
            
        full_table_id = table_metadata.get("full_name", "")
        table_size_gb = table_metadata.get("size_gb", 0)
        table_id = table_metadata.get("table_id", "")
            
        # Check if table is already partitioned
        if table_metadata.get("partitioning"):
            # For already partitioned tables, check if current partitioning is optimal
            current_partition = table_metadata["partitioning"]
            
            # For daily partitioned tables > 100GB, consider monthly partitioning
            if current_partition.get("type") == "DAY" and table_size_gb > 100:
                # Estimate savings (primarily from reduced management overhead and query efficiency)
                estimated_savings_pct = min(5, max(1, 3 * (table_size_gb / 200)))  # 1-5% based on size
                estimated_size_reduction_gb = table_size_gb * (estimated_savings_pct / 100)
                estimated_monthly_savings = estimated_size_reduction_gb * STORAGE_COST_PER_GB_PER_MONTH
                
                # Recommend changing to monthly partitioning
                recommendations.append({
                    "type": "partition_daily_to_monthly",
                    "table_id": table_id,
                    "full_table_id": full_table_id,
                    "current_state": f"Daily partitioning on '{current_partition.get('field')}'",
                    "recommendation": f"Change to MONTH partitioning on '{current_partition.get('field')}'",
                    "rationale": "Table is very large (>100GB); monthly partitioning would reduce partition count and management overhead while still providing good query performance for most use cases.",
                    "estimated_size_reduction_pct": estimated_savings_pct,
                    "estimated_size_reduction_gb": estimated_size_reduction_gb,
                    "estimated_monthly_savings": estimated_monthly_savings,
                    "priority": "medium",  # Medium priority as table is already partitioned
                    "implementation_complexity": "medium",  # Requires table recreation
                    "implementation_sql": self._generate_partition_change_sql(
                        table_metadata, current_partition.get("field"), "MONTH"
                    )
                })
                
            # For ingestion-time partitioned tables, recommend explicit field partitioning if suitable field exists
            if current_partition.get("type") and not current_partition.get("field"):
                date_timestamp_fields = self._find_date_timestamp_fields(table_metadata)
                if date_timestamp_fields:
                    best_field = date_timestamp_fields[0][0]
                    
                    # Conservative estimate: 5-15% for appropriate field-based partitioning
                    estimated_savings_pct = min(15, max(5, int(table_size_gb / 20)))
                    estimated_size_reduction_gb = table_size_gb * (estimated_savings_pct / 100)
                    estimated_monthly_savings = estimated_size_reduction_gb * STORAGE_COST_PER_GB_PER_MONTH
                    
                    recommendations.append({
                        "type": "partition_ingestion_to_field",
                        "table_id": table_id,
                        "full_table_id": full_table_id,
                        "current_state": "Ingestion-time partitioning without explicit field",
                        "recommendation": f"Change to field-based partitioning on '{best_field}'",
                        "rationale": f"Using explicit field '{best_field}' for partitioning allows for more efficient query filtering and partition pruning.",
                        "estimated_size_reduction_pct": estimated_savings_pct,
                        "estimated_size_reduction_gb": estimated_size_reduction_gb,
                        "estimated_monthly_savings": estimated_monthly_savings,
                        "priority": "high",
                        "implementation_complexity": "medium",  # Requires table recreation
                        "implementation_sql": self._generate_partition_change_sql(
                            table_metadata, best_field, current_partition.get("type", "DAY")
                        )
                    })
                    
            return recommendations
        
        # For non-partitioned tables, look for good partition candidates
        date_timestamp_fields = self._find_date_timestamp_fields(table_metadata)
        integer_fields = self._find_integer_partition_candidates(table_metadata)
        
        partition_candidates = date_timestamp_fields or integer_fields
        
        if partition_candidates:
            best_field, priority = partition_candidates[0]
            
            # Determine appropriate partition type based on table size and field type
            if table_size_gb > 100:
                partition_type = "MONTH"
            else:
                partition_type = "DAY"
                
            # For integer fields, use range partitioning
            if integer_fields and best_field == integer_fields[0][0]:
                partition_type = "RANGE"
                
            # Estimate savings (10-30% is typical for well-partitioned tables)
            estimated_savings_pct = min(30, max(10, int(table_size_gb / 10)))
            estimated_size_reduction_gb = table_size_gb * (estimated_savings_pct / 100)
            estimated_monthly_savings = estimated_size_reduction_gb * STORAGE_COST_PER_GB_PER_MONTH
            
            # Determine priority based on table size and potential savings
            if table_size_gb > 50:
                recommendation_priority = "high"
            elif table_size_gb > 10:
                recommendation_priority = "medium"
            else:
                recommendation_priority = "low"
                
            if partition_type == "RANGE":
                range_info = self._determine_range_partitioning(table_metadata, best_field)
                recommendation = {
                    "type": "add_range_partitioning",
                    "table_id": table_id,
                    "full_table_id": full_table_id,
                    "current_state": "No partitioning",
                    "recommendation": f"Add RANGE partitioning on field '{best_field}', range: {range_info['start']} to {range_info['end']} by {range_info['interval']}",
                    "rationale": f"Table is {table_size_gb:.1f} GB and field '{best_field}' is a good candidate for range partitioning.",
                    "estimated_size_reduction_pct": estimated_savings_pct,
                    "estimated_size_reduction_gb": estimated_size_reduction_gb,
                    "estimated_monthly_savings": estimated_monthly_savings,
                    "priority": recommendation_priority,
                    "implementation_complexity": "high",  # Requires table recreation
                    "implementation_sql": self._generate_range_partition_sql(table_metadata, best_field, range_info),
                    "range_start": range_info["start"],
                    "range_end": range_info["end"],
                    "range_interval": range_info["interval"]
                }
            else:
                recommendation = {
                    "type": "add_time_partitioning",
                    "table_id": table_id,
                    "full_table_id": full_table_id,
                    "current_state": "No partitioning",
                    "recommendation": f"Add {partition_type} partitioning on field '{best_field}'",
                    "rationale": f"Table is {table_size_gb:.1f} GB and field '{best_field}' is a good candidate for time partitioning.",
                    "estimated_size_reduction_pct": estimated_savings_pct,
                    "estimated_size_reduction_gb": estimated_size_reduction_gb,
                    "estimated_monthly_savings": estimated_monthly_savings,
                    "priority": recommendation_priority,
                    "implementation_complexity": "high",  # Requires table recreation
                    "implementation_sql": self._generate_partition_sql(table_metadata, best_field, partition_type)
                }
            
            recommendations.append(recommendation)
            
            # If high priority, also suggest partition filter requirement for the table
            if recommendation_priority == "high":
                recommendations.append({
                    "type": "require_partition_filter",
                    "table_id": table_id,
                    "full_table_id": full_table_id,
                    "current_state": "No partitioning",
                    "recommendation": f"After implementing partitioning, add require_partition_filter=True",
                    "rationale": "Requiring a partition filter ensures queries utilize partitioning and prevents accidental full table scans.",
                    "estimated_size_reduction_pct": 0,  # No direct size reduction
                    "estimated_size_reduction_gb": 0,
                    "estimated_monthly_savings": 0,  # This is more about query cost savings
                    "priority": "medium",
                    "implementation_complexity": "low",  # Simple option addition
                    "depends_on": recommendations[-1]["type"]
                })
        
        return recommendations
    
    def _analyze_clustering(self, table_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze clustering optimization opportunities for a table.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of clustering recommendations
        """
        recommendations = []
        
        # Skip small tables (< 1 GB)
        if table_metadata.get("size_gb", 0) < 1.0:
            return recommendations
            
        full_table_id = table_metadata.get("full_name", "")
        table_size_gb = table_metadata.get("size_gb", 0)
        table_id = table_metadata.get("table_id", "")
        
        # Skip if already clustered
        if table_metadata.get("clustering"):
            return recommendations
            
        # Identify if table is partitioned
        is_partitioned = bool(table_metadata.get("partitioning"))
        
        # Find potential clustering candidates
        clustering_candidates = self._find_clustering_candidates(table_metadata)
        
        if clustering_candidates and len(clustering_candidates) >= 1:
            # Take up to 4 fields (BigQuery max)
            top_fields = [f[0] for f in clustering_candidates[:4]]
            
            # Estimate savings (20-30% for well-clustered tables)
            estimated_savings_pct = min(30, max(10, int(table_size_gb / 5)))
            
            # Reduce expected savings by 30% if the table isn't partitioned
            if not is_partitioned:
                estimated_savings_pct = estimated_savings_pct * 0.7
                
            estimated_size_reduction_gb = table_size_gb * (estimated_savings_pct / 100)
            estimated_monthly_savings = estimated_size_reduction_gb * STORAGE_COST_PER_GB_PER_MONTH
            
            # Determine recommendation details
            recommendation_text = f"Add clustering on fields: {', '.join(top_fields)}"
            rationale = f"Table is {table_size_gb:.1f} GB and contains fields that would benefit from clustering."
            
            # If table isn't partitioned, suggest partitioning first
            priority = "medium"
            implementation_complexity = "medium"
            
            if not is_partitioned:
                recommendation_text += " (consider partitioning first)"
                rationale += " For best results, implement partitioning before clustering."
                implementation_complexity = "high"  # Higher complexity if need to add partitioning first
            elif table_size_gb > 20:
                priority = "high"
                
            recommendations.append({
                "type": "add_clustering",
                "table_id": table_id,
                "full_table_id": full_table_id,
                "current_state": "No clustering",
                "recommendation": recommendation_text,
                "rationale": rationale,
                "clustering_fields": top_fields,
                "estimated_size_reduction_pct": estimated_savings_pct,
                "estimated_size_reduction_gb": estimated_size_reduction_gb,
                "estimated_monthly_savings": estimated_monthly_savings,
                "priority": priority,
                "implementation_complexity": implementation_complexity,
                "implementation_sql": self._generate_clustering_sql(table_metadata, top_fields)
            })
        
        return recommendations
    
    def _analyze_long_term_storage(self, table_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze long-term storage opportunities for a table.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of LTS recommendations
        """
        recommendations = []
        
        # Skip small tables (< 5 GB)
        if table_metadata.get("size_gb", 0) < 5.0:
            return recommendations
            
        full_table_id = table_metadata.get("full_name", "")
        table_size_gb = table_metadata.get("size_gb", 0)
        table_id = table_metadata.get("table_id", "")
        
        # Check if table has been modified recently
        if "last_modified" in table_metadata:
            try:
                last_modified = datetime.fromisoformat(table_metadata["last_modified"])
                days_since_modified = (datetime.now() - last_modified).days
                
                # Check for query activity in the last 90 days
                query_count = table_metadata.get("query_count_30d", 0)
                active_usage = query_count > 0
                
                # If table hasn't been modified in > 30 days and has low query activity
                if days_since_modified > 30 and not active_usage:
                    # Calculate savings (difference between regular and LTS pricing)
                    savings_per_gb = STORAGE_COST_PER_GB_PER_MONTH - LTS_STORAGE_COST_PER_GB_PER_MONTH
                    estimated_monthly_savings = table_size_gb * savings_per_gb
                    
                    recommendations.append({
                        "type": "long_term_storage",
                        "table_id": table_id,
                        "full_table_id": full_table_id,
                        "current_state": "Regular storage pricing",
                        "recommendation": "Move table to long-term storage",
                        "rationale": f"Table has not been modified in {days_since_modified} days and has low query activity.",
                        "estimated_size_reduction_pct": 0,  # No size reduction, just cost reduction
                        "estimated_size_reduction_gb": 0,
                        "estimated_monthly_savings": estimated_monthly_savings,
                        "days_since_modified": days_since_modified,
                        "priority": "medium",
                        "implementation_complexity": "low",
                        "implementation_sql": self._generate_lts_sql(table_metadata)
                    })
            except Exception as e:
                logger.debug(f"Error analyzing long-term storage for {full_table_id}: {e}")
        
        return recommendations
    
    def _find_date_timestamp_fields(self, table_metadata: Dict[str, Any]) -> List[Tuple[str, int]]:
        """Find suitable date/timestamp fields for partitioning.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of tuples with (field_name, priority_score)
        """
        candidates = []
        
        for field in table_metadata.get("schema", []):
            if field["type"] in ("TIMESTAMP", "DATE", "DATETIME") and field["mode"] != "REPEATED":
                # Prioritize fields with certain names that are likely to be good partition keys
                priority = 0
                name_lower = field["name"].lower()
                
                # Check for common temporal field names
                if "date" in name_lower or "time" in name_lower:
                    priority += 5
                if "created" in name_lower or "updated" in name_lower or "timestamp" in name_lower:
                    priority += 3
                if name_lower.startswith("dt_") or name_lower.startswith("date_"):
                    priority += 3
                if "partition" in name_lower:
                    priority += 4
                    
                # Check for usage in query filters if available
                if "frequent_filters" in table_metadata:
                    for filter_info in table_metadata.get("frequent_filters", []):
                        if filter_info["column"] == field["name"]:
                            priority += filter_info["count"]
                
                if priority > 0:
                    candidates.append((field["name"], priority))
        
        # Sort candidates by priority
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
    
    def _find_integer_partition_candidates(self, table_metadata: Dict[str, Any]) -> List[Tuple[str, int]]:
        """Find suitable integer fields for range partitioning.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of tuples with (field_name, priority_score)
        """
        candidates = []
        
        for field in table_metadata.get("schema", []):
            if field["type"] == "INTEGER" and field["mode"] != "REPEATED":
                name_lower = field["name"].lower()
                priority = 0
                
                # Check for specific patterns that suggest good partition keys
                if "year" in name_lower or "month" in name_lower or "day" in name_lower:
                    priority += 5
                if "id" in name_lower and not ("uuid" in name_lower or "guid" in name_lower):
                    priority += 2
                if "shard" in name_lower or "partition" in name_lower:
                    priority += 5
                
                # Check for usage in query filters if available
                if "frequent_filters" in table_metadata:
                    for filter_info in table_metadata.get("frequent_filters", []):
                        if filter_info["column"] == field["name"]:
                            priority += filter_info["count"]
                
                if priority > 0:
                    candidates.append((field["name"], priority))
        
        # Sort candidates by priority
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
    
    def _find_clustering_candidates(self, table_metadata: Dict[str, Any]) -> List[Tuple[str, int]]:
        """Find suitable fields for clustering.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            List of tuples with (field_name, priority_score)
        """
        candidates = []
        
        # Consider good clustering field types: STRING, INTEGER, BOOL
        for field in table_metadata.get("schema", []):
            if field["type"] in ("STRING", "INTEGER", "BOOL") and field["mode"] != "REPEATED":
                name_lower = field["name"].lower()
                priority = 0
                
                # Prioritize fields with certain names that are likely to be good clustering keys
                if "id" in name_lower and field["type"] == "INTEGER":
                    priority += 4  # IDs are often good clustering keys
                if "status" in name_lower or "type" in name_lower or "category" in name_lower:
                    priority += 5  # Status/type fields are excellent for clustering
                if "code" in name_lower or "key" in name_lower:
                    priority += 3
                    
                # Avoid fields that might have extremely high cardinality
                if "uuid" in name_lower or "guid" in name_lower:
                    continue  # Skip UUID fields entirely
                if "name" == name_lower or "description" == name_lower or "comment" in name_lower:
                    continue  # Skip free text fields
                
                # Check for usage in query filters and joins
                if "frequent_filters" in table_metadata:
                    for filter_info in table_metadata.get("frequent_filters", []):
                        if filter_info["column"] == field["name"]:
                            priority += filter_info["count"]
                
                if "column_stats" in table_metadata:
                    for col_stats in table_metadata["column_stats"]:
                        if col_stats["name"] == field["name"]:
                            # Ideal clustering fields have moderate cardinality
                            if "distinct_values_count" in col_stats:
                                if col_stats["distinct_values_count"] > 100 and col_stats["distinct_values_count"] < 10000:
                                    priority += 3
                                elif col_stats["distinct_values_count"] > 10000:
                                    priority -= 5
                                elif col_stats["distinct_values_count"] < 10:
                                    priority -= 2
                
                if priority > 0:
                    candidates.append((field["name"], priority))
        
        # Sort candidates by priority
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
    
    def _determine_range_partitioning(self, table_metadata: Dict[str, Any], field_name: str) -> Dict[str, int]:
        """Determine appropriate range partitioning parameters.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            field_name: Integer field to partition on
            
        Returns:
            Dict with start, end, and interval values
        """
        # First try to determine from column stats if available
        for col_stats in table_metadata.get("column_stats", []):
            if col_stats["name"] == field_name and "min_value" in col_stats and "max_value" in col_stats:
                min_val = int(col_stats["min_value"])
                max_val = int(col_stats["max_value"])
                range_size = max_val - min_val
                
                # Aim for 50-100 partitions
                target_partitions = min(100, max(20, table_metadata.get("size_gb", 0) / 2))
                interval = max(1, range_size // target_partitions)
                
                return {
                    "start": min_val,
                    "end": max_val + interval,  # Add interval to ensure max value is included
                    "interval": interval
                }
        
        # If no column stats, make a generic recommendation
        # Common ID range: 0 to 1 billion by 10 million
        return {
            "start": 0,
            "end": 1000000000,
            "interval": 10000000
        }
    
    def _generate_partition_sql(self, table_metadata: Dict[str, Any], field_name: str, 
                              partition_type: str) -> str:
        """Generate SQL for adding time-based partitioning.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            field_name: Field to partition on
            partition_type: Partition type (DAY, MONTH, YEAR)
            
        Returns:
            SQL script for implementing the partitioning
        """
        project_id = self.project_id
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        sql = f"""-- Step 1: Create a backup of the original table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS 
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 2: Create a new partitioned table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_partitioned`
PARTITION BY {partition_type}({field_name})
AS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 3: Verify the row count matches
SELECT 
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_partitioned`) AS partitioned_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_partitioned`) AS counts_match;
  
-- Step 4: Swap tables (run these one at a time after verifying counts match)
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_partitioned` RENAME TO `{project_id}.{dataset_id}.{table_id}`;

-- Step 5: Drop old table after confirming everything is working
-- DROP TABLE `{project_id}.{dataset_id}.{table_id}_old`;
"""
        
        return sql
    
    def _generate_partition_change_sql(self, table_metadata: Dict[str, Any], field_name: str, 
                                    partition_type: str) -> str:
        """Generate SQL for changing partitioning scheme.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            field_name: Field to partition on
            partition_type: New partition type (DAY, MONTH, YEAR)
            
        Returns:
            SQL script for implementing the partitioning change
        """
        project_id = self.project_id
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        sql = f"""-- Step 1: Create a backup of the original table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS 
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 2: Create a new table with the modified partitioning
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_new_partition`
PARTITION BY {partition_type}({field_name})
AS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 3: Verify the row count matches
SELECT 
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_new_partition`) AS new_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_new_partition`) AS counts_match;
  
-- Step 4: Swap tables (run these one at a time after verifying counts match)
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_new_partition` RENAME TO `{project_id}.{dataset_id}.{table_id}`;

-- Step 5: Drop old table after confirming everything is working
-- DROP TABLE `{project_id}.{dataset_id}.{table_id}_old`;
"""
        
        return sql
    
    def _generate_range_partition_sql(self, table_metadata: Dict[str, Any], field_name: str, 
                                   range_info: Dict[str, int]) -> str:
        """Generate SQL for adding range partitioning.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            field_name: Field to partition on
            range_info: Range parameters (start, end, interval)
            
        Returns:
            SQL script for implementing range partitioning
        """
        project_id = self.project_id
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        start = range_info["start"]
        end = range_info["end"]
        interval = range_info["interval"]
        
        sql = f"""-- Step 1: Create a backup of the original table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS 
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 2: Create a new range-partitioned table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_partitioned`
PARTITION BY RANGE_BUCKET({field_name}, GENERATE_ARRAY({start}, {end}, {interval}))
AS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 3: Verify the row count matches
SELECT 
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_partitioned`) AS partitioned_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_partitioned`) AS counts_match;
  
-- Step 4: Swap tables (run these one at a time after verifying counts match)
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_partitioned` RENAME TO `{project_id}.{dataset_id}.{table_id}`;

-- Step 5: Drop old table after confirming everything is working
-- DROP TABLE `{project_id}.{dataset_id}.{table_id}_old`;
"""
        
        return sql
    
    def _generate_clustering_sql(self, table_metadata: Dict[str, Any], cluster_fields: List[str]) -> str:
        """Generate SQL for adding clustering.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            cluster_fields: Fields to cluster on
            
        Returns:
            SQL script for implementing clustering
        """
        project_id = self.project_id
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        # Check if table is already partitioned
        partition_clause = ""
        if table_metadata.get("partitioning"):
            partition_type = table_metadata["partitioning"].get("type", "")
            partition_field = table_metadata["partitioning"].get("field", "")
            
            if partition_type and partition_field:
                partition_clause = f"PARTITION BY {partition_type}({partition_field})"
        
        # Create cluster field list
        cluster_field_list = ", ".join([f"`{field}`" for field in cluster_fields])
        
        sql = f"""-- Step 1: Create a backup of the original table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS 
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 2: Create a new clustered table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_clustered`
{partition_clause}
CLUSTER BY {cluster_field_list}
AS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Step 3: Verify the row count matches
SELECT 
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_clustered`) AS clustered_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_clustered`) AS counts_match;
  
-- Step 4: Swap tables (run these one at a time after verifying counts match)
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_clustered` RENAME TO `{project_id}.{dataset_id}.{table_id}`;

-- Step 5: Drop old table after confirming everything is working
-- DROP TABLE `{project_id}.{dataset_id}.{table_id}_old`;
"""
        
        return sql
    
    def _generate_lts_sql(self, table_metadata: Dict[str, Any]) -> str:
        """Generate SQL for preparing a table for long-term storage.
        
        Args:
            table_metadata: Table metadata from MetadataExtractor
            
        Returns:
            SQL script for implementing LTS recommendations
        """
        project_id = self.project_id
        dataset_id = table_metadata.get("dataset_id", "")
        table_id = table_metadata.get("table_id", "")
        
        sql = f"""-- Option 1: Create a copy in a long-term storage dataset
-- First, create a long-term storage dataset if it doesn't exist
-- CREATE DATASET IF NOT EXISTS `{project_id}.{dataset_id}_archive`
--   OPTIONS(
--     description = 'Long-term storage for infrequently accessed tables',
--     location = '{table_metadata.get('location', 'US')}'
--   );

-- Create an archived copy with same schema and data
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}_archive.{table_id}` 
  AS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;

-- Option 2: Export to GCS for even cheaper storage
-- EXPORT DATA OPTIONS(
--   uri='gs://{project_id}-archive/{dataset_id}/{table_id}/*.parquet',
--   format='PARQUET',
--   compression='SNAPPY')
-- AS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;
"""
        
        return sql