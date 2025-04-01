"""Storage optimization analysis for BigQuery datasets."""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from google.cloud import bigquery

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def analyze_partitioning_options(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze dataset and recommend partitioning strategies.
    
    Args:
        metadata: Dataset metadata from metadata extractor
        
    Returns:
        List of partitioning recommendations
    """
    recommendations = []
    
    for table in metadata["tables"]:
        # Skip small tables (< 1 GB)
        if table["size_gb"] < 1.0:
            continue
            
        # Skip already partitioned tables
        if table["partitioning"]:
            # Check if current partitioning is optimal
            current_partition = table["partitioning"]
            if current_partition["type"] == "DAY" and table["size_gb"] > 100:
                # For very large tables, recommend monthly partitioning instead of daily
                recommendations.append({
                    "table_id": table["table_id"],
                    "type": "partitioning_change",
                    "current_state": f"Daily partitioning on {current_partition['field']}",
                    "recommendation": f"Change to MONTH partitioning on {current_partition['field']}",
                    "rationale": "Table is very large (>100GB); monthly partitioning would reduce partition count and management overhead while still providing good query performance for most use cases.",
                    "estimated_savings_pct": 5,
                    "priority": "medium",
                    "estimated_effort": "medium"  # Requires table recreation
                })
            continue
        
        # Look for datetime/timestamp fields that might be good partition candidates
        partition_candidates = []
        for field in table["schema"]:
            if field["type"] in ("TIMESTAMP", "DATE", "DATETIME") and field["mode"] != "REPEATED":
                # Prioritize fields with certain names that are likely to be good partition keys
                priority = 0
                name_lower = field["name"].lower()
                if "date" in name_lower or "time" in name_lower:
                    priority += 5
                if "created" in name_lower or "updated" in name_lower or "timestamp" in name_lower:
                    priority += 3
                if name_lower.startswith("dt_") or name_lower.startswith("date_"):
                    priority += 3
                partition_candidates.append((field["name"], priority))
        
        # Sort candidates by priority
        partition_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if partition_candidates:
            best_field = partition_candidates[0][0]
            
            # Estimate savings (10-20% is typical for well-partitioned tables)
            estimated_savings_pct = min(20, max(10, int(table["size_gb"] / 10)))
            
            # For tables over 50GB, partitioning is usually high priority
            priority = "high" if table["size_gb"] > 50 else "medium"
            
            # Recommend appropriate partition type based on table size
            partition_type = "DAY" if table["size_gb"] < 100 else "MONTH"
            
            recommendations.append({
                "table_id": table["table_id"],
                "type": "partitioning_add",
                "current_state": "No partitioning",
                "recommendation": f"Add {partition_type} partitioning on field '{best_field}'",
                "rationale": f"Table is {table['size_gb']:.1f} GB and contains a good candidate field for partitioning.",
                "estimated_savings_pct": estimated_savings_pct,
                "priority": priority,
                "estimated_effort": "high"  # Requires table recreation
            })
    
    return recommendations


def analyze_clustering_options(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze dataset and recommend clustering strategies.
    
    Args:
        metadata: Dataset metadata from metadata extractor
        
    Returns:
        List of clustering recommendations
    """
    recommendations = []
    
    for table in metadata["tables"]:
        # Skip small tables (< 1 GB)
        if table["size_gb"] < 1.0:
            continue
            
        # Skip tables that already have clustering
        if table["clustering"]:
            continue
            
        # Tables should ideally be partitioned before clustering
        has_partitioning = bool(table["partitioning"])
        
        # Look for good clustering candidates
        # Prefer fields with high cardinality but not too high (good selectivity)
        # Integer, string and bool fields are good candidates
        cluster_candidates = []
        
        for field in table["schema"]:
            if field["type"] in ("STRING", "INTEGER", "BOOL") and field["mode"] != "REPEATED":
                # Prioritize fields with certain names that are likely to be good clustering keys
                priority = 0
                name_lower = field["name"].lower()
                
                if "id" in name_lower and field["type"] == "INTEGER":
                    priority += 4  # IDs are often good clustering keys
                if "status" in name_lower or "type" in name_lower or "category" in name_lower:
                    priority += 5  # Status/type fields are excellent for clustering
                if "code" in name_lower or "key" in name_lower:
                    priority += 3
                    
                # Avoid fields that might have extremely high cardinality
                if "uuid" in name_lower or "guid" in name_lower:
                    priority -= 5
                if "name" == name_lower or "description" == name_lower or "comment" in name_lower:
                    priority -= 3  # Free text fields are usually poor clustering keys
                    
                if priority > 0:
                    cluster_candidates.append((field["name"], priority))
        
        # Sort candidates by priority
        cluster_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take up to 3 top candidates (BigQuery supports up to 4 clustering fields)
        top_candidates = [c[0] for c in cluster_candidates[:3]]
        
        if top_candidates:
            # Clustering typically saves 20-30% on well-clustered tables
            estimated_savings_pct = min(30, max(10, int(table["size_gb"] / 5)))
            
            # If no partitioning, suggest that first
            recommendation_text = f"Add clustering on fields: {', '.join(top_candidates)}"
            if not has_partitioning:
                recommendation_text += " (consider partitioning first)"
                priority = "medium"
            else:
                priority = "high" if table["size_gb"] > 20 else "medium"
                
            recommendations.append({
                "table_id": table["table_id"],
                "type": "clustering_add",
                "current_state": "No clustering",
                "recommendation": recommendation_text,
                "rationale": f"Table is {table['size_gb']:.1f} GB and contains fields that would benefit from clustering.",
                "estimated_savings_pct": estimated_savings_pct,
                "priority": priority,
                "estimated_effort": "high" if not has_partitioning else "medium"  # Harder if need to add partitioning first
            })
    
    return recommendations


def analyze_compression_options(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze dataset and recommend compression strategies.
    
    Args:
        metadata: Dataset metadata from metadata extractor
        
    Returns:
        List of compression recommendations
    """
    recommendations = []
    
    for table in metadata["tables"]:
        # Look for tables with TEXT/STRING fields that might benefit from compression
        has_text_fields = any(field["type"] == "STRING" for field in table["schema"])
        
        if has_text_fields and table["size_gb"] > 5.0:
            # Estimate compression savings (text-heavy tables can save 30-40%)
            text_field_ratio = sum(1 for field in table["schema"] if field["type"] == "STRING") / len(table["schema"])
            estimated_savings_pct = min(40, max(5, int(30 * text_field_ratio)))
            
            recommendations.append({
                "table_id": table["table_id"],
                "type": "compression_optimize",
                "current_state": "Default compression",
                "recommendation": "Export table to compressed Parquet files and reload",
                "rationale": f"Table is {table['size_gb']:.1f} GB and contains text fields that would benefit from improved compression.",
                "estimated_savings_pct": estimated_savings_pct,
                "priority": "medium",
                "estimated_effort": "medium"  # Requires table export/import
            })
    
    return recommendations
