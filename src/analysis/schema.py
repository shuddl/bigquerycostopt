"""Schema optimization analysis for BigQuery datasets."""

from typing import Dict, List, Any
from collections import defaultdict
import pandas as pd

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def analyze_schema_optimizations(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze dataset schemas and identify optimization opportunities.
    
    Args:
        metadata: Dataset metadata from metadata extractor
        
    Returns:
        List of schema optimization recommendations
    """
    recommendations = []
    
    for table in metadata["tables"]:
        # Skip small tables (< 1 GB)
        if table["size_gb"] < 1.0:
            continue
            
        # Check for unused columns
        unused_columns = identify_unused_columns(table)
        if unused_columns:
            recommendations.append({
                "table_id": table["table_id"],
                "type": "schema_remove_columns",
                "current_state": f"Table has potentially unused columns: {', '.join(unused_columns[:3])}" + 
                               (f" and {len(unused_columns) - 3} more" if len(unused_columns) > 3 else ""),
                "recommendation": f"Consider removing unused columns or moving them to a separate table",
                "rationale": "Removing unused columns reduces storage costs and improves query performance.",
                "columns": unused_columns,
                "estimated_savings_pct": min(30, len(unused_columns) / len(table["schema"]) * 100),
                "priority": "medium",
                "estimated_effort": "medium"  # Requires schema changes and data migration
            })
            
        # Check for suboptimal data types
        type_optimizations = identify_type_optimizations(table)
        if type_optimizations:
            for field_name, current_type, recommended_type, rationale in type_optimizations:
                recommendations.append({
                    "table_id": table["table_id"],
                    "type": "schema_optimize_type",
                    "current_state": f"Column '{field_name}' uses {current_type}",
                    "recommendation": f"Change column '{field_name}' to {recommended_type}",
                    "rationale": rationale,
                    "estimated_savings_pct": 5,  # Conservative estimate
                    "priority": "low",
                    "estimated_effort": "medium"  # Requires schema changes and data migration
                })
                
        # Check for nested and repeated fields that could be denormalized
        denormalization_recommendations = identify_denormalization_opportunities(table)
        recommendations.extend(denormalization_recommendations)
    
    return recommendations


def identify_unused_columns(table: Dict[str, Any]) -> List[str]:
    """Identify potentially unused columns in a table.
    
    Args:
        table: Table metadata
        
    Returns:
        List of potentially unused column names
    """
    # In a real implementation, this would analyze query history to find unused columns
    # For now, we'll use some heuristics
    
    unused_columns = []
    
    # Basic heuristic: look for columns that might be deprecated or temporary
    for field in table["schema"]:
        name_lower = field["name"].lower()
        if any(pattern in name_lower for pattern in ["deprecated", "temp", "old", "legacy", "_v1", "_bak"]):
            unused_columns.append(field["name"])
            
        # Look for columns with NULL values allowed but no description
        if field["mode"] == "NULLABLE" and not field["description"] and len(name_lower) < 4:
            # Short column names with no description might be legacy or poorly documented
            unused_columns.append(field["name"])
    
    return unused_columns


def identify_type_optimizations(table: Dict[str, Any]) -> List[tuple]:
    """Identify possible data type optimizations.
    
    Args:
        table: Table metadata
        
    Returns:
        List of tuples with (field_name, current_type, recommended_type, rationale)
    """
    optimizations = []
    
    for field in table["schema"]:
        name_lower = field["name"].lower()
        field_type = field["type"]
        
        # STRING to BYTES for binary data
        if field_type == "STRING" and any(hint in name_lower for hint in ["hash", "binary", "blob", "encoded"]):
            optimizations.append((field["name"], "STRING", "BYTES", 
                                "BYTES type is more efficient for binary data storage."))
            
        # INT64 to smaller integers where appropriate
        if field_type == "INTEGER" and "id" not in name_lower and "key" not in name_lower:
            if "flag" in name_lower or "bool" in name_lower or "indicator" in name_lower:
                optimizations.append((field["name"], "INTEGER", "BOOL", 
                                    "Column appears to store boolean values; BOOL type is more efficient."))
                                    
        # TIMESTAMP to DATE for date-only values
        if field_type == "TIMESTAMP" and ("date" in name_lower and "time" not in name_lower):
            optimizations.append((field["name"], "TIMESTAMP", "DATE", 
                                "Column appears to store dates only; DATE type is more efficient."))
    
    return optimizations


def identify_denormalization_opportunities(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify denormalization opportunities for nested and repeated fields.
    
    Args:
        table: Table metadata
        
    Returns:
        List of denormalization recommendations
    """
    recommendations = []
    
    # Count repeated and nested fields
    repeated_fields = [f for f in table["schema"] if f["mode"] == "REPEATED"]
    
    if len(repeated_fields) > 3 and table["size_gb"] > 10:
        recommendations.append({
            "table_id": table["table_id"],
            "type": "schema_denormalize",
            "current_state": f"Table has {len(repeated_fields)} repeated fields which may cause data explosion",
            "recommendation": "Consider using multiple tables with one-to-many relationships instead of repeated fields",
            "rationale": "Repeated fields can cause data explosion and make queries less efficient.",
            "estimated_savings_pct": 15,
            "priority": "medium",
            "estimated_effort": "high"  # Requires significant schema redesign
        })
    
    return recommendations
