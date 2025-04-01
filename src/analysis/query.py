"""Query pattern analysis for BigQuery cost optimization."""

from typing import Dict, List, Any
from google.cloud import bigquery
import pandas as pd
import re

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def analyze_query_patterns(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze query patterns and identify optimization opportunities.
    
    Args:
        metadata: Dataset metadata from metadata extractor
        
    Returns:
        List of query optimization recommendations
    """
    recommendations = []
    
    # Add sample query pattern recommendations
    # In a real implementation, this would analyze actual queries from the INFORMATION_SCHEMA
    for table in metadata["tables"]:
        # Skip tables with low query activity
        if table.get("query_count_30d", 0) < 10:
            continue
            
        # Analyze high-cost queries
        if table.get("total_bytes_processed_30d", 0) > 1024**4:  # > 1TB processed
            recommendations.extend(analyze_high_cost_queries(table))
            
        # Check for partitioning and query pattern alignment
        if table["partitioning"] and table.get("query_count_30d", 0) > 0:
            recommendations.extend(analyze_partition_usage(table))
    
    return recommendations


def analyze_high_cost_queries(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze high-cost queries for a specific table.
    
    Args:
        table: Table metadata
        
    Returns:
        List of query optimization recommendations
    """
    recommendations = []
    
    # Assume we have high-cost queries for this table
    # Check if the table has appropriate optimizations
    
    if not table["partitioning"] and table["size_gb"] > 10:
        # Recommend adding a WHERE clause to utilize partitioning
        # (once partitioning is implemented)
        recommendations.append({
            "table_id": table["table_id"],
            "type": "query_filter_add",
            "current_state": "Queries scanning full table without filters",
            "recommendation": "Add appropriate date/time filters to queries to utilize partitioning once implemented",
            "rationale": "High-cost queries could benefit from partition pruning to reduce data scanned.",
            "estimated_savings_pct": 60,  # Significant savings from partition pruning
            "priority": "high",
            "estimated_effort": "medium",
            "sql_example": f"-- Before:\nSELECT * FROM `{table['table_id']}`\n\n-- After (once partition column is added):\nSELECT * FROM `{table['table_id']}` WHERE partition_column >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)"
        })
    
    # Check for potential column pruning opportunities
    if table["field_count"] > 20 and table["size_gb"] > 5:
        recommendations.append({
            "table_id": table["table_id"],
            "type": "query_column_prune",
            "current_state": "Queries selecting all columns (SELECT *)",
            "recommendation": "Select only required columns instead of using SELECT *",
            "rationale": "Reducing columns scanned can significantly lower query costs.",
            "estimated_savings_pct": 40,  # Column pruning can save substantial costs
            "priority": "medium",
            "estimated_effort": "low",
            "sql_example": f"-- Before:\nSELECT * FROM `{table['table_id']}`\n\n-- After:\nSELECT id, name, status, created_at FROM `{table['table_id']}`"
        })
    
    return recommendations


def analyze_partition_usage(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze partition usage effectiveness.
    
    Args:
        table: Table metadata
        
    Returns:
        List of partition usage recommendations
    """
    recommendations = []
    
    # For partitioned tables, check if queries are effectively using the partitions
    if table["partitioning"]:
        partition_field = table["partitioning"]["field"]
        
        # In a real implementation, we would analyze actual queries from INFORMATION_SCHEMA
        # Here we'll create a sample recommendation
        recommendations.append({
            "table_id": table["table_id"],
            "type": "query_partition_filter",
            "current_state": f"Some queries may not filter on partition field '{partition_field}'",
            "recommendation": f"Ensure all queries include a filter on '{partition_field}'",
            "rationale": "Queries without partition filters scan the entire table, negating the benefits of partitioning.",
            "estimated_savings_pct": 50,
            "priority": "high",
            "estimated_effort": "low",
            "sql_example": f"-- Inefficient query (scans all partitions):\nSELECT * FROM `{table['table_id']}` WHERE status = 'COMPLETED'\n\n-- Efficient query (scans only relevant partitions):\nSELECT * FROM `{table['table_id']}` WHERE {partition_field} >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY) AND status = 'COMPLETED'"
        })
    
    return recommendations


def get_query_history(client: bigquery.Client, project_id: str, dataset_id: str, table_id: str, days: int = 30) -> pd.DataFrame:
    """Get query history for a specific table from INFORMATION_SCHEMA.
    
    Args:
        client: BigQuery client
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        days: Number of days of history to analyze
        
    Returns:
        DataFrame containing query history
    """
    query = f"""
    SELECT
      query_text,
      total_bytes_processed,
      total_slot_ms,
      creation_time,
      query_info.query_hashed_normalized
    FROM
      `{project_id}.region-us`.INFORMATION_SCHEMA.JOBS
    WHERE
      creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
      AND job_type = 'QUERY'
      AND state = 'DONE'
      AND REGEXP_CONTAINS(query, r'[^\w]`?{project_id}\.{dataset_id}\.{table_id}`?[^\w]')
    ORDER BY
      total_bytes_processed DESC
    LIMIT 100
    """
    
    try:
        return client.query(query).to_dataframe()
    except Exception as e:
        logger.warning(f"Error getting query history: {e}")
        return pd.DataFrame()


def extract_query_patterns(queries: pd.DataFrame) -> Dict[str, Any]:
    """Extract common patterns from query history.
    
    Args:
        queries: DataFrame containing query history
        
    Returns:
        Dict with analysis of query patterns
    """
    if queries.empty:
        return {}
        
    # Analyze query patterns - in a real implementation this would be more sophisticated
    patterns = {
        "select_star_count": 0,
        "missing_where_count": 0,
        "full_table_scan_count": 0,
        "high_cost_queries": []
    }
    
    for _, row in queries.iterrows():
        query_text = row["query_text"].lower()
        
        # Check for SELECT *
        if re.search(r"select\s+\*", query_text):
            patterns["select_star_count"] += 1
            
        # Check for missing WHERE clause
        if not re.search(r"where", query_text):
            patterns["missing_where_count"] += 1
            
        # Identify high-cost queries
        if row["total_bytes_processed"] > 1024**3:  # > 1GB processed
            patterns["high_cost_queries"].append({
                "bytes_processed": row["total_bytes_processed"],
                "slot_ms": row["total_slot_ms"],
                "query_hash": row["query_hashed_normalized"] if "query_hashed_normalized" in row else None
            })
    
    return patterns
