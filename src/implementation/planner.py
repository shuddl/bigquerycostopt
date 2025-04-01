"""Implementation plan generation for BigQuery cost optimization recommendations."""

from typing import Dict, List, Any
import json

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def generate_implementation_plan(recommendations: List[Dict[str, Any]], 
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an implementation plan for the given recommendations.
    
    Args:
        recommendations: List of recommendations from the recommendation engine
        metadata: Dataset metadata
        
    Returns:
        Dict containing the implementation plan
    """
    # Group recommendations by priority
    high_priority = [r for r in recommendations if r.get("priority") == "high"]
    medium_priority = [r for r in recommendations if r.get("priority") == "medium"]
    low_priority = [r for r in recommendations if r.get("priority") == "low"]
    
    # Group recommendations by table
    table_groups = {}
    for rec in recommendations:
        table_id = rec["table_id"]
        if table_id not in table_groups:
            table_groups[table_id] = []
        table_groups[table_id].append(rec)
    
    # Generate phases
    phases = [
        generate_phase("Phase 1: High Priority Optimizations", high_priority, metadata),
        generate_phase("Phase 2: Medium Priority Optimizations", medium_priority, metadata),
        generate_phase("Phase 3: Low Priority Optimizations", low_priority, metadata)
    ]
    
    # Generate ROI summary
    total_cost = sum(p["estimated_cost_usd"] for p in phases)
    total_annual_savings = sum(p["estimated_annual_savings_usd"] for p in phases)
    
    roi_summary = {
        "total_implementation_cost_usd": total_cost,
        "total_annual_savings_usd": total_annual_savings,
        "overall_roi": total_annual_savings / total_cost if total_cost > 0 else 0,
        "payback_period_months": (total_cost / (total_annual_savings / 12)) if total_annual_savings > 0 else 0
    }
    
    # Build final plan
    plan = {
        "dataset_id": metadata["dataset_id"],
        "project_id": metadata["project_id"],
        "total_recommendations": len(recommendations),
        "roi_summary": roi_summary,
        "phases": phases,
        "table_specific_plans": generate_table_specific_plans(table_groups, metadata)
    }
    
    return plan


def generate_phase(name: str, recommendations: List[Dict[str, Any]], 
                  metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single implementation phase.
    
    Args:
        name: Phase name
        recommendations: List of recommendations for this phase
        metadata: Dataset metadata
        
    Returns:
        Dict containing the phase details
    """
    steps = []
    estimated_cost = 0
    estimated_savings = 0
    
    for i, rec in enumerate(recommendations):
        # Generate implementation steps
        implementation_steps = generate_implementation_steps(rec, metadata)
        
        steps.append({
            "step_number": i + 1,
            "recommendation_id": rec.get("recommendation_id", f"rec_{i}"),
            "table_id": rec["table_id"],
            "recommendation": rec["recommendation"],
            "implementation_steps": implementation_steps,
            "effort_level": rec.get("estimated_effort", "medium"),
            "estimated_cost_usd": rec.get("implementation_cost_usd", 0),
            "estimated_savings_usd": rec.get("annual_savings_usd", 0)
        })
        
        estimated_cost += rec.get("implementation_cost_usd", 0)
        estimated_savings += rec.get("annual_savings_usd", 0)
    
    return {
        "name": name,
        "steps": steps,
        "recommendation_count": len(recommendations),
        "estimated_cost_usd": estimated_cost,
        "estimated_annual_savings_usd": estimated_savings,
        "roi": estimated_savings / estimated_cost if estimated_cost > 0 else 0
    }


def generate_implementation_steps(recommendation: Dict[str, Any], 
                                 metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate detailed implementation steps for a recommendation.
    
    Args:
        recommendation: Single recommendation
        metadata: Dataset metadata
        
    Returns:
        List of implementation steps
    """
    rec_type = recommendation.get("type", "")
    table_id = recommendation["table_id"]
    project_id = metadata["project_id"]
    dataset_id = metadata["dataset_id"]
    
    # Find the table metadata
    table_metadata = next((t for t in metadata["tables"] if t["table_id"] == table_id), {})
    
    steps = []
    
    # Implementation steps based on recommendation type
    if "partitioning_add" in rec_type:
        partition_field = recommendation["recommendation"].split("'")[1]  # Extract field name from recommendation
        partition_type = "DAY"
        if "MONTH" in recommendation["recommendation"]:
            partition_type = "MONTH"
        
        steps = [
            {
                "description": "Create a backup of the original table",
                "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}`;"
            },
            {
                "description": f"Create new table with {partition_type} partitioning",
                "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_partitioned`\nPARTITION BY {partition_type}({partition_field})\nAS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;"
            },
            {
                "description": "Verify data in the new partitioned table",
                "sql": f"SELECT COUNT(*) AS original_count FROM `{project_id}.{dataset_id}.{table_id}`;\nSELECT COUNT(*) AS new_count FROM `{project_id}.{dataset_id}.{table_id}_partitioned`;"
            },
            {
                "description": "Swap tables to implement the change",
                "sql": f"ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;\nALTER TABLE `{project_id}.{dataset_id}.{table_id}_partitioned` RENAME TO `{project_id}.{dataset_id}.{table_id}`;"
            }
        ]
        
    elif "clustering_add" in rec_type:
        # Extract clustering fields
        clustering_text = recommendation["recommendation"]
        clustering_fields = []
        
        if "fields:" in clustering_text:
            field_text = clustering_text.split("fields:")[1].split("(")[0].strip()
            clustering_fields = [f.strip() for f in field_text.split(",")]
        
        if clustering_fields:
            # Check if table is already partitioned
            is_partitioned = bool(table_metadata.get("partitioning", None))
            partition_clause = ""
            
            if is_partitioned:
                partition_field = table_metadata["partitioning"]["field"]
                partition_type = table_metadata["partitioning"]["type"]
                partition_clause = f"PARTITION BY {partition_type}({partition_field})"
            
            clustering_fields_str = ", ".join([f"`{field}`" for field in clustering_fields])
            
            steps = [
                {
                    "description": "Create a backup of the original table",
                    "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}`;"
                },
                {
                    "description": "Create new table with clustering",
                    "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_clustered`\n{partition_clause}\nCLUSTER BY {clustering_fields_str}\nAS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;"
                },
                {
                    "description": "Verify data in the new clustered table",
                    "sql": f"SELECT COUNT(*) AS original_count FROM `{project_id}.{dataset_id}.{table_id}`;\nSELECT COUNT(*) AS new_count FROM `{project_id}.{dataset_id}.{table_id}_clustered`;"
                },
                {
                    "description": "Swap tables to implement the change",
                    "sql": f"ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;\nALTER TABLE `{project_id}.{dataset_id}.{table_id}_clustered` RENAME TO `{project_id}.{dataset_id}.{table_id}`;"
                }
            ]
            
    elif "query_" in rec_type:
        # For query recommendations, provide the SQL example from the recommendation
        sql_example = recommendation.get("sql_example", "")
        
        steps = [
            {
                "description": "Identify queries that need modification",
                "sql": f"-- Query to find potentially inefficient queries:\nSELECT\n  query_text,\n  total_bytes_processed,\n  creation_time\nFROM\n  `{project_id}.region-us`.INFORMATION_SCHEMA.JOBS\nWHERE\n  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)\n  AND query LIKE '%{table_id}%'\n  AND total_bytes_processed > 1000000000\nORDER BY\n  total_bytes_processed DESC\nLIMIT 10;"
            },
            {
                "description": "Example query modification",
                "sql": sql_example
            }
        ]
        
    elif "schema_" in rec_type:
        if "schema_remove_columns" in rec_type and "columns" in recommendation:
            # Generate SQL to create a new table without the unused columns
            columns_to_remove = recommendation["columns"]
            
            # Get all columns except the ones to remove
            kept_columns = []
            for field in table_metadata["schema"]:
                if field["name"] not in columns_to_remove:
                    kept_columns.append(f"`{field['name']}`")
            
            kept_columns_str = ",\n  ".join(kept_columns)
            
            steps = [
                {
                    "description": "Create a backup of the original table",
                    "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}`;"
                },
                {
                    "description": "Create new table without unused columns",
                    "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_optimized` AS\nSELECT\n  {kept_columns_str}\nFROM `{project_id}.{dataset_id}.{table_id}`;"
                },
                {
                    "description": "Verify data in the new optimized table",
                    "sql": f"SELECT COUNT(*) AS original_count FROM `{project_id}.{dataset_id}.{table_id}`;\nSELECT COUNT(*) AS new_count FROM `{project_id}.{dataset_id}.{table_id}_optimized`;"
                },
                {
                    "description": "Swap tables to implement the change",
                    "sql": f"ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;\nALTER TABLE `{project_id}.{dataset_id}.{table_id}_optimized` RENAME TO `{project_id}.{dataset_id}.{table_id}`;"
                }
            ]
    
    # If no specific steps were generated, add generic steps
    if not steps:
        steps = [
            {
                "description": "Analyze current state",
                "sql": f"-- Analyze current state:\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 10;"
            },
            {
                "description": "Implement recommendation",
                "sql": "-- Implementation SQL would go here\n-- This is a placeholder for a custom implementation"
            },
            {
                "description": "Verify changes",
                "sql": f"-- Verify the changes were successfully implemented:\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 10;"
            }
        ]
    
    return steps


def generate_table_specific_plans(table_groups: Dict[str, List[Dict[str, Any]]], 
                                metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate implementation plans specific to each table.
    
    Args:
        table_groups: Recommendations grouped by table
        metadata: Dataset metadata
        
    Returns:
        List of table-specific implementation plans
    """
    table_plans = []
    
    for table_id, recommendations in table_groups.items():
        # Get table metadata
        table_metadata = next((t for t in metadata["tables"] if t["table_id"] == table_id), {})
        
        # Skip tables with no metadata or recommendations
        if not table_metadata or not recommendations:
            continue
        
        # Sort recommendations by priority and ROI
        sorted_recs = sorted(recommendations, 
                             key=lambda r: (0 if r.get("priority") == "high" else 
                                         1 if r.get("priority") == "medium" else 2,
                                         -1 * r.get("roi", 0)))
        
        # Generate combined implementation steps
        implementation_steps = []
        
        # Group the recommendations by type to implement related changes together
        rec_by_type = {}
        for rec in sorted_recs:
            rec_type = rec.get("type", "other")
            if rec_type not in rec_by_type:
                rec_by_type[rec_type] = []
            rec_by_type[rec_type].append(rec)
        
        # Process structural changes first (partition, cluster, schema)
        for rec_type in ["partitioning_add", "partitioning_change", "clustering_add", "schema_remove_columns", "schema_optimize_type"]:
            if rec_type in rec_by_type:
                for rec in rec_by_type[rec_type]:
                    rec_steps = generate_implementation_steps(rec, metadata)
                    implementation_steps.extend(rec_steps)
        
        # Then add query optimization steps
        for rec_type, recs in rec_by_type.items():
            if rec_type.startswith("query_"):
                for rec in recs:
                    rec_steps = generate_implementation_steps(rec, metadata)
                    implementation_steps.extend(rec_steps)
        
        # Calculate total savings for this table
        estimated_savings = sum(r.get("annual_savings_usd", 0) for r in recommendations)
        estimated_cost = sum(r.get("implementation_cost_usd", 0) for r in recommendations)
        
        table_plans.append({
            "table_id": table_id,
            "table_size_gb": table_metadata.get("size_gb", 0),
            "recommendation_count": len(recommendations),
            "estimated_annual_savings_usd": estimated_savings,
            "implementation_cost_usd": estimated_cost,
            "roi": estimated_savings / estimated_cost if estimated_cost > 0 else 0,
            "implementation_steps": implementation_steps
        })
    
    return table_plans
