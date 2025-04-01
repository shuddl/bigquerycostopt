"""Implementation plan generation for BigQuery cost optimization recommendations."""

from typing import Dict, List, Any, Tuple, Set, Optional
import json
import copy
from datetime import datetime, timedelta

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class ImplementationPlanGenerator:
    """Generator for detailed implementation plans based on BigQuery optimization recommendations."""
    
    def __init__(self):
        """Initialize the implementation plan generator."""
        # Map recommendation types to implementation templates
        self.implementation_templates = {
            "storage": {
                "partitioning_add": self._template_add_partitioning,
                "partitioning_daily_to_monthly": self._template_change_partition_granularity,
                "clustering_add": self._template_add_clustering,
                "long_term_storage": self._template_move_to_lts
            },
            "query": {
                "query_select_star": self._template_optimize_select_star,
                "query_filter_pushdown": self._template_filter_pushdown,
                "query_join_optimization": self._template_optimize_join,
                "query_partition_filter": self._template_add_partition_filter
            },
            "schema": {
                "datatype_float_to_int": self._template_change_datatype,
                "datatype_string_to_enum": self._template_string_to_enum,
                "datatype_string_to_bool": self._template_change_datatype,
                "datatype_timestamp_to_date": self._template_change_datatype,
                "normalize_repeated_field": self._template_normalize_repeated_field,
                "remove_unused_columns": self._template_remove_columns
            }
        }
    
    def generate_plan(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive implementation plan for a set of recommendations.
        
        Args:
            recommendations: The standardized recommendations to implement
            
        Returns:
            Dict containing the implementation plan
        """
        if not recommendations:
            return {
                "phases": [],
                "implementation_time_days": 0,
                "total_recommendations": 0
            }
        
        # Create deep copy to avoid modifying the original
        recs = copy.deepcopy(recommendations)
        
        # Analyze dependencies and conflicts
        dependency_graph = self._build_dependency_graph(recs)
        
        # Group recommendations by table and category
        grouped_recs, rec_by_id = self._group_recommendations(recs)
        
        # Create phased implementation plan
        phases = self._create_implementation_phases(grouped_recs, dependency_graph, rec_by_id)
        
        # Generate implementation steps for each recommendation
        for phase in phases:
            for step in phase["steps"]:
                rec_id = step["recommendation_id"]
                rec = rec_by_id.get(rec_id)
                if rec:
                    # Get implementation template based on recommendation type
                    template_func = self._get_implementation_template(rec)
                    step["implementation_steps"] = template_func(rec)
                    
                    # Add verification steps
                    step["verification_steps"] = self._generate_verification_steps(rec)
                    
                    # Add rollback instructions
                    step["rollback_procedure"] = self._generate_rollback_procedure(rec)
        
        # Calculate overall plan metrics
        total_implementation_time = sum(phase.get("estimated_days", 0) for phase in phases)
        total_cost = sum(phase.get("estimated_cost_usd", 0) for phase in phases)
        total_savings = sum(phase.get("estimated_annual_savings_usd", 0) for phase in phases)
        
        # Prepare final plan
        implementation_plan = {
            "plan_generated": datetime.now().isoformat(),
            "total_recommendations": len(recs),
            "implementation_time_days": total_implementation_time,
            "estimated_implementation_cost_usd": total_cost,
            "estimated_annual_savings_usd": total_savings,
            "roi": total_savings / total_cost if total_cost > 0 else 0,
            "phases": phases
        }
        
        return implementation_plan
    
    def _build_dependency_graph(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """Build a graph of dependencies between recommendations.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Dict mapping recommendation IDs to sets of dependency IDs
        """
        # Create dict of recommendation ID to dependencies
        dependency_graph = {}
        
        for rec in recommendations:
            rec_id = rec["recommendation_id"]
            dependency_graph[rec_id] = set()
            
            # Add explicit dependencies
            if "depends_on" in rec and rec["depends_on"]:
                dependency_graph[rec_id].update(rec["depends_on"])
        
        # Add implicit dependencies
        for rec in recommendations:
            rec_id = rec["recommendation_id"]
            category = rec["category"]
            table_id = rec["table_id"]
            
            # Partitioning should be implemented before clustering
            if category == "storage" and "clustering" in rec.get("type", ""):
                # Find partitioning recommendations for the same table
                for other_rec in recommendations:
                    if (other_rec["table_id"] == table_id and 
                        other_rec["category"] == "storage" and 
                        "partitioning" in other_rec.get("type", "")):
                        dependency_graph[rec_id].add(other_rec["recommendation_id"])
            
            # Schema changes depend on storage changes
            if category == "schema":
                # Find storage recommendations for the same table
                for other_rec in recommendations:
                    if (other_rec["table_id"] == table_id and 
                        other_rec["category"] == "storage"):
                        dependency_graph[rec_id].add(other_rec["recommendation_id"])
        
        return dependency_graph
    
    def _group_recommendations(self, recommendations: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Dict[str, Any]]]:
        """Group recommendations by table and category.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Tuple of (grouped recommendations, recommendation lookup by ID)
        """
        # Group by table and category
        grouped = {}
        rec_by_id = {}
        
        for rec in recommendations:
            rec_id = rec["recommendation_id"]
            table_id = rec["table_id"]
            category = rec["category"]
            
            if table_id not in grouped:
                grouped[table_id] = {
                    "storage": [],
                    "query": [],
                    "schema": []
                }
            
            grouped[table_id][category].append(rec)
            rec_by_id[rec_id] = rec
        
        return grouped, rec_by_id
    
    def _create_implementation_phases(self, 
                                    grouped_recs: Dict[str, Dict[str, List[Dict[str, Any]]]], 
                                    dependency_graph: Dict[str, Set[str]],
                                    rec_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create implementation phases based on recommendation dependencies and priorities.
        
        Args:
            grouped_recs: Recommendations grouped by table and category
            dependency_graph: Mapping of recommendation IDs to dependency IDs
            rec_by_id: Lookup for recommendations by ID
            
        Returns:
            List of implementation phases
        """
        # Create priority tiers
        high_priority_recs = []
        medium_priority_recs = []
        low_priority_recs = []
        
        # Sort recommendations into priority tiers
        for table_id, categories in grouped_recs.items():
            for category, recs in categories.items():
                for rec in recs:
                    priority = rec.get("priority", "medium")
                    
                    if priority == "high":
                        high_priority_recs.append(rec)
                    elif priority == "medium":
                        medium_priority_recs.append(rec)
                    else:
                        low_priority_recs.append(rec)
        
        # Sort recommendations within priority tiers by ROI
        high_priority_recs.sort(key=lambda r: r.get("roi", 0), reverse=True)
        medium_priority_recs.sort(key=lambda r: r.get("roi", 0), reverse=True)
        low_priority_recs.sort(key=lambda r: r.get("roi", 0), reverse=True)
        
        # Create phases
        phases = []
        
        # Phase 1: High priority with no dependencies
        phase1_steps = []
        for rec in high_priority_recs:
            rec_id = rec["recommendation_id"]
            dependencies = dependency_graph.get(rec_id, set())
            
            # Only include if no dependencies or all dependencies are in high priority
            if not dependencies or all(dep_id in [r["recommendation_id"] for r in high_priority_recs] for dep_id in dependencies):
                phase1_steps.append(self._create_implementation_step(rec))
        
        if phase1_steps:
            phases.append(self._create_phase("Phase 1: Critical Optimizations", phase1_steps, rec_by_id))
        
        # Phase 2: High priority with dependencies and medium priority with no dependencies
        phase2_steps = []
        for rec in high_priority_recs:
            rec_id = rec["recommendation_id"]
            if rec_id not in [step["recommendation_id"] for step in phase1_steps]:
                phase2_steps.append(self._create_implementation_step(rec))
        
        for rec in medium_priority_recs:
            rec_id = rec["recommendation_id"]
            dependencies = dependency_graph.get(rec_id, set())
            
            # Only include if no dependencies
            if not dependencies:
                phase2_steps.append(self._create_implementation_step(rec))
        
        if phase2_steps:
            phases.append(self._create_phase("Phase 2: Major Optimizations", phase2_steps, rec_by_id))
        
        # Phase 3: Remaining medium priority
        phase3_steps = []
        for rec in medium_priority_recs:
            rec_id = rec["recommendation_id"]
            if rec_id not in [step["recommendation_id"] for step in phase2_steps]:
                phase3_steps.append(self._create_implementation_step(rec))
        
        if phase3_steps:
            phases.append(self._create_phase("Phase 3: Standard Optimizations", phase3_steps, rec_by_id))
        
        # Phase 4: Low priority optimizations
        if low_priority_recs:
            phase4_steps = [self._create_implementation_step(rec) for rec in low_priority_recs]
            phases.append(self._create_phase("Phase 4: Minor Optimizations", phase4_steps, rec_by_id))
        
        return phases
    
    def _create_implementation_step(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Create an implementation step for a recommendation.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            Dict with implementation step details
        """
        return {
            "recommendation_id": recommendation["recommendation_id"],
            "table_id": recommendation["table_id"],
            "dataset_id": recommendation["dataset_id"],
            "project_id": recommendation["project_id"],
            "description": recommendation["description"],
            "type": recommendation["type"],
            "category": recommendation["category"],
            "priority": recommendation.get("priority", "medium"),
            "estimated_annual_savings_usd": recommendation.get("annual_savings_usd", 0),
            "estimated_implementation_cost_usd": recommendation.get("implementation_cost_usd", 0),
            "estimated_effort": recommendation.get("estimated_effort", "medium")
        }
    
    def _create_phase(self, name: str, steps: List[Dict[str, Any]], 
                    rec_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create an implementation phase.
        
        Args:
            name: Name of the phase
            steps: List of implementation steps
            rec_by_id: Lookup for recommendations by ID
            
        Returns:
            Dict with phase details
        """
        # Calculate phase metrics
        total_savings = sum(step.get("estimated_annual_savings_usd", 0) for step in steps)
        total_cost = sum(step.get("estimated_implementation_cost_usd", 0) for step in steps)
        
        # Estimate implementation time
        effort_days = {
            "low": 0.5,    # Half day
            "medium": 2,   # 2 days
            "high": 5      # 1 week
        }
        
        total_days = sum(effort_days.get(step.get("estimated_effort", "medium"), 2) for step in steps)
        
        # Account for parallelization and overhead
        adjusted_days = max(1, int(total_days * 0.7))  # Assume 30% parallelization efficiency
        
        return {
            "name": name,
            "steps": steps,
            "step_count": len(steps),
            "estimated_annual_savings_usd": total_savings,
            "estimated_cost_usd": total_cost,
            "estimated_days": adjusted_days,
            "roi": total_savings / total_cost if total_cost > 0 else 0
        }
    
    def _get_implementation_template(self, recommendation: Dict[str, Any]) -> callable:
        """Get the appropriate implementation template for a recommendation.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            Function to generate implementation steps
        """
        category = recommendation.get("category", "unknown")
        rec_type = recommendation.get("type", "unknown")
        
        # Try to find specific template
        if category in self.implementation_templates:
            templates = self.implementation_templates[category]
            
            # Check for exact type match
            if rec_type in templates:
                return templates[rec_type]
            
            # Check for partial type match
            for pattern, template_func in templates.items():
                if pattern in rec_type:
                    return template_func
        
        # Fall back to generic template
        return self._template_generic
    
    def _template_generic(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generic implementation template for any recommendation.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        
        # Use implementation SQL if provided
        implementation_sql = recommendation.get("implementation_sql", "")
        
        if implementation_sql:
            # Split into multiple statements if needed
            sql_statements = implementation_sql.split(";")
            sql_statements = [stmt.strip() for stmt in sql_statements if stmt.strip()]
            
            # Create steps for each SQL statement
            steps = []
            for i, sql in enumerate(sql_statements[:3]):  # Limit to first 3 statements for brevity
                steps.append({
                    "order": i + 1,
                    "description": f"Execute optimization SQL {i+1}",
                    "sql": sql + ";",
                    "estimated_time_minutes": 15
                })
            
            return steps
        
        # Create generic steps if no SQL provided
        return [
            {
                "order": 1,
                "description": "Analyze current state",
                "sql": f"-- Examine current structure and data\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 10;",
                "estimated_time_minutes": 10
            },
            {
                "order": 2,
                "description": "Implement optimization",
                "sql": f"-- Implementation SQL would be generated here\n-- for optimization: {recommendation.get('description', '')}",
                "estimated_time_minutes": 30
            },
            {
                "order": 3,
                "description": "Verify successful implementation",
                "sql": f"-- Verify the implementation\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 10;",
                "estimated_time_minutes": 10
            }
        ]
    
    def _template_add_partitioning(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for adding table partitioning.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Extract partition field from recommendation
        partition_field = None
        partition_type = "DAY"
        
        # Try to extract from recommendation text
        rec_text = recommendation.get("recommendation", "")
        if rec_text:
            # Look for field name in quotes
            if "'" in rec_text:
                parts = rec_text.split("'")
                if len(parts) >= 3:
                    partition_field = parts[1]
            
            # Check for partition type
            if "MONTH" in rec_text.upper():
                partition_type = "MONTH"
            elif "HOUR" in rec_text.upper():
                partition_type = "HOUR"
            elif "DAY" in rec_text.upper():
                partition_type = "DAY"
        
        # Fallback for partition field
        if not partition_field:
            # Look for common timestamp fields
            partition_field = "created_at"
        
        return [
            {
                "order": 1,
                "description": "Create a backup of the original table",
                "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}`;",
                "estimated_time_minutes": 30
            },
            {
                "order": 2,
                "description": f"Create new table with {partition_type} partitioning",
                "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_partitioned`\nPARTITION BY {partition_type}({partition_field})\nAS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;",
                "estimated_time_minutes": 45
            },
            {
                "order": 3,
                "description": "Verify row count in the new partitioned table",
                "sql": f"SELECT COUNT(*) AS original_count FROM `{project_id}.{dataset_id}.{table_id}`;\nSELECT COUNT(*) AS new_count FROM `{project_id}.{dataset_id}.{table_id}_partitioned`;",
                "estimated_time_minutes": 15
            },
            {
                "order": 4,
                "description": "Rename tables to implement the change",
                "sql": f"ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;\nALTER TABLE `{project_id}.{dataset_id}.{table_id}_partitioned` RENAME TO `{project_id}.{dataset_id}.{table_id}`;",
                "estimated_time_minutes": 10
            },
            {
                "order": 5,
                "description": "Verify queries work with the new partitioned table",
                "sql": f"-- Run a sample query that uses partitioning\nSELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`\nWHERE {partition_field} >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY);",
                "estimated_time_minutes": 10
            }
        ]
    
    def _template_change_partition_granularity(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for changing partition granularity.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Extract current and target partition type
        current_type = "DAY"
        target_type = "MONTH"
        partition_field = None
        
        # Try to extract from current state
        if "partitioning" in current_state:
            partition_info = current_state["partitioning"]
            if isinstance(partition_info, str) and "field:" in partition_info:
                field_part = partition_info.split("field:")[1].split(",")[0].strip()
                partition_field = field_part
        
        # Try to extract from recommendation text
        rec_text = recommendation.get("recommendation", "")
        if "MONTH" in rec_text.upper():
            target_type = "MONTH"
        
        # Fallback for partition field
        if not partition_field:
            partition_field = "created_at"
        
        return [
            {
                "order": 1,
                "description": "Create a backup of the original table",
                "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}`;",
                "estimated_time_minutes": 30
            },
            {
                "order": 2,
                "description": f"Create new table with {target_type} partitioning",
                "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_new_partition`\nPARTITION BY {target_type}({partition_field})\nAS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;",
                "estimated_time_minutes": 45
            },
            {
                "order": 3,
                "description": "Verify row count in the new partitioned table",
                "sql": f"SELECT COUNT(*) AS original_count FROM `{project_id}.{dataset_id}.{table_id}`;\nSELECT COUNT(*) AS new_count FROM `{project_id}.{dataset_id}.{table_id}_new_partition`;",
                "estimated_time_minutes": 15
            },
            {
                "order": 4,
                "description": "Rename tables to implement the change",
                "sql": f"ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;\nALTER TABLE `{project_id}.{dataset_id}.{table_id}_new_partition` RENAME TO `{project_id}.{dataset_id}.{table_id}`;",
                "estimated_time_minutes": 10
            },
            {
                "order": 5,
                "description": "Verify queries work with the new partition granularity",
                "sql": f"-- Run a sample query that uses partitioning\nSELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`\nWHERE {partition_field} >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY);",
                "estimated_time_minutes": 10
            }
        ]
    
    def _template_add_clustering(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for adding clustering to a table.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Extract clustering fields
        clustering_fields = []
        rec_text = recommendation.get("recommendation", "")
        
        # Try to extract from recommendation text
        if "CLUSTER BY" in rec_text.upper():
            cluster_part = rec_text.split("CLUSTER BY")[1].strip()
            if "(" in cluster_part:
                fields_part = cluster_part.split("(")[1].split(")")[0]
                clustering_fields = [f.strip() for f in fields_part.split(",")]
            else:
                fields_part = cluster_part.split(" ")[0]
                clustering_fields = [fields_part.strip()]
        
        # Fallback for clustering fields
        if not clustering_fields:
            clustering_fields = ["status"]
        
        # Check if table is already partitioned
        is_partitioned = "partitioning" in current_state and current_state["partitioning"] != "None"
        partition_clause = ""
        
        if is_partitioned:
            partition_info = current_state["partitioning"]
            partition_type = "DAY"
            partition_field = None
            
            # Try to extract field and type
            if isinstance(partition_info, str):
                if "type:" in partition_info:
                    type_part = partition_info.split("type:")[1].split(",")[0].strip()
                    partition_type = type_part
                if "field:" in partition_info:
                    field_part = partition_info.split("field:")[1].split(",")[0].strip()
                    partition_field = field_part
            
            # Create partition clause if field was found
            if partition_field:
                partition_clause = f"PARTITION BY {partition_type}({partition_field})"
        
        # Format clustering fields
        clustering_fields_str = ", ".join([f"`{field}`" for field in clustering_fields])
        
        return [
            {
                "order": 1,
                "description": "Create a backup of the original table",
                "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}`;",
                "estimated_time_minutes": 30
            },
            {
                "order": 2,
                "description": "Create new table with clustering",
                "sql": f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_clustered`\n{partition_clause}\nCLUSTER BY {clustering_fields_str}\nAS SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;",
                "estimated_time_minutes": 45
            },
            {
                "order": 3,
                "description": "Verify row count in the new clustered table",
                "sql": f"SELECT COUNT(*) AS original_count FROM `{project_id}.{dataset_id}.{table_id}`;\nSELECT COUNT(*) AS new_count FROM `{project_id}.{dataset_id}.{table_id}_clustered`;",
                "estimated_time_minutes": 15
            },
            {
                "order": 4,
                "description": "Rename tables to implement the change",
                "sql": f"ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;\nALTER TABLE `{project_id}.{dataset_id}.{table_id}_clustered` RENAME TO `{project_id}.{dataset_id}.{table_id}`;",
                "estimated_time_minutes": 10
            },
            {
                "order": 5,
                "description": "Verify queries work with the clustered table",
                "sql": f"-- Run a sample query that uses clustering\nSELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`\nWHERE {clustering_fields[0]} = 'active';",
                "estimated_time_minutes": 10
            }
        ]
    
    def _template_move_to_lts(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for moving a table to long-term storage.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Check if table is already in archive dataset
        if "archive" in dataset_id.lower():
            # Table is already in an archive dataset, create partition strategy
            return [
                {
                    "order": 1,
                    "description": "Set time travel retention policy for long-term storage pricing",
                    "sql": f"ALTER TABLE `{project_id}.{dataset_id}.{table_id}`\nSET OPTIONS (\n  time_partitioning_expiration_days = NULL,\n  require_partition_filter = FALSE\n);",
                    "estimated_time_minutes": 5
                },
                {
                    "order": 2,
                    "description": "Verify table settings for long-term storage",
                    "sql": f"SELECT * FROM `{project_id}.{dataset_id}.__TABLES__` WHERE table_id = '{table_id}';",
                    "estimated_time_minutes": 5
                },
                {
                    "order": 3,
                    "description": "Document archive table metadata",
                    "sql": f"-- Document the archive table in a metadata table if exists\n-- REPLACE INTO `{project_id}.metadata.archive_tables` (table_id, dataset_id, archive_date, retention_period)\n-- VALUES ('{table_id}', '{dataset_id}', CURRENT_DATE(), 'INDEFINITE');",
                    "estimated_time_minutes": 5
                }
            ]
        else:
            # Move to archive dataset
            archive_dataset = f"{dataset_id}_archive"
            
            return [
                {
                    "order": 1,
                    "description": "Create archive dataset if it doesn't exist",
                    "sql": f"-- Create archive dataset if it doesn't exist\n-- This command should be run using the bq command-line tool or console\n-- bq mk --dataset --description=\"Archive dataset for {dataset_id}\" {project_id}:{archive_dataset}",
                    "estimated_time_minutes": 5
                },
                {
                    "order": 2,
                    "description": "Copy table to archive dataset",
                    "sql": f"CREATE OR REPLACE TABLE `{project_id}.{archive_dataset}.{table_id}` AS\nSELECT * FROM `{project_id}.{dataset_id}.{table_id}`;",
                    "estimated_time_minutes": 45
                },
                {
                    "order": 3,
                    "description": "Verify row count in the archive table",
                    "sql": f"SELECT COUNT(*) AS original_count FROM `{project_id}.{dataset_id}.{table_id}`;\nSELECT COUNT(*) AS archive_count FROM `{project_id}.{archive_dataset}.{table_id}`;",
                    "estimated_time_minutes": 15
                },
                {
                    "order": 4,
                    "description": "Add archive table description",
                    "sql": f"ALTER TABLE `{project_id}.{archive_dataset}.{table_id}`\nSET OPTIONS (\n  description = 'Archived from {dataset_id}.{table_id} on {datetime.now().strftime('%Y-%m-%d')}'\n);",
                    "estimated_time_minutes": 5
                },
                {
                    "order": 5,
                    "description": "Remove original table (only after verification)",
                    "sql": f"-- Verify that the archive table is correct before running this\n-- DROP TABLE `{project_id}.{dataset_id}.{table_id}`;",
                    "estimated_time_minutes": 5
                }
            ]
    
    def _template_optimize_select_star(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for optimizing SELECT * queries.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Get original query pattern if available
        original_query = current_state.get("query_pattern", f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`")
        
        # Get optimized query if available
        optimized_query = recommendation.get("implementation_sql", "")
        
        if not optimized_query:
            # Generate a sample optimized query
            optimized_query = f"-- Replace SELECT * with specific columns\nSELECT \n  id, \n  created_at, \n  status, \n  -- Add only necessary columns here\nFROM `{project_id}.{dataset_id}.{table_id}`"
            
            # Add WHERE clause if present in original
            if "WHERE" in original_query:
                where_clause = original_query.split("WHERE")[1]
                optimized_query += f"\nWHERE {where_clause}"
        
        return [
            {
                "order": 1,
                "description": "Identify queries using SELECT *",
                "sql": f"""-- Find queries that use SELECT *
SELECT
  query_text,
  total_bytes_processed,
  creation_time
FROM
  `{project_id}.region-us`.INFORMATION_SCHEMA.JOBS
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND query LIKE '%SELECT%*%FROM%{table_id}%'
  AND query NOT LIKE '%INFORMATION_SCHEMA%'
ORDER BY
  total_bytes_processed DESC
LIMIT 10;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 2,
                "description": "Optimize query by selecting specific columns",
                "sql": f"""-- Original query:
{original_query}

-- Optimized query:
{optimized_query}""",
                "estimated_time_minutes": 15
            },
            {
                "order": 3,
                "description": "Verify query performance improvement",
                "sql": f"""-- Measure performance of original vs optimized query
-- Execute and compare the bytes processed:

-- 1. Original query (comment out when testing)
-- {original_query}

-- 2. Optimized query
{optimized_query}""",
                "estimated_time_minutes": 10
            },
            {
                "order": 4,
                "description": "Update application or dashboard code",
                "sql": f"""-- Update the query in your application code, dashboard, or stored procedures
-- No SQL to execute here - this is a code change in your application

-- Document the changes made for future reference:
-- Changed: SELECT * query on {table_id}
-- File locations: [list application files that were updated]
-- Dashboards: [list dashboards that were updated]""",
                "estimated_time_minutes": 30
            }
        ]
    
    def _template_filter_pushdown(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for filter pushdown optimization.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Get original query pattern if available
        original_query = current_state.get("query_pattern", "")
        
        # Get optimized query if available
        optimized_query = recommendation.get("implementation_sql", "")
        
        if not original_query:
            original_query = f"""SELECT
  t.id,
  t.name,
  t.created_at
FROM `{project_id}.{dataset_id}.{table_id}` t
JOIN `{project_id}.{dataset_id}.other_table` o ON t.id = o.id
WHERE o.status = 'active'"""
        
        if not optimized_query:
            optimized_query = f"""-- Optimized query with filter pushdown
SELECT
  t.id,
  t.name,
  t.created_at
FROM `{project_id}.{dataset_id}.{table_id}` t
JOIN (
  SELECT id FROM `{project_id}.{dataset_id}.other_table`
  WHERE status = 'active'
) o ON t.id = o.id"""
        
        return [
            {
                "order": 1,
                "description": "Identify queries with filter pushdown opportunities",
                "sql": f"""-- Find queries that could benefit from filter pushdown
SELECT
  query_text,
  total_bytes_processed,
  creation_time
FROM
  `{project_id}.region-us`.INFORMATION_SCHEMA.JOBS
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND query LIKE '%JOIN%{table_id}%WHERE%'
  AND total_bytes_processed > 1000000000
ORDER BY
  total_bytes_processed DESC
LIMIT 10;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 2,
                "description": "Optimize query with filter pushdown",
                "sql": f"""-- Original query:
{original_query}

-- Optimized query:
{optimized_query}""",
                "estimated_time_minutes": 20
            },
            {
                "order": 3,
                "description": "Compare query performance",
                "sql": f"""-- Execute both queries and compare the bytes processed
-- First run with a small LIMIT to ensure functionality is unchanged

-- Original query:
{original_query} LIMIT 100;

-- Optimized query:
{optimized_query} LIMIT 100;

-- Then check the detailed execution information in the BigQuery UI
-- or using the INFORMATION_SCHEMA.JOBS table""",
                "estimated_time_minutes": 15
            },
            {
                "order": 4,
                "description": "Update application code with optimized query",
                "sql": "-- No SQL to execute - update the query in application code or stored procedures",
                "estimated_time_minutes": 30
            }
        ]
    
    def _template_optimize_join(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for join optimization.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Get original query if available
        original_query = current_state.get("query_pattern", "")
        
        # Get optimized query if available
        optimized_query = recommendation.get("implementation_sql", "")
        
        if not original_query:
            original_query = f"""SELECT
  t.id,
  t.created_at,
  t.status,
  l.log_data
FROM `{project_id}.{dataset_id}.{table_id}` t
LEFT JOIN `{project_id}.{dataset_id}.logs` l ON t.id = l.entity_id
WHERE t.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)"""
        
        if not optimized_query:
            optimized_query = f"""-- Optimized join
SELECT
  t.id,
  t.created_at,
  t.status,
  l.log_data
FROM `{project_id}.{dataset_id}.{table_id}` t
LEFT JOIN (
  -- Prefilter the right side of the join
  SELECT entity_id, log_data 
  FROM `{project_id}.{dataset_id}.logs`
  WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
) l ON t.id = l.entity_id
WHERE t.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)"""
        
        return [
            {
                "order": 1,
                "description": "Identify inefficient join queries",
                "sql": f"""-- Find expensive join queries
SELECT
  query_text,
  total_bytes_processed,
  creation_time
FROM
  `{project_id}.region-us`.INFORMATION_SCHEMA.JOBS
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND query LIKE '%JOIN%{table_id}%'
  AND total_bytes_processed > 1000000000
ORDER BY
  total_bytes_processed DESC
LIMIT 10;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 2,
                "description": "Analyze current join strategy",
                "sql": f"""-- Original query:
{original_query}

-- Add EXPLAIN to examine the execution plan
EXPLAIN {original_query}""",
                "estimated_time_minutes": 15
            },
            {
                "order": 3,
                "description": "Implement optimized join",
                "sql": f"""-- Optimized join query:
{optimized_query}

-- Add EXPLAIN to examine the new execution plan
EXPLAIN {optimized_query}""",
                "estimated_time_minutes": 20
            },
            {
                "order": 4,
                "description": "Compare performance and verify results",
                "sql": f"""-- Run both queries with a small LIMIT and verify results match
-- Original query:
{original_query} LIMIT 100;

-- Optimized query:
{optimized_query} LIMIT 100;

-- Execute full queries and compare performance using BigQuery UI
-- or INFORMATION_SCHEMA.JOBS table""",
                "estimated_time_minutes": 15
            },
            {
                "order": 5,
                "description": "Update application code with optimized join",
                "sql": "-- No SQL to execute - update the query in application code or stored procedures",
                "estimated_time_minutes": 30
            }
        ]
    
    def _template_add_partition_filter(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for adding partition filters.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Get original query if available
        original_query = current_state.get("query_pattern", "")
        
        # Get optimized query if available
        optimized_query = recommendation.get("implementation_sql", "")
        
        # Get partition field if available
        partition_field = "created_at"  # Default
        if "partitioning" in current_state:
            partition_info = current_state["partitioning"]
            if isinstance(partition_info, str) and "field:" in partition_info:
                field_part = partition_info.split("field:")[1].split(",")[0].strip()
                partition_field = field_part
        
        if not original_query:
            original_query = f"""SELECT
  id,
  {partition_field},
  status,
  value
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE status = 'active'"""
        
        if not optimized_query:
            optimized_query = f"""-- Query with partition filter added
SELECT
  id,
  {partition_field},
  status,
  value
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE 
  status = 'active'
  AND {partition_field} >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)"""
        
        return [
            {
                "order": 1,
                "description": "Identify queries missing partition filters",
                "sql": f"""-- Find queries that don't use partition filters
SELECT
  query_text,
  total_bytes_processed,
  total_slot_ms,
  creation_time
FROM
  `{project_id}.region-us`.INFORMATION_SCHEMA.JOBS
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND query LIKE '%FROM%{table_id}%WHERE%'
  AND query NOT LIKE '%{partition_field}%'
  AND total_bytes_processed > 1000000000
ORDER BY
  total_bytes_processed DESC
LIMIT 10;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 2,
                "description": "Analyze table partitioning",
                "sql": f"""-- Get table partitioning information
SELECT
  table_name,
  type,
  time_partitioning_field
FROM
  `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
WHERE table_name = '{table_id}';

-- Look at partition distribution
SELECT
  _PARTITIONDATE as partition_date,
  COUNT(*) as row_count
FROM `{project_id}.{dataset_id}.{table_id}`
GROUP BY 1
ORDER BY 1 DESC
LIMIT 100;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 3,
                "description": "Add partition filter to queries",
                "sql": f"""-- Original query without partition filter:
{original_query}

-- Optimized query with partition filter:
{optimized_query}""",
                "estimated_time_minutes": 15
            },
            {
                "order": 4,
                "description": "Compare query performance",
                "sql": f"""-- Execute both queries and compare performance
-- Note: this step is just for demonstration; BigQuery will show the performance metrics

-- 1. Original query (high cost):
-- {original_query}

-- 2. Optimized query (should be much cheaper):
{optimized_query}

-- Check the slots and bytes processed in the BigQuery UI or job history""",
                "estimated_time_minutes": 15
            },
            {
                "order": 5,
                "description": "Update application code with partition filters",
                "sql": "-- No SQL to execute - update the query in application code, dashboards, or stored procedures",
                "estimated_time_minutes": 30
            }
        ]
    
    def _template_change_datatype(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for changing column data types.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Get column information
        column_name = current_state.get("column_name", "unknown_column")
        current_type = current_state.get("current_type", "STRING")
        target_type = current_state.get("recommended_type", "INT64")
        
        # Get implementation SQL if available
        implementation_sql = recommendation.get("implementation_sql", "")
        
        if not implementation_sql:
            # Generate basic implementation SQL
            cast_expr = f"CAST({column_name} AS {target_type})"
            
            if current_type == "STRING" and target_type == "BOOL":
                cast_expr = f"""CASE 
    WHEN LOWER({column_name}) IN ('true', 't', 'yes', 'y', '1') THEN TRUE
    WHEN LOWER({column_name}) IN ('false', 'f', 'no', 'n', '0') THEN FALSE
    ELSE NULL
END"""
            elif current_type == "STRING" and target_type == "DATE":
                cast_expr = f"PARSE_DATE('%Y-%m-%d', {column_name})"
            elif current_type == "STRING" and target_type == "TIMESTAMP":
                cast_expr = f"PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', {column_name})"
            elif current_type == "TIMESTAMP" and target_type == "DATE":
                cast_expr = f"DATE({column_name})"
            
            implementation_sql = f"""-- Create new table with modified column type
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_modified` AS
SELECT
  -- For each column in the table, if it's the target column, cast it
  -- This is placeholder SQL, you'll need to list all columns
  {cast_expr} AS {column_name},
  -- List all other columns here
  id,  -- Example, replace with actual columns
  name,
  created_at
FROM `{project_id}.{dataset_id}.{table_id}`;"""
        
        return [
            {
                "order": 1,
                "description": "Analyze current data in the column",
                "sql": f"""-- Check column data distribution
SELECT
  {column_name},
  COUNT(*) as count
FROM
  `{project_id}.{dataset_id}.{table_id}`
GROUP BY 1
ORDER BY 2 DESC
LIMIT 100;

-- Check for potential conversion issues
SELECT
  {column_name},
  COUNT(*) as error_count
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE SAFE_CAST({column_name} AS {target_type}) IS NULL
  AND {column_name} IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC
LIMIT 100;""",
                "estimated_time_minutes": 15
            },
            {
                "order": 2,
                "description": "Create backup of original table",
                "sql": f"""-- Create backup before making changes
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;""",
                "estimated_time_minutes": 20
            },
            {
                "order": 3,
                "description": f"Change data type of column {column_name} from {current_type} to {target_type}",
                "sql": implementation_sql,
                "estimated_time_minutes": 30
            },
            {
                "order": 4,
                "description": "Verify row counts match",
                "sql": f"""-- Verify row counts
SELECT
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_modified`) AS new_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_modified`) AS counts_match;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 5,
                "description": "Verify null percentages match",
                "sql": f"""-- Verify null percentages
SELECT
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}` WHERE {column_name} IS NULL) / 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_null_pct,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_modified` WHERE {column_name} IS NULL) / 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_modified`) AS new_null_pct;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 6,
                "description": "Swap tables to implement the change",
                "sql": f"""-- Swap tables to implement the change
-- IMPORTANT: Only run these after verifying the modified table is correct
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_modified` RENAME TO `{project_id}.{dataset_id}.{table_id}`;""",
                "estimated_time_minutes": 5
            }
        ]
    
    def _template_string_to_enum(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for converting string to enum.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Get column information
        column_name = current_state.get("column_name", "status")
        
        # Get implementation SQL if available
        implementation_sql = recommendation.get("implementation_sql", "")
        
        if not implementation_sql:
            implementation_sql = f"""-- Create ENUM type for the column
CREATE TYPE IF NOT EXISTS `{project_id}.{dataset_id}.{column_name}_enum` AS ENUM (
  -- Add your enum values here based on analyzing the column values
  'value1', 'value2', 'value3'
);

-- Create new table with ENUM type
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_enum` AS
SELECT
  -- Cast the string column to the ENUM type
  CAST({column_name} AS `{project_id}.{dataset_id}.{column_name}_enum`) AS {column_name},
  -- Include all other columns
  -- List all other columns here
  id,  -- Example, replace with actual columns
  name,
  created_at
FROM `{project_id}.{dataset_id}.{table_id}`;"""
        
        return [
            {
                "order": 1,
                "description": "Analyze current string values",
                "sql": f"""-- Check distinct values and their distribution
SELECT
  {column_name},
  COUNT(*) as count
FROM
  `{project_id}.{dataset_id}.{table_id}`
GROUP BY 1
ORDER BY 2 DESC;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 2,
                "description": "Create backup of original table",
                "sql": f"""-- Create backup before making changes
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;""",
                "estimated_time_minutes": 20
            },
            {
                "order": 3,
                "description": "Create ENUM type based on analysis",
                "sql": f"""-- Create ENUM type for {column_name} column
-- Note: Replace the enum values with the actual values from your analysis
CREATE OR REPLACE TYPE `{project_id}.{dataset_id}.{column_name}_enum` AS ENUM (
  -- Add the enum values from your analysis
  'value1', 'value2', 'value3'
);""",
                "estimated_time_minutes": 5
            },
            {
                "order": 4,
                "description": f"Create new table with {column_name} as ENUM type",
                "sql": implementation_sql,
                "estimated_time_minutes": 30
            },
            {
                "order": 5,
                "description": "Verify data conversion",
                "sql": f"""-- Verify row counts
SELECT
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_enum`) AS new_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_enum`) AS counts_match;

-- Verify value distribution
SELECT
  e.{column_name},
  COUNT(*) as count
FROM
  `{project_id}.{dataset_id}.{table_id}_enum` e
GROUP BY 1
ORDER BY 2 DESC;""",
                "estimated_time_minutes": 15
            },
            {
                "order": 6,
                "description": "Swap tables to implement the change",
                "sql": f"""-- Swap tables to implement the change
-- IMPORTANT: Only run these after verifying the modified table is correct
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_enum` RENAME TO `{project_id}.{dataset_id}.{table_id}`;""",
                "estimated_time_minutes": 5
            },
            {
                "order": 7,
                "description": "Update application code to handle ENUM type",
                "sql": """-- No SQL to execute - update the application code
-- This may involve changing any code that inserts or updates this column
-- to ensure it uses the correct ENUM values""",
                "estimated_time_minutes": 60
            }
        ]
    
    def _template_normalize_repeated_field(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for normalizing repeated fields.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Get repeated field information
        repeated_fields = []
        if "repeated_fields" in current_state:
            if isinstance(current_state["repeated_fields"], list):
                repeated_fields = current_state["repeated_fields"]
            else:
                repeated_fields = [current_state["repeated_fields"]]
        
        # If no repeated fields found, use a generic name
        if not repeated_fields:
            repeated_fields = ["items"]
        
        repeated_field = repeated_fields[0]  # Use the first repeated field
        
        # Get implementation SQL if available
        implementation_sql = recommendation.get("implementation_sql", "")
        
        if not implementation_sql:
            # Generate generic implementation SQL
            implementation_sql = f"""-- Create child table for the repeated field
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_{repeated_field}` AS
SELECT
  parent.id AS parent_id,
  child AS {repeated_field}_item,
  ROW_NUMBER() OVER(PARTITION BY parent.id) AS {repeated_field}_index
FROM
  `{project_id}.{dataset_id}.{table_id}` parent,
  UNNEST(parent.{repeated_field}) AS child;

-- Create parent table without the repeated field
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_normalized` AS
SELECT
  id,
  -- List all other non-repeated fields here
  -- Exclude the {repeated_field} field
  created_at,
  updated_at,
  name,
  description
FROM `{project_id}.{dataset_id}.{table_id}`;"""
        
        return [
            {
                "order": 1,
                "description": "Analyze the repeated field structure",
                "sql": f"""-- Check cardinality of repeated field
SELECT
  COUNT(*) as total_rows,
  AVG(ARRAY_LENGTH({repeated_field})) as avg_items_per_row,
  MAX(ARRAY_LENGTH({repeated_field})) as max_items_per_row
FROM `{project_id}.{dataset_id}.{table_id}`;

-- Examine repeated field structure
SELECT
  id,
  {repeated_field}
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE {repeated_field} IS NOT NULL
LIMIT 10;""",
                "estimated_time_minutes": 15
            },
            {
                "order": 2,
                "description": "Create backup of original table",
                "sql": f"""-- Create backup before making changes
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;""",
                "estimated_time_minutes": 20
            },
            {
                "order": 3,
                "description": f"Normalize the repeated field {repeated_field}",
                "sql": implementation_sql,
                "estimated_time_minutes": 30
            },
            {
                "order": 4,
                "description": "Verify data normalization",
                "sql": f"""-- Verify row counts in parent table
SELECT
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_normalized`) AS normalized_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_normalized`) AS counts_match;

-- Verify item counts in child table
SELECT
  (SELECT SUM(ARRAY_LENGTH({repeated_field})) FROM `{project_id}.{dataset_id}.{table_id}` WHERE {repeated_field} IS NOT NULL) AS original_items,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_{repeated_field}`) AS normalized_items;

-- Check join functionality
SELECT
  p.id,
  COUNT(c.{repeated_field}_item) as item_count
FROM 
  `{project_id}.{dataset_id}.{table_id}_normalized` p
LEFT JOIN
  `{project_id}.{dataset_id}.{table_id}_{repeated_field}` c
ON p.id = c.parent_id
GROUP BY 1
LIMIT 10;""",
                "estimated_time_minutes": 20
            },
            {
                "order": 5,
                "description": "Create view for backward compatibility",
                "sql": f"""-- Create a backward-compatible view that reconstructs the original structure
CREATE OR REPLACE VIEW `{project_id}.{dataset_id}.{table_id}_view` AS
SELECT
  p.*,
  ARRAY_AGG(c.{repeated_field}_item ORDER BY c.{repeated_field}_index) AS {repeated_field}
FROM
  `{project_id}.{dataset_id}.{table_id}_normalized` p
LEFT JOIN
  `{project_id}.{dataset_id}.{table_id}_{repeated_field}` c
ON p.id = c.parent_id
GROUP BY
  p.id,
  -- Include all other columns from the parent table
  p.created_at,
  p.updated_at,
  p.name,
  p.description;""",
                "estimated_time_minutes": 15
            },
            {
                "order": 6,
                "description": "Swap tables to implement the change",
                "sql": f"""-- Swap tables to implement the change
-- IMPORTANT: Only run these after thorough testing
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_normalized` RENAME TO `{project_id}.{dataset_id}.{table_id}`;
-- ALTER VIEW `{project_id}.{dataset_id}.{table_id}_view` RENAME TO `{project_id}.{dataset_id}.{table_id}_original`;""",
                "estimated_time_minutes": 5
            },
            {
                "order": 7,
                "description": "Update application code for new schema",
                "sql": """-- No SQL to execute - update application code
-- This is a significant schema change that requires application updates
-- Consider using the compatibility view during the transition period""",
                "estimated_time_minutes": 120
            }
        ]
    
    def _template_remove_columns(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps for removing unused columns.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            List of implementation steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        current_state = recommendation.get("current_state", {})
        
        # Get columns to remove
        columns_to_remove = []
        if "columns" in current_state:
            if isinstance(current_state["columns"], list):
                columns_to_remove = current_state["columns"]
            else:
                columns_to_remove = [current_state["columns"]]
        
        # Get implementation SQL if available
        implementation_sql = recommendation.get("implementation_sql", "")
        
        if not implementation_sql and columns_to_remove:
            # Generate SQL to remove columns
            columns_str = ", ".join([f"'{col}'" for col in columns_to_remove])
            implementation_sql = f"""-- Create new table without the unused columns
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_optimized` AS
SELECT
  * EXCEPT({', '.join(columns_to_remove)})
FROM `{project_id}.{dataset_id}.{table_id}`;"""
        
        columns_list = ", ".join(columns_to_remove)
        
        return [
            {
                "order": 1,
                "description": "Analyze column usage in queries",
                "sql": f"""-- Check if columns are used in recent queries
SELECT
  query_text,
  creation_time
FROM
  `{project_id}.region-us`.INFORMATION_SCHEMA.JOBS
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  AND query LIKE '%{table_id}%'
  AND (
    {" OR ".join([f"LOWER(query) LIKE '%{col.lower()}%'" for col in columns_to_remove])}
  )
ORDER BY
  creation_time DESC
LIMIT 20;""",
                "estimated_time_minutes": 20
            },
            {
                "order": 2,
                "description": "Check null percentage or distribution of values",
                "sql": f"""-- Check if columns are mostly NULL or contain meaningless values
SELECT
  COUNT(*) as total_rows,
  {", ".join([f"COUNTIF({col} IS NULL) AS {col}_null_count" for col in columns_to_remove])},
  {", ".join([f"COUNTIF({col} IS NULL) / COUNT(*) AS {col}_null_pct" for col in columns_to_remove])}
FROM `{project_id}.{dataset_id}.{table_id}`;""",
                "estimated_time_minutes": 15
            },
            {
                "order": 3,
                "description": "Create backup of original table",
                "sql": f"""-- Create backup before making changes
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}_backup` AS
SELECT * FROM `{project_id}.{dataset_id}.{table_id}`;""",
                "estimated_time_minutes": 20
            },
            {
                "order": 4,
                "description": f"Remove unused columns: {columns_list}",
                "sql": implementation_sql,
                "estimated_time_minutes": 30
            },
            {
                "order": 5,
                "description": "Verify row counts match",
                "sql": f"""-- Verify row counts
SELECT
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) AS original_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_optimized`) AS new_count,
  (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}`) = 
    (SELECT COUNT(*) FROM `{project_id}.{dataset_id}.{table_id}_optimized`) AS counts_match;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 6,
                "description": "Create backward compatibility view (optional)",
                "sql": f"""-- Create a backward-compatible view with NULL values for removed columns
CREATE OR REPLACE VIEW `{project_id}.{dataset_id}.{table_id}_full` AS
SELECT
  *,
  {", ".join([f"NULL AS {col}" for col in columns_to_remove])}
FROM `{project_id}.{dataset_id}.{table_id}_optimized`;""",
                "estimated_time_minutes": 10
            },
            {
                "order": 7,
                "description": "Swap tables to implement the change",
                "sql": f"""-- Swap tables to implement the change
-- IMPORTANT: Only run these after verifying no code depends on the columns
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}` RENAME TO `{project_id}.{dataset_id}.{table_id}_old`;
-- ALTER TABLE `{project_id}.{dataset_id}.{table_id}_optimized` RENAME TO `{project_id}.{dataset_id}.{table_id}`;""",
                "estimated_time_minutes": 5
            }
        ]
    
    def _generate_verification_steps(self, recommendation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate verification steps for a recommendation.
        
        Args:
            recommendation: The recommendation to verify
            
        Returns:
            List of verification steps
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        category = recommendation.get("category", "")
        rec_type = recommendation.get("type", "")
        
        verification_steps = []
        
        # Data integrity verification
        verification_steps.append({
            "description": "Verify data integrity",
            "details": "Ensure all data is preserved correctly after implementation",
            "sql": f"SELECT COUNT(*) AS row_count FROM `{project_id}.{dataset_id}.{table_id}`;"
        })
        
        # Category-specific verification
        if category == "storage":
            if "partitioning" in rec_type:
                verification_steps.append({
                    "description": "Verify partitioning configuration",
                    "details": "Confirm that partitioning is correctly applied",
                    "sql": f"""SELECT
  table_name,
  time_partitioning_type,
  time_partitioning_field
FROM
  `{dataset_id}.INFORMATION_SCHEMA.TABLES`
WHERE table_name = '{table_id}';"""
                })
                
                verification_steps.append({
                    "description": "Test partition filtering",
                    "details": "Verify queries correctly use partition filters",
                    "sql": f"""-- Check query performance with partition filter
-- Run a sample query with partition filter and verify reduced bytes processed"""
                })
            
            elif "clustering" in rec_type:
                verification_steps.append({
                    "description": "Verify clustering configuration",
                    "details": "Confirm that clustering is correctly applied",
                    "sql": f"""SELECT
  table_name,
  clustering_ordinal_position,
  clustering_column_name
FROM
  `{dataset_id}.INFORMATION_SCHEMA.CLUSTERING_COLUMNS`
WHERE table_name = '{table_id}'
ORDER BY clustering_ordinal_position;"""
                })
                
                verification_steps.append({
                    "description": "Test clustering effectiveness",
                    "details": "Verify queries correctly benefit from clustering",
                    "sql": f"""-- Check query performance with clustering columns
-- Run a sample query using clustering columns and verify reduced bytes processed"""
                })
        
        elif category == "schema":
            if "datatype" in rec_type:
                verification_steps.append({
                    "description": "Verify column data type",
                    "details": "Confirm that the column data type was correctly changed",
                    "sql": f"""SELECT
  column_name,
  data_type
FROM
  `{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '{table_id}'
ORDER BY ordinal_position;"""
                })
            
            elif "remove_unused" in rec_type:
                verification_steps.append({
                    "description": "Verify column removal",
                    "details": "Confirm that unused columns were successfully removed",
                    "sql": f"""SELECT
  column_name
FROM
  `{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '{table_id}'
ORDER BY ordinal_position;"""
                })
        
        # Add performance verification for all recommendations
        verification_steps.append({
            "description": "Verify performance improvement",
            "details": "Measure query performance and resource usage before and after implementation",
            "sql": f"""-- Compare query performance metrics
-- Check job statistics in BigQuery UI or use INFORMATION_SCHEMA.JOBS to compare:
-- - Bytes processed
-- - Slot milliseconds
-- - Query execution time"""
        })
        
        return verification_steps
    
    def _generate_rollback_procedure(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rollback procedure for a recommendation.
        
        Args:
            recommendation: The recommendation to implement
            
        Returns:
            Dict with rollback procedure
        """
        project_id = recommendation.get("project_id", "")
        dataset_id = recommendation.get("dataset_id", "")
        table_id = recommendation.get("table_id", "")
        category = recommendation.get("category", "")
        
        # Basic rollback procedure
        rollback = {
            "description": "Rollback procedure if implementation causes issues",
            "steps": [
                {
                    "description": "Restore from backup",
                    "sql": f"""-- Option 1: Restore from backup table (if available)
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{table_id}`;
ALTER TABLE `{project_id}.{dataset_id}.{table_id}_backup` RENAME TO `{project_id}.{dataset_id}.{table_id}`;"""
                },
                {
                    "description": "Restore from renamed original",
                    "sql": f"""-- Option 2: Restore from renamed original table (if available)
DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{table_id}`;
ALTER TABLE `{project_id}.{dataset_id}.{table_id}_old` RENAME TO `{project_id}.{dataset_id}.{table_id}`;"""
                }
            ],
            "impact_assessment": "Rolling back will lose any new data written to the optimized table since implementation",
            "safety_measures": [
                "Create a full backup before implementation",
                "Test thoroughly in a non-production environment first",
                "Monitor query patterns and performance after implementation",
                "Keep original table for at least one week before removing"
            ]
        }
        
        # Add category-specific rollback guidance
        if category == "query":
            rollback["steps"] = [
                {
                    "description": "Revert to original query",
                    "sql": "-- Simply revert to using the original query in your application code or stored procedures"
                }
            ]
            rollback["impact_assessment"] = "Reverting query optimizations will increase query costs but has no data impact"
        
        return rollback


def generate_implementation_plan(recommendations: List[Dict[str, Any]], 
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Standalone function to generate an implementation plan for backward compatibility.
    
    Args:
        recommendations: List of recommendations
        metadata: Dataset metadata
        
    Returns:
        Implementation plan
    """
    generator = ImplementationPlanGenerator()
    return generator.generate_plan(recommendations)