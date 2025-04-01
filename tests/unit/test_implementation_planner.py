"""Unit tests for the BigQuery Implementation Plan Generator module."""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
from datetime import datetime, timedelta

from bigquerycostopt.src.implementation.planner import ImplementationPlanGenerator

# Sample recommendations for testing
SAMPLE_STORAGE_REC = {
    "recommendation_id": "STORAGE_001",
    "category": "storage",
    "type": "partitioning_add",
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "description": "Add partitioning to table",
    "recommendation": "Partition table by 'created_at' field",
    "rationale": "Table is large and frequently queried by date",
    "annual_savings_usd": 120.0,
    "implementation_cost_usd": 24.0,
    "estimated_effort": "medium",
    "priority": "high",
    "current_state": {
        "partitioning": "None"
    }
}

SAMPLE_QUERY_REC = {
    "recommendation_id": "QUERY_001",
    "category": "query",
    "type": "query_select_star",
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "description": "Optimize SELECT * query",
    "recommendation": "Replace SELECT * with specific columns",
    "rationale": "Current query scans unnecessary columns",
    "annual_savings_usd": 180.0,
    "implementation_cost_usd": 12.0,
    "estimated_effort": "low",
    "priority": "high",
    "current_state": {
        "query_pattern": "SELECT * FROM test_dataset.test_table",
        "query_count": 120
    }
}

SAMPLE_SCHEMA_REC = {
    "recommendation_id": "SCHEMA_001",
    "category": "schema",
    "type": "datatype_float_to_int",
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "description": "Convert 'price' column from FLOAT64 to INT64",
    "recommendation": "Convert 'price' column from FLOAT64 to INT64",
    "rationale": "Column contains only integer values",
    "annual_savings_usd": 60.0,
    "implementation_cost_usd": 24.0,
    "estimated_effort": "medium",
    "priority": "medium",
    "current_state": {
        "column_name": "price",
        "current_type": "FLOAT64",
        "recommended_type": "INT64"
    }
}

SAMPLE_DEPENDENT_REC = {
    "recommendation_id": "SCHEMA_002",
    "category": "schema",
    "type": "normalize_repeated_field",
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "description": "Normalize 'tags' repeated field",
    "recommendation": "Normalize 'tags' repeated field into a separate table",
    "rationale": "Repeated field increases table size",
    "annual_savings_usd": 90.0,
    "implementation_cost_usd": 60.0,
    "estimated_effort": "high",
    "priority": "low",
    "current_state": {
        "repeated_fields": ["tags"]
    },
    "depends_on": ["STORAGE_001"]
}


class TestImplementationPlanGenerator(unittest.TestCase):
    """Test cases for the Implementation Plan Generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.planner = ImplementationPlanGenerator()
        
        # Create a list of test recommendations
        self.test_recommendations = [
            SAMPLE_STORAGE_REC,
            SAMPLE_QUERY_REC,
            SAMPLE_SCHEMA_REC,
            SAMPLE_DEPENDENT_REC
        ]
    
    def test_generate_plan(self):
        """Test generating a complete implementation plan."""
        # Generate plan
        plan = self.planner.generate_plan(self.test_recommendations)
        
        # Verify plan structure
        self.assertIn("plan_generated", plan)
        self.assertEqual(plan["total_recommendations"], 4)
        self.assertIn("implementation_time_days", plan)
        self.assertIn("estimated_implementation_cost_usd", plan)
        self.assertIn("estimated_annual_savings_usd", plan)
        self.assertIn("roi", plan)
        self.assertIn("phases", plan)
        
        # Verify phases
        self.assertGreaterEqual(len(plan["phases"]), 1)
        
        # Verify total cost and savings
        self.assertEqual(plan["estimated_implementation_cost_usd"], 120.0)  # Sum of all implementation costs
        self.assertEqual(plan["estimated_annual_savings_usd"], 450.0)  # Sum of all annual savings
        
        # Verify ROI calculation
        expected_roi = 450.0 / 120.0
        self.assertAlmostEqual(plan["roi"], expected_roi)
    
    def test_build_dependency_graph(self):
        """Test building the dependency graph."""
        # Build dependency graph
        dependency_graph = self.planner._build_dependency_graph(self.test_recommendations)
        
        # Verify explicit dependencies
        self.assertIn("SCHEMA_002", dependency_graph)
        self.assertIn("STORAGE_001", dependency_graph["SCHEMA_002"])
        
        # Verify implicit dependencies for schema changes
        self.assertIn("SCHEMA_001", dependency_graph)
        self.assertIn("STORAGE_001", dependency_graph["SCHEMA_001"])
    
    def test_group_recommendations(self):
        """Test grouping recommendations by table and category."""
        # Group recommendations
        grouped_recs, rec_by_id = self.planner._group_recommendations(self.test_recommendations)
        
        # Verify grouping
        self.assertIn("test_table", grouped_recs)
        
        # Verify categories
        table_groups = grouped_recs["test_table"]
        self.assertEqual(len(table_groups["storage"]), 1)
        self.assertEqual(len(table_groups["query"]), 1)
        self.assertEqual(len(table_groups["schema"]), 2)
        
        # Verify lookup by ID
        self.assertEqual(len(rec_by_id), 4)
        self.assertIn("STORAGE_001", rec_by_id)
        self.assertIn("QUERY_001", rec_by_id)
        self.assertIn("SCHEMA_001", rec_by_id)
        self.assertIn("SCHEMA_002", rec_by_id)
    
    def test_create_implementation_phases(self):
        """Test creating implementation phases."""
        # Group recommendations first
        grouped_recs, rec_by_id = self.planner._group_recommendations(self.test_recommendations)
        
        # Build dependency graph
        dependency_graph = self.planner._build_dependency_graph(self.test_recommendations)
        
        # Create phases
        phases = self.planner._create_implementation_phases(grouped_recs, dependency_graph, rec_by_id)
        
        # Verify phases
        self.assertGreaterEqual(len(phases), 3)  # Should have at least 3 phases
        
        # Verify phase 1 contains high priority independent recommendations
        phase1 = phases[0]
        self.assertEqual(phase1["name"], "Phase 1: Critical Optimizations")
        
        # Verify phase order - query optimization should be in phase 1 (high priority, no dependencies)
        phase1_rec_ids = [step["recommendation_id"] for step in phase1["steps"]]
        self.assertIn("QUERY_001", phase1_rec_ids)
        
        # Dependent recommendation should be in a later phase
        last_phase = phases[-1]
        last_phase_rec_ids = [step["recommendation_id"] for step in last_phase["steps"]]
        self.assertIn("SCHEMA_002", last_phase_rec_ids)
    
    def test_get_implementation_template(self):
        """Test template selection for different recommendation types."""
        # Test storage recommendation
        storage_template = self.planner._get_implementation_template(SAMPLE_STORAGE_REC)
        self.assertEqual(storage_template, self.planner._template_add_partitioning)
        
        # Test query recommendation
        query_template = self.planner._get_implementation_template(SAMPLE_QUERY_REC)
        self.assertEqual(query_template, self.planner._template_optimize_select_star)
        
        # Test schema recommendation
        schema_template = self.planner._get_implementation_template(SAMPLE_SCHEMA_REC)
        self.assertEqual(schema_template, self.planner._template_change_datatype)
        
        # Test recommendation with no specific template
        unknown_rec = {
            "recommendation_id": "UNKNOWN_001",
            "category": "unknown",
            "type": "unknown_type"
        }
        unknown_template = self.planner._get_implementation_template(unknown_rec)
        self.assertEqual(unknown_template, self.planner._template_generic)
    
    def test_template_add_partitioning(self):
        """Test implementation template for adding partitioning."""
        # Generate implementation steps
        steps = self.planner._template_add_partitioning(SAMPLE_STORAGE_REC)
        
        # Verify steps
        self.assertGreaterEqual(len(steps), 3)  # Should have at least 3 steps
        
        # Verify step structure
        for step in steps:
            self.assertIn("order", step)
            self.assertIn("description", step)
            self.assertIn("sql", step)
            self.assertIn("estimated_time_minutes", step)
        
        # Verify SQL content
        self.assertIn("CREATE OR REPLACE TABLE", steps[0]["sql"])
        self.assertIn("PARTITION BY", steps[1]["sql"])
        self.assertIn("RENAME TO", steps[3]["sql"])
    
    def test_template_optimize_select_star(self):
        """Test implementation template for optimizing SELECT * queries."""
        # Generate implementation steps
        steps = self.planner._template_optimize_select_star(SAMPLE_QUERY_REC)
        
        # Verify steps
        self.assertGreaterEqual(len(steps), 3)
        
        # Verify SQL content
        self.assertIn("SELECT * FROM", steps[1]["sql"])  # Original query
        self.assertIn("-- Optimized query", steps[1]["sql"])  # Optimized query
    
    def test_template_change_datatype(self):
        """Test implementation template for changing column data types."""
        # Generate implementation steps
        steps = self.planner._template_change_datatype(SAMPLE_SCHEMA_REC)
        
        # Verify steps
        self.assertGreaterEqual(len(steps), 4)
        
        # Verify SQL content for data type analysis
        self.assertIn("Check column data distribution", steps[0]["description"])
        self.assertIn("SAFE_CAST", steps[0]["sql"])
        
        # Verify SQL content for backup
        self.assertIn("Create backup", steps[1]["description"])
        
        # Verify SQL content for type change
        self.assertIn("Change data type", steps[2]["description"])
        self.assertIn("INT64", steps[2]["sql"])
    
    def test_generate_verification_steps(self):
        """Test generation of verification steps."""
        # Generate verification for storage recommendation
        storage_verification = self.planner._generate_verification_steps(SAMPLE_STORAGE_REC)
        
        # Verify structure
        self.assertGreaterEqual(len(storage_verification), 2)
        for step in storage_verification:
            self.assertIn("description", step)
            self.assertIn("details", step)
            self.assertIn("sql", step)
        
        # Verify partitioning-specific verification
        partition_verification = next((step for step in storage_verification if "partition" in step["description"].lower()), None)
        self.assertIsNotNone(partition_verification)
        self.assertIn("time_partitioning", partition_verification["sql"].lower())
        
        # Generate verification for schema recommendation
        schema_verification = self.planner._generate_verification_steps(SAMPLE_SCHEMA_REC)
        
        # Verify schema-specific verification
        datatype_verification = next((step for step in schema_verification if "data type" in step["description"].lower()), None)
        self.assertIsNotNone(datatype_verification)
        self.assertIn("data_type", datatype_verification["sql"].lower())
    
    def test_generate_rollback_procedure(self):
        """Test generation of rollback procedures."""
        # Generate rollback for storage recommendation
        storage_rollback = self.planner._generate_rollback_procedure(SAMPLE_STORAGE_REC)
        
        # Verify structure
        self.assertIn("description", storage_rollback)
        self.assertIn("steps", storage_rollback)
        self.assertIn("impact_assessment", storage_rollback)
        self.assertIn("safety_measures", storage_rollback)
        
        # Verify steps
        self.assertGreaterEqual(len(storage_rollback["steps"]), 1)
        for step in storage_rollback["steps"]:
            self.assertIn("description", step)
            self.assertIn("sql", step)
        
        # Generate rollback for query recommendation
        query_rollback = self.planner._generate_rollback_procedure(SAMPLE_QUERY_REC)
        
        # Verify query-specific rollback
        self.assertEqual(len(query_rollback["steps"]), 1)
        self.assertIn("Revert to original query", query_rollback["steps"][0]["description"])
    
    def test_backwards_compatibility(self):
        """Test backwards compatibility with standalone function."""
        # Create metadata for backward compatibility
        metadata = {
            "dataset_id": "test_dataset",
            "project_id": "test-project",
            "tables": [
                {
                    "table_id": "test_table",
                    "schema": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "price", "type": "FLOAT64"}
                    ]
                }
            ]
        }
        
        # Use standalone function
        from bigquerycostopt.src.implementation.planner import generate_implementation_plan
        plan = generate_implementation_plan(self.test_recommendations, metadata)
        
        # Verify that it still works
        self.assertIn("phases", plan)
        self.assertEqual(plan["total_recommendations"], 4)


if __name__ == "__main__":
    unittest.main()