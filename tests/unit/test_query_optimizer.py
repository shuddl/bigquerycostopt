"""Unit tests for the BigQuery Query Optimizer module."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import json
import os
from datetime import datetime, timedelta

from bigquerycostopt.src.analysis.query_optimizer import QueryOptimizer

# Sample test queries
SELECT_STAR_QUERY = """
SELECT * 
FROM `test-project.test_dataset.test_table`
WHERE created_at >= '2023-01-01'
"""

NO_PARTITION_FILTER_QUERY = """
SELECT id, name, value 
FROM `test-project.test_dataset.test_table`
WHERE status = 'active'
"""

INEFFICIENT_JOIN_QUERY = """
SELECT a.id, a.name, b.value 
FROM `test-project.test_dataset.table_a` a, 
     `test-project.test_dataset.table_b` b
WHERE a.status = 'active'
"""

SUBQUERY_QUERY = """
SELECT id, name,
    (SELECT MAX(value) FROM `test-project.test_dataset.table_b` WHERE user_id = test_table.id) as max_value
FROM `test-project.test_dataset.test_table`
"""

# Sample table metadata for testing
SAMPLE_TABLE_METADATA = {
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "full_name": "test-project.test_dataset.test_table",
    "size_gb": 50.0,
    "partitioning": {
        "type": "DAY",
        "field": "created_at",
        "expiration_ms": None
    },
    "schema": [
        {
            "name": "id",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Primary key"
        },
        {
            "name": "name",
            "type": "STRING",
            "mode": "REQUIRED",
            "description": "User name"
        },
        {
            "name": "created_at",
            "type": "TIMESTAMP",
            "mode": "REQUIRED",
            "description": "Creation timestamp"
        },
        {
            "name": "status",
            "type": "STRING",
            "mode": "REQUIRED",
            "description": "Status field"
        },
        {
            "name": "value",
            "type": "FLOAT",
            "mode": "NULLABLE",
            "description": "Value field"
        }
    ]
}

SAMPLE_QUERY_HISTORY = pd.DataFrame({
    'query_text': [
        SELECT_STAR_QUERY,
        NO_PARTITION_FILTER_QUERY,
        INEFFICIENT_JOIN_QUERY,
        SUBQUERY_QUERY
    ],
    'total_bytes_processed': [
        5000000000,  # 5 GB
        8000000000,  # 8 GB
        12000000000,  # 12 GB
        7000000000   # 7 GB
    ],
    'execution_count': [10, 5, 2, 8],
    'has_select_star': [True, False, False, False],
    'missing_where_clause': [False, False, False, False],
    'cache_hit_ratio': [0.1, 0.2, 0.0, 0.3],
    'user_emails': ['user1@example.com', 'user2@example.com', 'user1@example.com', 'user3@example.com'],
    'max_bytes_processed': [600000000, 900000000, 1200000000, 800000000],
    'avg_bytes_processed': [500000000, 800000000, 1200000000, 700000000],
    'last_execution': [
        datetime.now() - timedelta(days=1),
        datetime.now() - timedelta(days=2),
        datetime.now() - timedelta(days=3),
        datetime.now() - timedelta(days=1)
    ]
})


class TestQueryOptimizer(unittest.TestCase):
    """Test cases for BigQuery Query Optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock metadata extractor
        self.mock_metadata_extractor = MagicMock()
        self.mock_metadata_extractor.project_id = "test-project"
        
        # Create a mock connector
        self.mock_connector = MagicMock()
        self.mock_metadata_extractor.connector = self.mock_connector
        
        # Mock the query_to_dataframe method to return our sample query history
        self.mock_connector.query_to_dataframe.return_value = SAMPLE_QUERY_HISTORY
        
        # Mock extract_table_metadata to return our sample table metadata
        self.mock_metadata_extractor.extract_table_metadata.return_value = SAMPLE_TABLE_METADATA
        
        # Create a mock dataset metadata with our table
        mock_dataset_metadata = {
            "dataset_id": "test_dataset",
            "project_id": "test-project",
            "table_count": 1,
            "tables": [SAMPLE_TABLE_METADATA]
        }
        self.mock_metadata_extractor.extract_dataset_metadata.return_value = mock_dataset_metadata
        
        # Initialize QueryOptimizer with mock
        self.optimizer = QueryOptimizer(metadata_extractor=self.mock_metadata_extractor)
        
        # For testing query parsing directly
        self.tables_info = {"test_table": SAMPLE_TABLE_METADATA}
    
    def test_analyze_dataset_queries(self):
        """Test analyzing queries for a dataset."""
        # Run the analysis
        results = self.optimizer.analyze_dataset_queries("test_dataset")
        
        # Verify basic structure
        self.assertEqual(results["dataset_id"], "test_dataset")
        self.assertEqual(results["project_id"], "test-project")
        self.assertIn("queries_analyzed", results)
        self.assertIn("recommendations", results)
        self.assertIn("summary", results)
        
        # Verify that recommendations were generated
        self.assertGreater(len(results["recommendations"]), 0)
        
        # Verify summary metrics
        summary = results["summary"]
        self.assertIn("total_recommendations", summary)
        self.assertIn("estimated_savings_bytes", summary)
        self.assertIn("estimated_monthly_cost_savings", summary)
        self.assertIn("estimated_annual_cost_savings", summary)
    
    def test_analyze_table_queries(self):
        """Test analyzing queries for a specific table."""
        # Run the analysis
        results = self.optimizer.analyze_table_queries("test_dataset", "test_table")
        
        # Verify basic structure
        self.assertEqual(results["table_id"], "test_table")
        self.assertEqual(results["dataset_id"], "test_dataset")
        self.assertEqual(results["project_id"], "test-project")
        
        # Verify that recommendations were generated
        self.assertGreater(len(results["recommendations"]), 0)
    
    def test_analyze_query_text(self):
        """Test analyzing a specific query text."""
        # Run the analysis
        results = self.optimizer.analyze_query_text(SELECT_STAR_QUERY, "test_dataset")
        
        # Verify basic structure
        self.assertEqual(results["query_text"], SELECT_STAR_QUERY)
        self.assertEqual(results["dataset_id"], "test_dataset")
        
        # Verify that recommendations were generated
        self.assertGreater(len(results["recommendations"]), 0)
        
        # Since this is a SELECT * query, check for that specific recommendation type
        select_star_rec = next((r for r in results["recommendations"] if r["type"] == "select_star"), None)
        self.assertIsNotNone(select_star_rec)
    
    def test_check_select_star(self):
        """Test detection of SELECT * pattern."""
        # Parse query components (simplified for testing)
        query_info = {
            "has_select_star": True,
            "select_items": ["*"],
            "from_tables": ["`test-project.test_dataset.test_table`"]
        }
        
        # Run the check
        recommendation = self.optimizer._check_select_star(
            query_info, 
            self.tables_info, 
            5000000000,  # 5 GB
            "test_dataset"
        )
        
        # Verify recommendation
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation["type"], "select_star")
        self.assertIn("estimated_savings_bytes", recommendation)
        self.assertIn("implementation_difficulty", recommendation)
        self.assertIn("before", recommendation)
        self.assertIn("after", recommendation)
    
    def test_check_missing_partition_filter(self):
        """Test detection of missing partition filter."""
        # Parse query components (simplified for testing)
        query_info = {
            "has_where_clause": True,
            "where_conditions": ["status = 'active'"],
            "from_tables": ["`test-project.test_dataset.test_table`"]
        }
        
        # Run the check
        recommendation = self.optimizer._check_missing_partition_filter(
            query_info, 
            self.tables_info, 
            8000000000,  # 8 GB
            "test_dataset"
        )
        
        # Verify recommendation
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation["type"], "missing_partition_filter")
        self.assertIn("partition_field", recommendation)
        self.assertEqual(recommendation["partition_field"], "created_at")
        self.assertIn("estimated_savings_bytes", recommendation)
    
    def test_check_inefficient_joins(self):
        """Test detection of inefficient JOIN patterns."""
        # Parse query components (simplified for testing)
        query_info = {
            "from_tables": [
                "`test-project.test_dataset.table_a`", 
                "`test-project.test_dataset.table_b`"
            ],
            "join_conditions": []  # No join conditions
        }
        
        # Run the check
        recommendations = self.optimizer._check_inefficient_joins(
            query_info, 
            self.tables_info, 
            12000000000,  # 12 GB
            "test_dataset"
        )
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["type"], "cartesian_join")
        self.assertIn("estimated_savings_bytes", recommendations[0])
        self.assertIn("tables_joined", recommendations[0])
    
    def test_check_inefficient_subqueries(self):
        """Test detection of inefficient subqueries."""
        # Parse query components (simplified for testing)
        query_info = {
            "subqueries": [
                "(SELECT MAX(value) FROM `test-project.test_dataset.table_b` WHERE user_id = test_table.id)"
            ],
            "select_items": [
                "id", 
                "name", 
                "(SELECT MAX(value) FROM `test-project.test_dataset.table_b` WHERE user_id = test_table.id) as max_value"
            ]
        }
        
        # Run the check
        recommendations = self.optimizer._check_inefficient_subqueries(
            query_info, 
            self.tables_info, 
            7000000000,  # 7 GB
            "test_dataset"
        )
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["type"], "subquery_to_join")
        self.assertIn("estimated_savings_bytes", recommendations[0])
    
    def test_generate_recommendations_report(self):
        """Test generation of recommendation reports."""
        # Create sample recommendations
        recommendations = {
            "dataset_id": "test_dataset",
            "project_id": "test-project",
            "queries_analyzed": 10,
            "recommendations": [
                {
                    "type": "select_star",
                    "description": "Replace SELECT * with specific columns",
                    "rationale": "The query is reading all columns...",
                    "estimated_cost_savings": 25.0,
                    "before": "SELECT * FROM table",
                    "after": "SELECT id, name FROM table"
                },
                {
                    "type": "missing_partition_filter",
                    "description": "Add partition filter on created_at",
                    "rationale": "The query accesses partitioned table...",
                    "estimated_cost_savings": 40.0,
                    "before": "SELECT id FROM table",
                    "after": "SELECT id FROM table WHERE created_at > '2023-01-01'"
                }
            ],
            "summary": {
                "total_recommendations": 2,
                "estimated_monthly_cost_savings": 65.0,
                "estimated_annual_cost_savings": 780.0
            }
        }
        
        # Generate markdown report
        md_report = self.optimizer.generate_recommendations_report(recommendations, 'md')
        
        # Verify report structure
        self.assertIn("# Query Optimization Recommendations: test_dataset", md_report)
        self.assertIn("**Estimated Monthly Savings:** $65.00", md_report)
        self.assertIn("### 1. Select Star: Replace SELECT * with specific columns", md_report)
        
        # Generate text report
        text_report = self.optimizer.generate_recommendations_report(recommendations, 'text')
        
        # Verify text report structure
        self.assertIn("QUERY OPTIMIZATION RECOMMENDATIONS: test_dataset", text_report)
        self.assertIn("Estimated Monthly Savings: $65.00", text_report)
        self.assertIn("1. Select Star: Replace SELECT * with specific columns", text_report)


if __name__ == "__main__":
    unittest.main()