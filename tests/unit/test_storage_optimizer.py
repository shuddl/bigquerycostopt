"""Unit tests for the BigQuery Storage Optimizer module."""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
from datetime import datetime, timedelta

from src.analysis.storage_optimizer import StorageOptimizer

# Sample table metadata for testing
SAMPLE_TABLE_METADATA = {
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "full_name": "test-project.test_dataset.test_table",
    "size_gb": 50.0,
    "last_modified": (datetime.now() - timedelta(days=5)).isoformat(),
    "query_count_30d": 100,
    "partitioning": None,
    "clustering": None,
    "schema": [
        {
            "name": "id",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Primary key"
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
            "name": "user_id",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "User identifier"
        }
    ],
    "frequent_filters": [
        {"column": "created_at", "count": 80},
        {"column": "status", "count": 50},
        {"column": "user_id", "count": 30}
    ]
}

SAMPLE_DATASET_METADATA = {
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "total_size_gb": 150.0,
    "table_count": 3,
    "tables": [
        SAMPLE_TABLE_METADATA,
        {
            "table_id": "old_table",
            "dataset_id": "test_dataset",
            "full_name": "test-project.test_dataset.old_table",
            "size_gb": 80.0,
            "last_modified": (datetime.now() - timedelta(days=60)).isoformat(),
            "query_count_30d": 0,
            "partitioning": None,
            "clustering": None,
            "schema": [
                {
                    "name": "id",
                    "type": "INTEGER",
                    "mode": "REQUIRED",
                    "description": "Primary key"
                },
                {
                    "name": "event_date",
                    "type": "DATE",
                    "mode": "REQUIRED",
                    "description": "Event date"
                }
            ]
        },
        {
            "table_id": "partitioned_table",
            "dataset_id": "test_dataset",
            "full_name": "test-project.test_dataset.partitioned_table",
            "size_gb": 120.0,
            "last_modified": (datetime.now() - timedelta(days=1)).isoformat(),
            "query_count_30d": 500,
            "partitioning": {
                "type": "DAY",
                "field": "event_date",
                "expiration_ms": None
            },
            "clustering": None,
            "schema": [
                {
                    "name": "id",
                    "type": "INTEGER",
                    "mode": "REQUIRED"
                },
                {
                    "name": "event_date",
                    "type": "DATE",
                    "mode": "REQUIRED"
                },
                {
                    "name": "category",
                    "type": "STRING",
                    "mode": "REQUIRED"
                }
            ],
            "frequent_filters": [
                {"column": "event_date", "count": 400},
                {"column": "category", "count": 300}
            ]
        }
    ]
}


class TestStorageOptimizer(unittest.TestCase):
    """Test cases for BigQuery Storage Optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock metadata extractor
        self.mock_metadata_extractor = MagicMock()
        self.mock_metadata_extractor.project_id = "test-project"
        self.mock_metadata_extractor.extract_dataset_metadata.return_value = SAMPLE_DATASET_METADATA
        
        # Create a mock BigQuery client
        self.mock_client = MagicMock()
        self.mock_metadata_extractor.client = self.mock_client
        
        # Initialize StorageOptimizer with mocks
        self.optimizer = StorageOptimizer(metadata_extractor=self.mock_metadata_extractor)
    
    def test_analyze_dataset(self):
        """Test analyzing a dataset for storage optimizations."""
        # Run the analysis
        results = self.optimizer.analyze_dataset("test_dataset")
        
        # Verify basic structure
        self.assertEqual(results["dataset_id"], "test_dataset")
        self.assertEqual(results["project_id"], "test-project")
        self.assertEqual(results["total_size_gb"], 150.0)
        self.assertEqual(results["table_count"], 3)
        
        # Verify that recommendations were generated
        self.assertGreater(len(results["recommendations"]), 0)
        self.assertGreater(results["optimization_summary"]["total_recommendations"], 0)
        
        # Verify savings were calculated
        self.assertGreater(results["optimization_summary"]["estimated_monthly_savings"], 0)
    
    def test_analyze_partitioning(self):
        """Test partitioning recommendations."""
        # Run partitioning analysis directly
        recommendations = self.optimizer._analyze_partitioning(SAMPLE_TABLE_METADATA)
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        
        # Check that it recommended partitioning on created_at
        partition_rec = next((r for r in recommendations if r["type"] == "add_time_partitioning"), None)
        self.assertIsNotNone(partition_rec)
        self.assertIn("created_at", partition_rec["recommendation"])
        
        # Check that savings were calculated
        self.assertGreater(partition_rec["estimated_monthly_savings"], 0)
        self.assertGreater(partition_rec["estimated_size_reduction_gb"], 0)
        
        # Check SQL generation
        self.assertIn("CREATE OR REPLACE TABLE", partition_rec["implementation_sql"])
        self.assertIn("PARTITION BY", partition_rec["implementation_sql"])
    
    def test_analyze_clustering(self):
        """Test clustering recommendations."""
        # Run clustering analysis directly
        recommendations = self.optimizer._analyze_clustering(SAMPLE_TABLE_METADATA)
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        
        # Check that it found good clustering fields
        cluster_rec = recommendations[0]
        self.assertEqual(cluster_rec["type"], "add_clustering")
        self.assertIn("status", cluster_rec["recommendation"])
        
        # Check SQL generation
        self.assertIn("CLUSTER BY", cluster_rec["implementation_sql"])
    
    def test_analyze_long_term_storage(self):
        """Test long-term storage recommendations."""
        # Use the old table metadata which hasn't been queried recently
        old_table = SAMPLE_DATASET_METADATA["tables"][1]
        
        # Run LTS analysis directly
        recommendations = self.optimizer._analyze_long_term_storage(old_table)
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        
        # Check LTS recommendation
        lts_rec = recommendations[0]
        self.assertEqual(lts_rec["type"], "long_term_storage")
        self.assertEqual(lts_rec["table_id"], "old_table")
        
        # Check savings calculation
        # Should be 0.01 (price difference) * table size in GB
        expected_savings = 0.01 * old_table["size_gb"]
        self.assertAlmostEqual(lts_rec["estimated_monthly_savings"], expected_savings, places=2)
    
    def test_daily_to_monthly_partitioning(self):
        """Test recommendations for changing daily to monthly partitioning."""
        # Use the already partitioned table
        partitioned_table = SAMPLE_DATASET_METADATA["tables"][2]
        
        # Modify it to be very large
        large_table = dict(partitioned_table)
        large_table["size_gb"] = 200.0
        
        # Run partitioning analysis
        recommendations = self.optimizer._analyze_partitioning(large_table)
        
        # Check for daily to monthly recommendation
        daily_to_monthly = next((r for r in recommendations if r["type"] == "partition_daily_to_monthly"), None)
        self.assertIsNotNone(daily_to_monthly)
        self.assertIn("MONTH", daily_to_monthly["recommendation"])
    
    def test_recommendation_priorities(self):
        """Test that recommendation priorities are set correctly."""
        # Run the analysis
        results = self.optimizer.analyze_dataset("test_dataset")
        
        # Check that priorities exist
        for rec in results["recommendations"]:
            self.assertIn("priority", rec)
            self.assertIn(rec["priority"], ["high", "medium", "low"])
            
            # Large tables should have higher priority
            if rec["table_id"] == "partitioned_table" and "clustering" in rec["type"]:
                self.assertEqual(rec["priority"], "high")
    
    def test_implementation_sql_generation(self):
        """Test that SQL scripts are generated correctly."""
        # Run the analysis
        results = self.optimizer.analyze_dataset("test_dataset")
        
        # Check all recommendations have implementation SQL
        for rec in results["recommendations"]:
            if rec.get("depends_on"):
                # Skip recommendations that depend on others
                continue
                
            self.assertIn("implementation_sql", rec)
            sql = rec["implementation_sql"]
            
            # Basic SQL validation
            self.assertIn("CREATE", sql)
            
            # Check specific SQL content based on recommendation type
            if "partition" in rec["type"]:
                self.assertIn("PARTITION BY", sql)
            elif "clustering" in rec["type"]:
                self.assertIn("CLUSTER BY", sql)


if __name__ == "__main__":
    unittest.main()