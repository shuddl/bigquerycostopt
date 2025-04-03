"""Unit tests for the BigQuery Schema Optimizer module."""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
from datetime import datetime, timedelta

from src.analysis.schema_optimizer import SchemaOptimizer

# Sample table metadata for testing
SAMPLE_TABLE_METADATA = {
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "full_name": "test-project.test_dataset.test_table",
    "size_gb": 50.0,
    "size_bytes": 53687091200,  # 50 GB in bytes
    "num_rows": 1000000,
    "last_modified": (datetime.now() - timedelta(days=5)).isoformat(),
    "query_count_30d": 100,
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
            "name": "created_date",
            "type": "TIMESTAMP",
            "mode": "REQUIRED", 
            "description": "Creation date only, stored as timestamp"
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
        },
        {
            "name": "is_active",
            "type": "STRING",
            "mode": "NULLABLE",
            "description": "Active status stored as string 'true'/'false'"
        },
        {
            "name": "price",
            "type": "FLOAT64",
            "mode": "NULLABLE",
            "description": "Price value"
        },
        {
            "name": "tags",
            "type": "STRING",
            "mode": "REPEATED",
            "description": "Associated tags"
        },
        {
            "name": "metadata",
            "type": "RECORD",
            "mode": "NULLABLE",
            "description": "Associated metadata"
        },
        {
            "name": "legacy_field",
            "type": "STRING",
            "mode": "NULLABLE",
            "description": "Deprecated field"
        },
        {
            "name": "temp_value",
            "type": "STRING",
            "mode": "NULLABLE",
            "description": ""
        }
    ],
    "column_stats": [
        {"name": "id", "null_percentage": 0, "distinct_values_count": 1000000},
        {"name": "created_at", "null_percentage": 0},
        {"name": "created_date", "null_percentage": 0, "is_date_only": True},
        {"name": "status", "null_percentage": 0, "distinct_values_count": 3},
        {"name": "user_id", "null_percentage": 0, "distinct_values_count": 50000},
        {"name": "is_active", "null_percentage": 5, "distinct_values_count": 2},
        {"name": "price", "null_percentage": 10, "all_integers": True},
        {"name": "tags", "null_percentage": 20},
        {"name": "metadata", "null_percentage": 30},
        {"name": "legacy_field", "null_percentage": 90},
        {"name": "temp_value", "null_percentage": 95}
    ],
    "frequent_filters": [
        {"column": "created_at", "count": 80},
        {"column": "status", "count": 50},
        {"column": "user_id", "count": 30}
    ],
    "frequently_joined_tables": [
        {"table": "test-project.test_dataset.user_details", "count": 15}
    ]
}

SAMPLE_DATASET_METADATA = {
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "total_size_gb": 150.0,
    "tables": [
        SAMPLE_TABLE_METADATA,
        {
            "table_id": "small_table",
            "dataset_id": "test_dataset",
            "project_id": "test-project",
            "full_name": "test-project.test_dataset.small_table",
            "size_gb": 0.5,
            "size_bytes": 536870912,  # 0.5 GB in bytes
            "num_rows": 10000,
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
                }
            ]
        },
        {
            "table_id": "repeated_fields_table",
            "dataset_id": "test_dataset",
            "project_id": "test-project",
            "full_name": "test-project.test_dataset.repeated_fields_table",
            "size_gb": 120.0,
            "size_bytes": 128849018880,  # 120 GB in bytes
            "num_rows": 5000000,
            "schema": [
                {
                    "name": "id",
                    "type": "INTEGER",
                    "mode": "REQUIRED"
                },
                {
                    "name": "user_id",
                    "type": "INTEGER",
                    "mode": "REQUIRED"
                },
                {
                    "name": "events",
                    "type": "RECORD",
                    "mode": "REPEATED",
                    "fields": [
                        {"name": "event_id", "type": "INTEGER", "mode": "REQUIRED"},
                        {"name": "event_type", "type": "STRING", "mode": "REQUIRED"},
                        {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"}
                    ]
                },
                {
                    "name": "locations",
                    "type": "RECORD",
                    "mode": "REPEATED",
                    "fields": [
                        {"name": "location_id", "type": "INTEGER", "mode": "REQUIRED"},
                        {"name": "city", "type": "STRING", "mode": "REQUIRED"},
                        {"name": "coordinates", "type": "STRING", "mode": "REQUIRED"}
                    ]
                },
                {
                    "name": "tags",
                    "type": "STRING",
                    "mode": "REPEATED"
                }
            ]
        }
    ]
}


class TestSchemaOptimizer(unittest.TestCase):
    """Test cases for BigQuery Schema Optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock metadata extractor
        self.mock_metadata_extractor = MagicMock()
        self.mock_metadata_extractor.project_id = "test-project"
        self.mock_metadata_extractor.extract_dataset_metadata.return_value = SAMPLE_DATASET_METADATA
        self.mock_metadata_extractor.extract_table_metadata.return_value = SAMPLE_TABLE_METADATA
        
        # Create a mock BigQuery client
        self.mock_client = MagicMock()
        self.mock_metadata_extractor.client = self.mock_client
        self.mock_metadata_extractor.connector = MagicMock()
        
        # Initialize SchemaOptimizer with mocks
        self.optimizer = SchemaOptimizer(metadata_extractor=self.mock_metadata_extractor)
    
    def test_analyze_dataset_schemas(self):
        """Test analyzing a dataset for schema optimizations."""
        # Run the analysis
        results = self.optimizer.analyze_dataset_schemas("test_dataset")
        
        # Verify basic structure
        self.assertEqual(results["dataset_id"], "test_dataset")
        self.assertEqual(results["project_id"], "test-project")
        self.assertEqual(results["total_tables"], 3)
        self.assertGreaterEqual(results["tables_analyzed"], 2)  # Should analyze at least the tables > 1 GB
        
        # Verify that recommendations were generated
        self.assertGreater(len(results["recommendations"]), 0)
        self.assertGreater(results["summary"]["total_recommendations"], 0)
        
        # Verify savings were calculated
        self.assertGreater(results["summary"]["estimated_storage_savings_gb"], 0)
        self.assertGreater(results["summary"]["estimated_monthly_cost_savings"], 0)
    
    def test_analyze_data_types(self):
        """Test data type optimization recommendations."""
        # Run data type analysis directly
        recommendations = self.optimizer._analyze_data_types(SAMPLE_TABLE_METADATA)
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 3)  # Should have at least 3 recommendations
        
        # Check specific type recommendations
        
        # Check TIMESTAMP to DATE conversion
        date_rec = next((r for r in recommendations if r["type"] == "datatype_timestamp_to_date"), None)
        self.assertIsNotNone(date_rec)
        self.assertEqual(date_rec["column_name"], "created_date")
        self.assertEqual(date_rec["current_type"], "TIMESTAMP")
        self.assertEqual(date_rec["recommended_type"], "DATE")
        
        # Check FLOAT64 to INT64 conversion
        float_rec = next((r for r in recommendations if r["type"] == "datatype_float_to_int"), None)
        self.assertIsNotNone(float_rec)
        self.assertEqual(float_rec["column_name"], "price")
        self.assertEqual(float_rec["current_type"], "FLOAT64")
        self.assertEqual(float_rec["recommended_type"], "INT64")
        
        # Check STRING to BOOL conversion
        bool_rec = next((r for r in recommendations if r["type"] == "datatype_string_to_bool"), None)
        self.assertIsNotNone(bool_rec)
        self.assertEqual(bool_rec["column_name"], "is_active")
        self.assertEqual(bool_rec["current_type"], "STRING")
        self.assertEqual(bool_rec["recommended_type"], "BOOL")
    
    def test_analyze_string_column(self):
        """Test string column optimization recommendations."""
        column_stats_dict = {stat["name"]: stat for stat in SAMPLE_TABLE_METADATA["column_stats"]}
        
        # Run string column analysis directly
        string_column = {"name": "status", "type": "STRING", "mode": "REQUIRED"}
        recommendations = self.optimizer._analyze_string_column("status", string_column, 
                                                             SAMPLE_TABLE_METADATA, column_stats_dict)
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        
        # Check for ENUM recommendation
        enum_rec = next((r for r in recommendations if r["type"] == "datatype_string_to_enum"), None)
        self.assertIsNotNone(enum_rec)
        self.assertEqual(enum_rec["column_name"], "status")
        self.assertEqual(enum_rec["current_type"], "STRING")
        self.assertEqual(enum_rec["recommended_type"], "ENUM (or INT64 mapping)")
        self.assertGreater(enum_rec["estimated_storage_savings_gb"], 0)
    
    def test_analyze_column_usage(self):
        """Test unused column recommendations."""
        # Run column usage analysis directly
        recommendations = self.optimizer._analyze_column_usage(SAMPLE_TABLE_METADATA)
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        
        # Check for legacy field recommendation
        unused_rec = recommendations[0]
        self.assertEqual(unused_rec["type"], "remove_unused_columns")
        
        # The legacy_field and temp_value columns should be identified as unused
        self.assertIn("legacy_field", unused_rec["columns"])
        self.assertIn("temp_value", unused_rec["columns"])
        
        # Check that storage savings were calculated
        self.assertGreater(unused_rec["estimated_storage_savings_gb"], 0)
    
    def test_analyze_repeated_fields(self):
        """Test repeated fields optimization recommendations."""
        # Use the repeated fields table
        repeated_table = SAMPLE_DATASET_METADATA["tables"][2]
        
        # Run repeated fields analysis directly
        recommendations = self.optimizer._analyze_repeated_fields(repeated_table)
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        
        # Check for repeated fields recommendation
        repeated_rec = next((r for r in recommendations if r["type"] == "denormalize_repeated_fields"), None)
        self.assertIsNotNone(repeated_rec)
        self.assertGreaterEqual(len(repeated_rec["repeated_fields"]), 3)
        
        # Check that SQL was generated
        self.assertIn("normalization", repeated_rec["implementation_sql"].lower())
    
    def test_analyze_denormalization(self):
        """Test denormalization recommendations."""
        # Run denormalization analysis directly
        recommendations = self.optimizer._analyze_denormalization(SAMPLE_TABLE_METADATA)
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        
        # Check for denormalization recommendation
        denorm_rec = recommendations[0]
        self.assertEqual(denorm_rec["type"], "consider_denormalization")
        self.assertEqual(denorm_rec["joined_table"], "test-project.test_dataset.user_details")
        self.assertEqual(denorm_rec["join_count"], 15)
        
        # Check that implementation SQL was generated
        self.assertIn("denormalization", denorm_rec["implementation_sql"].lower())
    
    def test_estimate_column_size_contribution(self):
        """Test column size estimation."""
        # Test for STRING column
        string_size = self.optimizer._estimate_column_size_contribution("status", "STRING", SAMPLE_TABLE_METADATA)
        self.assertGreater(string_size, 0)
        
        # Test for FLOAT column
        float_size = self.optimizer._estimate_column_size_contribution("price", "FLOAT64", SAMPLE_TABLE_METADATA)
        self.assertGreater(float_size, 0)
        
        # Test for REPEATED column
        repeated_size = self.optimizer._estimate_column_size_contribution("tags", "STRING", SAMPLE_TABLE_METADATA)
        self.assertGreater(repeated_size, 0)
        
        # Verify that string columns typically use more space than numeric columns
        self.assertGreater(string_size, float_size)
    
    def test_prioritize_recommendations(self):
        """Test recommendation prioritization."""
        # Create test recommendations with different priorities
        test_recs = [
            {
                "type": "datatype_string_to_enum",
                "priority_score": 80,
                "estimated_storage_savings_gb": 5.0,
                "project_id": "test-project",
                "dataset_id": "test_dataset",
                "table_id": "test_table"
            },
            {
                "type": "datatype_float_to_int",
                "priority_score": 70,
                "estimated_storage_savings_gb": 2.0,
                "project_id": "test-project",
                "dataset_id": "test_dataset",
                "table_id": "test_table"
            },
            {
                "type": "remove_unused_columns",
                "priority_score": 65,
                "estimated_storage_savings_gb": 10.0,
                "project_id": "test-project",
                "dataset_id": "test_dataset",
                "table_id": "test_table"
            }
        ]
        
        # Run prioritization
        prioritized = self.optimizer._prioritize_recommendations(test_recs)
        
        # Verify that recommendations are ordered by priority score
        self.assertEqual(prioritized[0]["type"], "datatype_string_to_enum")
        self.assertEqual(prioritized[1]["type"], "datatype_float_to_int")
        self.assertEqual(prioritized[2]["type"], "remove_unused_columns")
    
    def test_generate_type_change_sql(self):
        """Test SQL generation for type changes."""
        # Test SQL for TIMESTAMP to DATE conversion
        timestamp_sql = self.optimizer._generate_type_change_sql(
            SAMPLE_TABLE_METADATA, "created_date", "DATE")
        
        # Verify SQL content
        self.assertIn("CREATE OR REPLACE TABLE", timestamp_sql)
        self.assertIn("created_date", timestamp_sql)
        self.assertIn("DATE", timestamp_sql)
        
        # Test SQL for STRING to BOOL conversion
        bool_sql = self.optimizer._generate_type_change_sql(
            SAMPLE_TABLE_METADATA, "is_active", "BOOL", is_boolean_string=True)
        
        # Verify SQL contains CASE statement for boolean conversion
        self.assertIn("CASE", bool_sql)
        self.assertIn("WHEN LOWER(is_active) IN ('true'", bool_sql)
    
    def test_generate_recommendations_report(self):
        """Test generation of recommendation reports."""
        # Create sample recommendations
        recommendations = {
            "dataset_id": "test_dataset",
            "project_id": "test-project",
            "total_tables": 3,
            "tables_analyzed": 2,
            "summary": {
                "total_recommendations": 5,
                "estimated_storage_savings_gb": 20.0,
                "estimated_storage_savings_percentage": 13.3,
                "estimated_monthly_cost_savings": 0.4,
                "estimated_annual_cost_savings": 4.8
            },
            "recommendations": [
                {
                    "type": "datatype_string_to_enum",
                    "column_name": "status",
                    "description": "Convert column 'status' to ENUM",
                    "rationale": "The column contains only 3 unique values",
                    "estimated_storage_savings_gb": 5.0,
                    "estimated_monthly_cost_savings": 0.1,
                    "table_id": "test_table",
                    "implementation_complexity": "medium",
                    "backward_compatibility_risk": "medium",
                    "implementation_sql": "-- SQL for conversion"
                }
            ]
        }
        
        # Generate report in markdown format
        md_report = self.optimizer.generate_recommendations_report(recommendations, format='md')
        
        # Verify report content
        self.assertIn("# Schema Optimization Recommendations", md_report)
        self.assertIn("**Estimated Monthly Savings:** $0.40", md_report)
        self.assertIn("**Storage Reduction:** 20.00 GB (13.3%)", md_report)
        
        # Generate report in plain text format
        text_report = self.optimizer.generate_recommendations_report(recommendations, format='text')
        
        # Verify report content
        self.assertIn("SCHEMA OPTIMIZATION RECOMMENDATIONS", text_report)
        self.assertIn("Estimated Monthly Savings: $0.40", text_report)


if __name__ == "__main__":
    unittest.main()