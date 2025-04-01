"""Unit tests for the BigQuery Recommendation Engine module."""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
from datetime import datetime, timedelta

from bigquerycostopt.src.recommender.engine import RecommendationEngine
from bigquerycostopt.src.recommender.roi import ROICalculator, calculate_roi

# Sample storage recommendation for testing
SAMPLE_STORAGE_REC = {
    "type": "partitioning_add",
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "description": "Add partitioning to table",
    "recommendation": "Partition table by 'created_at' field",
    "rationale": "Table is large and frequently queried by date",
    "estimated_savings_pct": 40.0,
    "estimated_size_reduction_gb": 20.0,
    "estimated_monthly_savings": 10.0,
    "implementation_complexity": "medium",
    "priority": "high"
}

# Sample query recommendation for testing
SAMPLE_QUERY_REC = {
    "type": "query_select_star",
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "description": "Optimize SELECT * query",
    "recommendation": "Replace SELECT * with specific columns",
    "rationale": "Current query scans unnecessary columns",
    "estimated_savings_pct": 60.0,
    "estimated_bytes_reduction": 1000000000000,  # 1 TB
    "estimated_monthly_cost_savings": 15.0,
    "implementation_complexity": "low",
    "priority": "high",
    "original_query": "SELECT * FROM test_dataset.test_table",
    "optimized_query": "SELECT id, name FROM test_dataset.test_table"
}

# Sample schema recommendation for testing
SAMPLE_SCHEMA_REC = {
    "type": "datatype_float_to_int",
    "table_id": "test_table",
    "dataset_id": "test_dataset",
    "project_id": "test-project",
    "description": "Convert 'price' column from FLOAT64 to INT64",
    "recommendation": "Convert 'price' column from FLOAT64 to INT64",
    "rationale": "Column contains only integer values",
    "estimated_storage_savings_pct": 40.0,
    "estimated_storage_savings_gb": 5.0,
    "estimated_monthly_cost_savings": 5.0,
    "implementation_complexity": "low",
    "priority_score": 80,
    "column_name": "price",
    "current_type": "FLOAT64",
    "recommended_type": "INT64"
}


class TestRecommendationEngine(unittest.TestCase):
    """Test cases for BigQuery Recommendation Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock optimizer components
        self.mock_storage_optimizer = MagicMock()
        self.mock_query_optimizer = MagicMock()
        self.mock_schema_optimizer = MagicMock()
        
        # Create mock ROI calculator
        self.mock_roi_calculator = MagicMock()
        
        # Configure mocks
        self.mock_storage_optimizer.analyze_dataset.return_value = {
            "recommendations": [SAMPLE_STORAGE_REC]
        }
        
        self.mock_query_optimizer.analyze_dataset_queries.return_value = {
            "recommendations": [SAMPLE_QUERY_REC]
        }
        
        self.mock_schema_optimizer.analyze_dataset_schemas.return_value = {
            "recommendations": [SAMPLE_SCHEMA_REC]
        }
        
        # Mock ROI calculation
        self.mock_roi_calculator.calculate_roi.return_value = {
            "roi": 5.0,
            "annual_savings_usd": 100.0,
            "implementation_cost_usd": 20.0,
            "payback_period_months": 2.4,
            "npv_3yr_usd": 250.0,
            "risk_adjusted_roi": 4.5,
            "risk_factor": 0.9
        }
        
        # Create the recommendation engine and inject mocks
        with patch('bigquerycostopt.src.recommender.engine.StorageOptimizer') as mock_storage_class, \
             patch('bigquerycostopt.src.recommender.engine.QueryOptimizer') as mock_query_class, \
             patch('bigquerycostopt.src.recommender.engine.SchemaOptimizer') as mock_schema_class, \
             patch('bigquerycostopt.src.recommender.engine.ROICalculator') as mock_roi_class, \
             patch('bigquerycostopt.src.recommender.engine.ImplementationPlanGenerator'):
                
            mock_storage_class.return_value = self.mock_storage_optimizer
            mock_query_class.return_value = self.mock_query_optimizer
            mock_schema_class.return_value = self.mock_schema_optimizer
            mock_roi_class.return_value = self.mock_roi_calculator
            
            self.engine = RecommendationEngine(project_id="test-project")
    
    def test_analyze_dataset(self):
        """Test analyzing a dataset for all optimization types."""
        # Run the analysis
        results = self.engine.analyze_dataset("test_dataset")
        
        # Verify basic structure
        self.assertEqual(results["dataset_id"], "test_dataset")
        self.assertEqual(results["project_id"], "test-project")
        self.assertIn("analysis_timestamp", results)
        self.assertIn("recommendations", results)
        self.assertIn("implementation_plan", results)
        self.assertIn("summary", results)
        
        # Verify that each optimizer was called once
        self.mock_storage_optimizer.analyze_dataset.assert_called_once_with(
            "test_dataset", min_table_size_gb=1.0
        )
        self.mock_query_optimizer.analyze_dataset_queries.assert_called_once_with(
            "test_dataset", days=30
        )
        self.mock_schema_optimizer.analyze_dataset_schemas.assert_called_once_with(
            "test_dataset", min_table_size_gb=1.0
        )
        
        # Verify recommendations were standardized
        self.assertEqual(len(results["recommendations"]), 3)
        
        # Verify ROI was calculated
        self.assertEqual(self.mock_roi_calculator.calculate_roi.call_count, 3)
    
    def test_analyze_dataset_with_filters(self):
        """Test analyzing a dataset with filters for specific optimization types."""
        # Run the analysis with only storage and query optimizations
        results = self.engine.analyze_dataset(
            "test_dataset",
            include_storage=True,
            include_query=True,
            include_schema=False
        )
        
        # Verify that only storage and query optimizers were called
        self.mock_storage_optimizer.analyze_dataset.assert_called_once()
        self.mock_query_optimizer.analyze_dataset_queries.assert_called_once()
        self.mock_schema_optimizer.analyze_dataset_schemas.assert_not_called()
        
        # Verify that only 2 recommendations were standardized
        self.assertEqual(len(results["recommendations"]), 2)
    
    def test_standardize_recommendations(self):
        """Test standardization of recommendations from different optimizer modules."""
        # Set the raw recommendations
        self.engine.raw_recommendations = {
            "storage": [SAMPLE_STORAGE_REC],
            "query": [SAMPLE_QUERY_REC],
            "schema": [SAMPLE_SCHEMA_REC]
        }
        
        # Standardize recommendations
        self.engine._standardize_recommendations()
        
        # Check that standardized recommendations have standard fields
        self.assertEqual(len(self.engine.recommendations), 3)
        
        for rec in self.engine.recommendations:
            # Check required fields
            self.assertIn("recommendation_id", rec)
            self.assertIn("category", rec)
            self.assertIn("description", rec)
            self.assertIn("table_id", rec)
            self.assertIn("dataset_id", rec)
            self.assertIn("project_id", rec)
            self.assertIn("current_state", rec)
            
            # Check that category is correctly set
            category = rec["category"]
            self.assertIn(category, ["storage", "query", "schema"])
            
            # Check for ID format based on category
            if category == "storage":
                self.assertTrue(rec["recommendation_id"].startswith("STORAGE_"))
            elif category == "query":
                self.assertTrue(rec["recommendation_id"].startswith("QUERY_"))
            elif category == "schema":
                self.assertTrue(rec["recommendation_id"].startswith("SCHEMA_"))
    
    def test_calculate_roi_and_prioritize(self):
        """Test ROI calculation and prioritization of recommendations."""
        # Create test recommendations
        test_recs = [
            {
                "recommendation_id": "TEST_001",
                "category": "storage",
                "description": "High ROI recommendation",
                "estimated_storage_savings_gb": 50.0,
                "estimated_monthly_savings": 20.0,
                "estimated_effort": "low"
            },
            {
                "recommendation_id": "TEST_002",
                "category": "query",
                "description": "Medium ROI recommendation",
                "estimated_query_bytes_reduction": 500000000000,
                "estimated_monthly_savings": 10.0,
                "estimated_effort": "medium"
            },
            {
                "recommendation_id": "TEST_003",
                "category": "schema",
                "description": "Low ROI recommendation",
                "estimated_storage_savings_gb": 2.0,
                "estimated_monthly_savings": 2.0,
                "estimated_effort": "high"
            }
        ]
        
        # Set up mock ROI calculator responses
        self.mock_roi_calculator.calculate_roi.side_effect = [
            {
                "roi": 10.0,
                "annual_savings_usd": 240.0,
                "implementation_cost_usd": 24.0,
                "payback_period_months": 1.2,
                "priority_score": 90
            },
            {
                "roi": 5.0,
                "annual_savings_usd": 120.0,
                "implementation_cost_usd": 24.0,
                "payback_period_months": 2.4,
                "priority_score": 60
            },
            {
                "roi": 0.5,
                "annual_savings_usd": 24.0,
                "implementation_cost_usd": 48.0,
                "payback_period_months": 24.0,
                "priority_score": 30
            }
        ]
        
        # Set recommendations and calculate ROI
        self.engine.recommendations = test_recs
        self.engine._calculate_roi_and_prioritize()
        
        # Verify that ROI calculator was called for each recommendation
        self.assertEqual(self.mock_roi_calculator.calculate_roi.call_count, 3)
        
        # Verify that recommendations are sorted by priority_score (descending)
        self.assertEqual(len(self.engine.recommendations), 3)
        self.assertEqual(self.engine.recommendations[0]["roi"], 10.0)
        self.assertEqual(self.engine.recommendations[1]["roi"], 5.0)
        self.assertEqual(self.engine.recommendations[2]["roi"], 0.5)
        
        # Verify priority assignments
        self.assertEqual(self.engine.recommendations[0]["priority"], "high")
        self.assertEqual(self.engine.recommendations[1]["priority"], "medium")
        self.assertEqual(self.engine.recommendations[2]["priority"], "low")
    
    def test_generate_summary(self):
        """Test summary generation from recommendations."""
        # Create test recommendations with ROI info
        self.engine.recommendations = [
            {
                "recommendation_id": "STORAGE_001",
                "category": "storage",
                "description": "Add partitioning",
                "estimated_storage_savings_gb": 50.0,
                "annual_savings_usd": 240.0,
                "implementation_cost_usd": 24.0,
                "priority": "high",
                "estimated_effort": "low"
            },
            {
                "recommendation_id": "QUERY_001",
                "category": "query",
                "description": "Optimize query",
                "annual_savings_usd": 120.0,
                "implementation_cost_usd": 24.0,
                "priority": "medium",
                "estimated_effort": "medium"
            },
            {
                "recommendation_id": "SCHEMA_001",
                "category": "schema",
                "description": "Optimize schema",
                "estimated_storage_savings_gb": 2.0,
                "annual_savings_usd": 24.0,
                "implementation_cost_usd": 48.0,
                "priority": "low",
                "estimated_effort": "high"
            }
        ]
        
        # Generate summary
        summary = self.engine._generate_summary()
        
        # Verify summary structure
        self.assertEqual(summary["total_recommendations"], 3)
        self.assertEqual(summary["priority_breakdown"]["high"], 1)
        self.assertEqual(summary["priority_breakdown"]["medium"], 1)
        self.assertEqual(summary["priority_breakdown"]["low"], 1)
        
        # Verify category breakdown
        self.assertEqual(summary["category_breakdown"]["storage"], 1)
        self.assertEqual(summary["category_breakdown"]["query"], 1)
        self.assertEqual(summary["category_breakdown"]["schema"], 1)
        
        # Verify savings calculations
        self.assertEqual(summary["savings_summary"]["total_storage_savings_gb"], 52.0)
        self.assertEqual(summary["savings_summary"]["total_annual_savings_usd"], 384.0)
        
        # Verify implementation summary
        self.assertEqual(summary["implementation_summary"]["total_implementation_cost_usd"], 96.0)
        self.assertGreater(summary["implementation_summary"]["estimated_implementation_days"], 0)
    
    def test_format_for_bigquery(self):
        """Test formatting recommendations for BigQuery storage."""
        # Create test recommendations
        self.engine.recommendations = [
            {
                "recommendation_id": "STORAGE_001",
                "category": "storage",
                "type": "partitioning_add",
                "description": "Add partitioning",
                "table_id": "test_table",
                "dataset_id": "test_dataset",
                "project_id": "test-project",
                "rationale": "Improve query performance",
                "recommendation": "Partition by date",
                "annual_savings_usd": 240.0,
                "priority": "high",
                "current_state": {"partitioning": "None"}
            }
        ]
        
        # Format for BigQuery
        bigquery_recs = self.engine.format_for_bigquery()
        
        # Verify format
        self.assertEqual(len(bigquery_recs), 1)
        rec = bigquery_recs[0]
        
        # Check that structure is correct for BigQuery
        self.assertEqual(rec["recommendation_id"], "STORAGE_001")
        self.assertEqual(rec["category"], "storage")
        self.assertEqual(rec["type"], "partitioning_add")
        self.assertEqual(rec["description"], "Add partitioning")
        self.assertEqual(rec["table_id"], "test_table")
        self.assertEqual(rec["dataset_id"], "test_dataset")
        self.assertEqual(rec["project_id"], "test-project")
        
        # Check that nested objects are serialized to JSON strings
        self.assertEqual(rec["current_state"], '{"partitioning": "None"}')
        self.assertEqual(rec["depends_on"], "[]")
        self.assertEqual(rec["conflicts_with"], "[]")
        
        # Check timestamp
        self.assertIn("timestamp", rec)
    
    def test_format_for_dashboard(self):
        """Test formatting recommendations for dashboard display."""
        # Create test recommendations
        self.engine.recommendations = [
            {
                "recommendation_id": "STORAGE_001",
                "category": "storage",
                "type": "partitioning_add",
                "description": "Add partitioning",
                "table_id": "test_table",
                "dataset_id": "test_dataset",
                "project_id": "test-project",
                "estimated_monthly_savings": 20.0,
                "annual_savings_usd": 240.0,
                "estimated_storage_savings_gb": 50.0,
                "priority": "high",
                "implementation_complexity": "low"
            }
        ]
        
        # Format for dashboard
        dashboard_data = self.engine.format_for_dashboard()
        
        # Verify format
        self.assertIn("timestamp", dashboard_data)
        self.assertEqual(dashboard_data["project_id"], "test-project")
        
        # Check summary
        self.assertEqual(dashboard_data["summary"]["total_recommendations"], 1)
        self.assertEqual(dashboard_data["summary"]["monthly_savings_usd"], 20.0)
        self.assertEqual(dashboard_data["summary"]["annual_savings_usd"], 240.0)
        self.assertEqual(dashboard_data["summary"]["storage_savings_gb"], 50.0)
        self.assertEqual(dashboard_data["summary"]["priority_breakdown"]["high"], 1)
        
        # Check recommendation categories
        self.assertEqual(len(dashboard_data["recommendations"]["storage"]), 1)
        self.assertEqual(len(dashboard_data["recommendations"]["query"]), 0)
        self.assertEqual(len(dashboard_data["recommendations"]["schema"]), 0)
        
        # Check chart data
        self.assertIn("charts", dashboard_data)
        self.assertIn("savings_by_category", dashboard_data["charts"])
        self.assertIn("recommendations_by_priority", dashboard_data["charts"])
        self.assertIn("top_tables_by_savings", dashboard_data["charts"])


class TestROICalculator(unittest.TestCase):
    """Test cases for ROI Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = ROICalculator()
    
    def test_calculate_roi_for_storage_recommendation(self):
        """Test ROI calculation for storage recommendation."""
        # Clone the sample storage recommendation
        recommendation = SAMPLE_STORAGE_REC.copy()
        
        # Calculate ROI
        roi_data = self.calculator.calculate_roi(recommendation)
        
        # Verify ROI structure
        self.assertIn("roi", roi_data)
        self.assertIn("annual_savings_usd", roi_data)
        self.assertIn("implementation_cost_usd", roi_data)
        self.assertIn("payback_period_months", roi_data)
        self.assertIn("npv_3yr_usd", roi_data)
        self.assertIn("risk_adjusted_roi", roi_data)
        self.assertIn("risk_factor", roi_data)
        
        # Verify calculations
        monthly_savings = recommendation["estimated_monthly_savings"]
        annual_savings = monthly_savings * 12
        
        self.assertAlmostEqual(roi_data["annual_savings_usd"], annual_savings)
        self.assertGreater(roi_data["implementation_cost_usd"], 0)
        self.assertGreater(roi_data["roi"], 0)
        self.assertLess(roi_data["payback_period_months"], 12)  # High ROI should have short payback
    
    def test_calculate_roi_for_query_recommendation(self):
        """Test ROI calculation for query recommendation."""
        # Clone the sample query recommendation
        recommendation = SAMPLE_QUERY_REC.copy()
        
        # Calculate ROI
        roi_data = self.calculator.calculate_roi(recommendation)
        
        # Verify ROI structure and calculations
        self.assertGreater(roi_data["roi"], 0)
        self.assertGreater(roi_data["annual_savings_usd"], 0)
        
        # Verify implementation cost based on effort
        expected_cost = self.calculator.hourly_engineering_rate * 4  # 'low' effort = 4 hours
        self.assertEqual(roi_data["implementation_cost_usd"], expected_cost)
    
    def test_calculate_roi_for_schema_recommendation(self):
        """Test ROI calculation for schema recommendation."""
        # Clone the sample schema recommendation
        recommendation = SAMPLE_SCHEMA_REC.copy()
        
        # Calculate ROI
        roi_data = self.calculator.calculate_roi(recommendation)
        
        # Verify ROI structure and calculations
        self.assertGreater(roi_data["roi"], 0)
        self.assertGreater(roi_data["annual_savings_usd"], 0)
        
        # Verify that schema changes have appropriate risk factor
        self.assertLess(roi_data["risk_factor"], 1.0)  # Should be risk-adjusted
    
    def test_calculate_npv(self):
        """Test NPV calculation for different timeframes."""
        # Test 1-year NPV
        npv_1yr = self.calculator._calculate_npv(
            implementation_cost=1000,
            annual_savings=500,
            years=1
        )
        
        # Test 3-year NPV
        npv_3yr = self.calculator._calculate_npv(
            implementation_cost=1000,
            annual_savings=500,
            years=3
        )
        
        # Test 5-year NPV
        npv_5yr = self.calculator._calculate_npv(
            implementation_cost=1000,
            annual_savings=500,
            years=5
        )
        
        # Verify NPVs
        self.assertLess(npv_1yr, 0)  # 1-year NPV should be negative (not recovered cost yet)
        self.assertGreater(npv_3yr, 0)  # 3-year NPV should be positive
        self.assertGreater(npv_5yr, npv_3yr)  # 5-year NPV should be greater than 3-year
    
    def test_risk_factor_calculation(self):
        """Test risk factor calculation for different recommendation types."""
        # Test low risk recommendation (simple query optimization)
        low_risk_rec = {
            "category": "query",
            "type": "query_select_star",
            "implementation_complexity": "low"
        }
        low_risk_factor = self.calculator._calculate_risk_factor(low_risk_rec)
        
        # Test medium risk recommendation (storage optimization)
        medium_risk_rec = {
            "category": "storage",
            "type": "partitioning_add",
            "implementation_complexity": "medium"
        }
        medium_risk_factor = self.calculator._calculate_risk_factor(medium_risk_rec)
        
        # Test high risk recommendation (schema change with dependencies)
        high_risk_rec = {
            "category": "schema",
            "type": "schema_remove_columns",
            "implementation_complexity": "high",
            "depends_on": ["STORAGE_001"]
        }
        high_risk_factor = self.calculator._calculate_risk_factor(high_risk_rec)
        
        # Verify risk factors
        self.assertGreater(low_risk_factor, medium_risk_factor)
        self.assertGreater(medium_risk_factor, high_risk_factor)
    
    def test_backwards_compatibility(self):
        """Test backwards compatibility with standalone calculate_roi function."""
        # Create a recommendation
        recommendation = {
            "estimated_savings_pct": 30.0,
            "estimated_effort": "medium",
            "type": "partitioning_add"
        }
        
        # Use standalone function
        roi_data = calculate_roi(recommendation, dataset_size_gb=100.0, table_size_gb=50.0)
        
        # Verify that it still works
        self.assertIn("roi", roi_data)
        self.assertIn("annual_savings_usd", roi_data)
        self.assertIn("implementation_cost_usd", roi_data)
        self.assertGreater(roi_data["roi"], 0)


if __name__ == "__main__":
    unittest.main()