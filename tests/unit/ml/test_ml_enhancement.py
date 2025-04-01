"""Unit tests for the Machine Learning Enhancement Module."""

import unittest
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.ml.enhancer import MLEnhancementModule
from src.ml.models import CostImpactClassifier, UsagePatternClustering, AnomalyDetector
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.recommendation_enhancer import RecommendationEnhancer
from src.ml.feedback import FeedbackCollector
from src.ml.evaluation import EvaluationMetrics


class TestMLEnhancementModule(unittest.TestCase):
    """Test cases for the ML Enhancement Module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for model storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name)
        
        # Mock components
        self.mock_feature_pipeline = MagicMock(spec=FeatureEngineeringPipeline)
        self.mock_cost_classifier = MagicMock(spec=CostImpactClassifier)
        self.mock_pattern_clustering = MagicMock(spec=UsagePatternClustering)
        self.mock_anomaly_detector = MagicMock(spec=AnomalyDetector)
        self.mock_recommendation_enhancer = MagicMock(spec=RecommendationEnhancer)
        self.mock_feedback_collector = MagicMock(spec=FeedbackCollector)
        self.mock_evaluation_metrics = MagicMock(spec=EvaluationMetrics)
        
        # Prepare test data
        self.project_id = "test-project"
        
        # Sample recommendations
        self.recommendations = [
            {
                "recommendation_id": "rec_1",
                "recommendation_type": "partition",
                "target_table": "project.dataset.table1",
                "estimated_savings": {"total": 1000, "monthly": 100},
                "complexity": 3,
                "risk_level": "low",
                "priority_score": 8.5
            },
            {
                "recommendation_id": "rec_2",
                "recommendation_type": "cluster",
                "target_table": "project.dataset.table2",
                "estimated_savings": {"total": 500, "monthly": 50},
                "complexity": 2,
                "risk_level": "medium",
                "priority_score": 6.2
            }
        ]
        
        # Sample dataset metadata
        self.dataset_metadata = {
            "project_id": "test-project",
            "dataset_id": "dataset",
            "tables": [
                {
                    "table_ref": "project.dataset.table1",
                    "table_id": "table1",
                    "size_bytes": 1000000,
                    "num_rows": 10000,
                    "schema": [{"name": "col1", "type": "STRING"}],
                    "usage_statistics": {
                        "query_count_last_30d": 100,
                        "bytes_processed_last_30d": 5000000
                    }
                },
                {
                    "table_ref": "project.dataset.table2",
                    "table_id": "table2",
                    "size_bytes": 2000000,
                    "num_rows": 20000,
                    "schema": [{"name": "col1", "type": "STRING"}],
                    "usage_statistics": {
                        "query_count_last_30d": 50,
                        "bytes_processed_last_30d": 3000000
                    }
                }
            ]
        }
        
        # Sample ML insights
        self.ml_insights = {
            "cost_impact": {
                "business_impact_category": ["high", "medium"],
                "business_impact": [850, 400],
                "recommendation_ids": ["rec_1", "rec_2"]
            },
            "pattern_clusters": {
                "cluster_id": [0, 1],
                "pattern_name": ["batch_etl_workload", "interactive_reporting"],
                "pattern_description": ["Regular batch ETL processing", "Interactive reporting queries"],
                "confidence_score": [0.8, 0.9],
                "recommendation_ids": ["rec_1", "rec_2"]
            },
            "anomalies": {
                "is_anomaly": [True, False],
                "anomaly_score": [0.75, 0.2],
                "anomaly_features": [{"bytes_processed_last_30d": 2.5}, {}],
                "recommendation_ids": ["rec_1", "rec_2"]
            }
        }
        
        # Sample feedback data
        self.feedback_data = {
            "recommendations": {
                "rec_1": {
                    "success": True,
                    "actual_cost_savings": 900,
                    "implementation_time_minutes": 120,
                    "user_rating": 4,
                    "complexity_rating": 3,
                    "business_impact": 800,
                    "business_impact_category": "high",
                    "comments": "Great recommendation"
                }
            },
            "dataset_id": "dataset",
            "feedback_source": "user",
            "feedback_version": "1.0"
        }
        
        # Set up feature pipeline mock to return sample features
        self.features_df = pd.DataFrame({
            "recommendation_id": ["rec_1", "rec_2"],
            "recommendation_type": ["partition", "cluster"],
            "table_size_bytes": [1000000, 2000000],
            "query_count_last_30d": [100, 50]
        })
        self.mock_feature_pipeline.extract_features.return_value = self.features_df
        
        # Set up model mocks
        self.mock_cost_classifier.predict.return_value = self.ml_insights["cost_impact"]
        self.mock_pattern_clustering.predict.return_value = self.ml_insights["pattern_clusters"]
        self.mock_anomaly_detector.detect_anomalies.return_value = self.ml_insights["anomalies"]
        
        # Set up recommendation enhancer mock
        self.enhanced_recommendations = [
            {
                "recommendation_id": "rec_1",
                "recommendation_type": "partition",
                "target_table": "project.dataset.table1",
                "estimated_savings": {"total": 1000, "monthly": 100},
                "complexity": 3,
                "risk_level": "low",
                "priority_score": 9.2,
                "ml_influenced_priority": True,
                "ml_insights": {
                    "business_impact_category": "high",
                    "business_impact": 850,
                    "pattern_name": "batch_etl_workload",
                    "pattern_description": "Regular batch ETL processing",
                    "pattern_confidence": 0.8,
                    "is_anomaly": True,
                    "anomaly_score": 0.75,
                    "anomaly_features": {"bytes_processed_last_30d": 2.5},
                    "business_context": "This table shows ETL workload patterns."
                }
            },
            {
                "recommendation_id": "rec_2",
                "recommendation_type": "cluster",
                "target_table": "project.dataset.table2",
                "estimated_savings": {"total": 500, "monthly": 50},
                "complexity": 2,
                "risk_level": "medium",
                "priority_score": 6.8,
                "ml_influenced_priority": True,
                "ml_insights": {
                    "business_impact_category": "medium",
                    "business_impact": 400,
                    "pattern_name": "interactive_reporting",
                    "pattern_description": "Interactive reporting queries",
                    "pattern_confidence": 0.9,
                    "is_anomaly": False,
                    "anomaly_score": 0.2,
                    "business_context": "This table supports interactive reporting."
                }
            }
        ]
        self.mock_recommendation_enhancer.enhance_recommendations.return_value = self.enhanced_recommendations
        
        # Set up evaluation metrics mock
        self.evaluation_results = {
            "summary": {
                "total_recommendations": 2,
                "ml_enhanced_count": 2,
                "ml_enhanced_percentage": 100.0,
                "avg_priority_score_change": 0.65
            }
        }
        self.mock_evaluation_metrics.evaluate_recommendations.return_value = self.evaluation_results
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    @patch('src.ml.enhancer.FeatureEngineeringPipeline')
    @patch('src.ml.enhancer.CostImpactClassifier')
    @patch('src.ml.enhancer.UsagePatternClustering')
    @patch('src.ml.enhancer.AnomalyDetector')
    @patch('src.ml.enhancer.RecommendationEnhancer')
    @patch('src.ml.enhancer.FeedbackCollector')
    @patch('src.ml.enhancer.EvaluationMetrics')
    def test_init(self, mock_eval_cls, mock_feedback_cls, mock_enhancer_cls,
                mock_anomaly_cls, mock_pattern_cls, mock_cost_cls, mock_feature_cls):
        """Test initialization of the ML Enhancement Module."""
        # Set up mocks
        mock_feature_cls.return_value = self.mock_feature_pipeline
        mock_cost_cls.return_value = self.mock_cost_classifier
        mock_pattern_cls.return_value = self.mock_pattern_clustering
        mock_anomaly_cls.return_value = self.mock_anomaly_detector
        mock_enhancer_cls.return_value = self.mock_recommendation_enhancer
        mock_feedback_cls.return_value = self.mock_feedback_collector
        mock_eval_cls.return_value = self.mock_evaluation_metrics
        
        # Initialize module
        module = MLEnhancementModule(
            project_id=self.project_id,
            model_dir=str(self.model_dir),
            use_pretrained=False
        )
        
        # Verify initialization
        self.assertEqual(module.project_id, self.project_id)
        self.assertEqual(module.model_dir, self.model_dir)
        
        # Verify components were initialized
        mock_feature_cls.assert_called_once()
        mock_cost_cls.assert_called_once()
        mock_pattern_cls.assert_called_once()
        mock_anomaly_cls.assert_called_once()
        mock_enhancer_cls.assert_called_once()
        mock_feedback_cls.assert_called_once()
        mock_eval_cls.assert_called_once()
    
    @patch('src.ml.enhancer.FeatureEngineeringPipeline')
    @patch('src.ml.enhancer.CostImpactClassifier')
    @patch('src.ml.enhancer.UsagePatternClustering')
    @patch('src.ml.enhancer.AnomalyDetector')
    @patch('src.ml.enhancer.RecommendationEnhancer')
    @patch('src.ml.enhancer.FeedbackCollector')
    @patch('src.ml.enhancer.EvaluationMetrics')
    def test_enhance_recommendations(self, mock_eval_cls, mock_feedback_cls, mock_enhancer_cls,
                                   mock_anomaly_cls, mock_pattern_cls, mock_cost_cls, mock_feature_cls):
        """Test enhancing recommendations with ML insights."""
        # Set up mocks
        mock_feature_cls.return_value = self.mock_feature_pipeline
        mock_cost_cls.return_value = self.mock_cost_classifier
        mock_pattern_cls.return_value = self.mock_pattern_clustering
        mock_anomaly_cls.return_value = self.mock_anomaly_detector
        mock_enhancer_cls.return_value = self.mock_recommendation_enhancer
        mock_feedback_cls.return_value = self.mock_feedback_collector
        mock_eval_cls.return_value = self.mock_evaluation_metrics
        
        # Initialize module
        module = MLEnhancementModule(
            project_id=self.project_id,
            model_dir=str(self.model_dir),
            use_pretrained=False
        )
        
        # Enhance recommendations
        result = module.enhance_recommendations(
            recommendations=self.recommendations,
            dataset_metadata=self.dataset_metadata
        )
        
        # Verify feature extraction was called
        self.mock_feature_pipeline.extract_features.assert_called_once_with(
            self.recommendations, self.dataset_metadata
        )
        
        # Verify models were called
        self.mock_cost_classifier.predict.assert_called_once()
        self.mock_pattern_clustering.predict.assert_called_once()
        self.mock_anomaly_detector.detect_anomalies.assert_called_once()
        
        # Verify enhancement was called
        self.mock_recommendation_enhancer.enhance_recommendations.assert_called_once()
        
        # Verify evaluation was called
        self.mock_evaluation_metrics.evaluate_recommendations.assert_called_once_with(
            original_recommendations=self.recommendations,
            enhanced_recommendations=self.enhanced_recommendations
        )
        
        # Verify result
        self.assertEqual(result, self.enhanced_recommendations)
    
    @patch('src.ml.enhancer.FeatureEngineeringPipeline')
    @patch('src.ml.enhancer.CostImpactClassifier')
    @patch('src.ml.enhancer.UsagePatternClustering')
    @patch('src.ml.enhancer.AnomalyDetector')
    @patch('src.ml.enhancer.RecommendationEnhancer')
    @patch('src.ml.enhancer.FeedbackCollector')
    @patch('src.ml.enhancer.EvaluationMetrics')
    def test_collect_feedback(self, mock_eval_cls, mock_feedback_cls, mock_enhancer_cls,
                            mock_anomaly_cls, mock_pattern_cls, mock_cost_cls, mock_feature_cls):
        """Test collecting and processing feedback."""
        # Set up mocks
        mock_feature_cls.return_value = self.mock_feature_pipeline
        mock_cost_cls.return_value = self.mock_cost_classifier
        mock_pattern_cls.return_value = self.mock_pattern_clustering
        mock_anomaly_cls.return_value = self.mock_anomaly_detector
        mock_enhancer_cls.return_value = self.mock_recommendation_enhancer
        mock_feedback_cls.return_value = self.mock_feedback_collector
        mock_eval_cls.return_value = self.mock_evaluation_metrics
        
        # Mock feature extraction from feedback
        feedback_features = pd.DataFrame({
            "recommendation_id": ["rec_1"],
            "actual_cost_savings": [900],
            "implementation_success": [True]
        })
        self.mock_feature_pipeline.extract_features_from_feedback.return_value = feedback_features
        
        # Initialize module
        module = MLEnhancementModule(
            project_id=self.project_id,
            model_dir=str(self.model_dir),
            use_pretrained=False
        )
        
        # Collect feedback
        module.collect_feedback(
            implemented_recommendations=[self.recommendations[0]],
            feedback_data=self.feedback_data
        )
        
        # Verify feedback storage was called
        self.mock_feedback_collector.store_feedback.assert_called_once_with(
            [self.recommendations[0]], self.feedback_data
        )
        
        # Verify feature extraction from feedback was called
        self.mock_feature_pipeline.extract_features_from_feedback.assert_called_once()
        
        # Verify models were updated with feedback
        self.mock_cost_classifier.update_with_feedback.assert_called_once()
        self.mock_pattern_clustering.update_with_feedback.assert_called_once()
        self.mock_anomaly_detector.update_with_feedback.assert_called_once()
    
    @patch('src.ml.enhancer.FeatureEngineeringPipeline')
    @patch('src.ml.enhancer.CostImpactClassifier')
    @patch('src.ml.enhancer.UsagePatternClustering')
    @patch('src.ml.enhancer.AnomalyDetector')
    @patch('src.ml.enhancer.RecommendationEnhancer')
    @patch('src.ml.enhancer.FeedbackCollector')
    @patch('src.ml.enhancer.EvaluationMetrics')
    def test_train_models(self, mock_eval_cls, mock_feedback_cls, mock_enhancer_cls,
                        mock_anomaly_cls, mock_pattern_cls, mock_cost_cls, mock_feature_cls):
        """Test training ML models."""
        # Set up mocks
        mock_feature_cls.return_value = self.mock_feature_pipeline
        mock_cost_cls.return_value = self.mock_cost_classifier
        mock_pattern_cls.return_value = self.mock_pattern_clustering
        mock_anomaly_cls.return_value = self.mock_anomaly_detector
        mock_enhancer_cls.return_value = self.mock_recommendation_enhancer
        mock_feedback_cls.return_value = self.mock_feedback_collector
        mock_eval_cls.return_value = self.mock_evaluation_metrics
        
        # Mock feature extraction from training data
        training_features = pd.DataFrame({
            "recommendation_id": ["rec_1", "rec_2"],
            "table_size_bytes": [1000000, 2000000],
            "actual_cost_savings": [900, 0]
        })
        self.mock_feature_pipeline.extract_features_from_training_data.return_value = training_features
        
        # Mock model training results
        cost_metrics = {"accuracy": 0.85, "precision": 0.88}
        pattern_metrics = {"silhouette": 0.75, "inertia": 120}
        anomaly_metrics = {"anomaly_percentage": 8.5}
        
        self.mock_cost_classifier.train.return_value = cost_metrics
        self.mock_pattern_clustering.train.return_value = pattern_metrics
        self.mock_anomaly_detector.train.return_value = anomaly_metrics
        
        # Initialize module
        module = MLEnhancementModule(
            project_id=self.project_id,
            model_dir=str(self.model_dir),
            use_pretrained=False
        )
        
        # Training data
        training_data = {
            "historical_recommendations": self.recommendations,
            "dataset_metadata": self.dataset_metadata,
            "feedback_data": self.feedback_data,
            "labels": {
                "business_impact_category": ["high", "medium"],
                "business_impact": [900, 400]
            }
        }
        
        # Train models
        result = module.train_models(training_data)
        
        # Verify feature extraction was called
        self.mock_feature_pipeline.extract_features_from_training_data.assert_called_once_with(training_data)
        
        # Verify models were trained
        self.mock_cost_classifier.train.assert_called_once()
        self.mock_pattern_clustering.train.assert_called_once()
        self.mock_anomaly_detector.train.assert_called_once()
        
        # Verify results include all model metrics
        self.assertEqual(result["cost_impact_classifier"], cost_metrics)
        self.assertEqual(result["pattern_clustering"], pattern_metrics)
        self.assertEqual(result["anomaly_detector"], anomaly_metrics)
    
    @patch('src.ml.enhancer.FeatureEngineeringPipeline')
    @patch('src.ml.enhancer.CostImpactClassifier')
    @patch('src.ml.enhancer.UsagePatternClustering')
    @patch('src.ml.enhancer.AnomalyDetector')
    @patch('src.ml.enhancer.RecommendationEnhancer')
    @patch('src.ml.enhancer.FeedbackCollector')
    @patch('src.ml.enhancer.EvaluationMetrics')
    def test_save_and_load_models(self, mock_eval_cls, mock_feedback_cls, mock_enhancer_cls,
                               mock_anomaly_cls, mock_pattern_cls, mock_cost_cls, mock_feature_cls):
        """Test saving and loading ML models."""
        # Set up mocks
        mock_feature_cls.return_value = self.mock_feature_pipeline
        mock_cost_cls.return_value = self.mock_cost_classifier
        mock_pattern_cls.return_value = self.mock_pattern_clustering
        mock_anomaly_cls.return_value = self.mock_anomaly_detector
        mock_enhancer_cls.return_value = self.mock_recommendation_enhancer
        mock_feedback_cls.return_value = self.mock_feedback_collector
        mock_eval_cls.return_value = self.mock_evaluation_metrics
        
        # Initialize module
        module = MLEnhancementModule(
            project_id=self.project_id,
            model_dir=str(self.model_dir),
            use_pretrained=False
        )
        
        # Save models
        module.save_models()
        
        # Verify models were saved
        self.mock_cost_classifier.save.assert_called_once()
        self.mock_pattern_clustering.save.assert_called_once()
        self.mock_anomaly_detector.save.assert_called_once()
        
        # Reset mocks
        self.mock_cost_classifier.save.reset_mock()
        self.mock_pattern_clustering.save.reset_mock()
        self.mock_anomaly_detector.save.reset_mock()
        
        # Load models
        module.load_models()
        
        # Verify models were loaded
        self.mock_cost_classifier.load.assert_called_once()
        self.mock_pattern_clustering.load.assert_called_once()
        self.mock_anomaly_detector.load.assert_called_once()
    
    @patch('src.ml.enhancer.FeatureEngineeringPipeline')
    @patch('src.ml.enhancer.CostImpactClassifier')
    @patch('src.ml.enhancer.UsagePatternClustering')
    @patch('src.ml.enhancer.AnomalyDetector')
    @patch('src.ml.enhancer.RecommendationEnhancer')
    @patch('src.ml.enhancer.FeedbackCollector')
    @patch('src.ml.enhancer.EvaluationMetrics')
    def test_generate_ml_insights_report(self, mock_eval_cls, mock_feedback_cls, mock_enhancer_cls,
                                      mock_anomaly_cls, mock_pattern_cls, mock_cost_cls, mock_feature_cls):
        """Test generating ML insights report."""
        # Set up mocks
        mock_feature_cls.return_value = self.mock_feature_pipeline
        mock_cost_cls.return_value = self.mock_cost_classifier
        mock_pattern_cls.return_value = self.mock_pattern_clustering
        mock_anomaly_cls.return_value = self.mock_anomaly_detector
        mock_enhancer_cls.return_value = self.mock_recommendation_enhancer
        mock_feedback_cls.return_value = self.mock_feedback_collector
        mock_eval_cls.return_value = self.mock_evaluation_metrics
        
        # Initialize module
        module = MLEnhancementModule(
            project_id=self.project_id,
            model_dir=str(self.model_dir),
            use_pretrained=False
        )
        
        # Generate report
        report = module.generate_ml_insights_report(self.enhanced_recommendations)
        
        # Verify report structure
        self.assertIn("patterns", report)
        self.assertIn("anomalies", report)
        self.assertIn("anomaly_count", report)
        self.assertIn("impact_distribution", report)
        self.assertIn("ml_enhanced_count", report)
        
        # Verify patterns were extracted
        self.assertEqual(report["ml_enhanced_count"], 2)
        self.assertEqual(report["anomaly_count"], 1)
        self.assertEqual(len(report["patterns"]), 2)


if __name__ == '__main__':
    unittest.main()