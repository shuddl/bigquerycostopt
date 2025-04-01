"""Machine Learning Enhancement Module for BigQuery Cost Intelligence Engine.

This module provides ML-based enhancements to the basic rule-based recommendations
from the optimizer modules. It detects patterns, identifies anomalies, adds business
context, and improves recommendation quality over time through feedback loops.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
import pandas as pd
import numpy as np
import logging
import json
import datetime
import hashlib
from pathlib import Path

from .feature_engineering import FeatureEngineeringPipeline
from .models import CostImpactClassifier, UsagePatternClustering, AnomalyDetector
from .recommendation_enhancer import RecommendationEnhancer
from .feedback import FeedbackCollector
from .evaluation import EvaluationMetrics
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class MLEnhancementModule:
    """Machine Learning Enhancement Module for BigQuery recommendations.
    
    This class coordinates all ML functionality, interfaces with the recommendation engine,
    and manages the models and prediction pipelines to enhance recommendations with
    ML-derived insights and business context.
    """
    
    def __init__(self, 
                 project_id: str, 
                 model_dir: Optional[str] = None,
                 credentials_path: Optional[str] = None,
                 use_pretrained: bool = True):
        """Initialize the ML Enhancement Module.
        
        Args:
            project_id: GCP project ID
            model_dir: Directory to store/load trained models (default: 'models/' in package dir)
            credentials_path: Path to GCP service account credentials
            use_pretrained: Whether to use pre-trained models (if available)
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        
        # Set model directory
        if model_dir is None:
            # Default to a 'models' directory in the package
            self.model_dir = Path(__file__).parent.parent.parent / "models"
        else:
            self.model_dir = Path(model_dir)
            
        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_pipeline = FeatureEngineeringPipeline(
            project_id=project_id, 
            credentials_path=credentials_path
        )
        
        # Initialize ML models
        self.cost_impact_classifier = CostImpactClassifier(
            model_path=self.model_dir / "cost_impact_classifier",
            use_pretrained=use_pretrained
        )
        
        self.pattern_clustering = UsagePatternClustering(
            model_path=self.model_dir / "usage_pattern_clustering",
            use_pretrained=use_pretrained
        )
        
        self.anomaly_detector = AnomalyDetector(
            model_path=self.model_dir / "anomaly_detector",
            use_pretrained=use_pretrained
        )
        
        # Initialize recommendation enhancer
        self.recommendation_enhancer = RecommendationEnhancer(
            models={
                "cost_impact": self.cost_impact_classifier,
                "pattern_clustering": self.pattern_clustering,
                "anomaly_detector": self.anomaly_detector
            }
        )
        
        # Initialize feedback collector
        self.feedback_collector = FeedbackCollector(
            project_id=project_id,
            credentials_path=credentials_path
        )
        
        # Initialize evaluation metrics
        self.evaluation_metrics = EvaluationMetrics()
        
        logger.info(f"Initialized ML Enhancement Module for project {project_id}")
        
    def enhance_recommendations(self, 
                              recommendations: List[Dict[str, Any]], 
                              dataset_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance recommendations with ML-derived insights.
        
        This method applies ML models to enhance recommendations with business context,
        improved prioritization, anomaly detection, and pattern recognition.
        
        Args:
            recommendations: List of standardized recommendations from optimizer modules
            dataset_metadata: Metadata about the dataset being analyzed
            
        Returns:
            Enhanced recommendations with ML-derived insights
        """
        logger.info(f"Enhancing {len(recommendations)} recommendations with ML insights")
        
        # Extract features from recommendations and dataset metadata
        features = self.feature_pipeline.extract_features(recommendations, dataset_metadata)
        
        # Perform cost impact classification
        cost_impact_results = self.cost_impact_classifier.predict(features)
        
        # Perform usage pattern clustering
        pattern_clusters = self.pattern_clustering.predict(features)
        
        # Perform anomaly detection
        anomalies = self.anomaly_detector.detect_anomalies(features)
        
        # Combine ML insights
        ml_insights = {
            "cost_impact": cost_impact_results,
            "pattern_clusters": pattern_clusters,
            "anomalies": anomalies
        }
        
        # Enhance recommendations with ML insights
        enhanced_recommendations = self.recommendation_enhancer.enhance_recommendations(
            recommendations, 
            ml_insights,
            dataset_metadata
        )
        
        # Calculate evaluation metrics
        evaluation_results = self.evaluation_metrics.evaluate_recommendations(
            original_recommendations=recommendations,
            enhanced_recommendations=enhanced_recommendations
        )
        
        logger.info(f"Enhanced recommendations: {evaluation_results['summary']}")
        
        return enhanced_recommendations
    
    def collect_feedback(self, 
                        implemented_recommendations: List[Dict[str, Any]], 
                        feedback_data: Dict[str, Any]) -> None:
        """Collect feedback on implemented recommendations to improve future recommendations.
        
        Args:
            implemented_recommendations: List of recommendations that were implemented
            feedback_data: Feedback data about the implementations
        """
        logger.info(f"Collecting feedback on {len(implemented_recommendations)} implemented recommendations")
        
        # Store feedback in the feedback collection system
        self.feedback_collector.store_feedback(implemented_recommendations, feedback_data)
        
        # Update models based on feedback
        feature_data = self.feature_pipeline.extract_features_from_feedback(
            implemented_recommendations, 
            feedback_data
        )
        
        # Update each model with the feedback data
        self.cost_impact_classifier.update_with_feedback(feature_data, feedback_data)
        self.pattern_clustering.update_with_feedback(feature_data, feedback_data)
        self.anomaly_detector.update_with_feedback(feature_data, feedback_data)
        
        logger.info("Models updated with feedback data")
    
    def train_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train or retrain all ML models using provided training data.
        
        Args:
            training_data: Training data for ML models
            
        Returns:
            Training results and metrics
        """
        logger.info("Training ML models")
        
        # Extract features from training data
        features = self.feature_pipeline.extract_features_from_training_data(training_data)
        
        # Train each model
        cost_impact_metrics = self.cost_impact_classifier.train(features, training_data)
        pattern_clustering_metrics = self.pattern_clustering.train(features, training_data)
        anomaly_detector_metrics = self.anomaly_detector.train(features, training_data)
        
        # Combine training metrics
        training_results = {
            "cost_impact_classifier": cost_impact_metrics,
            "pattern_clustering": pattern_clustering_metrics,
            "anomaly_detector": anomaly_detector_metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Models trained successfully: {json.dumps(training_results, default=str)}")
        
        return training_results
    
    def generate_ml_insights_report(self, 
                                  enhanced_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a report of ML-derived insights from enhanced recommendations.
        
        Args:
            enhanced_recommendations: List of ML-enhanced recommendations
            
        Returns:
            Report of ML insights and patterns detected
        """
        logger.info("Generating ML insights report")
        
        # Extract ML insights from enhanced recommendations
        ml_insights = {}
        
        # Get unique patterns and their frequencies
        patterns = {}
        for rec in enhanced_recommendations:
            if "ml_insights" in rec and "pattern_name" in rec["ml_insights"]:
                pattern = rec["ml_insights"]["pattern_name"]
                if pattern not in patterns:
                    patterns[pattern] = {
                        "count": 0,
                        "avg_impact": 0,
                        "recommendations": []
                    }
                patterns[pattern]["count"] += 1
                patterns[pattern]["avg_impact"] += rec.get("ml_insights", {}).get("business_impact", 0)
                patterns[pattern]["recommendations"].append(rec["recommendation_id"])
        
        # Calculate averages
        for pattern in patterns:
            if patterns[pattern]["count"] > 0:
                patterns[pattern]["avg_impact"] /= patterns[pattern]["count"]
        
        # Get anomalies
        anomalies = [rec for rec in enhanced_recommendations if rec.get("ml_insights", {}).get("is_anomaly", False)]
        
        # Get business impact distribution
        impact_distribution = {}
        for rec in enhanced_recommendations:
            impact = rec.get("ml_insights", {}).get("business_impact_category", "unknown")
            if impact not in impact_distribution:
                impact_distribution[impact] = 0
            impact_distribution[impact] += 1
        
        # Build report
        ml_insights = {
            "patterns": patterns,
            "anomalies": [a["recommendation_id"] for a in anomalies],
            "anomaly_count": len(anomalies),
            "impact_distribution": impact_distribution,
            "ml_enhanced_count": len([r for r in enhanced_recommendations if "ml_insights" in r]),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Generated ML insights report with {len(patterns)} patterns and {len(anomalies)} anomalies")
        
        return ml_insights
    
    def save_models(self, output_dir: Optional[str] = None) -> None:
        """Save all ML models to disk.
        
        Args:
            output_dir: Directory to save models (default: self.model_dir)
        """
        save_dir = Path(output_dir) if output_dir else self.model_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving ML models to {save_dir}")
        
        # Save each model
        self.cost_impact_classifier.save(save_dir / "cost_impact_classifier")
        self.pattern_clustering.save(save_dir / "usage_pattern_clustering")
        self.anomaly_detector.save(save_dir / "anomaly_detector")
        
        logger.info("Models saved successfully")
    
    def load_models(self, model_dir: Optional[str] = None) -> None:
        """Load all ML models from disk.
        
        Args:
            model_dir: Directory to load models from (default: self.model_dir)
        """
        load_dir = Path(model_dir) if model_dir else self.model_dir
        
        logger.info(f"Loading ML models from {load_dir}")
        
        # Load each model
        self.cost_impact_classifier.load(load_dir / "cost_impact_classifier")
        self.pattern_clustering.load(load_dir / "usage_pattern_clustering")
        self.anomaly_detector.load(load_dir / "anomaly_detector")
        
        logger.info("Models loaded successfully")