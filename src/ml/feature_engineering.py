"""Feature Engineering Pipeline for ML Enhancement Module.

This module provides functionality for extracting features from BigQuery metadata,
usage patterns, and recommendation data to feed into ML models.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
import pandas as pd
import numpy as np
import logging
from google.cloud import bigquery
from pathlib import Path
import datetime
import json

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class FeatureEngineeringPipeline:
    """Feature Engineering Pipeline for extracting ML features from BigQuery data.
    
    This class extracts features from BigQuery metadata, usage patterns, and
    recommendations data to feed into ML models for pattern recognition,
    anomaly detection, and cost impact prediction.
    """
    
    def __init__(self, 
                 project_id: str, 
                 credentials_path: Optional[str] = None):
        """Initialize the Feature Engineering Pipeline.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to GCP service account credentials (optional)
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        
        # Initialize BigQuery client
        try:
            if credentials_path:
                self.client = bigquery.Client.from_service_account_json(credentials_path)
            else:
                self.client = bigquery.Client(project=project_id)
            logger.info(f"Initialized BigQuery client for project {project_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize BigQuery client: {e}")
            self.client = None
        
        # Define feature sets
        self.feature_sets = {
            "table_metadata": self._extract_table_metadata_features,
            "query_patterns": self._extract_query_pattern_features,
            "recommendation": self._extract_recommendation_features,
            "usage_patterns": self._extract_usage_pattern_features
        }
        
    def extract_features(self, 
                        recommendations: List[Dict[str, Any]], 
                        dataset_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Extract features from recommendations and dataset metadata.
        
        Args:
            recommendations: List of standardized recommendations
            dataset_metadata: Metadata about the dataset being analyzed
            
        Returns:
            DataFrame containing features for each recommendation
        """
        logger.info(f"Extracting features from {len(recommendations)} recommendations")
        
        # Initialize an empty list to store feature dictionaries
        feature_dicts = []
        
        # Process each recommendation
        for rec in recommendations:
            # Initialize a dictionary for this recommendation's features
            rec_features = {
                "recommendation_id": rec.get("recommendation_id", "unknown"),
                "recommendation_type": rec.get("recommendation_type", "unknown")
            }
            
            # Extract table metadata features
            if "target_table" in rec:
                table_features = self._extract_table_metadata_features(
                    rec["target_table"], 
                    dataset_metadata
                )
                rec_features.update(table_features)
            
            # Extract query pattern features if applicable
            if rec.get("recommendation_type") in ["query_optimization", "materialized_view"]:
                query_features = self._extract_query_pattern_features(
                    rec,
                    dataset_metadata
                )
                rec_features.update(query_features)
            
            # Extract recommendation-specific features
            rec_specific_features = self._extract_recommendation_features(rec)
            rec_features.update(rec_specific_features)
            
            # Extract usage pattern features
            usage_features = self._extract_usage_pattern_features(
                rec,
                dataset_metadata
            )
            rec_features.update(usage_features)
            
            # Add the recommendation's features to our list
            feature_dicts.append(rec_features)
        
        # Convert to DataFrame
        if feature_dicts:
            features_df = pd.DataFrame(feature_dicts)
            
            # Fill NaN values with appropriate defaults
            features_df = features_df.fillna({
                col: 0 if col.endswith('_count') or col.endswith('_size') or col.endswith('_bytes') 
                     else 'unknown' if col.endswith('_type') or col.endswith('_name')
                     else np.nan
                for col in features_df.columns
            })
            
            logger.info(f"Generated feature dataframe with shape {features_df.shape}")
            return features_df
        else:
            logger.warning("No features could be extracted, returning empty DataFrame")
            return pd.DataFrame()
    
    def extract_features_from_feedback(self,
                                     implemented_recommendations: List[Dict[str, Any]],
                                     feedback_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract features from feedback on implemented recommendations.
        
        Args:
            implemented_recommendations: List of recommendations that were implemented
            feedback_data: Feedback data about the implementations
            
        Returns:
            DataFrame containing features for each implemented recommendation with feedback
        """
        logger.info(f"Extracting features from {len(implemented_recommendations)} recommendation feedbacks")
        
        # Initialize an empty list to store feature dictionaries
        feature_dicts = []
        
        # Process each implemented recommendation
        for rec in implemented_recommendations:
            # Get the recommendation ID
            rec_id = rec.get("recommendation_id", "unknown")
            
            # Check if we have feedback for this recommendation
            if rec_id not in feedback_data.get("recommendations", {}):
                continue
                
            rec_feedback = feedback_data["recommendations"][rec_id]
            
            # Initialize a dictionary for this recommendation's features
            rec_features = {
                "recommendation_id": rec_id,
                "recommendation_type": rec.get("recommendation_type", "unknown"),
                "implemented": True,
                "implementation_success": rec_feedback.get("success", False),
                "actual_cost_savings": rec_feedback.get("actual_cost_savings", 0),
                "expected_cost_savings": rec.get("estimated_savings", {}).get("total", 0),
                "savings_accuracy": 0,  # Will calculate below
                "implementation_time": rec_feedback.get("implementation_time_minutes", 0),
                "complexity_actual": rec_feedback.get("complexity_rating", 3),
                "complexity_predicted": rec.get("complexity", 3),
                "complexity_accuracy": 0,  # Will calculate below
                "feedback_date": rec_feedback.get("date", datetime.datetime.now().isoformat())
            }
            
            # Calculate accuracy metrics
            if rec_features["expected_cost_savings"] > 0:
                rec_features["savings_accuracy"] = (
                    rec_features["actual_cost_savings"] / rec_features["expected_cost_savings"]
                )
            
            if rec_features["complexity_predicted"] > 0:
                rec_features["complexity_accuracy"] = (
                    1 - abs(rec_features["complexity_actual"] - rec_features["complexity_predicted"]) / 5
                )
            
            # Extract the basic recommendation features 
            basic_features = self.extract_features([rec], feedback_data.get("dataset_metadata", {}))
            
            # If basic features were successfully extracted, merge them
            if not basic_features.empty:
                rec_row = basic_features.iloc[0].to_dict()
                rec_features.update(rec_row)
            
            # Add feedback-specific features
            rec_features.update({
                "user_rating": rec_feedback.get("user_rating", 0),
                "business_impact_actual": rec_feedback.get("business_impact", 0),
                "comments": rec_feedback.get("comments", ""),
                "has_negative_impact": "negative_impacts" in rec_feedback and len(rec_feedback["negative_impacts"]) > 0
            })
            
            # Add the recommendation's features to our list
            feature_dicts.append(rec_features)
        
        # Convert to DataFrame
        if feature_dicts:
            features_df = pd.DataFrame(feature_dicts)
            logger.info(f"Generated feedback feature dataframe with shape {features_df.shape}")
            return features_df
        else:
            logger.warning("No feedback features could be extracted, returning empty DataFrame")
            return pd.DataFrame()
    
    def extract_features_from_training_data(self, training_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract features from provided training data.
        
        Args:
            training_data: Training data for ML models
            
        Returns:
            DataFrame containing features for training
        """
        logger.info("Extracting features from training data")
        
        # Initialize feature dataframes list
        feature_dfs = []
        
        # Process historical recommendations if available
        if "historical_recommendations" in training_data and "dataset_metadata" in training_data:
            historical_features = self.extract_features(
                training_data["historical_recommendations"],
                training_data["dataset_metadata"]
            )
            if not historical_features.empty:
                historical_features["data_source"] = "historical"
                feature_dfs.append(historical_features)
        
        # Process feedback data if available
        if "feedback_data" in training_data:
            feedback_features = self.extract_features_from_feedback(
                training_data.get("implemented_recommendations", []),
                training_data["feedback_data"]
            )
            if not feedback_features.empty:
                feedback_features["data_source"] = "feedback"
                feature_dfs.append(feedback_features)
        
        # Process synthetic data if available
        if "synthetic_data" in training_data:
            synthetic_features = pd.DataFrame(training_data["synthetic_data"])
            if not synthetic_features.empty:
                synthetic_features["data_source"] = "synthetic"
                feature_dfs.append(synthetic_features)
        
        # Combine all feature dataframes
        if feature_dfs:
            combined_features = pd.concat(feature_dfs, ignore_index=True)
            logger.info(f"Generated combined feature dataframe with shape {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No training features could be extracted, returning empty DataFrame")
            return pd.DataFrame()
    
    def _extract_table_metadata_features(self, 
                                       table_ref: str, 
                                       dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from table metadata.
        
        Args:
            table_ref: Table reference (project.dataset.table)
            dataset_metadata: Dataset metadata
            
        Returns:
            Dictionary of table metadata features
        """
        features = {}
        
        # Find the table in the dataset metadata
        table_metadata = None
        for table in dataset_metadata.get("tables", []):
            if table.get("table_ref") == table_ref:
                table_metadata = table
                break
        
        if not table_metadata:
            logger.warning(f"Table metadata not found for {table_ref}")
            return features
        
        # Table size features
        features.update({
            "table_size_bytes": table_metadata.get("size_bytes", 0),
            "table_num_rows": table_metadata.get("num_rows", 0),
            "table_num_columns": len(table_metadata.get("schema", [])),
            "is_partitioned": "partitioning" in table_metadata and len(table_metadata["partitioning"]) > 0,
            "is_clustered": "clustering" in table_metadata and len(table_metadata["clustering"]) > 0,
            "creation_date_days": 0  # Will calculate below
        })
        
        # Calculate age of table in days
        if "creation_time" in table_metadata:
            try:
                creation_time = datetime.datetime.fromisoformat(table_metadata["creation_time"].replace('Z', '+00:00'))
                now = datetime.datetime.now(datetime.timezone.utc)
                features["creation_date_days"] = (now - creation_time).days
            except (ValueError, TypeError):
                pass
        
        # Schema complexity features
        nested_field_count = 0
        repeated_field_count = 0
        data_types = set()
        
        for column in table_metadata.get("schema", []):
            data_types.add(column.get("type", "").lower())
            
            if "mode" in column and column["mode"] == "REPEATED":
                repeated_field_count += 1
                
            if "fields" in column:
                nested_field_count += 1
        
        features.update({
            "nested_field_count": nested_field_count,
            "repeated_field_count": repeated_field_count,
            "data_type_diversity": len(data_types),
            "has_geography": "geography" in data_types,
            "has_json": "json" in data_types
        })
        
        # Usage statistics
        features.update({
            "query_count_last_30d": table_metadata.get("usage_statistics", {}).get("query_count_last_30d", 0),
            "bytes_processed_last_30d": table_metadata.get("usage_statistics", {}).get("bytes_processed_last_30d", 0),
            "avg_query_execution_time": table_metadata.get("usage_statistics", {}).get("avg_execution_time_ms", 0) / 1000 if "usage_statistics" in table_metadata and "avg_execution_time_ms" in table_metadata["usage_statistics"] else 0,
            "unique_users_last_30d": len(table_metadata.get("usage_statistics", {}).get("unique_users_last_30d", [])),
            "slot_ms_last_30d": table_metadata.get("usage_statistics", {}).get("slot_ms_last_30d", 0)
        })
        
        return features
    
    def _extract_query_pattern_features(self,
                                      recommendation: Dict[str, Any],
                                      dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from query patterns related to a recommendation.
        
        Args:
            recommendation: Recommendation data
            dataset_metadata: Dataset metadata
            
        Returns:
            Dictionary of query pattern features
        """
        features = {}
        
        # Get query patterns from the recommendation
        query_patterns = recommendation.get("query_patterns", [])
        
        if not query_patterns:
            return features
        
        # Basic query statistics
        features.update({
            "query_pattern_count": len(query_patterns),
            "total_query_executions": sum(qp.get("execution_count", 0) for qp in query_patterns),
            "avg_query_cost": sum(qp.get("avg_cost", 0) * qp.get("execution_count", 0) 
                                for qp in query_patterns) / sum(qp.get("execution_count", 1) for qp in query_patterns)
                                if sum(qp.get("execution_count", 0) for qp in query_patterns) > 0 else 0,
            "max_query_bytes_processed": max((qp.get("avg_bytes_processed", 0) for qp in query_patterns), default=0),
            "has_frequent_queries": any(qp.get("execution_count", 0) > 10 for qp in query_patterns)
        })
        
        # Query complexity features
        complexity_metrics = []
        for qp in query_patterns:
            # Calculate complexity score based on query properties
            complexity = 0
            complexity += 2 if qp.get("has_joins", False) else 0
            complexity += 3 if qp.get("has_subqueries", False) else 0
            complexity += 1 if qp.get("has_window_functions", False) else 0
            complexity += 2 if qp.get("has_wildcards", False) else 0
            complexity += qp.get("join_count", 0)
            complexity += qp.get("subquery_count", 0) * 2
            complexity_metrics.append(complexity)
        
        if complexity_metrics:
            features.update({
                "avg_query_complexity": sum(complexity_metrics) / len(complexity_metrics),
                "max_query_complexity": max(complexity_metrics),
                "has_complex_queries": any(c > 5 for c in complexity_metrics)
            })
        
        # Query pattern features
        has_filters = False
        has_joins = False
        has_group_by = False
        has_order_by = False
        has_limit = False
        
        for qp in query_patterns:
            has_filters = has_filters or qp.get("has_filters", False)
            has_joins = has_joins or qp.get("has_joins", False)
            has_group_by = has_group_by or qp.get("has_group_by", False)
            has_order_by = has_order_by or qp.get("has_order_by", False)
            has_limit = has_limit or qp.get("has_limit", False)
        
        features.update({
            "has_filters": has_filters,
            "has_joins": has_joins,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_limit": has_limit
        })
        
        return features
    
    def _extract_recommendation_features(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features specific to the recommendation itself.
        
        Args:
            recommendation: Recommendation data
            
        Returns:
            Dictionary of recommendation-specific features
        """
        features = {
            "recommendation_type": recommendation.get("recommendation_type", "unknown"),
            "estimated_savings_total": recommendation.get("estimated_savings", {}).get("total", 0),
            "estimated_savings_monthly": recommendation.get("estimated_savings", {}).get("monthly", 0),
            "implementation_complexity": recommendation.get("complexity", 3),
            "has_implementation_risk": recommendation.get("risk_level", "low") != "low",
            "risk_level_encoded": {"low": 1, "medium": 2, "high": 3}.get(recommendation.get("risk_level", "low"), 1),
            "priority_score": recommendation.get("priority_score", 0)
        }
        
        # Type-specific features
        if recommendation.get("recommendation_type") == "partition":
            features.update({
                "partition_type": recommendation.get("partition_type", "unknown"),
                "partition_column": recommendation.get("partition_column", "unknown"),
                "partition_count_estimate": recommendation.get("partition_count_estimate", 0)
            })
        elif recommendation.get("recommendation_type") == "cluster":
            features.update({
                "cluster_columns": len(recommendation.get("clustering_columns", [])),
                "cluster_columns_cardinality": recommendation.get("column_cardinality_score", 0)
            })
        elif recommendation.get("recommendation_type") == "schema_optimization":
            features.update({
                "column_count": len(recommendation.get("columns", [])),
                "is_type_change": recommendation.get("optimization_type", "") == "type_change"
            })
        elif recommendation.get("recommendation_type") == "materialized_view":
            features.update({
                "query_count_affected": recommendation.get("affected_query_count", 0),
                "view_refresh_frequency": recommendation.get("refresh_frequency_hours", 24)
            })
        
        return features
    
    def _extract_usage_pattern_features(self,
                                      recommendation: Dict[str, Any],
                                      dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from usage patterns related to a recommendation.
        
        Args:
            recommendation: Recommendation data
            dataset_metadata: Dataset metadata
            
        Returns:
            Dictionary of usage pattern features
        """
        features = {}
        
        # Extract target table usage patterns if available
        if "target_table" in recommendation:
            table_ref = recommendation["target_table"]
            table_metadata = None
            
            # Find the table in the dataset metadata
            for table in dataset_metadata.get("tables", []):
                if table.get("table_ref") == table_ref:
                    table_metadata = table
                    break
            
            if table_metadata and "usage_patterns" in table_metadata:
                usage_patterns = table_metadata["usage_patterns"]
                
                # Calculate usage pattern features
                features.update({
                    "usage_frequency": usage_patterns.get("usage_frequency", "unknown"),
                    "usage_frequency_encoded": {"unknown": 0, "rare": 1, "occasional": 2, "frequent": 3, "very_frequent": 4}.get(
                        usage_patterns.get("usage_frequency", "unknown"), 0
                    ),
                    "usage_growth_trend": usage_patterns.get("growth_trend", 0),
                    "usage_seasonality": usage_patterns.get("has_seasonality", False),
                    "peak_usage_hour": usage_patterns.get("peak_usage_hour", -1),
                    "has_weekday_pattern": usage_patterns.get("has_weekday_pattern", False),
                    "has_monthly_pattern": usage_patterns.get("has_monthly_pattern", False),
                    "data_staleness_days": usage_patterns.get("data_staleness_days", 0),
                    "has_increasing_trend": usage_patterns.get("growth_trend", 0) > 0.05,
                    "has_decreasing_trend": usage_patterns.get("growth_trend", 0) < -0.05
                })
                
                # User diversity features
                features.update({
                    "user_count": usage_patterns.get("unique_user_count", 0),
                    "user_diversity": usage_patterns.get("user_diversity", "low"),
                    "user_diversity_encoded": {"unknown": 0, "low": 1, "medium": 2, "high": 3}.get(
                        usage_patterns.get("user_diversity", "low"), 0
                    ),
                    "has_automated_queries": usage_patterns.get("has_automated_queries", False),
                    "interactive_query_ratio": usage_patterns.get("interactive_query_ratio", 0)
                })
                
                # Query type distribution
                query_types = usage_patterns.get("query_type_distribution", {})
                if query_types:
                    features.update({
                        "export_query_ratio": query_types.get("export", 0),
                        "reporting_query_ratio": query_types.get("reporting", 0),
                        "etl_query_ratio": query_types.get("etl", 0),
                        "ad_hoc_query_ratio": query_types.get("ad_hoc", 0),
                        "dominant_query_type": max(query_types.items(), key=lambda x: x[1])[0] if query_types else "unknown"
                    })
        
        return features