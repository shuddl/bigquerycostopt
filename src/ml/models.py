"""Machine Learning Models for BigQuery Cost Intelligence Engine.

This module provides implementation of various ML models used for enhancing 
cost optimization recommendations.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
import pandas as pd
import numpy as np
import pickle
import json
import datetime
import logging
from pathlib import Path
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class BaseModel:
    """Base class for all ML models in the system."""
    
    def __init__(self, model_path: Optional[Path] = None, use_pretrained: bool = True):
        """Initialize the base model.
        
        Args:
            model_path: Path to save/load the model
            use_pretrained: Whether to use pre-trained models if available
        """
        self.model_path = model_path
        self.use_pretrained = use_pretrained
        self.model = None
        self.preprocessor = None
        self.metrics = {}
        self.feature_importance = {}
        self.training_time = None
        self.model_version = "0.1.0"
        
        # Create model directory if it doesn't exist
        if model_path:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Try to load pre-trained model if requested
        if use_pretrained and model_path and model_path.exists():
            try:
                self.load(model_path)
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")
    
    def preprocess(self, features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess features for model input.
        
        This method should be implemented by subclasses to handle feature preprocessing.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Tuple of (preprocessed_features, feature_names)
        """
        raise NotImplementedError("Subclasses must implement preprocess method")
    
    def train(self, features: pd.DataFrame, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model on the provided features.
        
        This method should be implemented by subclasses to handle model training.
        
        Args:
            features: DataFrame of features
            training_data: Additional training data
            
        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, features: pd.DataFrame) -> Any:
        """Make predictions using the trained model.
        
        This method should be implemented by subclasses to handle predictions.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Model predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def update_with_feedback(self, 
                           features: pd.DataFrame, 
                           feedback_data: Dict[str, Any]) -> None:
        """Update the model with feedback data.
        
        This method should be implemented by subclasses to handle model updates.
        
        Args:
            features: DataFrame of features from feedback
            feedback_data: Feedback data
        """
        raise NotImplementedError("Subclasses must implement update_with_feedback method")
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model (default: self.model_path)
        """
        save_path = path if path else self.model_path
        if not save_path:
            logger.warning("No save path provided, model not saved")
            return
            
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model metadata
        metadata = {
            "model_type": self.__class__.__name__,
            "model_version": self.model_version,
            "training_time": self.training_time,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            # Save model
            if self.model is not None:
                joblib.dump(self.model, save_path / "model.joblib")
            
            # Save preprocessor
            if self.preprocessor is not None:
                joblib.dump(self.preprocessor, save_path / "preprocessor.joblib")
            
            # Save metadata
            with open(save_path / "metadata.json", "w") as f:
                json.dump(metadata, f, default=str, indent=2)
                
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load the model from disk.
        
        Args:
            path: Path to load the model from (default: self.model_path)
        """
        load_path = path if path else self.model_path
        if not load_path:
            logger.warning("No load path provided, model not loaded")
            return
            
        try:
            # Load model
            model_file = load_path / "model.joblib"
            if model_file.exists():
                self.model = joblib.load(model_file)
            else:
                logger.warning(f"Model file not found at {model_file}")
            
            # Load preprocessor
            preprocessor_file = load_path / "preprocessor.joblib"
            if preprocessor_file.exists():
                self.preprocessor = joblib.load(preprocessor_file)
            else:
                logger.warning(f"Preprocessor file not found at {preprocessor_file}")
            
            # Load metadata
            metadata_file = load_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                self.model_version = metadata.get("model_version", "0.1.0")
                self.metrics = metadata.get("metrics", {})
                self.feature_importance = metadata.get("feature_importance", {})
                self.training_time = metadata.get("training_time")
                
                logger.info(f"Model loaded from {load_path} (version {self.model_version})")
            else:
                logger.warning(f"Metadata file not found at {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


class CostImpactClassifier(BaseModel):
    """Classifier for predicting the cost impact of recommendations.
    
    This model classifies recommendations into different cost impact categories
    and predicts the expected business impact for each recommendation.
    """
    
    def __init__(self, model_path: Optional[Path] = None, use_pretrained: bool = True):
        """Initialize the Cost Impact Classifier.
        
        Args:
            model_path: Path to save/load the model
            use_pretrained: Whether to use pre-trained models if available
        """
        super().__init__(model_path, use_pretrained)
        
        # Define categorical and numerical features for preprocessing
        self.categorical_features = [
            "recommendation_type", "dominant_query_type", "usage_frequency",
            "user_diversity", "table_name", "partition_type"
        ]
        
        self.numerical_features = [
            "table_size_bytes", "table_num_rows", "table_num_columns", 
            "query_count_last_30d", "bytes_processed_last_30d", "slot_ms_last_30d",
            "usage_growth_trend", "peak_usage_hour", "data_staleness_days",
            "user_count", "export_query_ratio", "reporting_query_ratio",
            "etl_query_ratio", "ad_hoc_query_ratio", "estimated_savings_total",
            "implementation_complexity", "risk_level_encoded", "priority_score",
            "nested_field_count", "repeated_field_count", "data_type_diversity"
        ]
        
        # Initialize model if not loaded from disk
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        
        # Initialize preprocessor if not loaded from disk
        if self.preprocessor is None:
            self._init_preprocessor()
    
    def _init_preprocessor(self) -> None:
        """Initialize the feature preprocessor pipeline."""
        # Create transformers for categorical and numerical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Combine transformers in a column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_features),
                ('num', numerical_transformer, self.numerical_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )
    
    def preprocess(self, features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess features for model input.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Tuple of (preprocessed_features, feature_names)
        """
        # Ensure all necessary columns exist, with defaults if missing
        for col in self.categorical_features:
            if col not in features.columns:
                features[col] = "unknown"
                
        for col in self.numerical_features:
            if col not in features.columns:
                features[col] = 0
        
        # Prepare feature names
        feature_names = self.categorical_features + self.numerical_features
        
        # Apply preprocessing
        if self.preprocessor is None:
            self._init_preprocessor()
            # Fit the preprocessor if not already fit
            self.preprocessor.fit(features[feature_names])
            
        # Transform the features
        processed_features = self.preprocessor.transform(features[feature_names])
        
        return processed_features, feature_names
    
    def train(self, features: pd.DataFrame, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the cost impact classifier.
        
        Args:
            features: DataFrame of features
            training_data: Additional training data containing labels
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training Cost Impact Classifier")
        
        # Get target variables from training data
        if "labels" not in training_data or "business_impact" not in training_data["labels"]:
            raise ValueError("Training data must contain 'labels' with 'business_impact' field")
            
        labels = training_data["labels"]
        
        # Extract target variables
        y_impact_category = pd.Series(labels.get("business_impact_category", []))
        y_impact_value = pd.Series(labels.get("business_impact", []))
        
        # Ensure we have valid target data
        if len(y_impact_category) != len(features) or len(y_impact_value) != len(features):
            raise ValueError(f"Labels length ({len(y_impact_category)}, {len(y_impact_value)}) " +
                            f"doesn't match features length ({len(features)})")
        
        # Preprocess features
        X, feature_names = self.preprocess(features)
        
        # Split data into train and test sets
        X_train, X_test, y_cat_train, y_cat_test, y_val_train, y_val_test = train_test_split(
            X, y_impact_category, y_impact_value, test_size=0.2, random_state=42
        )
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Train category classifier
        self.model.fit(X_train, y_cat_train)
        
        # Train regression model for impact value prediction
        self.impact_value_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        self.impact_value_model.fit(X_train, y_val_train)
        
        # Record training time
        self.training_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Evaluate models
        y_cat_pred = self.model.predict(X_test)
        y_val_pred = self.impact_value_model.predict(X_test)
        
        # Calculate category classification metrics
        category_metrics = {
            "accuracy": accuracy_score(y_cat_test, y_cat_pred),
            "precision": precision_score(y_cat_test, y_cat_pred, average='weighted'),
            "recall": recall_score(y_cat_test, y_cat_pred, average='weighted'),
            "f1": f1_score(y_cat_test, y_cat_pred, average='weighted')
        }
        
        # Calculate impact value regression metrics
        value_metrics = {
            "mse": mean_squared_error(y_val_test, y_val_pred),
            "mae": mean_absolute_error(y_val_test, y_val_pred),
            "r2": r2_score(y_val_test, y_val_pred)
        }
        
        # Calculate feature importance
        self.feature_importance = {
            "category_classifier": dict(zip(
                self.categorical_features + self.numerical_features,
                self.model.feature_importances_
            )),
            "value_regressor": dict(zip(
                self.categorical_features + self.numerical_features,
                self.impact_value_model.feature_importances_
            ))
        }
        
        # Store metrics
        self.metrics = {
            "category_classification": category_metrics,
            "impact_value_regression": value_metrics,
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "training_time_seconds": self.training_time
        }
        
        logger.info(f"Cost Impact Classifier trained successfully: " +
                  f"Category accuracy: {category_metrics['accuracy']:.4f}, " +
                  f"Impact MAE: {value_metrics['mae']:.4f}")
        
        return self.metrics
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict cost impact category and value for recommendations.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Dictionary with predicted impact categories and values
        """
        if self.model is None:
            logger.warning("Model not trained or loaded, returning empty predictions")
            return {
                "business_impact_category": ["unknown"] * len(features),
                "business_impact": [0.0] * len(features)
            }
        
        # Preprocess features
        X, _ = self.preprocess(features)
        
        # Predict impact category
        impact_categories = self.model.predict(X)
        
        # Predict impact value if value model exists
        if hasattr(self, 'impact_value_model') and self.impact_value_model is not None:
            impact_values = self.impact_value_model.predict(X)
        else:
            impact_values = np.zeros(len(features))
        
        # Organize results
        results = {
            "business_impact_category": impact_categories.tolist(),
            "business_impact": impact_values.tolist(),
            "recommendation_ids": features.get("recommendation_id", [f"rec_{i}" for i in range(len(features))]).tolist()
        }
        
        return results
    
    def update_with_feedback(self, 
                           features: pd.DataFrame, 
                           feedback_data: Dict[str, Any]) -> None:
        """Update the model with feedback data.
        
        This performs incremental training of the model using feedback data.
        
        Args:
            features: DataFrame of features from feedback
            feedback_data: Feedback data including actual impact
        """
        if self.model is None:
            logger.warning("Model not trained or loaded, cannot update with feedback")
            return
        
        if features.empty:
            logger.warning("No feedback features provided, skipping update")
            return
        
        logger.info(f"Updating Cost Impact Classifier with {len(features)} feedback samples")
        
        # Extract labels from feedback
        actual_impact_category = []
        actual_impact_value = []
        
        for rec_id in features["recommendation_id"]:
            rec_feedback = feedback_data.get("recommendations", {}).get(rec_id, {})
            actual_impact_category.append(rec_feedback.get("business_impact_category", "unknown"))
            actual_impact_value.append(rec_feedback.get("business_impact", 0.0))
        
        # Preprocess features
        X, _ = self.preprocess(features)
        
        # Update category classifier with feedback
        self.model.n_estimators += 10  # Add more trees for the new data
        self.model.fit(X, actual_impact_category)
        
        # Update impact value regressor with feedback
        if hasattr(self, 'impact_value_model') and self.impact_value_model is not None:
            self.impact_value_model.n_estimators += 10
            self.impact_value_model.fit(X, actual_impact_value)
        
        logger.info("Cost Impact Classifier updated successfully with feedback data")


class UsagePatternClustering(BaseModel):
    """Clustering model for identifying usage patterns in BigQuery resources.
    
    This model clusters recommendations based on usage patterns to identify
    common patterns and trends that can be used to enhance recommendations.
    """
    
    def __init__(self, model_path: Optional[Path] = None, use_pretrained: bool = True):
        """Initialize the Usage Pattern Clustering model.
        
        Args:
            model_path: Path to save/load the model
            use_pretrained: Whether to use pre-trained models if available
        """
        super().__init__(model_path, use_pretrained)
        
        # Define clustering features
        self.clustering_features = [
            # Usage metrics
            "query_count_last_30d", "bytes_processed_last_30d", "avg_query_execution_time",
            "unique_users_last_30d", "slot_ms_last_30d",
            
            # Query patterns
            "export_query_ratio", "reporting_query_ratio", "etl_query_ratio", 
            "ad_hoc_query_ratio",
            
            # Usage patterns
            "usage_growth_trend", "peak_usage_hour", "data_staleness_days",
            "interactive_query_ratio", "has_weekday_pattern", "has_monthly_pattern",
            
            # Query complexity
            "avg_query_complexity", "has_complex_queries",
            
            # Table features
            "table_size_bytes", "table_num_rows", "table_num_columns",
            "is_partitioned", "is_clustered"
        ]
        
        # Initialize model if not loaded from disk
        if self.model is None:
            self.model = KMeans(n_clusters=5, random_state=42)
        
        # Initialize preprocessor if not loaded from disk
        if self.preprocessor is None:
            self._init_preprocessor()
            
        # Pattern definitions
        self.pattern_definitions = {
            0: {"name": "rare_infrequent_access", "description": "Rarely accessed data with low query frequency"},
            1: {"name": "batch_etl_workload", "description": "Regular batch ETL processing with high data volume"},
            2: {"name": "interactive_reporting", "description": "Interactive reporting queries with frequent access patterns"},
            3: {"name": "complex_analytical", "description": "Complex analytical queries with high compute usage"},
            4: {"name": "export_heavy", "description": "Export-heavy workload with large data transfers"}
        }
    
    def _init_preprocessor(self) -> None:
        """Initialize the feature preprocessor pipeline."""
        # Create numerical feature transformer
        self.preprocessor = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
    
    def preprocess(self, features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess features for model input.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Tuple of (preprocessed_features, feature_names)
        """
        # Ensure all necessary columns exist, with defaults if missing
        for col in self.clustering_features:
            if col not in features.columns:
                features[col] = 0
                
        # Handle boolean features
        bool_features = ['is_partitioned', 'is_clustered', 'has_weekday_pattern', 
                        'has_monthly_pattern', 'has_complex_queries']
        for col in bool_features:
            if col in features.columns:
                features[col] = features[col].astype(int)
        
        # Subset to relevant features
        X = features[self.clustering_features].copy()
        
        # Apply preprocessing
        if self.preprocessor is None:
            self._init_preprocessor()
            # Fit the preprocessor if not already fit
            self.preprocessor.fit(X)
            
        # Transform the features
        processed_features = self.preprocessor.transform(X)
        
        return processed_features, self.clustering_features
    
    def train(self, features: pd.DataFrame, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the usage pattern clustering model.
        
        Args:
            features: DataFrame of features
            training_data: Additional training data
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training Usage Pattern Clustering model")
        
        # Preprocess features
        X, feature_names = self.preprocess(features)
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Determine optimal number of clusters if needed (3-7 range)
        if "optimal_clusters" in training_data.get("clustering_params", {}):
            n_clusters = training_data["clustering_params"]["optimal_clusters"]
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            # Default to 5 clusters if not specified
            n_clusters = 5
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Train the model
        self.model.fit(X)
        
        # Record training time
        self.training_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Calculate cluster metrics
        labels = self.model.labels_
        cluster_counts = np.bincount(labels)
        centroids = self.model.cluster_centers_
        
        # Calculate inertia (sum of squared distances to closest centroid)
        inertia = self.model.inertia_
        
        # Calculate silhouette score if sklearn's silhouette_score is available
        try:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(X, labels)
        except ImportError:
            silhouette = 0
        
        # Update pattern definitions if provided in training data
        if "pattern_definitions" in training_data:
            self.pattern_definitions = training_data["pattern_definitions"]
        
        # Calculate feature importance for each cluster
        feature_importance = {}
        for i in range(n_clusters):
            # Calculate the distance of each feature from the global mean for this cluster
            cluster_importance = {}
            for j, feature in enumerate(feature_names):
                importance = abs(centroids[i, j])
                cluster_importance[feature] = float(importance)
            feature_importance[f"cluster_{i}"] = cluster_importance
        
        self.feature_importance = feature_importance
        
        # Store metrics
        self.metrics = {
            "n_clusters": n_clusters,
            "inertia": float(inertia),
            "silhouette_score": silhouette,
            "cluster_sizes": cluster_counts.tolist(),
            "training_samples": len(X),
            "training_time_seconds": self.training_time
        }
        
        logger.info(f"Usage Pattern Clustering model trained successfully: " +
                  f"{n_clusters} clusters, inertia: {inertia:.2f}, silhouette: {silhouette:.4f}")
        
        return self.metrics
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict usage patterns for recommendations.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Dictionary with predicted patterns
        """
        if self.model is None:
            logger.warning("Model not trained or loaded, returning empty predictions")
            return {
                "cluster_id": [0] * len(features),
                "pattern_name": ["unknown"] * len(features),
                "pattern_description": ["unknown"] * len(features)
            }
        
        # Preprocess features
        X, _ = self.preprocess(features)
        
        # Predict clusters
        cluster_ids = self.model.predict(X)
        
        # Map clusters to pattern names and descriptions
        pattern_names = []
        pattern_descriptions = []
        
        for cluster_id in cluster_ids:
            pattern = self.pattern_definitions.get(int(cluster_id), 
                                                {"name": "unknown", "description": "Unknown pattern"})
            pattern_names.append(pattern["name"])
            pattern_descriptions.append(pattern["description"])
        
        # Calculate distance to cluster centroid (as a confidence measure)
        distances = []
        for i, point in enumerate(X):
            cluster_id = cluster_ids[i]
            centroid = self.model.cluster_centers_[cluster_id]
            distance = np.linalg.norm(point - centroid)
            distances.append(float(distance))
        
        # Normalize distances to 0-1 range to get confidence scores (1 = close to centroid)
        max_dist = max(distances) if distances else 1
        confidence_scores = [1 - (d / max_dist) for d in distances]
        
        # Organize results
        results = {
            "cluster_id": cluster_ids.tolist(),
            "pattern_name": pattern_names,
            "pattern_description": pattern_descriptions,
            "confidence_score": confidence_scores,
            "recommendation_ids": features.get("recommendation_id", [f"rec_{i}" for i in range(len(features))]).tolist()
        }
        
        return results
    
    def update_with_feedback(self, 
                           features: pd.DataFrame, 
                           feedback_data: Dict[str, Any]) -> None:
        """Update the model with feedback data.
        
        Args:
            features: DataFrame of features from feedback
            feedback_data: Feedback data
        """
        if self.model is None:
            logger.warning("Model not trained or loaded, cannot update with feedback")
            return
        
        if features.empty:
            logger.warning("No feedback features provided, skipping update")
            return
        
        logger.info(f"Updating Usage Pattern Clustering model with {len(features)} feedback samples")
        
        # For clustering, we can update pattern definitions based on feedback
        if "pattern_feedback" in feedback_data:
            pattern_feedback = feedback_data["pattern_feedback"]
            
            # Update pattern definitions with feedback
            for cluster_id, feedback in pattern_feedback.items():
                cluster_id = int(cluster_id)
                if cluster_id in self.pattern_definitions:
                    if "name" in feedback:
                        self.pattern_definitions[cluster_id]["name"] = feedback["name"]
                    if "description" in feedback:
                        self.pattern_definitions[cluster_id]["description"] = feedback["description"]
        
        # Note: We don't retrain the clustering model with each feedback batch
        # as it would change all cluster assignments. Instead, we update pattern 
        # definitions and periodically retrain with a full dataset.
        
        logger.info("Usage Pattern Clustering model updated successfully with feedback data")


class AnomalyDetector(BaseModel):
    """Anomaly detection model for identifying unusual BigQuery usage patterns.
    
    This model identifies anomalies in usage patterns and resource consumption
    that may indicate optimization opportunities or potential issues.
    """
    
    def __init__(self, model_path: Optional[Path] = None, use_pretrained: bool = True):
        """Initialize the Anomaly Detector model.
        
        Args:
            model_path: Path to save/load the model
            use_pretrained: Whether to use pre-trained models if available
        """
        super().__init__(model_path, use_pretrained)
        
        # Define anomaly detection features
        self.anomaly_features = [
            # Resource usage metrics
            "table_size_bytes", "query_count_last_30d", "bytes_processed_last_30d",
            "avg_query_execution_time", "slot_ms_last_30d",
            
            # Usage patterns
            "usage_growth_trend", "data_staleness_days", "user_count",
            "interactive_query_ratio",
            
            # Query complexity
            "max_query_complexity", "max_query_bytes_processed",
            
            # Resource design
            "nested_field_count", "repeated_field_count",
            
            # Calculated metrics
            "bytes_per_user", "queries_per_row", "cost_per_query"
        ]
        
        # Initialize model if not loaded from disk
        if self.model is None:
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.05,  # Expect about 5% of data to be anomalous
                random_state=42
            )
        
        # Initialize preprocessor if not loaded from disk
        if self.preprocessor is None:
            self._init_preprocessor()
    
    def _init_preprocessor(self) -> None:
        """Initialize the feature preprocessor pipeline."""
        # Create numerical feature transformer
        self.preprocessor = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
    
    def _calculate_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features for anomaly detection.
        
        Args:
            features: DataFrame of features
            
        Returns:
            DataFrame with additional derived features
        """
        df = features.copy()
        
        # Bytes per user (resource usage efficiency)
        df["bytes_per_user"] = (df["bytes_processed_last_30d"] / 
                              df["unique_users_last_30d"].clip(lower=1))
        
        # Queries per row (access frequency relative to size)
        df["queries_per_row"] = (df["query_count_last_30d"] / 
                               df["table_num_rows"].clip(lower=1))
        
        # Cost per query (efficiency metric)
        df["cost_per_query"] = (df["bytes_processed_last_30d"] / 
                              df["query_count_last_30d"].clip(lower=1))
        
        return df
    
    def preprocess(self, features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess features for model input.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Tuple of (preprocessed_features, feature_names)
        """
        # Calculate derived features
        df = self._calculate_derived_features(features)
        
        # Ensure all necessary columns exist, with defaults if missing
        for col in self.anomaly_features:
            if col not in df.columns:
                df[col] = 0
        
        # Subset to relevant features
        X = df[self.anomaly_features].copy()
        
        # Handle any NaN values
        X = X.fillna(0)
        
        # Apply preprocessing
        if self.preprocessor is None:
            self._init_preprocessor()
            # Fit the preprocessor if not already fit
            self.preprocessor.fit(X)
            
        # Transform the features
        processed_features = self.preprocessor.transform(X)
        
        return processed_features, self.anomaly_features
    
    def train(self, features: pd.DataFrame, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the anomaly detection model.
        
        Args:
            features: DataFrame of features
            training_data: Additional training data
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training Anomaly Detector model")
        
        # Preprocess features
        X, feature_names = self.preprocess(features)
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Update model parameters if provided
        if "anomaly_params" in training_data:
            params = training_data["anomaly_params"]
            contamination = params.get("contamination", 0.05)
            n_estimators = params.get("n_estimators", 100)
            
            self.model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42
            )
        
        # Train the model
        self.model.fit(X)
        
        # Record training time
        self.training_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Make predictions on training data to get anomaly counts
        y_pred = self.model.predict(X)
        anomaly_count = (y_pred == -1).sum()
        
        # Get anomaly scores for training data
        anomaly_scores = self.model.decision_function(X)
        
        # Calculate metrics
        metrics = {
            "anomaly_count": int(anomaly_count),
            "anomaly_percentage": float(anomaly_count / len(X) * 100),
            "min_anomaly_score": float(anomaly_scores.min()),
            "max_anomaly_score": float(anomaly_scores.max()),
            "mean_anomaly_score": float(anomaly_scores.mean()),
            "contamination": float(self.model.contamination),
            "training_samples": len(X),
            "training_time_seconds": self.training_time
        }
        
        self.metrics = metrics
        
        logger.info(f"Anomaly Detector trained successfully: " +
                  f"{anomaly_count} anomalies ({metrics['anomaly_percentage']:.2f}%) detected in training data")
        
        return metrics
    
    def detect_anomalies(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the provided features.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Dictionary with anomaly detection results
        """
        if self.model is None:
            logger.warning("Model not trained or loaded, returning empty anomaly detection")
            return {
                "is_anomaly": [False] * len(features),
                "anomaly_score": [0.0] * len(features),
                "anomaly_features": [{}] * len(features)
            }
        
        # Preprocess features
        X, feature_names = self.preprocess(features)
        
        # Detect anomalies
        y_pred = self.model.predict(X)
        is_anomaly = (y_pred == -1)
        
        # Get anomaly scores
        anomaly_scores = self.model.decision_function(X)
        
        # Normalize scores to 0-1 range where 1 is most anomalous
        max_score = abs(anomaly_scores).max() if len(anomaly_scores) > 0 else 1
        normalized_scores = [1 - (score / max_score) for score in anomaly_scores]
        
        # For each anomaly, identify contributing features
        anomaly_features = []
        for i in range(len(X)):
            if is_anomaly[i]:
                # Identify features with the most extreme values
                feature_scores = {}
                for j, feature in enumerate(feature_names):
                    # Get the standardized value for this feature
                    feature_val = X[i, j]
                    # Calculate how anomalous this feature is
                    feature_score = abs(feature_val)
                    if feature_score > 1.0:  # More than 1 std deviation from mean
                        feature_scores[feature] = float(feature_score)
                
                # Sort and keep top 3 contributing features
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                anomaly_features.append(dict(sorted_features))
            else:
                anomaly_features.append({})
        
        # Organize results
        results = {
            "is_anomaly": is_anomaly.tolist(),
            "anomaly_score": normalized_scores,
            "anomaly_features": anomaly_features,
            "recommendation_ids": features.get("recommendation_id", [f"rec_{i}" for i in range(len(features))]).tolist()
        }
        
        return results
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Alias for detect_anomalies to maintain consistent API with other models.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Dictionary with anomaly detection results
        """
        return self.detect_anomalies(features)
    
    def update_with_feedback(self, 
                           features: pd.DataFrame, 
                           feedback_data: Dict[str, Any]) -> None:
        """Update the model with feedback data.
        
        Args:
            features: DataFrame of features from feedback
            feedback_data: Feedback data
        """
        if self.model is None:
            logger.warning("Model not trained or loaded, cannot update with feedback")
            return
        
        if features.empty:
            logger.warning("No feedback features provided, skipping update")
            return
        
        logger.info(f"Updating Anomaly Detector with {len(features)} feedback samples")
        
        # For anomaly detection, we can adjust the model based on false positive/negative feedback
        if "anomaly_feedback" in feedback_data:
            # Extract anomaly feedback
            feedback = []
            true_anomalies = []
            true_normals = []
            
            for rec_id, anomaly_feedback in feedback_data["anomaly_feedback"].items():
                if rec_id in features["recommendation_id"].values:
                    # Get the features for this recommendation
                    rec_features = features[features["recommendation_id"] == rec_id]
                    
                    if anomaly_feedback.get("is_true_anomaly", False):
                        true_anomalies.append(rec_features)
                    elif anomaly_feedback.get("is_false_positive", False):
                        true_normals.append(rec_features)
            
            # Combine feedback datasets
            true_anomaly_features = pd.concat(true_anomalies) if true_anomalies else pd.DataFrame()
            true_normal_features = pd.concat(true_normals) if true_normals else pd.DataFrame()
            
            # If we have enough feedback, retrain the model
            if len(true_anomaly_features) + len(true_normal_features) >= 10:
                # Preprocess feedback features
                X_feedback = []
                
                if not true_anomaly_features.empty:
                    X_anomaly, _ = self.preprocess(true_anomaly_features)
                    X_feedback.append(X_anomaly)
                
                if not true_normal_features.empty:
                    X_normal, _ = self.preprocess(true_normal_features)
                    X_feedback.append(X_normal)
                
                # Combine into a single array if we have any feedback
                if X_feedback:
                    # Create labels for semi-supervised learning
                    y_feedback = []
                    if not true_anomaly_features.empty:
                        y_feedback.extend([-1] * len(true_anomaly_features))  # -1 for anomalies
                    if not true_normal_features.empty:
                        y_feedback.extend([1] * len(true_normal_features))    # 1 for normal samples
                    
                    # Adjust model based on feedback
                    # Note: This is a simplified approach; in practice, you might
                    # want to blend this feedback with the original model more carefully
                    self.model.n_estimators += 10  # Add more trees
                    
                    # Get a subset of the original training data (if available)
                    X_orig, _ = self.preprocess(features)
                    
                    # Combine with feedback
                    X_combined = np.vstack(X_feedback + [X_orig])
                    
                    # Retrain the model
                    self.model.fit(X_combined)
        
        logger.info("Anomaly Detector updated successfully with feedback data")