"""Advanced cost anomaly detection for BigQuery using machine learning.

This module extends the basic cost anomaly detection with more sophisticated 
machine learning approaches, including time series forecasting and clustering
to detect complex anomaly patterns in BigQuery costs.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import joblib
import os
import json
from pathlib import Path
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from ..utils.logging import setup_logger
from ..analysis.cost_attribution import CostAttributionAnalyzer, CostAnomalyDetector
from ..ml.models import BaseModel

logger = setup_logger(__name__)


class TimeSeriesForecaster:
    """Time series forecasting for BigQuery costs using statistical models."""
    
    def __init__(self, attribution_analyzer: CostAttributionAnalyzer):
        """Initialize the Time Series Forecaster.
        
        Args:
            attribution_analyzer: CostAttributionAnalyzer instance
        """
        self.attribution_analyzer = attribution_analyzer
        
        # Initialize parameters
        self.forecast_days = 7  # Default forecast horizon
        self.confidence_level = 0.95  # For prediction intervals
        
        # Cache for storing results
        self._cache = {}
    
    def forecast_daily_costs(self, training_days: int = 90,
                          forecast_days: Optional[int] = None) -> Dict[str, Any]:
        """Forecast daily costs using exponential smoothing.
        
        Args:
            training_days: Number of days to use for training
            forecast_days: Number of days to forecast (default: self.forecast_days)
            
        Returns:
            Dictionary with forecast results including prediction intervals
        """
        # Set forecast horizon
        if forecast_days is not None:
            self.forecast_days = forecast_days
        
        # Get historical cost data
        job_history = self.attribution_analyzer.get_job_history(training_days)
        
        # Aggregate by day
        job_history['date'] = job_history['creation_time'].dt.date
        daily_costs = job_history.groupby('date').agg({
            'estimated_cost_usd': 'sum',
            'total_bytes_processed': 'sum',
            'job_id': 'count'
        }).reset_index()
        
        daily_costs.columns = ['date', 'total_cost_usd', 'total_bytes_processed', 'query_count']
        
        # Ensure the data is sorted by date
        daily_costs = daily_costs.sort_values('date')
        
        # Fill in missing dates
        date_range = pd.date_range(
            start=daily_costs['date'].min(),
            end=daily_costs['date'].max(),
            freq='D'
        )
        
        full_date_df = pd.DataFrame({'date': date_range})
        daily_costs = pd.merge(full_date_df, daily_costs, on='date', how='left')
        
        # Fill missing values
        daily_costs['total_cost_usd'] = daily_costs['total_cost_usd'].fillna(0)
        daily_costs['total_bytes_processed'] = daily_costs['total_bytes_processed'].fillna(0)
        daily_costs['query_count'] = daily_costs['query_count'].fillna(0)
        
        # Check if we have enough data
        if len(daily_costs) < 14:  # Need at least two weeks of data
            raise ValueError("Insufficient data for forecasting. Need at least 14 days.")
        
        # Apply exponential smoothing
        try:
            import statsmodels.api as sm
            
            # Create time series
            cost_series = daily_costs.set_index('date')['total_cost_usd']
            
            # Fit exponential smoothing model
            model = sm.tsa.ExponentialSmoothing(
                cost_series,
                trend='add',
                seasonal='add',
                seasonal_periods=7  # Weekly seasonality
            ).fit()
            
            # Generate forecast
            forecast = model.forecast(self.forecast_days)
            
            # Generate prediction intervals
            forecast_df = forecast.reset_index()
            forecast_df.columns = ['date', 'forecasted_cost_usd']
            
            # Calculate prediction intervals
            alpha = 1 - self.confidence_level
            forecast_df['lower_bound'] = model.forecasts.predicted_mean - \
                                      model.forecasts.se_mean * stats.norm.ppf(1 - alpha/2)
            forecast_df['upper_bound'] = model.forecasts.predicted_mean + \
                                      model.forecasts.se_mean * stats.norm.ppf(1 - alpha/2)
            
            # Ensure lower bound is not negative
            forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
            
            # Prepare result
            result = {
                'forecast': forecast_df.to_dict('records'),
                'historical_data': daily_costs.to_dict('records'),
                'model_info': {
                    'method': 'ExponentialSmoothing',
                    'params': {
                        'trend': 'add',
                        'seasonal': 'add',
                        'seasonal_periods': 7
                    },
                    'fit_quality': {
                        'aic': model.aic,
                        'bic': model.bic,
                        'mse': model.mse
                    }
                },
                'generated_at': datetime.now().isoformat(),
                'forecast_days': self.forecast_days,
                'confidence_level': self.confidence_level
            }
            
            return result
            
        except ImportError:
            # Fall back to simple moving average if statsmodels is not available
            logger.warning("statsmodels not available, using simple moving average instead")
            return self._forecast_with_moving_average(daily_costs)
    
    def _forecast_with_moving_average(self, daily_costs: pd.DataFrame) -> Dict[str, Any]:
        """Forecast using simple moving average when statsmodels is not available.
        
        Args:
            daily_costs: DataFrame with daily cost data
            
        Returns:
            Dictionary with forecast results
        """
        # Calculate 7-day moving average
        daily_costs['ma7'] = daily_costs['total_cost_usd'].rolling(window=7).mean()
        
        # Fill NaN values in the beginning
        daily_costs['ma7'] = daily_costs['ma7'].fillna(daily_costs['total_cost_usd'])
        
        # Use the last 7 values to forecast
        last_week_avg = daily_costs['total_cost_usd'].tail(7).mean()
        
        # Calculate day-of-week factors (to account for weekly seasonality)
        daily_costs['dayofweek'] = pd.to_datetime(daily_costs['date']).dt.dayofweek
        dow_factors = daily_costs.groupby('dayofweek')['total_cost_usd'].mean() / last_week_avg
        
        # Generate forecast dates
        last_date = daily_costs['date'].max()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(self.forecast_days)]
        
        # Generate forecasts with day-of-week seasonality
        forecasts = []
        for date in forecast_dates:
            dow = date.weekday()
            factor = dow_factors.get(dow, 1.0)
            forecasts.append({
                'date': date,
                'forecasted_cost_usd': last_week_avg * factor,
                'lower_bound': last_week_avg * factor * 0.8,  # Simple 20% lower bound
                'upper_bound': last_week_avg * factor * 1.2   # Simple 20% upper bound
            })
        
        # Prepare result
        result = {
            'forecast': forecasts,
            'historical_data': daily_costs.to_dict('records'),
            'model_info': {
                'method': 'MovingAverage',
                'params': {
                    'window': 7,
                    'seasonal': True
                }
            },
            'generated_at': datetime.now().isoformat(),
            'forecast_days': self.forecast_days,
            'confidence_level': 0.80  # Approximation for the simple bounds
        }
        
        return result
    
    def detect_trend_changes(self, training_days: int = 90) -> List[Dict[str, Any]]:
        """Detect significant changes in cost trends.
        
        Args:
            training_days: Number of days to analyze
            
        Returns:
            List of detected trend changes
        """
        # Get historical cost data
        job_history = self.attribution_analyzer.get_job_history(training_days)
        
        # Aggregate by day
        job_history['date'] = job_history['creation_time'].dt.date
        daily_costs = job_history.groupby('date').agg({
            'estimated_cost_usd': 'sum'
        }).reset_index()
        
        # Ensure the data is sorted by date
        daily_costs = daily_costs.sort_values('date')
        
        # Calculate 7-day moving average to smooth out noise
        daily_costs['ma7'] = daily_costs['estimated_cost_usd'].rolling(window=7).mean()
        
        # Calculate daily changes
        daily_costs['ma7_change'] = daily_costs['ma7'].diff()
        
        # Calculate z-scores of changes
        if len(daily_costs) >= 14:  # Need at least 14 days for meaningful z-scores
            daily_costs['z_score'] = stats.zscore(daily_costs['ma7_change'].fillna(0))
            
            # Detect significant changes (z-score > 2 or < -2)
            changes = daily_costs[(abs(daily_costs['z_score']) > 2) & 
                               (daily_costs['date'] >= daily_costs['date'].min() + timedelta(days=7))]
            
            # Format results
            results = []
            for _, change in changes.iterrows():
                results.append({
                    'date': change['date'].strftime('%Y-%m-%d'),
                    'cost_usd': float(change['estimated_cost_usd']),
                    'ma7_cost_usd': float(change['ma7']),
                    'ma7_change_usd': float(change['ma7_change']),
                    'z_score': float(change['z_score']),
                    'direction': 'increase' if change['ma7_change'] > 0 else 'decrease',
                    'significance': 'high' if abs(change['z_score']) > 3 else 'medium'
                })
            
            return results
        
        return []


class MLCostAnomalyDetector(BaseModel):
    """Machine learning based cost anomaly detector."""
    
    def __init__(self, model_path: Optional[Path] = None, use_pretrained: bool = True):
        """Initialize the ML Cost Anomaly Detector.
        
        Args:
            model_path: Path to save/load the model
            use_pretrained: Whether to use pre-trained models if available
        """
        super().__init__(model_path, use_pretrained)
        
        # Define features for anomaly detection
        self.features = [
            'total_cost_usd', 'query_count', 'unique_users', 'avg_duration',
            'total_bytes_processed', 'total_slot_ms', 
            'cost_per_query', 'bytes_per_query', 'slots_per_query',
            'day_of_week', 'is_weekend', 'cost_vs_prev_day', 'cost_vs_prev_week'
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
        self.preprocessor = Pipeline([
            ('scaler', StandardScaler())
        ])
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and compute features for anomaly detection.
        
        Args:
            df: DataFrame with cost data
            
        Returns:
            DataFrame with extracted features
        """
        # Ensure date field is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Extract day of week and weekend flag
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Calculate derived metrics
        if 'total_cost_usd' in df.columns and 'query_count' in df.columns:
            df['cost_per_query'] = df['total_cost_usd'] / df['query_count'].clip(lower=1)
        
        if 'total_bytes_processed' in df.columns and 'query_count' in df.columns:
            df['bytes_per_query'] = df['total_bytes_processed'] / df['query_count'].clip(lower=1)
        
        if 'total_slot_ms' in df.columns and 'query_count' in df.columns:
            df['slots_per_query'] = df['total_slot_ms'] / df['query_count'].clip(lower=1)
        
        # Calculate comparison with previous day and week
        df = df.sort_values('date')
        df['cost_vs_prev_day'] = df['total_cost_usd'].diff()
        df['cost_vs_prev_week'] = df['total_cost_usd'] - df['total_cost_usd'].shift(7)
        
        # Fill missing values
        for col in df.columns:
            if col != 'date' and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data for anomaly detection.
        
        Args:
            df: DataFrame with cost data
            
        Returns:
            Tuple of (preprocessed_features, feature_names)
        """
        # Extract features
        df_features = self._extract_features(df)
        
        # Ensure all required columns exist
        for col in self.features:
            if col not in df_features.columns:
                df_features[col] = 0
        
        # Select feature columns
        X = df_features[self.features].values
        
        # Fit or transform
        if hasattr(self.preprocessor, 'fit_transform'):
            X_scaled = self.preprocessor.fit_transform(X)
        else:
            X_scaled = self.preprocessor.transform(X)
        
        return X_scaled, self.features
    
    def train(self, df: pd.DataFrame, training_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train the anomaly detection model.
        
        Args:
            df: DataFrame with cost data
            training_data: Optional additional training data
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training ML Cost Anomaly Detector")
        
        if training_data and 'contamination' in training_data:
            # Update model with custom contamination parameter
            self.model = IsolationForest(
                n_estimators=100,
                contamination=training_data['contamination'],
                random_state=42
            )
        
        # Preprocess data
        X, _ = self.preprocess(df)
        
        # Record start time
        start_time = datetime.now()
        
        # Train the model
        self.model.fit(X)
        
        # Record training time
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions on training data
        y_pred = self.model.predict(X)
        scores = self.model.decision_function(X)
        
        # Calculate metrics
        anomaly_count = (y_pred == -1).sum()
        anomaly_pct = 100 * anomaly_count / len(X)
        
        metrics = {
            'anomaly_count': int(anomaly_count),
            'anomaly_percentage': float(anomaly_pct),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'avg_score': float(np.mean(scores)),
            'training_samples': len(X),
            'training_time_seconds': self.training_time
        }
        
        self.metrics = metrics
        
        logger.info(f"Training completed: found {anomaly_count} anomalies ({anomaly_pct:.2f}%)")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the provided data.
        
        Args:
            df: DataFrame with cost data
            
        Returns:
            Dictionary with anomaly detection results
        """
        if self.model is None:
            logger.warning("Model not trained or loaded. Returning empty result.")
            return {
                'is_anomaly': [False] * len(df),
                'anomaly_score': [0.0] * len(df),
                'dates': df['date'].dt.strftime('%Y-%m-%d').tolist() if 'date' in df.columns else []
            }
        
        # Extract features
        df_features = self._extract_features(df)
        
        # Preprocess data
        X, _ = self.preprocess(df)
        
        # Make predictions
        y_pred = self.model.predict(X)
        scores = self.model.decision_function(X)
        
        # Normalize scores to 0-1 range (1 = most anomalous)
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            normalized_scores = 1 - (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(scores)
        
        # Create result dictionary
        result = {
            'is_anomaly': (y_pred == -1).tolist(),
            'anomaly_score': normalized_scores.tolist(),
            'raw_score': scores.tolist()
        }
        
        # Add dates if available
        if 'date' in df.columns:
            result['dates'] = df['date'].dt.strftime('%Y-%m-%d').tolist()
        
        # Add top contributors for each anomaly
        if 'is_anomaly' in result and any(result['is_anomaly']):
            # Find feature importance for each anomaly
            anomaly_features = []
            for i, is_anomaly in enumerate(result['is_anomaly']):
                if is_anomaly:
                    # Get feature values and contribution for this point
                    feature_values = df_features.iloc[i][self.features].to_dict()
                    
                    # Get z-scores for each feature value
                    feature_z_scores = {}
                    for feature in self.features:
                        feature_mean = df_features[feature].mean()
                        feature_std = df_features[feature].std()
                        if feature_std > 0:
                            z_score = (feature_values[feature] - feature_mean) / feature_std
                            feature_z_scores[feature] = float(z_score)
                    
                    # Sort by absolute z-score
                    sorted_features = sorted(
                        feature_z_scores.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    
                    # Keep top 3 contributing features
                    top_features = {k: v for k, v in sorted_features[:3]}
                    anomaly_features.append(top_features)
                else:
                    anomaly_features.append({})
            
            result['anomaly_features'] = anomaly_features
        
        return result


class CostAttributionClusterer(BaseModel):
    """Clusters BigQuery users and teams based on cost patterns."""
    
    def __init__(self, model_path: Optional[Path] = None, use_pretrained: bool = True):
        """Initialize the Cost Attribution Clusterer.
        
        Args:
            model_path: Path to save/load the model
            use_pretrained: Whether to use pre-trained models if available
        """
        super().__init__(model_path, use_pretrained)
        
        # Define clustering features
        self.user_features = [
            'total_cost_usd', 'query_count', 'avg_duration', 'total_bytes_processed',
            'total_slot_ms', 'cache_hit_ratio', 'cost_per_query', 'bytes_per_query'
        ]
        
        # Initialize model if not loaded from disk
        if self.model is None:
            self.model = KMeans(
                n_clusters=5,
                random_state=42
            )
            
        # Initialize preprocessor if not loaded from disk
        if self.preprocessor is None:
            self._init_preprocessor()
        
        # Cluster definitions - will be updated during training
        self.cluster_definitions = {
            0: {"name": "low_usage", "description": "Low volume, infrequent users"},
            1: {"name": "medium_usage", "description": "Medium volume, regular users"},
            2: {"name": "high_usage", "description": "High volume power users"},
            3: {"name": "inefficient_usage", "description": "Users with inefficient query patterns"},
            4: {"name": "export_heavy", "description": "Users with many large data exports"}
        }
    
    def _init_preprocessor(self) -> None:
        """Initialize the feature preprocessor pipeline."""
        self.preprocessor = Pipeline([
            ('scaler', StandardScaler())
        ])
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and compute features for clustering.
        
        Args:
            df: DataFrame with cost attribution data
            
        Returns:
            DataFrame with extracted features
        """
        # Calculate derived metrics
        if 'total_cost_usd' in df.columns and 'query_count' in df.columns:
            df['cost_per_query'] = df['total_cost_usd'] / df['query_count'].clip(lower=1)
        
        if 'total_bytes_processed' in df.columns and 'query_count' in df.columns:
            df['bytes_per_query'] = df['total_bytes_processed'] / df['query_count'].clip(lower=1)
        
        if 'total_slot_ms' in df.columns and 'query_count' in df.columns:
            df['slots_per_query'] = df['total_slot_ms'] / df['query_count'].clip(lower=1)
        
        # Fill missing values
        for col in df.columns:
            if col not in ['user_email', 'team'] and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data for clustering.
        
        Args:
            df: DataFrame with cost attribution data
            
        Returns:
            Tuple of (preprocessed_features, feature_names)
        """
        # Extract features
        df_features = self._extract_features(df)
        
        # Ensure all required columns exist
        for col in self.user_features:
            if col not in df_features.columns:
                df_features[col] = 0
        
        # Select feature columns
        X = df_features[self.user_features].values
        
        # Fit or transform
        if hasattr(self.preprocessor, 'fit_transform'):
            X_scaled = self.preprocessor.fit_transform(X)
        else:
            X_scaled = self.preprocessor.transform(X)
        
        return X_scaled, self.user_features
    
    def train(self, df: pd.DataFrame, training_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train the clustering model.
        
        Args:
            df: DataFrame with cost attribution data
            training_data: Optional additional training data
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training Cost Attribution Clusterer")
        
        if training_data and 'n_clusters' in training_data:
            # Update model with custom number of clusters
            self.model = KMeans(
                n_clusters=training_data['n_clusters'],
                random_state=42
            )
        
        # Preprocess data
        X, _ = self.preprocess(df)
        
        # Record start time
        start_time = datetime.now()
        
        # Train the model
        self.model.fit(X)
        
        # Record training time
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        # Get cluster labels and centroids
        labels = self.model.labels_
        centroids = self.model.cluster_centers_
        
        # Calculate metrics
        cluster_counts = np.bincount(labels)
        inertia = self.model.inertia_
        
        # Calculate silhouette score if scikit-learn is available
        silhouette = 0.0
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
                silhouette = silhouette_score(X, labels)
        except ImportError:
            logger.warning("sklearn.metrics.silhouette_score not available")
        
        # Update cluster definitions based on centroids
        n_clusters = len(centroids)
        self.cluster_definitions = {}
        
        # Interpret clusters based on centroid values
        for i in range(n_clusters):
            centroid = centroids[i]
            
            # Initialize with default name
            cluster_def = {
                "name": f"cluster_{i}",
                "description": f"Cluster {i}",
                "centroid_values": {feature: float(centroid[j]) for j, feature in enumerate(self.user_features)}
            }
            
            # Analyze centroid to characterize the cluster
            feature_dict = {feature: centroid[j] for j, feature in enumerate(self.user_features)}
            
            # High cost users
            if feature_dict['total_cost_usd'] > np.mean([c[0] for c in centroids]) * 1.5:
                cluster_def["name"] = "high_cost_users"
                cluster_def["description"] = "Users with high total cost"
            
            # Inefficient users (high cost per query)
            elif feature_dict['cost_per_query'] > np.mean([c[self.user_features.index('cost_per_query')] for c in centroids]) * 1.5:
                cluster_def["name"] = "inefficient_users"
                cluster_def["description"] = "Users with high cost per query"
            
            # Cache-friendly users
            elif 'cache_hit_ratio' in feature_dict and feature_dict['cache_hit_ratio'] > np.mean([c[self.user_features.index('cache_hit_ratio')] for c in centroids]) * 1.5:
                cluster_def["name"] = "cache_friendly_users"
                cluster_def["description"] = "Users with high cache hit ratio"
            
            # Low volume users
            elif feature_dict['query_count'] < np.mean([c[self.user_features.index('query_count')] for c in centroids]) * 0.5:
                cluster_def["name"] = "low_usage_users"
                cluster_def["description"] = "Users with low query volume"
            
            # Batch processing users
            elif feature_dict['avg_duration'] > np.mean([c[self.user_features.index('avg_duration')] for c in centroids]) * 1.5:
                cluster_def["name"] = "batch_process_users"
                cluster_def["description"] = "Users running long-duration batch queries"
            
            self.cluster_definitions[i] = cluster_def
        
        # Update training data if provided
        if training_data and 'cluster_definitions' in training_data:
            # Override automatically generated cluster definitions
            for cluster_id, definition in training_data['cluster_definitions'].items():
                if int(cluster_id) in self.cluster_definitions:
                    self.cluster_definitions[int(cluster_id)].update(definition)
        
        metrics = {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_counts.tolist(),
            'inertia': float(inertia),
            'silhouette_score': float(silhouette),
            'training_samples': len(X),
            'training_time_seconds': self.training_time,
            'cluster_definitions': self.cluster_definitions
        }
        
        self.metrics = metrics
        
        logger.info(f"Training completed with {n_clusters} clusters, silhouette: {silhouette:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assign clusters to new data points.
        
        Args:
            df: DataFrame with cost attribution data
            
        Returns:
            Dictionary with clustering results
        """
        if self.model is None:
            logger.warning("Model not trained or loaded. Returning empty result.")
            return {
                'cluster_id': [0] * len(df),
                'cluster_name': ["unknown"] * len(df),
                'cluster_description': ["Unknown cluster"] * len(df)
            }
        
        # Preprocess data
        X, _ = self.preprocess(df)
        
        # Make predictions
        labels = self.model.predict(X)
        
        # Calculate distance to centroid (as a confidence measure)
        centroids = self.model.cluster_centers_
        distances = np.zeros(len(X))
        
        for i, point in enumerate(X):
            cluster_id = labels[i]
            centroid = centroids[cluster_id]
            distances[i] = np.linalg.norm(point - centroid)
        
        # Normalize distances to 0-1 range (1 = closest to centroid)
        max_dist = np.max(distances) if len(distances) > 0 else 1
        confidence_scores = 1 - (distances / max_dist)
        
        # Map cluster IDs to names and descriptions
        cluster_names = []
        cluster_descriptions = []
        
        for cluster_id in labels:
            cluster_def = self.cluster_definitions.get(int(cluster_id), 
                                                   {"name": "unknown", "description": "Unknown cluster"})
            cluster_names.append(cluster_def["name"])
            cluster_descriptions.append(cluster_def["description"])
        
        # Create result dictionary
        result = {
            'cluster_id': labels.tolist(),
            'cluster_name': cluster_names,
            'cluster_description': cluster_descriptions,
            'confidence_score': confidence_scores.tolist()
        }
        
        # Add user/team info if available
        if 'user_email' in df.columns:
            result['user_email'] = df['user_email'].tolist()
        
        if 'team' in df.columns:
            result['team'] = df['team'].tolist()
        
        return result


def detect_anomalies_with_ml(project_id: str, days_back: int = 90, 
                          model_path: Optional[str] = None) -> Dict[str, Any]:
    """Detect cost anomalies using machine learning.
    
    This function provides a simplified interface for detecting cost anomalies
    with ML without needing to instantiate the classes directly.
    
    Args:
        project_id: GCP project ID
        days_back: Number of days to analyze
        model_path: Optional path to load/save models
        
    Returns:
        Dictionary with detected anomalies and forecasts
    """
    # Initialize analyzers
    analyzer = CostAttributionAnalyzer(project_id=project_id)
    costs = analyzer.attribute_costs(days_back)
    
    # Get daily costs
    daily_costs = costs['cost_by_day']
    
    # Initialize ML detector
    model_path_obj = Path(model_path) if model_path else None
    detector = MLCostAnomalyDetector(model_path=model_path_obj)
    
    # Train if no model loaded
    if detector.model is None or detector.preprocessor is None:
        detector.train(daily_costs)
    
    # Detect anomalies
    anomalies = detector.predict(daily_costs)
    
    # Generate forecast
    forecaster = TimeSeriesForecaster(analyzer)
    try:
        forecast = forecaster.forecast_daily_costs(training_days=days_back)
    except Exception as e:
        logger.warning(f"Could not generate forecast: {e}")
        forecast = {"error": str(e)}
    
    # Cluster users
    user_costs = costs['cost_by_user']
    clusterer = CostAttributionClusterer(model_path=model_path_obj)
    
    # Train if no model loaded
    if clusterer.model is None or clusterer.preprocessor is None:
        clusterer.train(user_costs)
    
    # Assign user clusters
    user_clusters = clusterer.predict(user_costs)
    
    # Prepare result
    result = {
        'daily_anomalies': anomalies,
        'forecast': forecast,
        'user_clusters': user_clusters,
        'analysis_period_days': days_back,
        'ml_models_used': {
            'anomaly_detector': detector.__class__.__name__,
            'forecaster': forecaster.__class__.__name__,
            'clusterer': clusterer.__class__.__name__
        },
        'generated_at': datetime.now().isoformat()
    }
    
    return result