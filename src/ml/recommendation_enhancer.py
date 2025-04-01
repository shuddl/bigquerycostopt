"""Recommendation Enhancer for ML Enhancement Module.

This module enhances recommendations with ML-derived insights to provide
business context, improved prioritization, and pattern recognition.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
import pandas as pd
import numpy as np
import logging
import datetime
import json
import hashlib

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class RecommendationEnhancer:
    """Enhances recommendations with ML-derived insights.
    
    This class applies ML models to enhance recommendations with business context,
    improved prioritization, pattern recognition, and anomaly detection.
    """
    
    def __init__(self, models: Dict[str, Any]):
        """Initialize the Recommendation Enhancer.
        
        Args:
            models: Dictionary of ML models to use for enhancing recommendations
                - cost_impact: model for predicting business impact
                - pattern_clustering: model for identifying usage patterns
                - anomaly_detector: model for detecting anomalies
        """
        self.models = models
        
        # Define enhancement strategies
        self.enhancement_strategies = {
            "partition": self._enhance_partition_recommendation,
            "cluster": self._enhance_cluster_recommendation,
            "schema_optimization": self._enhance_schema_recommendation,
            "materialized_view": self._enhance_materialized_view_recommendation,
            "query_optimization": self._enhance_query_optimization_recommendation,
            "slot_allocation": self._enhance_slot_allocation_recommendation,
            "caching": self._enhance_caching_recommendation
        }
    
    def enhance_recommendations(self,
                              recommendations: List[Dict[str, Any]],
                              ml_insights: Dict[str, Any],
                              dataset_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance recommendations with ML-derived insights.
        
        Args:
            recommendations: List of standardized recommendations
            ml_insights: Dictionary of ML insights from models
            dataset_metadata: Metadata about the dataset being analyzed
            
        Returns:
            Enhanced recommendations with ML-derived insights
        """
        logger.info(f"Enhancing {len(recommendations)} recommendations")
        
        # Create enhanced recommendations list
        enhanced_recommendations = []
        
        # Create a lookup of recommendation IDs for faster access
        recommendation_ids = [rec.get("recommendation_id", "") for rec in recommendations]
        
        # Process each recommendation
        for i, recommendation in enumerate(recommendations):
            rec_id = recommendation.get("recommendation_id", "")
            rec_type = recommendation.get("recommendation_type", "unknown")
            
            # Create a copy of the recommendation to enhance
            enhanced_rec = recommendation.copy()
            
            # Add ML insights container if not present
            if "ml_insights" not in enhanced_rec:
                enhanced_rec["ml_insights"] = {}
            
            # Apply cost impact insights
            if "cost_impact" in ml_insights:
                cost_impact = ml_insights["cost_impact"]
                idx = cost_impact["recommendation_ids"].index(rec_id) if rec_id in cost_impact["recommendation_ids"] else -1
                
                if idx >= 0:
                    enhanced_rec["ml_insights"]["business_impact_category"] = cost_impact["business_impact_category"][idx]
                    enhanced_rec["ml_insights"]["business_impact"] = cost_impact["business_impact"][idx]
            
            # Apply pattern clustering insights
            if "pattern_clusters" in ml_insights:
                patterns = ml_insights["pattern_clusters"]
                idx = patterns["recommendation_ids"].index(rec_id) if rec_id in patterns["recommendation_ids"] else -1
                
                if idx >= 0:
                    enhanced_rec["ml_insights"]["pattern_name"] = patterns["pattern_name"][idx]
                    enhanced_rec["ml_insights"]["pattern_description"] = patterns["pattern_description"][idx]
                    enhanced_rec["ml_insights"]["pattern_confidence"] = patterns["confidence_score"][idx]
            
            # Apply anomaly detection insights
            if "anomalies" in ml_insights:
                anomalies = ml_insights["anomalies"]
                idx = anomalies["recommendation_ids"].index(rec_id) if rec_id in anomalies["recommendation_ids"] else -1
                
                if idx >= 0:
                    enhanced_rec["ml_insights"]["is_anomaly"] = anomalies["is_anomaly"][idx]
                    enhanced_rec["ml_insights"]["anomaly_score"] = anomalies["anomaly_score"][idx]
                    enhanced_rec["ml_insights"]["anomaly_features"] = anomalies["anomaly_features"][idx]
            
            # Apply recommendation type-specific enhancements
            if rec_type in self.enhancement_strategies:
                enhanced_rec = self.enhancement_strategies[rec_type](
                    enhanced_rec, 
                    ml_insights,
                    dataset_metadata
                )
            
            # Update priority score based on ML insights
            if "business_impact" in enhanced_rec["ml_insights"]:
                # Get the original priority score
                original_priority = enhanced_rec.get("priority_score", 0)
                
                # Get the business impact
                business_impact = enhanced_rec["ml_insights"]["business_impact"]
                
                # Adjust priority score based on business impact (50% original, 50% ML-derived)
                enhanced_rec["priority_score"] = 0.5 * original_priority + 0.5 * (business_impact * 10)
                
                # Add ML influence flag
                enhanced_rec["ml_influenced_priority"] = True
            
            # Add enhancement timestamp
            enhanced_rec["ml_insights"]["enhancement_timestamp"] = datetime.datetime.now().isoformat()
            
            # Add to enhanced recommendations
            enhanced_recommendations.append(enhanced_rec)
        
        logger.info(f"Enhanced {len(enhanced_recommendations)} recommendations with ML insights")
        
        return enhanced_recommendations
    
    def _enhance_partition_recommendation(self,
                                        recommendation: Dict[str, Any],
                                        ml_insights: Dict[str, Any],
                                        dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance partition recommendations with ML insights.
        
        Args:
            recommendation: Recommendation to enhance
            ml_insights: ML model insights
            dataset_metadata: Dataset metadata
            
        Returns:
            Enhanced recommendation
        """
        enhanced_rec = recommendation.copy()
        
        # Get ML insights specifically for this recommendation
        if "pattern_name" in enhanced_rec["ml_insights"]:
            pattern = enhanced_rec["ml_insights"]["pattern_name"]
            
            # Add business context based on usage pattern
            if pattern == "batch_etl_workload":
                enhanced_rec["ml_insights"]["business_context"] = (
                    "This table shows ETL workload patterns. Partitioning will significantly reduce " +
                    "query costs for your batch processing jobs by avoiding full table scans."
                )
            elif pattern == "interactive_reporting":
                enhanced_rec["ml_insights"]["business_context"] = (
                    "This table is used for interactive reporting. Partitioning on your filters " +
                    "will improve query performance and reduce costs for your reporting dashboards."
                )
            elif pattern == "rare_infrequent_access":
                enhanced_rec["ml_insights"]["business_context"] = (
                    "This table is rarely accessed. Consider if partitioning is worth the " +
                    "implementation effort given the low usage frequency."
                )
        
        # Add ML-derived implementation complexity adjustment
        partition_type = recommendation.get("partition_type", "")
        if partition_type == "time-unit":
            # Check if we have data on query patterns to refine complexity
            if "query_patterns" in recommendation:
                date_filtered_queries = sum(1 for qp in recommendation.get("query_patterns", []) 
                                        if qp.get("has_date_filter", False))
                total_queries = len(recommendation.get("query_patterns", []))
                
                # If date filtering isn't common, increase complexity due to adaptation needed
                if total_queries > 0 and date_filtered_queries / total_queries < 0.3:
                    enhanced_rec["ml_insights"]["complexity_adjustment"] = 1
                    enhanced_rec["ml_insights"]["complexity_context"] = (
                        "Implementation complexity increased because date filtering is not common " +
                        "in current queries. Query patterns will need to be modified."
                    )
            
        # Add ML-derived performance impact insights
        if "is_anomaly" in enhanced_rec["ml_insights"] and enhanced_rec["ml_insights"]["is_anomaly"]:
            anomaly_features = enhanced_rec["ml_insights"].get("anomaly_features", {})
            if "bytes_processed_last_30d" in anomaly_features:
                enhanced_rec["ml_insights"]["performance_insight"] = (
                    "This table has anomalously high data processing volume. " +
                    "Partitioning could provide exceptional cost savings."
                )
        
        return enhanced_rec
    
    def _enhance_cluster_recommendation(self,
                                      recommendation: Dict[str, Any],
                                      ml_insights: Dict[str, Any],
                                      dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance cluster recommendations with ML insights.
        
        Args:
            recommendation: Recommendation to enhance
            ml_insights: ML model insights
            dataset_metadata: Dataset metadata
            
        Returns:
            Enhanced recommendation
        """
        enhanced_rec = recommendation.copy()
        
        # Get ML insights specifically for this recommendation
        if "pattern_name" in enhanced_rec["ml_insights"]:
            pattern = enhanced_rec["ml_insights"]["pattern_name"]
            
            # Add business context based on usage pattern
            if pattern == "complex_analytical":
                enhanced_rec["ml_insights"]["business_context"] = (
                    "This table shows complex analytical query patterns. Clustering will " +
                    "significantly improve query performance for your analytical workloads."
                )
            elif pattern == "interactive_reporting":
                enhanced_rec["ml_insights"]["business_context"] = (
                    "This table supports interactive reporting. Clustering will reduce " +
                    "response times for your dashboards and improve user experience."
                )
        
        # Add ML-derived implementation complexity adjustment
        clustering_columns = recommendation.get("clustering_columns", [])
        if len(clustering_columns) > 0:
            # Check if clustering columns are commonly used in filters
            column_usage = {}
            for qp in recommendation.get("query_patterns", []):
                for col in qp.get("filter_columns", []):
                    if col not in column_usage:
                        column_usage[col] = 0
                    column_usage[col] += 1
            
            # Check if recommended clustering columns are frequently used in filters
            cluster_cols_in_filters = sum(column_usage.get(col, 0) for col in clustering_columns)
            
            # If clustering columns aren't commonly used in filters, increase complexity
            if cluster_cols_in_filters == 0:
                enhanced_rec["ml_insights"]["complexity_adjustment"] = 1
                enhanced_rec["ml_insights"]["complexity_context"] = (
                    "Implementation complexity increased because recommended clustering columns " +
                    "are not commonly used in query filters. Query patterns may need to be modified."
                )
        
        return enhanced_rec
    
    def _enhance_schema_recommendation(self,
                                     recommendation: Dict[str, Any],
                                     ml_insights: Dict[str, Any],
                                     dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance schema optimization recommendations with ML insights.
        
        Args:
            recommendation: Recommendation to enhance
            ml_insights: ML model insights
            dataset_metadata: Dataset metadata
            
        Returns:
            Enhanced recommendation
        """
        enhanced_rec = recommendation.copy()
        
        # Get optimization type
        optimization_type = recommendation.get("optimization_type", "")
        
        # Add business context based on optimization type
        if optimization_type == "type_change":
            enhanced_rec["ml_insights"]["business_context"] = (
                "Changing column types to more efficient representations will reduce " +
                "storage costs and improve query performance."
            )
        elif optimization_type == "drop_unused_columns":
            # Check if pattern suggests rare usage
            if "pattern_name" in enhanced_rec["ml_insights"]:
                pattern = enhanced_rec["ml_insights"]["pattern_name"]
                if pattern == "rare_infrequent_access":
                    enhanced_rec["ml_insights"]["business_context"] = (
                        "This table is rarely accessed. Dropping unused columns will provide " +
                        "storage cost savings without disrupting existing workflows."
                    )
                else:
                    enhanced_rec["ml_insights"]["business_context"] = (
                        "Dropping unused columns will reduce storage costs and simplify " +
                        "schema management for this actively used table."
                    )
        
        # Add ML-derived risk assessment
        if "is_anomaly" in enhanced_rec["ml_insights"] and enhanced_rec["ml_insights"]["is_anomaly"]:
            anomaly_features = enhanced_rec["ml_insights"].get("anomaly_features", {})
            if "user_count" in anomaly_features and anomaly_features["user_count"] > 3:
                # If many users access this table, increase risk assessment
                if "risk_level" in enhanced_rec and enhanced_rec["risk_level"] == "low":
                    enhanced_rec["ml_insights"]["risk_adjustment"] = "medium"
                    enhanced_rec["ml_insights"]["risk_context"] = (
                        "Risk level increased because this table has an unusually high number " +
                        "of users. Schema changes may impact more stakeholders than typical."
                    )
        
        return enhanced_rec
    
    def _enhance_materialized_view_recommendation(self,
                                               recommendation: Dict[str, Any],
                                               ml_insights: Dict[str, Any],
                                               dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance materialized view recommendations with ML insights.
        
        Args:
            recommendation: Recommendation to enhance
            ml_insights: ML model insights
            dataset_metadata: Dataset metadata
            
        Returns:
            Enhanced recommendation
        """
        enhanced_rec = recommendation.copy()
        
        # Get ML insights specifically for this recommendation
        if "pattern_name" in enhanced_rec["ml_insights"]:
            pattern = enhanced_rec["ml_insights"]["pattern_name"]
            
            # Add business context based on usage pattern
            if pattern == "interactive_reporting":
                enhanced_rec["ml_insights"]["business_context"] = (
                    "This materialized view will significantly improve dashboard performance " +
                    "by pre-aggregating data for your interactive reporting queries."
                )
            elif pattern == "export_heavy":
                enhanced_rec["ml_insights"]["business_context"] = (
                    "This materialized view will reduce costs for your export processes " +
                    "by pre-computing frequently exported data subsets."
                )
        
        # Add ML-derived refresh frequency recommendation
        if "usage_statistics" in dataset_metadata:
            # Check for usage patterns to determine optimal refresh frequency
            peak_hours = set()
            for table in dataset_metadata.get("tables", []):
                if "usage_patterns" in table:
                    patterns = table["usage_patterns"]
                    if "peak_usage_hour" in patterns:
                        peak_hours.add(patterns["peak_usage_hour"])
            
            # If we have peak hours, recommend refreshing before those times
            if peak_hours:
                min_peak = min(peak_hours)
                refresh_hour = (min_peak - 1) % 24  # Refresh 1 hour before peak
                
                enhanced_rec["ml_insights"]["refresh_recommendation"] = {
                    "frequency_hours": 24,
                    "optimal_time_utc": f"{refresh_hour:02d}:00",
                    "rationale": f"Schedule refresh before peak usage time (UTC {min_peak:02d}:00)"
                }
        
        return enhanced_rec
    
    def _enhance_query_optimization_recommendation(self,
                                                recommendation: Dict[str, Any],
                                                ml_insights: Dict[str, Any],
                                                dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance query optimization recommendations with ML insights.
        
        Args:
            recommendation: Recommendation to enhance
            ml_insights: ML model insights
            dataset_metadata: Dataset metadata
            
        Returns:
            Enhanced recommendation
        """
        enhanced_rec = recommendation.copy()
        
        # Get query type and optimization suggestions
        query_id = recommendation.get("query_id", "")
        optimization_type = recommendation.get("optimization_type", "")
        
        # Add business context based on query usage patterns
        if "query_patterns" in recommendation:
            query_patterns = recommendation.get("query_patterns", [])
            if query_patterns:
                qp = query_patterns[0]  # Use the first query pattern for simplicity
                
                # Get execution stats
                execution_count = qp.get("execution_count", 0)
                avg_execution_time = qp.get("avg_execution_time_ms", 0) / 1000  # Convert to seconds
                
                if execution_count > 50 and avg_execution_time > 10:
                    enhanced_rec["ml_insights"]["business_context"] = (
                        f"This query runs frequently ({execution_count} times in the last 30 days) " +
                        f"and has a high average execution time ({avg_execution_time:.1f}s). " +
                        "Optimizing it will have significant impact on cost and performance."
                    )
                elif execution_count > 50:
                    enhanced_rec["ml_insights"]["business_context"] = (
                        f"This query runs very frequently ({execution_count} times in the last 30 days). " +
                        "Even small optimizations will have a cumulative significant impact."
                    )
                elif avg_execution_time > 30:
                    enhanced_rec["ml_insights"]["business_context"] = (
                        f"This query has a very high execution time ({avg_execution_time:.1f}s). " +
                        "Optimizing it will significantly improve user experience and reduce costs."
                    )
        
        # Add ML-derived complexity assessment
        if optimization_type == "join_optimization":
            # Check if pattern suggests complex analytical workload
            if "pattern_name" in enhanced_rec["ml_insights"]:
                pattern = enhanced_rec["ml_insights"]["pattern_name"]
                if pattern == "complex_analytical":
                    enhanced_rec["ml_insights"]["complexity_adjustment"] = 1
                    enhanced_rec["ml_insights"]["complexity_context"] = (
                        "Join optimization for complex analytical queries may require " +
                        "more extensive testing and validation to ensure correctness."
                    )
        
        return enhanced_rec
    
    def _enhance_slot_allocation_recommendation(self,
                                             recommendation: Dict[str, Any],
                                             ml_insights: Dict[str, Any],
                                             dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance slot allocation recommendations with ML insights.
        
        Args:
            recommendation: Recommendation to enhance
            ml_insights: ML model insights
            dataset_metadata: Dataset metadata
            
        Returns:
            Enhanced recommendation
        """
        enhanced_rec = recommendation.copy()
        
        # Add business context based on usage patterns
        if "usage_pattern" in recommendation:
            usage_pattern = recommendation.get("usage_pattern", {})
            
            if "peak_slots" in usage_pattern and "average_slots" in usage_pattern:
                peak = usage_pattern["peak_slots"]
                avg = usage_pattern["average_slots"]
                
                # Calculate peak-to-average ratio
                ratio = peak / avg if avg > 0 else 1
                
                if ratio > 5:
                    enhanced_rec["ml_insights"]["business_context"] = (
                        f"This project has a very high peak-to-average slot usage ratio ({ratio:.1f}x). " +
                        "Optimizing slot allocation will significantly reduce costs while " +
                        "maintaining performance for peak workloads."
                    )
                elif ratio > 2:
                    enhanced_rec["ml_insights"]["business_context"] = (
                        f"This project has a moderate peak-to-average slot usage ratio ({ratio:.1f}x). " +
                        "Optimizing slot allocation will reduce costs while maintaining adequate performance."
                    )
        
        return enhanced_rec
    
    def _enhance_caching_recommendation(self,
                                      recommendation: Dict[str, Any],
                                      ml_insights: Dict[str, Any],
                                      dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance query result caching recommendations with ML insights.
        
        Args:
            recommendation: Recommendation to enhance
            ml_insights: ML model insights
            dataset_metadata: Dataset metadata
            
        Returns:
            Enhanced recommendation
        """
        enhanced_rec = recommendation.copy()
        
        # Get cache opportunity details
        cache_opportunity = recommendation.get("cache_opportunity", {})
        
        # Add business context based on usage patterns
        if "repeated_query_count" in cache_opportunity and "potential_savings" in cache_opportunity:
            count = cache_opportunity.get("repeated_query_count", 0)
            savings = cache_opportunity.get("potential_savings", 0)
            
            # Get ML pattern insights
            if "pattern_name" in enhanced_rec["ml_insights"]:
                pattern = enhanced_rec["ml_insights"]["pattern_name"]
                
                if pattern == "interactive_reporting":
                    enhanced_rec["ml_insights"]["business_context"] = (
                        f"Enabling query result caching will significantly improve dashboard performance " +
                        f"by eliminating repeated execution of {count} queries. This pattern of " +
                        f"interactive reporting would benefit greatly from caching."
                    )
                else:
                    enhanced_rec["ml_insights"]["business_context"] = (
                        f"Enabling query result caching will save approximately ${savings:.2f} " +
                        f"by eliminating repeated execution of {count} queries."
                    )
        
        return enhanced_rec