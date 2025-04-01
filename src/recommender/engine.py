"""Recommendation engine for BigQuery cost optimization."""

from typing import Dict, List, Any
import json
from datetime import datetime

from ..analysis.storage import analyze_partitioning_options, analyze_clustering_options, analyze_compression_options
from ..analysis.query import analyze_query_patterns
from ..analysis.schema import analyze_schema_optimizations
from ..recommender.roi import calculate_roi
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class RecommendationEngine:
    """Engine for generating BigQuery cost optimization recommendations."""
    
    def __init__(self, metadata: Dict[str, Any]):
        """Initialize the recommendation engine.
        
        Args:
            metadata: Dataset metadata from metadata extractor
        """
        self.metadata = metadata
        self.recommendations = []
        
    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate comprehensive optimization recommendations.
        
        Returns:
            Dict containing all recommendations and summary information
        """
        # Run all analysis modules
        storage_recs = self._analyze_storage_optimizations()
        query_recs = self._analyze_query_optimizations()
        schema_recs = self._analyze_schema_optimizations()
        
        # Combine all recommendations
        self.recommendations = storage_recs + query_recs + schema_recs
        
        # Calculate ROI for each recommendation
        self._calculate_roi()
        
        # Sort recommendations by ROI (highest first)
        self.recommendations.sort(key=lambda x: x.get("roi", 0), reverse=True)
        
        # Generate summary
        summary = self._generate_summary()
        
        # Build final recommendation object
        result = {
            "dataset_id": self.metadata["dataset_id"],
            "project_id": self.metadata["project_id"],
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "total_size_gb": self.metadata["total_size_gb"],
            "table_count": self.metadata["table_count"],
            "summary": summary,
            "recommendations": self.recommendations
        }
        
        return result
        
    def _analyze_storage_optimizations(self) -> List[Dict[str, Any]]:
        """Run storage optimization analyses."""
        try:
            partitioning_recs = analyze_partitioning_options(self.metadata)
            clustering_recs = analyze_clustering_options(self.metadata)
            compression_recs = analyze_compression_options(self.metadata)
            
            return partitioning_recs + clustering_recs + compression_recs
        except Exception as e:
            logger.exception(f"Error in storage optimization analysis: {e}")
            return []
            
    def _analyze_query_optimizations(self) -> List[Dict[str, Any]]:
        """Run query optimization analyses."""
        try:
            return analyze_query_patterns(self.metadata)
        except Exception as e:
            logger.exception(f"Error in query optimization analysis: {e}")
            return []
            
    def _analyze_schema_optimizations(self) -> List[Dict[str, Any]]:
        """Run schema optimization analyses."""
        try:
            return analyze_schema_optimizations(self.metadata)
        except Exception as e:
            logger.exception(f"Error in schema optimization analysis: {e}")
            return []
            
    def _calculate_roi(self) -> None:
        """Calculate ROI for each recommendation."""
        for rec in self.recommendations:
            try:
                roi_data = calculate_roi(
                    recommendation=rec,
                    dataset_size_gb=self.metadata["total_size_gb"],
                    table_size_gb=next(
                        (t["size_gb"] for t in self.metadata["tables"] if t["table_id"] == rec["table_id"]),
                        0
                    )
                )
                rec.update(roi_data)
            except Exception as e:
                logger.warning(f"Could not calculate ROI for recommendation: {e}")
                rec["roi"] = 0
                rec["annual_savings_usd"] = 0
                
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all recommendations."""
        total_recommendations = len(self.recommendations)
        high_priority = sum(1 for r in self.recommendations if r.get("priority") == "high")
        medium_priority = sum(1 for r in self.recommendations if r.get("priority") == "medium")
        low_priority = sum(1 for r in self.recommendations if r.get("priority") == "low")
        
        # Calculate potential savings
        total_annual_savings = sum(r.get("annual_savings_usd", 0) for r in self.recommendations)
        
        # Group by recommendation type
        rec_types = {}
        for rec in self.recommendations:
            rec_type = rec.get("type", "other")
            if rec_type not in rec_types:
                rec_types[rec_type] = 0
            rec_types[rec_type] += 1
            
        return {
            "total_recommendations": total_recommendations,
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority,
            "recommendation_types": rec_types,
            "total_annual_savings_usd": total_annual_savings,
            "estimated_implementation_days": self._estimate_implementation_time()
        }
        
    def _estimate_implementation_time(self) -> int:
        """Estimate total implementation time in days."""
        # Rough estimates based on effort levels
        effort_days = {
            "high": 5,
            "medium": 2,
            "low": 0.5
        }
        
        # Sum up estimated days
        total_days = sum(effort_days.get(r.get("estimated_effort", "medium"), 2) for r in self.recommendations)
        
        # Cap at reasonable maximum (assume some parallelization for many recommendations)
        return min(int(total_days), 60)
