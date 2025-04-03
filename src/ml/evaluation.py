"""Evaluation Metrics for ML Enhancement Module.

This module provides functionality for evaluating the quality and impact of
ML-enhanced recommendations compared to base rule-based recommendations.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
import pandas as pd
import numpy as np
import logging
import datetime
import json
from pathlib import Path

# Conditionally import matplotlib
try:
    import matplotlib.pyplot as plt
    _has_matplotlib = True
except ImportError:
    _has_matplotlib = False

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class EvaluationMetrics:
    """Evaluation metrics for ML-enhanced recommendations.
    
    This class provides functionality for calculating metrics to evaluate
    the quality and impact of ML-enhanced recommendations compared to
    base rule-based recommendations.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the Evaluation Metrics calculator.
        
        Args:
            output_dir: Directory to save evaluation reports and visualizations
        """
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to 'reports' directory in the package
            self.output_dir = Path(__file__).parent.parent.parent / "reports"
            
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_recommendations(self,
                               original_recommendations: List[Dict[str, Any]],
                               enhanced_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the quality and impact of ML-enhanced recommendations.
        
        Args:
            original_recommendations: List of original rule-based recommendations
            enhanced_recommendations: List of ML-enhanced recommendations
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating ML-enhanced recommendations")
        
        # Ensure we have matching recommendation sets
        if len(original_recommendations) != len(enhanced_recommendations):
            logger.warning(f"Recommendation count mismatch: original={len(original_recommendations)}, " +
                         f"enhanced={len(enhanced_recommendations)}")
            # Try to match by recommendation ID
            original_by_id = {r.get("recommendation_id", ""): r for r in original_recommendations}
            enhanced_by_id = {r.get("recommendation_id", ""): r for r in enhanced_recommendations}
            
            # Use only recommendations that exist in both sets
            common_ids = set(original_by_id.keys()) & set(enhanced_by_id.keys())
            
            original_recommendations = [original_by_id[id] for id in common_ids]
            enhanced_recommendations = [enhanced_by_id[id] for id in common_ids]
            
            logger.info(f"Matched {len(common_ids)} recommendations by ID")
        
        # Calculate priority score changes
        priority_changes = self._calculate_priority_changes(original_recommendations, enhanced_recommendations)
        
        # Calculate ML insight metrics
        ml_insight_metrics = self._calculate_ml_insight_metrics(enhanced_recommendations)
        
        # Calculate business context metrics
        business_context_metrics = self._calculate_business_context_metrics(enhanced_recommendations)
        
        # Calculate anomaly detection metrics
        anomaly_metrics = self._calculate_anomaly_metrics(enhanced_recommendations)
        
        # Calculate pattern recognition metrics
        pattern_metrics = self._calculate_pattern_metrics(enhanced_recommendations)
        
        # Compile summary metrics
        summary = {
            "total_recommendations": len(enhanced_recommendations),
            "ml_enhanced_count": ml_insight_metrics["ml_enhanced_count"],
            "ml_enhanced_percentage": ml_insight_metrics["ml_enhanced_percentage"],
            "avg_priority_score_change": priority_changes["avg_change"],
            "priority_increases": priority_changes["increases"],
            "priority_decreases": priority_changes["decreases"],
            "business_context_count": business_context_metrics["context_count"],
            "anomaly_count": anomaly_metrics["anomaly_count"],
            "pattern_count": pattern_metrics["total_patterns"],
            "top_pattern": pattern_metrics["top_pattern"] if pattern_metrics["patterns"] else "none"
        }
        
        # Combine all metrics
        evaluation_metrics = {
            "summary": summary,
            "priority_changes": priority_changes,
            "ml_insight_metrics": ml_insight_metrics,
            "business_context_metrics": business_context_metrics,
            "anomaly_metrics": anomaly_metrics,
            "pattern_metrics": pattern_metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save evaluation report
        self._save_evaluation_report(evaluation_metrics)
        
        logger.info(f"Evaluation complete: {len(enhanced_recommendations)} recommendations, " +
                  f"{ml_insight_metrics['ml_enhanced_count']} ML-enhanced ({ml_insight_metrics['ml_enhanced_percentage']:.1f}%)")
        
        return evaluation_metrics
    
    def _calculate_priority_changes(self,
                                  original_recommendations: List[Dict[str, Any]],
                                  enhanced_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate changes in priority scores.
        
        Args:
            original_recommendations: List of original rule-based recommendations
            enhanced_recommendations: List of ML-enhanced recommendations
            
        Returns:
            Dictionary of priority change metrics
        """
        changes = []
        increases = 0
        decreases = 0
        significant_changes = 0  # Changes > 10%
        
        for i in range(len(original_recommendations)):
            orig_score = original_recommendations[i].get("priority_score", 0)
            enhanced_score = enhanced_recommendations[i].get("priority_score", 0)
            
            change = enhanced_score - orig_score
            pct_change = (change / orig_score * 100) if orig_score != 0 else 0
            
            changes.append({
                "recommendation_id": enhanced_recommendations[i].get("recommendation_id", ""),
                "recommendation_type": enhanced_recommendations[i].get("recommendation_type", ""),
                "original_score": orig_score,
                "enhanced_score": enhanced_score,
                "change": change,
                "percent_change": pct_change
            })
            
            if change > 0:
                increases += 1
            elif change < 0:
                decreases += 1
                
            if abs(pct_change) > 10:
                significant_changes += 1
        
        # Calculate average and median changes
        if changes:
            avg_change = sum(c["change"] for c in changes) / len(changes)
            median_change = sorted(changes, key=lambda x: x["change"])[len(changes) // 2]["change"]
            avg_pct_change = sum(c["percent_change"] for c in changes) / len(changes)
        else:
            avg_change = 0
            median_change = 0
            avg_pct_change = 0
        
        # Prepare results
        return {
            "changes": changes,
            "increases": increases,
            "decreases": decreases,
            "no_change": len(changes) - increases - decreases,
            "significant_changes": significant_changes,
            "avg_change": avg_change,
            "median_change": median_change,
            "avg_percent_change": avg_pct_change
        }
    
    def _calculate_ml_insight_metrics(self, enhanced_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics related to ML insights.
        
        Args:
            enhanced_recommendations: List of ML-enhanced recommendations
            
        Returns:
            Dictionary of ML insight metrics
        """
        # Count recommendations with ML insights
        ml_enhanced_count = sum(1 for r in enhanced_recommendations if "ml_insights" in r)
        ml_enhanced_percentage = ml_enhanced_count / len(enhanced_recommendations) * 100 if enhanced_recommendations else 0
        
        # Count by ML insight type
        business_impact_count = sum(1 for r in enhanced_recommendations 
                                  if "ml_insights" in r and "business_impact" in r["ml_insights"])
        
        anomaly_count = sum(1 for r in enhanced_recommendations 
                          if "ml_insights" in r and "is_anomaly" in r["ml_insights"] and r["ml_insights"]["is_anomaly"])
        
        pattern_count = sum(1 for r in enhanced_recommendations 
                          if "ml_insights" in r and "pattern_name" in r["ml_insights"])
        
        context_count = sum(1 for r in enhanced_recommendations 
                          if "ml_insights" in r and "business_context" in r["ml_insights"])
        
        # Count by recommendation type
        by_type = {}
        for r in enhanced_recommendations:
            rec_type = r.get("recommendation_type", "unknown")
            
            if rec_type not in by_type:
                by_type[rec_type] = {
                    "total": 0,
                    "ml_enhanced": 0
                }
                
            by_type[rec_type]["total"] += 1
            
            if "ml_insights" in r:
                by_type[rec_type]["ml_enhanced"] += 1
        
        # Calculate percentages by type
        for rec_type in by_type:
            if by_type[rec_type]["total"] > 0:
                by_type[rec_type]["percentage"] = (by_type[rec_type]["ml_enhanced"] / 
                                                by_type[rec_type]["total"] * 100)
            else:
                by_type[rec_type]["percentage"] = 0
        
        return {
            "ml_enhanced_count": ml_enhanced_count,
            "ml_enhanced_percentage": ml_enhanced_percentage,
            "business_impact_count": business_impact_count,
            "anomaly_count": anomaly_count,
            "pattern_count": pattern_count,
            "context_count": context_count,
            "by_recommendation_type": by_type
        }
    
    def _calculate_business_context_metrics(self, enhanced_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics related to business context.
        
        Args:
            enhanced_recommendations: List of ML-enhanced recommendations
            
        Returns:
            Dictionary of business context metrics
        """
        # Count recommendations with business context
        context_count = sum(1 for r in enhanced_recommendations 
                          if "ml_insights" in r and "business_context" in r["ml_insights"])
        
        context_percentage = context_count / len(enhanced_recommendations) * 100 if enhanced_recommendations else 0
        
        # Count by recommendation type
        by_type = {}
        for r in enhanced_recommendations:
            if "ml_insights" not in r or "business_context" not in r["ml_insights"]:
                continue
                
            rec_type = r.get("recommendation_type", "unknown")
            
            if rec_type not in by_type:
                by_type[rec_type] = 0
                
            by_type[rec_type] += 1
        
        # Count unique business contexts
        unique_contexts = set()
        for r in enhanced_recommendations:
            if "ml_insights" in r and "business_context" in r["ml_insights"]:
                unique_contexts.add(r["ml_insights"]["business_context"])
        
        return {
            "context_count": context_count,
            "context_percentage": context_percentage,
            "unique_context_count": len(unique_contexts),
            "by_recommendation_type": by_type
        }
    
    def _calculate_anomaly_metrics(self, enhanced_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics related to anomaly detection.
        
        Args:
            enhanced_recommendations: List of ML-enhanced recommendations
            
        Returns:
            Dictionary of anomaly metrics
        """
        # Count anomalous recommendations
        anomaly_count = sum(1 for r in enhanced_recommendations 
                          if "ml_insights" in r and "is_anomaly" in r["ml_insights"] and r["ml_insights"]["is_anomaly"])
        
        anomaly_percentage = anomaly_count / len(enhanced_recommendations) * 100 if enhanced_recommendations else 0
        
        # Collect anomaly scores
        anomaly_scores = []
        for r in enhanced_recommendations:
            if "ml_insights" in r and "anomaly_score" in r["ml_insights"]:
                anomaly_scores.append(r["ml_insights"]["anomaly_score"])
        
        # Calculate statistics on anomaly scores
        avg_score = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0
        median_score = sorted(anomaly_scores)[len(anomaly_scores) // 2] if anomaly_scores else 0
        max_score = max(anomaly_scores) if anomaly_scores else 0
        
        # Count by recommendation type
        by_type = {}
        for r in enhanced_recommendations:
            if ("ml_insights" not in r or "is_anomaly" not in r["ml_insights"] or 
                not r["ml_insights"]["is_anomaly"]):
                continue
                
            rec_type = r.get("recommendation_type", "unknown")
            
            if rec_type not in by_type:
                by_type[rec_type] = 0
                
            by_type[rec_type] += 1
        
        # Count common anomaly features
        anomaly_features = {}
        for r in enhanced_recommendations:
            if ("ml_insights" not in r or "is_anomaly" not in r["ml_insights"] or 
                not r["ml_insights"]["is_anomaly"] or "anomaly_features" not in r["ml_insights"]):
                continue
                
            for feature in r["ml_insights"]["anomaly_features"]:
                if feature not in anomaly_features:
                    anomaly_features[feature] = 0
                    
                anomaly_features[feature] += 1
        
        # Sort anomaly features by frequency
        sorted_features = sorted(anomaly_features.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "anomaly_count": anomaly_count,
            "anomaly_percentage": anomaly_percentage,
            "avg_anomaly_score": avg_score,
            "median_anomaly_score": median_score,
            "max_anomaly_score": max_score,
            "by_recommendation_type": by_type,
            "top_anomaly_features": dict(sorted_features[:5]) if sorted_features else {}
        }
    
    def _calculate_pattern_metrics(self, enhanced_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics related to pattern recognition.
        
        Args:
            enhanced_recommendations: List of ML-enhanced recommendations
            
        Returns:
            Dictionary of pattern metrics
        """
        # Count patterns
        patterns = {}
        for r in enhanced_recommendations:
            if "ml_insights" not in r or "pattern_name" not in r["ml_insights"]:
                continue
                
            pattern = r["ml_insights"]["pattern_name"]
            
            if pattern not in patterns:
                patterns[pattern] = 0
                
            patterns[pattern] += 1
        
        # Find top pattern
        top_pattern = max(patterns.items(), key=lambda x: x[1])[0] if patterns else None
        
        # Count by recommendation type
        by_type = {}
        for r in enhanced_recommendations:
            if "ml_insights" not in r or "pattern_name" not in r["ml_insights"]:
                continue
                
            rec_type = r.get("recommendation_type", "unknown")
            pattern = r["ml_insights"]["pattern_name"]
            
            if rec_type not in by_type:
                by_type[rec_type] = {}
                
            if pattern not in by_type[rec_type]:
                by_type[rec_type][pattern] = 0
                
            by_type[rec_type][pattern] += 1
        
        return {
            "patterns": patterns,
            "total_patterns": sum(patterns.values()),
            "unique_pattern_count": len(patterns),
            "top_pattern": top_pattern,
            "by_recommendation_type": by_type
        }
    
    def _save_evaluation_report(self, evaluation_metrics: Dict[str, Any]) -> None:
        """Save evaluation metrics to a report file.
        
        Args:
            evaluation_metrics: Dictionary of evaluation metrics
        """
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics as JSON
        report_file = self.output_dir / f"ml_evaluation_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(evaluation_metrics, f, default=str, indent=2)
            
        logger.info(f"Evaluation report saved to {report_file}")
        
        # Generate visualizations if matplotlib is available
        try:
            self._generate_visualizations(evaluation_metrics, timestamp)
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
    
    def _generate_visualizations(self, evaluation_metrics: Dict[str, Any], timestamp: str) -> None:
        """Generate visualizations for evaluation metrics.
        
        Args:
            evaluation_metrics: Dictionary of evaluation metrics
            timestamp: Timestamp string for filenames
        """
        # Skip if matplotlib is not available
        if not _has_matplotlib:
            logger.warning("Matplotlib not available, skipping visualization generation")
            return
            
        # Create directory for visualizations
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Pattern distribution pie chart
        if evaluation_metrics["pattern_metrics"]["patterns"]:
            plt.figure(figsize=(10, 6))
            patterns = evaluation_metrics["pattern_metrics"]["patterns"]
            labels = list(patterns.keys())
            sizes = list(patterns.values())
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Usage Pattern Distribution')
            
            plt.savefig(vis_dir / f"pattern_distribution_{timestamp}.png")
            plt.close()
        
        # Priority score changes histogram
        if evaluation_metrics["priority_changes"]["changes"]:
            plt.figure(figsize=(10, 6))
            changes = [c["percent_change"] for c in evaluation_metrics["priority_changes"]["changes"]]
            
            plt.hist(changes, bins=20)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.xlabel('Percent Change in Priority Score')
            plt.ylabel('Count')
            plt.title('ML Impact on Priority Scores')
            
            plt.savefig(vis_dir / f"priority_changes_{timestamp}.png")
            plt.close()
        
        # ML enhancement by recommendation type
        if evaluation_metrics["ml_insight_metrics"]["by_recommendation_type"]:
            plt.figure(figsize=(12, 6))
            by_type = evaluation_metrics["ml_insight_metrics"]["by_recommendation_type"]
            
            types = list(by_type.keys())
            total = [by_type[t]["total"] for t in types]
            enhanced = [by_type[t]["ml_enhanced"] for t in types]
            
            x = range(len(types))
            width = 0.35
            
            plt.bar(x, total, width, label='Total')
            plt.bar([i + width for i in x], enhanced, width, label='ML Enhanced')
            
            plt.xlabel('Recommendation Type')
            plt.ylabel('Count')
            plt.title('ML Enhancement by Recommendation Type')
            plt.xticks([i + width/2 for i in x], types, rotation=45)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(vis_dir / f"ml_by_type_{timestamp}.png")
            plt.close()
        
        logger.info(f"Evaluation visualizations saved to {vis_dir}")
    
    def compare_feedback_to_predictions(self, 
                                      enhanced_recommendations: List[Dict[str, Any]],
                                      feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare ML predictions to actual feedback.
        
        Args:
            enhanced_recommendations: List of ML-enhanced recommendations
            feedback_data: Feedback data on implemented recommendations
            
        Returns:
            Dictionary of comparison metrics
        """
        logger.info("Comparing ML predictions to feedback data")
        
        # Initialize comparison metrics
        comparison = {
            "business_impact": {
                "predicted": [],
                "actual": [],
                "error": [],
                "mae": 0,
                "mse": 0
            },
            "complexity": {
                "predicted": [],
                "actual": [],
                "error": [],
                "mae": 0,
                "mse": 0
            },
            "anomaly_detection": {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0
            }
        }
        
        # Process each recommendation
        for rec in enhanced_recommendations:
            rec_id = rec.get("recommendation_id", "")
            
            # Skip if no feedback for this recommendation
            if rec_id not in feedback_data.get("recommendations", {}):
                continue
                
            feedback = feedback_data["recommendations"][rec_id]
            
            # Compare business impact
            if "ml_insights" in rec and "business_impact" in rec["ml_insights"] and "business_impact" in feedback:
                predicted = rec["ml_insights"]["business_impact"]
                actual = feedback["business_impact"]
                error = actual - predicted
                
                comparison["business_impact"]["predicted"].append(predicted)
                comparison["business_impact"]["actual"].append(actual)
                comparison["business_impact"]["error"].append(error)
            
            # Compare complexity
            if "complexity" in rec and "complexity_rating" in feedback:
                predicted = rec["complexity"]
                actual = feedback["complexity_rating"]
                error = actual - predicted
                
                comparison["complexity"]["predicted"].append(predicted)
                comparison["complexity"]["actual"].append(actual)
                comparison["complexity"]["error"].append(error)
            
            # Compare anomaly detection
            if "ml_insights" in rec and "is_anomaly" in rec["ml_insights"]:
                predicted_anomaly = rec["ml_insights"]["is_anomaly"]
                actual_anomaly = feedback.get("is_anomaly", False)
                
                if predicted_anomaly and actual_anomaly:
                    comparison["anomaly_detection"]["true_positives"] += 1
                elif predicted_anomaly and not actual_anomaly:
                    comparison["anomaly_detection"]["false_positives"] += 1
                elif not predicted_anomaly and not actual_anomaly:
                    comparison["anomaly_detection"]["true_negatives"] += 1
                elif not predicted_anomaly and actual_anomaly:
                    comparison["anomaly_detection"]["false_negatives"] += 1
        
        # Calculate business impact metrics
        if comparison["business_impact"]["error"]:
            mae = sum(abs(e) for e in comparison["business_impact"]["error"]) / len(comparison["business_impact"]["error"])
            mse = sum(e**2 for e in comparison["business_impact"]["error"]) / len(comparison["business_impact"]["error"])
            
            comparison["business_impact"]["mae"] = mae
            comparison["business_impact"]["mse"] = mse
        
        # Calculate complexity metrics
        if comparison["complexity"]["error"]:
            mae = sum(abs(e) for e in comparison["complexity"]["error"]) / len(comparison["complexity"]["error"])
            mse = sum(e**2 for e in comparison["complexity"]["error"]) / len(comparison["complexity"]["error"])
            
            comparison["complexity"]["mae"] = mae
            comparison["complexity"]["mse"] = mse
        
        # Calculate anomaly detection metrics
        ad = comparison["anomaly_detection"]
        tp = ad["true_positives"]
        fp = ad["false_positives"]
        fn = ad["false_negatives"]
        
        if tp + fp > 0:
            ad["precision"] = tp / (tp + fp)
        
        if tp + fn > 0:
            ad["recall"] = tp / (tp + fn)
        
        if ad["precision"] + ad["recall"] > 0:
            ad["f1"] = 2 * (ad["precision"] * ad["recall"]) / (ad["precision"] + ad["recall"])
        
        logger.info(f"Comparison complete: Business Impact MAE={comparison['business_impact']['mae']:.2f}, " +
                  f"Anomaly F1={ad['f1']:.2f}")
        
        return comparison