"""Recommendation engine for BigQuery cost optimization."""

from typing import Dict, List, Any, Optional, Tuple, Set
import json
import uuid
import pandas as pd
from datetime import datetime, timedelta

from ..analysis.storage_optimizer import StorageOptimizer
from ..analysis.query_optimizer import QueryOptimizer
from ..analysis.schema_optimizer import SchemaOptimizer
from ..recommender.roi import ROICalculator
from ..implementation.planner import ImplementationPlanGenerator
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class RecommendationEngine:
    """Engine for generating, consolidating, and prioritizing BigQuery cost optimization recommendations."""
    
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        """Initialize the recommendation engine.
        
        Args:
            project_id: The GCP project ID to analyze
            credentials_path: Optional path to service account credentials
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        
        # Initialize optimizer components
        self.storage_optimizer = StorageOptimizer(
            project_id=project_id, 
            credentials_path=credentials_path
        )
        
        self.query_optimizer = QueryOptimizer(
            project_id=project_id,
            credentials_path=credentials_path
        )
        
        self.schema_optimizer = SchemaOptimizer(
            project_id=project_id,
            credentials_path=credentials_path
        )
        
        # Initialize ROI calculator
        self.roi_calculator = ROICalculator()
        
        # Initialize implementation plan generator
        self.plan_generator = ImplementationPlanGenerator()
        
        # Storage for consolidated recommendations
        self.recommendations = []
        self.raw_recommendations = {
            "storage": [],
            "query": [],
            "schema": []
        }
        
        logger.info(f"Initialized RecommendationEngine for project {project_id}")
    
    def analyze_dataset(self, dataset_id: str, 
                        days: int = 30,
                        min_table_size_gb: float = 1.0,
                        include_storage: bool = True,
                        include_query: bool = True,
                        include_schema: bool = True) -> Dict[str, Any]:
        """Run a comprehensive analysis of a dataset to generate optimization recommendations.
        
        Args:
            dataset_id: The BigQuery dataset ID to analyze
            days: Number of days of query history to analyze
            min_table_size_gb: Minimum table size in GB to analyze
            include_storage: Whether to include storage optimization analysis
            include_query: Whether to include query optimization analysis
            include_schema: Whether to include schema optimization analysis
            
        Returns:
            Dict containing all recommendations and summary information
        """
        logger.info(f"Starting comprehensive analysis of dataset {dataset_id}")
        
        # Reset recommendations for this analysis
        self.recommendations = []
        self.raw_recommendations = {
            "storage": [],
            "query": [],
            "schema": []
        }
        
        # Run analyses based on parameters
        if include_storage:
            logger.info("Running storage optimization analysis")
            storage_results = self.storage_optimizer.analyze_dataset(
                dataset_id, 
                min_table_size_gb=min_table_size_gb
            )
            self.raw_recommendations["storage"] = storage_results.get("recommendations", [])
        
        if include_query:
            logger.info("Running query optimization analysis")
            query_results = self.query_optimizer.analyze_dataset_queries(
                dataset_id,
                days=days
            )
            self.raw_recommendations["query"] = query_results.get("recommendations", [])
        
        if include_schema:
            logger.info("Running schema optimization analysis")
            schema_results = self.schema_optimizer.analyze_dataset_schemas(
                dataset_id,
                min_table_size_gb=min_table_size_gb
            )
            self.raw_recommendations["schema"] = schema_results.get("recommendations", [])
        
        # Standardize and consolidate all recommendations
        self._standardize_recommendations()
        
        # Calculate ROI and prioritize recommendations
        self._calculate_roi_and_prioritize()
        
        # Generate implementation plans
        implementation_plan = self.plan_generator.generate_plan(self.recommendations)
        
        # Generate summary statistics
        summary = self._generate_summary()
        
        # Build final result
        result = {
            "dataset_id": dataset_id,
            "project_id": self.project_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "recommendations": self.recommendations,
            "implementation_plan": implementation_plan,
            "summary": summary
        }
        
        logger.info(f"Completed analysis of dataset {dataset_id}, " 
                   f"found {len(self.recommendations)} recommendations")
        
        return result
    
    def _standardize_recommendations(self) -> None:
        """Standardize recommendations from different optimizer modules into a uniform format."""
        standardized_recs = []
        rec_counter = 1
        
        # Process storage recommendations
        for rec in self.raw_recommendations["storage"]:
            # Add a unique identifier
            rec_id = f"STORAGE_{rec_counter:03d}"
            rec_counter += 1
            
            # Standardize to common format
            std_rec = {
                "recommendation_id": rec_id,
                "category": "storage",
                "type": rec.get("type", "unknown"),
                "table_id": rec.get("table_id", ""),
                "dataset_id": rec.get("dataset_id", ""),
                "project_id": rec.get("project_id", self.project_id),
                "description": rec.get("description", ""),
                "rationale": rec.get("rationale", ""),
                "recommendation": rec.get("recommendation", ""),
                "estimated_savings_pct": rec.get("estimated_savings_pct", 0),
                "estimated_storage_savings_gb": rec.get("estimated_size_reduction_gb", 0),
                "estimated_monthly_savings": rec.get("estimated_monthly_savings", 0),
                "estimated_effort": rec.get("effort_level", "medium"),
                "implementation_complexity": rec.get("implementation_complexity", "medium"),
                "implementation_sql": rec.get("implementation_sql", ""),
                "priority": rec.get("priority", "medium"),
                "current_state": self._extract_current_state(rec, "storage"),
                "affected_components": [rec.get("table_id", "")]
            }
            
            standardized_recs.append(std_rec)
        
        # Process query recommendations
        for rec in self.raw_recommendations["query"]:
            # Add a unique identifier
            rec_id = f"QUERY_{rec_counter:03d}"
            rec_counter += 1
            
            # Standardize to common format
            std_rec = {
                "recommendation_id": rec_id,
                "category": "query",
                "type": rec.get("type", "unknown"),
                "table_id": rec.get("table_id", ""),
                "dataset_id": rec.get("dataset_id", ""),
                "project_id": rec.get("project_id", self.project_id),
                "description": rec.get("description", ""),
                "rationale": rec.get("rationale", ""),
                "recommendation": rec.get("recommendation", ""),
                "estimated_savings_pct": rec.get("estimated_savings_pct", 0),
                "estimated_query_bytes_reduction": rec.get("estimated_bytes_reduction", 0),
                "estimated_monthly_savings": rec.get("estimated_monthly_cost_savings", 0),
                "estimated_effort": rec.get("effort_level", "medium"),
                "implementation_complexity": rec.get("implementation_complexity", "medium"),
                "implementation_sql": rec.get("optimized_query", ""),
                "priority": rec.get("priority", "medium"),
                "current_state": self._extract_current_state(rec, "query"),
                "affected_components": [rec.get("table_id", "")] if rec.get("table_id") else []
            }
            
            # Handle pattern-level recommendations that affect multiple tables
            if "affected_tables" in rec:
                std_rec["affected_components"] = rec["affected_tables"]
            
            standardized_recs.append(std_rec)
        
        # Process schema recommendations
        for rec in self.raw_recommendations["schema"]:
            # Add a unique identifier
            rec_id = f"SCHEMA_{rec_counter:03d}"
            rec_counter += 1
            
            # Standardize to common format
            std_rec = {
                "recommendation_id": rec_id,
                "category": "schema",
                "type": rec.get("type", "unknown"),
                "table_id": rec.get("table_id", ""),
                "dataset_id": rec.get("dataset_id", ""),
                "project_id": rec.get("project_id", self.project_id),
                "description": rec.get("description", ""),
                "rationale": rec.get("rationale", ""),
                "recommendation": rec.get("description", ""),  # Schema optimizer uses description as the recommendation
                "estimated_savings_pct": rec.get("estimated_storage_savings_pct", 0),
                "estimated_storage_savings_gb": rec.get("estimated_storage_savings_gb", 0),
                "estimated_monthly_savings": rec.get("estimated_monthly_cost_savings", 0),
                "estimated_effort": self._convert_complexity_to_effort(rec.get("implementation_complexity", "medium")),
                "implementation_complexity": rec.get("implementation_complexity", "medium"),
                "implementation_sql": rec.get("implementation_sql", ""),
                "priority": self._convert_score_to_priority(rec.get("priority_score", 50)),
                "current_state": self._extract_current_state(rec, "schema"),
                "affected_components": [rec.get("table_id", "")]
            }
            
            standardized_recs.append(std_rec)
        
        # Detect and handle recommendations with dependencies
        self._identify_dependencies(standardized_recs)
        
        self.recommendations = standardized_recs
    
    def _extract_current_state(self, recommendation: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Extract current state information from a recommendation.
        
        Args:
            recommendation: The recommendation to extract from
            category: The category of the recommendation (storage, query, schema)
            
        Returns:
            Dict with current state information
        """
        current_state = {}
        
        if category == "storage":
            # Extract current storage configuration
            if "type" in recommendation:
                rec_type = recommendation["type"]
                
                if "partitioning" in rec_type:
                    current_state["partitioning"] = recommendation.get("current_partitioning", "None")
                elif "clustering" in rec_type:
                    current_state["clustering"] = recommendation.get("current_clustering", "None")
                elif "compress" in rec_type:
                    current_state["compression"] = recommendation.get("current_compression", "None")
                elif "long_term_storage" in rec_type:
                    current_state["last_modified"] = recommendation.get("last_modified", "")
                    current_state["query_count_30d"] = recommendation.get("query_count_30d", 0)
        
        elif category == "query":
            # Extract current query pattern information
            if "original_query" in recommendation:
                current_state["query_pattern"] = recommendation["original_query"]
            if "query_count" in recommendation:
                current_state["query_count"] = recommendation["query_count"]
            if "avg_bytes_processed" in recommendation:
                current_state["avg_bytes_processed"] = recommendation["avg_bytes_processed"]
            
        elif category == "schema":
            # Extract current schema information
            if "column_name" in recommendation:
                current_state["column_name"] = recommendation["column_name"]
                current_state["current_type"] = recommendation.get("current_type", "")
                current_state["recommended_type"] = recommendation.get("recommended_type", "")
            elif "columns" in recommendation:
                current_state["columns"] = recommendation["columns"]
            elif "repeated_fields" in recommendation:
                current_state["repeated_fields"] = recommendation["repeated_fields"]
            elif "joined_table" in recommendation:
                current_state["joined_table"] = recommendation["joined_table"]
                current_state["join_count"] = recommendation.get("join_count", 0)
        
        return current_state
    
    def _convert_complexity_to_effort(self, complexity: str) -> str:
        """Convert implementation complexity to effort level.
        
        Args:
            complexity: Implementation complexity (low, medium, high)
            
        Returns:
            Corresponding effort level (low, medium, high)
        """
        # For now, they're the same, but we could have a more nuanced mapping in the future
        return complexity
    
    def _convert_score_to_priority(self, score: int) -> str:
        """Convert priority score to priority level.
        
        Args:
            score: Numeric priority score (0-100)
            
        Returns:
            Priority level (low, medium, high)
        """
        if score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"
    
    def _identify_dependencies(self, recommendations: List[Dict[str, Any]]) -> None:
        """Identify dependencies between recommendations.
        
        Args:
            recommendations: List of standardized recommendations
        """
        # Group recommendations by table
        table_recs = {}
        for rec in recommendations:
            table_id = rec["table_id"]
            if table_id not in table_recs:
                table_recs[table_id] = []
            table_recs[table_id].append(rec)
        
        # Check for dependencies within each table
        for table_id, recs in table_recs.items():
            storage_recs = [r for r in recs if r["category"] == "storage"]
            schema_recs = [r for r in recs if r["category"] == "schema"]
            
            # Schema changes often depend on storage recommendations
            # e.g., implementing partitioning before changing column data types
            if storage_recs and schema_recs:
                for schema_rec in schema_recs:
                    for storage_rec in storage_recs:
                        if "partitioning" in storage_rec["type"] or "clustering" in storage_rec["type"]:
                            if "depends_on" not in schema_rec:
                                schema_rec["depends_on"] = []
                            schema_rec["depends_on"].append(storage_rec["recommendation_id"])
            
            # Check for conflicting schema recommendations
            # e.g., removing a column vs changing its type
            schema_by_column = {}
            for rec in schema_recs:
                if "column_name" in rec.get("current_state", {}):
                    col_name = rec["current_state"]["column_name"]
                    if col_name not in schema_by_column:
                        schema_by_column[col_name] = []
                    schema_by_column[col_name].append(rec)
            
            # Mark conflicting recommendations
            for col_name, col_recs in schema_by_column.items():
                if len(col_recs) > 1:
                    for rec in col_recs:
                        if "conflicts_with" not in rec:
                            rec["conflicts_with"] = []
                        for other_rec in col_recs:
                            if other_rec != rec:
                                rec["conflicts_with"].append(other_rec["recommendation_id"])
    
    def _calculate_roi_and_prioritize(self) -> None:
        """Calculate ROI for all recommendations and prioritize them."""
        # Calculate ROI for each recommendation
        for rec in self.recommendations:
            try:
                roi_data = self.roi_calculator.calculate_roi(rec)
                rec.update(roi_data)
            except Exception as e:
                logger.warning(f"Failed to calculate ROI for recommendation {rec['recommendation_id']}: {e}")
                rec["roi"] = 0
                rec["payback_period_months"] = 0
                rec["annual_savings_usd"] = rec.get("estimated_monthly_savings", 0) * 12
                rec["implementation_cost_usd"] = 0
        
        # Group recommendations by type for relative impact assessment
        rec_by_type = {}
        for rec in self.recommendations:
            rec_type = rec.get("type", "unknown")
            if rec_type not in rec_by_type:
                rec_by_type[rec_type] = []
            rec_by_type[rec_type].append(rec)
        
        # For each type, calculate relative impact percentile
        for rec_type, recs in rec_by_type.items():
            if len(recs) <= 1:
                # If only one recommendation of this type, set it to 50th percentile
                for rec in recs:
                    rec["impact_percentile"] = 50
                continue
                
            # Sort by estimated savings
            sorted_recs = sorted(recs, key=lambda r: r.get("annual_savings_usd", 0))
            
            # Assign percentiles
            for i, rec in enumerate(sorted_recs):
                rec["impact_percentile"] = int((i / (len(sorted_recs) - 1)) * 100)
        
        # Calculate overall priority score based on ROI, impact, and effort
        for rec in self.recommendations:
            roi_score = min(100, rec.get("roi", 0) * 20)  # Scale ROI to 0-100
            impact_score = rec.get("impact_percentile", 50)
            
            # Convert effort to score (low effort = high score)
            effort_level = rec.get("estimated_effort", "medium")
            effort_score = {
                "low": 100,
                "medium": 50,
                "high": 20
            }.get(effort_level, 50)
            
            # Calculate weighted overall score
            # Weight ROI and impact more heavily than effort
            overall_score = (roi_score * 0.4) + (impact_score * 0.4) + (effort_score * 0.2)
            rec["priority_score"] = overall_score
            
            # Determine priority level from score
            if overall_score >= 70:
                rec["priority"] = "high"
            elif overall_score >= 40:
                rec["priority"] = "medium"
            else:
                rec["priority"] = "low"
        
        # Sort recommendations by priority score (descending)
        self.recommendations = sorted(
            self.recommendations, 
            key=lambda r: r.get("priority_score", 0), 
            reverse=True
        )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for all recommendations.
        
        Returns:
            Dict with summary statistics
        """
        # Count recommendations by priority
        high_priority = sum(1 for r in self.recommendations if r.get("priority") == "high")
        medium_priority = sum(1 for r in self.recommendations if r.get("priority") == "medium")
        low_priority = sum(1 for r in self.recommendations if r.get("priority") == "low")
        
        # Count recommendations by category
        storage_count = sum(1 for r in self.recommendations if r.get("category") == "storage")
        query_count = sum(1 for r in self.recommendations if r.get("category") == "query")
        schema_count = sum(1 for r in self.recommendations if r.get("category") == "schema")
        
        # Calculate potential savings
        total_monthly_savings = sum(r.get("estimated_monthly_savings", 0) for r in self.recommendations)
        total_annual_savings = sum(r.get("annual_savings_usd", 0) for r in self.recommendations)
        total_storage_savings_gb = sum(r.get("estimated_storage_savings_gb", 0) for r in self.recommendations)
        
        # Calculate implementation costs
        total_implementation_cost = sum(r.get("implementation_cost_usd", 0) for r in self.recommendations)
        
        # Calculate ROI
        overall_roi = total_annual_savings / total_implementation_cost if total_implementation_cost > 0 else 0
        payback_period_months = (total_implementation_cost / (total_annual_savings / 12)) if total_annual_savings > 0 else 0
        
        # Estimate implementation time
        implementation_days = self._estimate_implementation_time()
        
        # Top recommendations by savings (top 3)
        top_by_savings = sorted(
            self.recommendations, 
            key=lambda r: r.get("annual_savings_usd", 0), 
            reverse=True
        )[:3]
        
        top_recs = [{
            "recommendation_id": r["recommendation_id"],
            "description": r["description"],
            "annual_savings_usd": r.get("annual_savings_usd", 0),
            "priority": r.get("priority", "medium")
        } for r in top_by_savings]
        
        return {
            "total_recommendations": len(self.recommendations),
            "priority_breakdown": {
                "high": high_priority,
                "medium": medium_priority,
                "low": low_priority
            },
            "category_breakdown": {
                "storage": storage_count,
                "query": query_count,
                "schema": schema_count
            },
            "savings_summary": {
                "total_monthly_savings_usd": total_monthly_savings,
                "total_annual_savings_usd": total_annual_savings,
                "total_storage_savings_gb": total_storage_savings_gb
            },
            "implementation_summary": {
                "total_implementation_cost_usd": total_implementation_cost,
                "overall_roi": overall_roi,
                "payback_period_months": payback_period_months,
                "estimated_implementation_days": implementation_days
            },
            "top_recommendations": top_recs
        }
    
    def _estimate_implementation_time(self) -> int:
        """Estimate total implementation time in days.
        
        Returns:
            Estimated implementation time in days
        """
        # Effort level to days mapping
        effort_days = {
            "low": 0.5,    # Half day
            "medium": 2,   # 2 days
            "high": 5      # 1 week
        }
        
        # Calculate days for each recommendation
        total_days = 0
        for rec in self.recommendations:
            effort_level = rec.get("estimated_effort", "medium")
            total_days += effort_days.get(effort_level, 2)
        
        # Account for parallelization and overhead
        adjusted_days = int(total_days * 0.7)  # Assume 30% parallelization efficiency
        
        # Add planning and testing overhead
        adjusted_days += 3  # Planning, coordination, and testing overhead
        
        # Cap at reasonable maximum
        return min(adjusted_days, 60)
    
    def format_for_bigquery(self) -> List[Dict[str, Any]]:
        """Format recommendations for storage in BigQuery.
        
        Returns:
            List of recommendations formatted for BigQuery insertion
        """
        formatted_recs = []
        
        for rec in self.recommendations:
            # Format nested JSON fields as strings
            current_state = json.dumps(rec.get("current_state", {}))
            
            formatted_rec = {
                "recommendation_id": rec["recommendation_id"],
                "dataset_id": rec["dataset_id"],
                "table_id": rec["table_id"],
                "project_id": rec["project_id"],
                "category": rec["category"],
                "type": rec["type"],
                "description": rec["description"],
                "rationale": rec["rationale"],
                "recommendation": rec["recommendation"],
                "estimated_monthly_savings_usd": rec.get("estimated_monthly_savings", 0),
                "annual_savings_usd": rec.get("annual_savings_usd", 0),
                "estimated_storage_savings_gb": rec.get("estimated_storage_savings_gb", 0),
                "roi": rec.get("roi", 0),
                "implementation_cost_usd": rec.get("implementation_cost_usd", 0),
                "payback_period_months": rec.get("payback_period_months", 0),
                "priority": rec.get("priority", "medium"),
                "priority_score": rec.get("priority_score", 0),
                "implementation_complexity": rec.get("implementation_complexity", "medium"),
                "current_state": current_state,
                "depends_on": json.dumps(rec.get("depends_on", [])),
                "conflicts_with": json.dumps(rec.get("conflicts_with", [])),
                "timestamp": datetime.now().isoformat()
            }
            
            formatted_recs.append(formatted_rec)
        
        return formatted_recs
    
    def format_for_dashboard(self) -> Dict[str, Any]:
        """Format recommendations for Retool dashboard presentation.
        
        Returns:
            Dict with recommendations formatted for dashboard display
        """
        # Prepare data for dashboard visualization
        by_category = {
            "storage": [],
            "query": [],
            "schema": []
        }
        
        for rec in self.recommendations:
            category = rec.get("category", "other")
            if category in by_category:
                dashboard_rec = {
                    "id": rec["recommendation_id"],
                    "description": rec["description"],
                    "table": rec["table_id"],
                    "priority": rec.get("priority", "medium"),
                    "savings": rec.get("annual_savings_usd", 0),
                    "effort": rec.get("implementation_complexity", "medium"),
                    "roi": rec.get("roi", 0),
                    "type": rec["type"]
                }
                by_category[category].append(dashboard_rec)
        
        # Generate summary metrics
        monthly_savings = sum(r.get("estimated_monthly_savings", 0) for r in self.recommendations)
        annual_savings = sum(r.get("annual_savings_usd", 0) for r in self.recommendations)
        storage_savings = sum(r.get("estimated_storage_savings_gb", 0) for r in self.recommendations)
        
        # Count by priority
        high_priority = sum(1 for r in self.recommendations if r.get("priority") == "high")
        medium_priority = sum(1 for r in self.recommendations if r.get("priority") == "medium")
        low_priority = sum(1 for r in self.recommendations if r.get("priority") == "low")
        
        # Format for dashboard
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "summary": {
                "total_recommendations": len(self.recommendations),
                "monthly_savings_usd": monthly_savings,
                "annual_savings_usd": annual_savings,
                "storage_savings_gb": storage_savings,
                "priority_breakdown": {
                    "high": high_priority,
                    "medium": medium_priority,
                    "low": low_priority
                }
            },
            "recommendations": by_category,
            "charts": self._generate_dashboard_charts()
        }
        
        return dashboard_data
    
    def _generate_dashboard_charts(self) -> Dict[str, Any]:
        """Generate chart data for dashboard visualizations.
        
        Returns:
            Dict with chart data
        """
        # Create savings by category
        savings_by_category = {
            "storage": 0,
            "query": 0,
            "schema": 0
        }
        
        for rec in self.recommendations:
            category = rec.get("category", "other")
            if category in savings_by_category:
                savings_by_category[category] += rec.get("annual_savings_usd", 0)
        
        # Create recommendations by priority
        recs_by_priority = {
            "high": sum(1 for r in self.recommendations if r.get("priority") == "high"),
            "medium": sum(1 for r in self.recommendations if r.get("priority") == "medium"),
            "low": sum(1 for r in self.recommendations if r.get("priority") == "low")
        }
        
        # Create top tables by potential savings
        savings_by_table = {}
        for rec in self.recommendations:
            table_id = rec["table_id"]
            if table_id not in savings_by_table:
                savings_by_table[table_id] = 0
            savings_by_table[table_id] += rec.get("annual_savings_usd", 0)
        
        # Sort and get top 5 tables
        top_tables = dict(sorted(savings_by_table.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Create ROI distribution data
        roi_values = [rec.get("roi", 0) for rec in self.recommendations]
        roi_bins = [0, 1, 2, 3, 5, 10, float('inf')]
        roi_distribution = {f"{a}-{b}": sum(1 for r in roi_values if a <= r < b) 
                           for a, b in zip(roi_bins[:-1], roi_bins[1:])}
        roi_distribution[f"{roi_bins[-2]}+"] = roi_distribution.pop(f"{roi_bins[-2]}-{roi_bins[-1]}")
        
        return {
            "savings_by_category": savings_by_category,
            "recommendations_by_priority": recs_by_priority,
            "top_tables_by_savings": top_tables,
            "roi_distribution": roi_distribution
        }