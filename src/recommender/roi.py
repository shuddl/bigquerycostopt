"""ROI calculation logic for BigQuery optimization recommendations."""

from typing import Dict, Any, Optional
from datetime import datetime, date, timedelta
import math

# BigQuery cost constants (in USD)
BQ_STORAGE_COST_PER_GB_PER_MONTH = 0.02  # $0.02 per GB/month for active storage
BQ_LTS_STORAGE_COST_PER_GB_PER_MONTH = 0.01  # $0.01 per GB/month for long-term storage
BQ_QUERY_COST_PER_TB = 5.0  # $5 per TB for on-demand queries


class ROICalculator:
    """Calculator for ROI metrics for BigQuery optimization recommendations."""
    
    def __init__(self):
        """Initialize the ROI calculator."""
        self.hourly_engineering_rate = 150  # $150/hour for engineering time
        self.annual_discount_rate = 0.05  # 5% discount rate for NPV calculations
    
    def calculate_roi(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ROI metrics for a recommendation.
        
        Args:
            recommendation: The standardized recommendation to evaluate
            
        Returns:
            Dict with ROI metrics including:
            - roi: Return on investment ratio
            - annual_savings_usd: Estimated annual savings in USD
            - implementation_cost_usd: Estimated implementation cost in USD
            - payback_period_months: Months to recoup implementation costs
            - npv: Net present value of the recommendation over 3 years
        """
        # Extract base values
        category = recommendation.get("category", "unknown")
        rec_type = recommendation.get("type", "unknown")
        monthly_savings = recommendation.get("estimated_monthly_savings", 0)
        
        # Calculate implementation cost based on effort
        implementation_cost_usd = self._calculate_implementation_cost(recommendation)
        
        # Calculate annual savings based on category and type
        annual_savings_usd = self._calculate_annual_savings(recommendation)
        
        # Calculate ROI metrics
        roi = annual_savings_usd / implementation_cost_usd if implementation_cost_usd > 0 else 0
        monthly_savings = annual_savings_usd / 12
        payback_period_months = implementation_cost_usd / monthly_savings if monthly_savings > 0 else float('inf')
        
        # Calculate Net Present Value over 3 years
        npv = self._calculate_npv(implementation_cost_usd, annual_savings_usd, years=3)
        
        # Calculate risk-adjusted ROI
        risk_factor = self._calculate_risk_factor(recommendation)
        risk_adjusted_roi = roi * risk_factor
        
        return {
            "roi": roi,
            "annual_savings_usd": annual_savings_usd,
            "implementation_cost_usd": implementation_cost_usd,
            "payback_period_months": min(payback_period_months, 999),  # Cap at reasonable value
            "npv_3yr_usd": npv,
            "risk_adjusted_roi": risk_adjusted_roi,
            "risk_factor": risk_factor
        }
    
    def _calculate_implementation_cost(self, recommendation: Dict[str, Any]) -> float:
        """Calculate the implementation cost for a recommendation.
        
        Args:
            recommendation: The recommendation to evaluate
            
        Returns:
            Implementation cost in USD
        """
        # Get effort level
        effort_level = recommendation.get("estimated_effort", "medium")
        
        # Map effort level to hours
        implementation_hours = {
            "high": 40,    # 1 week
            "medium": 16,  # 2 days
            "high-medium": 24,  # 3 days
            "medium-low": 8,  # 1 day
            "low": 4       # Half day
        }.get(effort_level, 16)  # Default to medium
        
        # Adjust hours based on recommendation category
        category = recommendation.get("category", "unknown")
        rec_type = recommendation.get("type", "unknown")
        
        # Schema changes typically require more testing
        if category == "schema":
            implementation_hours *= 1.2  # 20% more for testing
            
            # Complex schema changes like denormalization require even more time
            if "denormalization" in rec_type or "normalize" in rec_type:
                implementation_hours *= 1.5
        
        # Partitioning implementation often requires more time for large tables
        if "partitioning" in rec_type:
            # Check if it's a large table
            storage_gb = recommendation.get("estimated_storage_savings_gb", 0)
            if storage_gb > 100:  # If it can save >100GB, it's probably a large table
                implementation_hours *= 1.25  # 25% more for large tables
        
        # Calculate cost
        return implementation_hours * self.hourly_engineering_rate
    
    def _calculate_annual_savings(self, recommendation: Dict[str, Any]) -> float:
        """Calculate annual savings for a recommendation.
        
        Args:
            recommendation: The recommendation to evaluate
            
        Returns:
            Annual savings in USD
        """
        category = recommendation.get("category", "unknown")
        rec_type = recommendation.get("type", "unknown")
        
        # Use existing monthly savings if provided
        if "estimated_monthly_savings" in recommendation and recommendation["estimated_monthly_savings"] > 0:
            return recommendation["estimated_monthly_savings"] * 12
        
        annual_savings = 0
        
        if category == "storage":
            # Extract storage savings from recommendation
            storage_savings_gb = recommendation.get("estimated_storage_savings_gb", 0)
            savings_pct = recommendation.get("estimated_savings_pct", 0) / 100
            
            if "partitioning" in rec_type or "clustering" in rec_type:
                # Both storage and query savings
                storage_savings = storage_savings_gb * BQ_STORAGE_COST_PER_GB_PER_MONTH * 12
                
                # Estimate query savings based on table size
                table_size_gb = storage_savings_gb / savings_pct if savings_pct > 0 else 0
                monthly_query_tb = max(1, table_size_gb / 100) * 10  # Approximate: 10 TB processed per month per 100GB
                query_savings = monthly_query_tb * BQ_QUERY_COST_PER_TB * 12 * (savings_pct * 0.7)  # 70% query savings
                
                annual_savings = storage_savings + query_savings
                
            elif "long_term_storage" in rec_type:
                # Storage cost difference between active and LTS
                cost_diff = BQ_STORAGE_COST_PER_GB_PER_MONTH - BQ_LTS_STORAGE_COST_PER_GB_PER_MONTH
                annual_savings = storage_savings_gb * cost_diff * 12
                
            else:  # General storage savings
                annual_savings = storage_savings_gb * BQ_STORAGE_COST_PER_GB_PER_MONTH * 12
                
        elif category == "query":
            # Extract query savings from recommendation
            bytes_reduction = recommendation.get("estimated_query_bytes_reduction", 0)
            savings_pct = recommendation.get("estimated_savings_pct", 0) / 100
            
            # Convert bytes to TB
            tb_reduction = bytes_reduction / (1024 ** 4)  # bytes to TB
            
            # Calculate annual query savings
            query_frequency = recommendation.get("current_state", {}).get("query_count", 12)  # Default to monthly
            annual_query_tb = tb_reduction * (query_frequency / 30) * 365  # Scale to annual
            annual_savings = annual_query_tb * BQ_QUERY_COST_PER_TB
            
        elif category == "schema":
            # Extract schema savings information
            storage_savings_gb = recommendation.get("estimated_storage_savings_gb", 0)
            
            # Most schema changes primarily affect storage costs
            storage_savings = storage_savings_gb * BQ_STORAGE_COST_PER_GB_PER_MONTH * 12
            
            # For certain types, also include query savings
            if "datatype" in rec_type or "column" in rec_type:
                # Improved data types and column removal can improve query performance
                query_savings = storage_savings * 0.5  # Rough estimate: 50% of storage savings
                annual_savings = storage_savings + query_savings
            else:
                annual_savings = storage_savings
        
        return max(0, annual_savings)  # Ensure non-negative
    
    def _calculate_npv(self, implementation_cost: float, annual_savings: float, years: int = 3) -> float:
        """Calculate Net Present Value over specified years.
        
        Args:
            implementation_cost: One-time implementation cost in USD
            annual_savings: Annual savings in USD
            years: Number of years to calculate NPV for
            
        Returns:
            Net Present Value in USD
        """
        monthly_discount_rate = self.annual_discount_rate / 12
        
        # Upfront cost (negative)
        npv = -1 * implementation_cost
        
        # Monthly savings for specified years
        monthly_savings = annual_savings / 12
        for month in range(1, years * 12 + 1):
            npv += monthly_savings / ((1 + monthly_discount_rate) ** month)
        
        return npv
    
    def _calculate_risk_factor(self, recommendation: Dict[str, Any]) -> float:
        """Calculate risk adjustment factor for ROI.
        
        Args:
            recommendation: The recommendation to evaluate
            
        Returns:
            Risk factor (0-1) where 1 is no risk adjustment
        """
        base_risk = 0.9  # Start with 90% confidence
        
        # Consider implementation complexity
        complexity = recommendation.get("implementation_complexity", "medium")
        complexity_factor = {
            "low": 0.95,
            "medium": 0.85,
            "high": 0.7
        }.get(complexity, 0.85)
        
        # Consider recommendation category
        category = recommendation.get("category", "unknown")
        rec_type = recommendation.get("type", "unknown")
        
        category_factor = {
            "storage": 0.9,  # Storage changes have medium risk
            "query": 0.95,   # Query optimizations are typically low risk
            "schema": 0.8    # Schema changes are higher risk
        }.get(category, 0.9)
        
        # Additional risk factors for specific recommendation types
        type_factor = 1.0
        
        if "schema_remove" in rec_type:
            type_factor = 0.7  # Removing columns is high risk
        elif "denormalization" in rec_type:
            type_factor = 0.8  # Denormalization can be complex
        elif "partitioning" in rec_type and "daily_to_monthly" in rec_type:
            type_factor = 0.85  # Changing partition granularity has some risk
        
        # Dependencies increase risk
        if "depends_on" in recommendation and recommendation["depends_on"]:
            dependency_factor = 0.9  # 10% risk increase due to dependencies
        else:
            dependency_factor = 1.0
        
        # Combine all factors
        risk_factor = base_risk * complexity_factor * category_factor * type_factor * dependency_factor
        
        # Ensure factor is between 0.5 and 1
        return max(0.5, min(1.0, risk_factor))


def calculate_roi(recommendation: Dict[str, Any], dataset_size_gb: Optional[float] = None, 
                table_size_gb: Optional[float] = None) -> Dict[str, Any]:
    """Standalone function to calculate ROI for backward compatibility.
    
    Args:
        recommendation: The recommendation to evaluate
        dataset_size_gb: Optional dataset size for context
        table_size_gb: Optional table size for context
        
    Returns:
        Dict with ROI metrics
    """
    calculator = ROICalculator()
    
    # Ensure table size information is available
    if "estimated_storage_savings_gb" not in recommendation and table_size_gb:
        savings_pct = recommendation.get("estimated_savings_pct", 0) / 100
        recommendation["estimated_storage_savings_gb"] = table_size_gb * savings_pct
    
    return calculator.calculate_roi(recommendation)