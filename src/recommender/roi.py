"""ROI calculation logic for BigQuery optimization recommendations."""

from typing import Dict, Any

# BigQuery cost constants (in USD)
BQ_STORAGE_COST_PER_GB = 0.02  # $0.02 per GB/month for active storage
BQ_QUERY_COST_PER_TB = 5.0  # $5 per TB for on-demand queries


def calculate_roi(recommendation: Dict[str, Any], dataset_size_gb: float, 
                 table_size_gb: float) -> Dict[str, Any]:
    """Calculate ROI metrics for a recommendation.
    
    Args:
        recommendation: The recommendation to evaluate
        dataset_size_gb: Total dataset size in GB
        table_size_gb: Size of the specific table in GB
        
    Returns:
        Dict with ROI metrics including:
        - roi: Return on investment ratio
        - annual_savings_usd: Estimated annual savings in USD
        - implementation_cost_usd: Estimated implementation cost in USD
        - payback_period_months: Months to recoup implementation costs
    """
    # Get estimated savings percentage from recommendation
    savings_pct = recommendation.get("estimated_savings_pct", 0) / 100.0
    
    # Different calculation methods based on recommendation type
    rec_type = recommendation.get("type", "")
    
    # Calculate implementation cost based on effort
    effort_level = recommendation.get("estimated_effort", "medium")
    implementation_hours = {
        "high": 40,  # 1 week
        "medium": 16,  # 2 days
        "low": 4  # Half day
    }.get(effort_level, 16)
    
    # Assume average cost of $150/hour for engineering time
    hourly_rate = 150  
    implementation_cost_usd = implementation_hours * hourly_rate
    
    # Calculate annual savings based on recommendation type
    annual_savings_usd = 0
    
    if "partitioning" in rec_type or "clustering" in rec_type:
        # Both storage and query savings
        storage_savings = table_size_gb * BQ_STORAGE_COST_PER_GB * 12 * (savings_pct * 0.3)  # 30% of savings from storage
        
        # Estimate query volume based on table size
        monthly_query_tb = max(1, table_size_gb / 100) * 10  # Rough estimate: 10 TB queried per month per 100GB
        query_savings = monthly_query_tb * BQ_QUERY_COST_PER_TB * 12 * (savings_pct * 0.7)  # 70% of savings from queries
        
        annual_savings_usd = storage_savings + query_savings
        
    elif "compression" in rec_type:
        # Primarily storage savings
        annual_savings_usd = table_size_gb * BQ_STORAGE_COST_PER_GB * 12 * savings_pct
        
    elif "schema" in rec_type:
        # Both storage and query savings
        storage_savings = table_size_gb * BQ_STORAGE_COST_PER_GB * 12 * savings_pct
        query_savings = (table_size_gb / 100) * 10 * BQ_QUERY_COST_PER_TB * 12 * (savings_pct * 0.5)  # Half the savings pct for queries
        
        annual_savings_usd = storage_savings + query_savings
        
    elif "query" in rec_type:
        # Query savings only
        monthly_query_tb = max(1, table_size_gb / 50) * 10  # Higher estimate for query-focused recommendations
        annual_savings_usd = monthly_query_tb * BQ_QUERY_COST_PER_TB * 12 * savings_pct
    
    # Calculate ROI (return divided by investment)
    roi = annual_savings_usd / implementation_cost_usd if implementation_cost_usd > 0 else 0
    
    # Calculate payback period in months
    monthly_savings = annual_savings_usd / 12
    payback_period_months = implementation_cost_usd / monthly_savings if monthly_savings > 0 else 0
    
    return {
        "roi": roi,
        "annual_savings_usd": annual_savings_usd,
        "implementation_cost_usd": implementation_cost_usd,
        "payback_period_months": payback_period_months
    }
