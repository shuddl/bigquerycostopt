"""API endpoints for the BigQuery Cost Attribution Dashboard with Anomaly Detection."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import os

from flask import Blueprint, request, jsonify, current_app
import pandas as pd

from ..utils.logging import setup_logger
from ..utils.security import validate_request
from ..analysis.cost_attribution import (
    CostAttributionAnalyzer, 
    CostAnomalyDetector,
    CostAlertSystem
)
from ..ml.cost_anomaly_detection import (
    TimeSeriesForecaster,
    MLCostAnomalyDetector,
    CostAttributionClusterer,
    detect_anomalies_with_ml
)

# Initialize logger
logger = setup_logger(__name__)

# Create blueprint for cost dashboard API
cost_dashboard_bp = Blueprint('cost_dashboard', __name__, url_prefix='/api/v1/cost-dashboard')

# Cache for frequent requests
_cache = {}
_cache_ttl = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

def get_from_cache(key: str) -> Optional[Any]:
    """Get data from cache if available and not expired.
    
    Args:
        key: Cache key
        
    Returns:
        Cached data or None if not available
    """
    if key in _cache and key in _cache_ttl:
        # Check if cache is still valid
        if datetime.now().timestamp() < _cache_ttl[key]:
            return _cache[key]
    
    return None

def set_in_cache(key: str, data: Any, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
    """Store data in cache.
    
    Args:
        key: Cache key
        data: Data to cache
        ttl_seconds: Time to live in seconds
    """
    _cache[key] = data
    _cache_ttl[key] = datetime.now().timestamp() + ttl_seconds

@cost_dashboard_bp.route('/summary', methods=['GET'])
def get_cost_summary():
    """Get cost summary for the specified period.
    
    Query parameters:
        - project_id: GCP project ID
        - days: Number of days to analyze (default: 30)
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get request parameters
    project_id = request.args.get('project_id')
    days = int(request.args.get('days', 30))
    
    if not project_id:
        return jsonify({"error": "Missing required parameter: project_id"}), 400
    
    # Check cache
    cache_key = f"cost_summary_{project_id}_{days}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
    
    try:
        # Initialize analyzer
        analyzer = CostAttributionAnalyzer(project_id=project_id)
        
        # Get cost summary
        summary = analyzer.get_cost_summary(days_back=days)
        
        # Add metadata
        result = {
            "summary": summary,
            "query_params": {
                "project_id": project_id,
                "days": days
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result
        set_in_cache(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error generating cost summary: {e}")
        return jsonify({"error": str(e)}), 500

@cost_dashboard_bp.route('/attribution', methods=['GET'])
def get_cost_attribution():
    """Get detailed cost attribution data for the specified period.
    
    Query parameters:
        - project_id: GCP project ID
        - days: Number of days to analyze (default: 30)
        - dimensions: Comma-separated list of attribution dimensions to include 
                     (default: all - user,team,pattern,day,table)
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get request parameters
    project_id = request.args.get('project_id')
    days = int(request.args.get('days', 30))
    dimensions = request.args.get('dimensions', 'user,team,pattern,day,table')
    dimensions_list = dimensions.split(',')
    
    if not project_id:
        return jsonify({"error": "Missing required parameter: project_id"}), 400
    
    # Check cache
    cache_key = f"cost_attribution_{project_id}_{days}_{dimensions}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
    
    try:
        # Initialize analyzer
        analyzer = CostAttributionAnalyzer(project_id=project_id)
        
        # Get cost attribution data
        costs = analyzer.attribute_costs(days_back=days)
        
        # Filter dimensions
        filtered_costs = {}
        dimension_map = {
            'user': 'cost_by_user',
            'team': 'cost_by_team',
            'pattern': 'cost_by_pattern',
            'day': 'cost_by_day',
            'table': 'cost_by_table'
        }
        
        for dim in dimensions_list:
            if dim in dimension_map and dimension_map[dim] in costs:
                filtered_costs[dimension_map[dim]] = costs[dimension_map[dim]].to_dict('records')
        
        # Add metadata
        result = {
            "attribution": filtered_costs,
            "query_params": {
                "project_id": project_id,
                "days": days,
                "dimensions": dimensions
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result
        set_in_cache(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error generating cost attribution: {e}")
        return jsonify({"error": str(e)}), 500

@cost_dashboard_bp.route('/trends', methods=['GET'])
def get_cost_trends():
    """Get cost trends over time.
    
    Query parameters:
        - project_id: GCP project ID
        - days: Number of days to analyze (default: 90)
        - granularity: Time granularity (day, week, month) (default: day)
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get request parameters
    project_id = request.args.get('project_id')
    days = int(request.args.get('days', 90))
    granularity = request.args.get('granularity', 'day')
    
    if not project_id:
        return jsonify({"error": "Missing required parameter: project_id"}), 400
    
    if granularity not in ['day', 'week', 'month']:
        return jsonify({"error": "Invalid granularity. Must be one of: day, week, month"}), 400
    
    # Check cache
    cache_key = f"cost_trends_{project_id}_{days}_{granularity}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
    
    try:
        # Initialize analyzer
        analyzer = CostAttributionAnalyzer(project_id=project_id)
        
        # Get cost trends
        trends = analyzer.get_cost_trends(days_back=days, granularity=granularity)
        
        # Convert to serializable format (handle date objects)
        trends_data = []
        for _, row in trends.iterrows():
            record = row.to_dict()
            record['period'] = record['period'].strftime('%Y-%m-%d')
            trends_data.append(record)
        
        # Add metadata
        result = {
            "trends": trends_data,
            "query_params": {
                "project_id": project_id,
                "days": days,
                "granularity": granularity
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result
        set_in_cache(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error generating cost trends: {e}")
        return jsonify({"error": str(e)}), 500

@cost_dashboard_bp.route('/compare-periods', methods=['GET'])
def compare_cost_periods():
    """Compare costs between two time periods.
    
    Query parameters:
        - project_id: GCP project ID
        - current_days: Number of days in current period (default: 30)
        - previous_days: Number of days in previous period (default: 30)
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get request parameters
    project_id = request.args.get('project_id')
    current_days = int(request.args.get('current_days', 30))
    previous_days = int(request.args.get('previous_days', 30))
    
    if not project_id:
        return jsonify({"error": "Missing required parameter: project_id"}), 400
    
    # Check cache
    cache_key = f"cost_compare_{project_id}_{current_days}_{previous_days}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
    
    try:
        # Initialize analyzer
        analyzer = CostAttributionAnalyzer(project_id=project_id)
        
        # Compare periods
        comparison = analyzer.compare_periods(current_days=current_days, previous_days=previous_days)
        
        # Add metadata
        result = {
            "comparison": comparison,
            "query_params": {
                "project_id": project_id,
                "current_days": current_days,
                "previous_days": previous_days
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result
        set_in_cache(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error comparing cost periods: {e}")
        return jsonify({"error": str(e)}), 500

@cost_dashboard_bp.route('/expensive-queries', methods=['GET'])
def get_expensive_queries():
    """Get the most expensive queries.
    
    Query parameters:
        - project_id: GCP project ID
        - days: Number of days to analyze (default: 30)
        - limit: Maximum number of queries to return (default: 100)
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get request parameters
    project_id = request.args.get('project_id')
    days = int(request.args.get('days', 30))
    limit = int(request.args.get('limit', 100))
    
    if not project_id:
        return jsonify({"error": "Missing required parameter: project_id"}), 400
    
    # Check cache
    cache_key = f"expensive_queries_{project_id}_{days}_{limit}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
    
    try:
        # Initialize analyzer
        analyzer = CostAttributionAnalyzer(project_id=project_id)
        
        # Get expensive queries
        queries = analyzer.get_expensive_queries(days_back=days, limit=limit)
        
        # Convert to serializable format (handle datetime objects)
        queries_data = []
        for _, row in queries.iterrows():
            record = row.to_dict()
            record['creation_time'] = record['creation_time'].isoformat() if 'creation_time' in record else None
            queries_data.append(record)
        
        # Add metadata
        result = {
            "queries": queries_data,
            "query_params": {
                "project_id": project_id,
                "days": days,
                "limit": limit
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result
        set_in_cache(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error retrieving expensive queries: {e}")
        return jsonify({"error": str(e)}), 500

@cost_dashboard_bp.route('/anomalies', methods=['GET'])
def get_cost_anomalies():
    """Get cost anomalies for the specified period.
    
    Query parameters:
        - project_id: GCP project ID
        - days: Number of days to analyze (default: 30)
        - anomaly_types: Comma-separated list of anomaly types to detect 
                        (default: daily,user,team,pattern)
        - use_ml: Whether to use ML-enhanced anomaly detection (default: false)
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get request parameters
    project_id = request.args.get('project_id')
    days = int(request.args.get('days', 30))
    anomaly_types = request.args.get('anomaly_types', 'daily,user,team,pattern')
    use_ml = request.args.get('use_ml', 'false').lower() == 'true'
    
    if not project_id:
        return jsonify({"error": "Missing required parameter: project_id"}), 400
    
    # Check cache
    cache_key = f"cost_anomalies_{project_id}_{days}_{anomaly_types}_{use_ml}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
    
    try:
        # If ML-enhanced anomaly detection requested
        if use_ml:
            # Get ML anomalies
            ml_anomalies = detect_anomalies_with_ml(project_id=project_id, days_back=days)
            
            # Add metadata
            result = {
                "anomalies": ml_anomalies,
                "query_params": {
                    "project_id": project_id,
                    "days": days,
                    "anomaly_types": anomaly_types,
                    "use_ml": use_ml
                },
                "generated_at": datetime.now().isoformat()
            }
        else:
            # Initialize analyzer and detector
            analyzer = CostAttributionAnalyzer(project_id=project_id)
            detector = CostAnomalyDetector(analyzer)
            
            # Generate anomaly report
            comparison_days = days  # Use same period for comparison by default
            report = detector.generate_anomaly_report(days_back=days, comparison_days=comparison_days)
            
            # Filter by requested anomaly types
            anomaly_types_list = anomaly_types.split(',')
            filtered_report = {k: v for k, v in report.items() if k not in [
                'daily_anomalies', 'user_anomalies', 'team_anomalies', 'pattern_anomalies'
            ]}
            
            # Add requested anomaly types
            type_map = {
                'daily': 'daily_anomalies',
                'user': 'user_anomalies',
                'team': 'team_anomalies',
                'pattern': 'pattern_anomalies'
            }
            
            for anomaly_type in anomaly_types_list:
                if anomaly_type in type_map and type_map[anomaly_type] in report:
                    filtered_report[type_map[anomaly_type]] = report[type_map[anomaly_type]]
            
            # Add metadata
            result = {
                "anomalies": filtered_report,
                "query_params": {
                    "project_id": project_id,
                    "days": days,
                    "anomaly_types": anomaly_types,
                    "use_ml": use_ml
                },
                "generated_at": datetime.now().isoformat()
            }
        
        # Cache result
        set_in_cache(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error detecting cost anomalies: {e}")
        return jsonify({"error": str(e)}), 500

@cost_dashboard_bp.route('/forecast', methods=['GET'])
def get_cost_forecast():
    """Get cost forecast for the specified period.
    
    Query parameters:
        - project_id: GCP project ID
        - training_days: Number of days to use for training (default: 90)
        - forecast_days: Number of days to forecast (default: 7)
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get request parameters
    project_id = request.args.get('project_id')
    training_days = int(request.args.get('training_days', 90))
    forecast_days = int(request.args.get('forecast_days', 7))
    
    if not project_id:
        return jsonify({"error": "Missing required parameter: project_id"}), 400
    
    # Check cache
    cache_key = f"cost_forecast_{project_id}_{training_days}_{forecast_days}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
    
    try:
        # Initialize analyzer and forecaster
        analyzer = CostAttributionAnalyzer(project_id=project_id)
        forecaster = TimeSeriesForecaster(analyzer)
        
        # Generate forecast
        forecast = forecaster.forecast_daily_costs(
            training_days=training_days,
            forecast_days=forecast_days
        )
        
        # Add metadata
        result = {
            "forecast": forecast,
            "query_params": {
                "project_id": project_id,
                "training_days": training_days,
                "forecast_days": forecast_days
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result
        set_in_cache(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error generating cost forecast: {e}")
        return jsonify({"error": str(e)}), 500

@cost_dashboard_bp.route('/user-clusters', methods=['GET'])
def get_user_clusters():
    """Get user behavior clusters for the specified period.
    
    Query parameters:
        - project_id: GCP project ID
        - days: Number of days to analyze (default: 90)
        - n_clusters: Number of clusters to generate (default: 5)
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get request parameters
    project_id = request.args.get('project_id')
    days = int(request.args.get('days', 90))
    n_clusters = int(request.args.get('n_clusters', 5))
    
    if not project_id:
        return jsonify({"error": "Missing required parameter: project_id"}), 400
    
    # Check cache
    cache_key = f"user_clusters_{project_id}_{days}_{n_clusters}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
    
    try:
        # Initialize analyzer and clusterer
        analyzer = CostAttributionAnalyzer(project_id=project_id)
        costs = analyzer.attribute_costs(days_back=days)
        user_costs = costs['cost_by_user']
        
        clusterer = CostAttributionClusterer()
        
        # Train with specified number of clusters
        training_data = {"n_clusters": n_clusters}
        metrics = clusterer.train(user_costs, training_data)
        
        # Get clusters
        clusters = clusterer.predict(user_costs)
        
        # Add metadata
        result = {
            "clusters": clusters,
            "metrics": metrics,
            "query_params": {
                "project_id": project_id,
                "days": days,
                "n_clusters": n_clusters
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result
        set_in_cache(cache_key, result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error generating user clusters: {e}")
        return jsonify({"error": str(e)}), 500

@cost_dashboard_bp.route('/team-mapping', methods=['POST'])
def update_team_mapping():
    """Update the team mapping for user attribution.
    
    Expected JSON body:
        - project_id: GCP project ID
        - mapping: Dictionary mapping user emails to team names
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Parse request data
    try:
        data = request.get_json()
        
        if 'project_id' not in data or 'mapping' not in data:
            return jsonify({"error": "Missing required fields: project_id, mapping"}), 400
        
        project_id = data['project_id']
        mapping = data['mapping']
        
        if not isinstance(mapping, dict):
            return jsonify({"error": "Mapping must be a dictionary"}), 400
        
        # Initialize analyzer
        analyzer = CostAttributionAnalyzer(project_id=project_id)
        
        # Update team mapping
        analyzer.set_team_mapping(mapping)
        
        # Return success
        return jsonify({
            "status": "success",
            "message": f"Team mapping updated with {len(mapping)} entries",
            "updated_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.exception(f"Error updating team mapping: {e}")
        return jsonify({"error": str(e)}), 500

@cost_dashboard_bp.route('/alerts', methods=['GET'])
def get_cost_alerts():
    """Get cost alerts for the specified period.
    
    Query parameters:
        - project_id: GCP project ID
        - days: Number of days to look back (default: 7)
        - min_cost_increase_usd: Minimum cost increase to trigger alert (default: 100.0)
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get request parameters
    project_id = request.args.get('project_id')
    days = int(request.args.get('days', 7))
    min_cost_increase_usd = float(request.args.get('min_cost_increase_usd', 100.0))
    
    if not project_id:
        return jsonify({"error": "Missing required parameter: project_id"}), 400
    
    # Check cache
    cache_key = f"cost_alerts_{project_id}_{days}_{min_cost_increase_usd}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)
    
    try:
        # Initialize analyzer, detector, and alert system
        analyzer = CostAttributionAnalyzer(project_id=project_id)
        detector = CostAnomalyDetector(analyzer)
        alert_system = CostAlertSystem(detector)
        
        # Generate alerts
        alerts = alert_system.check_and_generate_alerts(
            days_back=days,
            min_cost_increase_usd=min_cost_increase_usd
        )
        
        # Add metadata
        result = {
            "alerts": alerts,
            "alert_count": len(alerts),
            "query_params": {
                "project_id": project_id,
                "days": days,
                "min_cost_increase_usd": min_cost_increase_usd
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache result (shorter TTL for alerts)
        set_in_cache(cache_key, result, ttl_seconds=60)  # 1 minute TTL
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error generating cost alerts: {e}")
        return jsonify({"error": str(e)}), 500