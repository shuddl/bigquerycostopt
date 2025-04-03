"""FastAPI implementation of the BigQuery Cost Intelligence Engine API."""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import uuid

from fastapi import FastAPI, HTTPException, Depends, Query, Body, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
from pydantic import BaseModel, Field
import pandas as pd

from ..utils.logging import setup_logger
from ..utils.security import validate_request, validate_signature
from ..analysis.cost_attribution import (
    CostAttributionAnalyzer, 
    CostAnomalyDetector,
    CostAlertSystem,
    get_cost_attribution_data
)
from ..ml.cost_anomaly_detection import (
    TimeSeriesForecaster,
    MLCostAnomalyDetector,
    CostAttributionClusterer,
    detect_anomalies_with_ml
)

# Initialize logger
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BigQuery Cost Intelligence Engine API",
    description="API for analyzing and optimizing BigQuery costs, with attribution dashboard and anomaly detection",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to responses."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        response.headers["Server"] = "BigQueryCostOpt"
        response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        return response

# Request timing middleware
class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware for timing requests and logging performance metrics."""
    
    async def dispatch(self, request: Request, call_next):
        """Time request processing and log metrics."""
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log request timing
        logger.debug(f"Request to {request.url.path} processed in {process_time:.6f} seconds")
        
        # Log slow requests
        if process_time > 1.0:  # Log requests taking more than 1 second
            logger.warning(f"Slow request: {request.method} {request.url.path} took {process_time:.6f} seconds")
        
        return response

# Set up CORS with restricted origins for production
origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(self, app, requests_per_minute=60):
        """Initialize with configurable rate limit."""
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        self.reset_interval = 60  # Reset counts every 60 seconds
        
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        # Get client identifier (API key or IP)
        client_id = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not client_id:
            client_id = request.client.host
            
        # Get current timestamp
        now = time.time()
        
        # Clear expired entries
        self._clear_expired(now)
        
        # Check and update rate limit
        if client_id in self.request_counts:
            count, timestamp = self.request_counts[client_id]
            if now - timestamp < self.reset_interval:
                # Within the time window, check limit
                if count >= self.requests_per_minute:
                    # Rate limit exceeded
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "message": f"Maximum {self.requests_per_minute} requests per minute allowed"
                        }
                    )
                # Update count
                self.request_counts[client_id] = (count + 1, timestamp)
            else:
                # Reset for new time window
                self.request_counts[client_id] = (1, now)
        else:
            # First request for this client
            self.request_counts[client_id] = (1, now)
        
        # Process the request normally
        return await call_next(request)
    
    def _clear_expired(self, current_time):
        """Clear expired entries from request count tracking."""
        expired_clients = []
        for client_id, (count, timestamp) in self.request_counts.items():
            if current_time - timestamp >= self.reset_interval:
                expired_clients.append(client_id)
                
        for client_id in expired_clients:
            del self.request_counts[client_id]

# Add timing middleware
app.add_middleware(TimingMiddleware)

# Add rate limiting middleware - 60 requests per minute
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

# Health check endpoint
@app.get("/api/health", tags=["Health"], summary="API Health Check")
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Cache for frequent requests
_cache = {}
_cache_ttl = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

def get_from_cache(key: str) -> Optional[Any]:
    """Get data from cache if available and not expired."""
    if key in _cache and key in _cache_ttl:
        # Check if cache is still valid
        if datetime.now().timestamp() < _cache_ttl[key]:
            return _cache[key]
    
    return None

def set_in_cache(key: str, data: Any, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
    """Store data in cache."""
    _cache[key] = data
    _cache_ttl[key] = datetime.now().timestamp() + ttl_seconds

async def check_auth(request: Request) -> bool:
    """Validate authentication token."""
    if not validate_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# Pydantic models for request/response validation
class AnalysisRequest(BaseModel):
    project_id: str
    dataset_id: str
    callback_url: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str

class TeamMapping(BaseModel):
    project_id: str
    mapping: Dict[str, str]

@app.get("/api/v1/cost-dashboard/summary")
async def get_cost_summary(
    project_id: str,
    days: int = Query(30, description="Number of days to analyze"),
    auth: bool = Depends(check_auth)
):
    """Get cost summary for the specified period."""
    # Check cache
    cache_key = f"cost_summary_{project_id}_{days}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
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
        
        return result
    
    except Exception as e:
        logger.exception(f"Error generating cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cost-dashboard/attribution")
async def get_cost_attribution(
    project_id: str,
    days: int = Query(30, description="Number of days to analyze"),
    dimensions: str = Query("user,team,pattern,day,table", description="Comma-separated list of attribution dimensions"),
    auth: bool = Depends(check_auth)
):
    """Get detailed cost attribution data for the specified period."""
    dimensions_list = dimensions.split(',')
    
    # Check cache
    cache_key = f"cost_attribution_{project_id}_{days}_{dimensions}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
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
        
        return result
    
    except Exception as e:
        logger.exception(f"Error generating cost attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cost-dashboard/trends")
async def get_cost_trends(
    project_id: str,
    days: int = Query(90, description="Number of days to analyze"),
    granularity: str = Query("day", description="Time granularity (day, week, month)"),
    auth: bool = Depends(check_auth)
):
    """Get cost trends over time."""
    if granularity not in ['day', 'week', 'month']:
        raise HTTPException(status_code=400, detail="Invalid granularity. Must be one of: day, week, month")
    
    # Check cache
    cache_key = f"cost_trends_{project_id}_{days}_{granularity}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
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
        
        return result
    
    except Exception as e:
        logger.exception(f"Error generating cost trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cost-dashboard/compare-periods")
async def compare_cost_periods(
    project_id: str,
    current_days: int = Query(30, description="Number of days in current period"),
    previous_days: int = Query(30, description="Number of days in previous period"),
    auth: bool = Depends(check_auth)
):
    """Compare costs between two time periods."""
    # Check cache
    cache_key = f"cost_compare_{project_id}_{current_days}_{previous_days}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
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
        
        return result
    
    except Exception as e:
        logger.exception(f"Error comparing cost periods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cost-dashboard/expensive-queries")
async def get_expensive_queries(
    project_id: str,
    days: int = Query(30, description="Number of days to analyze"),
    limit: int = Query(100, description="Maximum number of queries to return"),
    auth: bool = Depends(check_auth)
):
    """Get the most expensive queries."""
    # Check cache
    cache_key = f"expensive_queries_{project_id}_{days}_{limit}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
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
        
        return result
    
    except Exception as e:
        logger.exception(f"Error retrieving expensive queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cost-dashboard/anomalies")
async def get_cost_anomalies(
    project_id: str,
    days: int = Query(30, description="Number of days to analyze"),
    anomaly_types: str = Query("daily,user,team,pattern", description="Comma-separated list of anomaly types"),
    use_ml: bool = Query(False, description="Whether to use ML-enhanced anomaly detection"),
    auth: bool = Depends(check_auth)
):
    """Get cost anomalies for the specified period."""
    # Check cache
    cache_key = f"cost_anomalies_{project_id}_{days}_{anomaly_types}_{use_ml}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
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
        
        return result
    
    except Exception as e:
        logger.exception(f"Error detecting cost anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cost-dashboard/forecast")
async def get_cost_forecast(
    project_id: str,
    training_days: int = Query(90, description="Number of days to use for training"),
    forecast_days: int = Query(7, description="Number of days to forecast"),
    auth: bool = Depends(check_auth)
):
    """Get cost forecast for the specified period."""
    # Check cache
    cache_key = f"cost_forecast_{project_id}_{training_days}_{forecast_days}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
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
        
        return result
    
    except Exception as e:
        logger.exception(f"Error generating cost forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cost-dashboard/user-clusters")
async def get_user_clusters(
    project_id: str,
    days: int = Query(90, description="Number of days to analyze"),
    n_clusters: int = Query(5, description="Number of clusters to generate"),
    auth: bool = Depends(check_auth)
):
    """Get user behavior clusters for the specified period."""
    # Check cache
    cache_key = f"user_clusters_{project_id}_{days}_{n_clusters}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
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
        
        return result
    
    except Exception as e:
        logger.exception(f"Error generating user clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/cost-dashboard/team-mapping")
async def update_team_mapping(
    mapping_data: TeamMapping,
    auth: bool = Depends(check_auth)
):
    """Update the team mapping for user attribution."""
    try:
        # Initialize analyzer
        analyzer = CostAttributionAnalyzer(project_id=mapping_data.project_id)
        
        # Update team mapping
        analyzer.set_team_mapping(mapping_data.mapping)
        
        # Return success
        return {
            "status": "success",
            "message": f"Team mapping updated with {len(mapping_data.mapping)} entries",
            "updated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.exception(f"Error updating team mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cost-dashboard/alerts")
async def get_cost_alerts(
    project_id: str,
    days: int = Query(7, description="Number of days to look back"),
    min_cost_increase_usd: float = Query(100.0, description="Minimum cost increase to trigger alert"),
    auth: bool = Depends(check_auth)
):
    """Get cost alerts for the specified period."""
    # Check cache
    cache_key = f"cost_alerts_{project_id}_{days}_{min_cost_increase_usd}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
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
        
        return result
    
    except Exception as e:
        logger.exception(f"Error generating cost alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for basic analysis (existing functionality)
@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def trigger_analysis(
    analysis: AnalysisRequest,
    x_user_id: Optional[str] = Header(None),
    auth: bool = Depends(check_auth)
):
    """Trigger a BigQuery dataset analysis."""
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # In a real implementation, this would submit a message to Pub/Sub
        # For now, return a simulated response
        return {
            "analysis_id": analysis_id,
            "status": "pending",
            "message": "Analysis request submitted successfully"
        }
    
    except Exception as e:
        logger.exception(f"Error processing analysis request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analysis/{analysis_id}")
async def get_analysis_status(
    analysis_id: str,
    auth: bool = Depends(check_auth)
):
    """Get the status of a previously submitted analysis."""
    # This would query the status from a database
    # For now, return a placeholder response
    return {
        "analysis_id": analysis_id,
        "status": "in_progress",
        "progress": 45,
        "message": "Analyzing query patterns"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bigquerycostopt.src.api.fastapi_server:app", host="0.0.0.0", port=port, reload=False)