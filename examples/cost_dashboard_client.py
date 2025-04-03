#!/usr/bin/env python
"""Example client for the BigQuery Cost Attribution Dashboard API.

This script demonstrates how to interact with the cost dashboard API endpoints
and process the results.
"""

import argparse
import json
import os
import sys
import requests
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import time

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bigquerycostopt.src.utils.logging import setup_logger

logger = setup_logger(__name__)

class CostDashboardClient:
    """Client for interacting with the BigQuery Cost Dashboard API."""
    
    def __init__(self, base_url="http://localhost:8080", api_key=None):
        """Initialize the cost dashboard client.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Add user ID header for attribution
        self.headers["X-User-ID"] = os.environ.get("USER", "example-user")
    
    def get_cost_summary(self, project_id, days=30):
        """Get cost summary for the specified period.
        
        Args:
            project_id: GCP project ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with cost summary data
        """
        url = f"{self.base_url}/api/v1/cost-dashboard/summary"
        params = {
            "project_id": project_id,
            "days": days
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def get_cost_attribution(self, project_id, days=30, dimensions="user,team,pattern,day,table"):
        """Get detailed cost attribution data.
        
        Args:
            project_id: GCP project ID
            days: Number of days to analyze
            dimensions: Comma-separated list of attribution dimensions
            
        Returns:
            Dictionary with cost attribution data by dimension
        """
        url = f"{self.base_url}/api/v1/cost-dashboard/attribution"
        params = {
            "project_id": project_id,
            "days": days,
            "dimensions": dimensions
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def get_cost_trends(self, project_id, days=90, granularity="day"):
        """Get cost trends over time.
        
        Args:
            project_id: GCP project ID
            days: Number of days to analyze
            granularity: Time granularity (day, week, month)
            
        Returns:
            Dictionary with cost trend data
        """
        url = f"{self.base_url}/api/v1/cost-dashboard/trends"
        params = {
            "project_id": project_id,
            "days": days,
            "granularity": granularity
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def get_anomalies(self, project_id, days=30, anomaly_types="daily,user,team,pattern", use_ml=False):
        """Get cost anomalies for the specified period.
        
        Args:
            project_id: GCP project ID
            days: Number of days to analyze
            anomaly_types: Comma-separated list of anomaly types
            use_ml: Whether to use ML-enhanced anomaly detection
            
        Returns:
            Dictionary with anomaly detection results
        """
        url = f"{self.base_url}/api/v1/cost-dashboard/anomalies"
        params = {
            "project_id": project_id,
            "days": days,
            "anomaly_types": anomaly_types,
            "use_ml": str(use_ml).lower()
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def get_forecast(self, project_id, training_days=90, forecast_days=7):
        """Get cost forecast for the specified period.
        
        Args:
            project_id: GCP project ID
            training_days: Number of days to use for training
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        url = f"{self.base_url}/api/v1/cost-dashboard/forecast"
        params = {
            "project_id": project_id,
            "training_days": training_days,
            "forecast_days": forecast_days
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def update_team_mapping(self, project_id, mapping):
        """Update the team mapping for user attribution.
        
        Args:
            project_id: GCP project ID
            mapping: Dictionary mapping user emails to team names
            
        Returns:
            Dictionary with update status
        """
        url = f"{self.base_url}/api/v1/cost-dashboard/team-mapping"
        payload = {
            "project_id": project_id,
            "mapping": mapping
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def get_alerts(self, project_id, days=7, min_cost_increase_usd=100.0):
        """Get cost alerts for the specified period.
        
        Args:
            project_id: GCP project ID
            days: Number of days to look back
            min_cost_increase_usd: Minimum cost increase to trigger alert
            
        Returns:
            Dictionary with cost alerts
        """
        url = f"{self.base_url}/api/v1/cost-dashboard/alerts"
        params = {
            "project_id": project_id,
            "days": days,
            "min_cost_increase_usd": min_cost_increase_usd
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        
        return response.json()

def plot_cost_trends(trends_data):
    """Plot cost trends.
    
    Args:
        trends_data: Cost trends data from API
    """
    # Convert to DataFrame
    trends = pd.DataFrame(trends_data["trends"])
    trends["period"] = pd.to_datetime(trends["period"])
    
    # Plot cost trend
    plt.figure(figsize=(12, 6))
    plt.plot(trends["period"], trends["total_cost_usd"], marker="o", label="Daily Cost")
    
    # Plot moving average if available
    if "cost_ma" in trends.columns:
        plt.plot(trends["period"], trends["cost_ma"], linestyle="--", label="Moving Avg")
    
    plt.title("BigQuery Cost Trend")
    plt.xlabel("Date")
    plt.ylabel("Cost (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig("cost_trend.png")
    logger.info("Cost trend plot saved to cost_trend.png")

def plot_cost_forecast(forecast_data):
    """Plot cost forecast.
    
    Args:
        forecast_data: Cost forecast data from API
    """
    # Extract historical and forecast data
    historical = pd.DataFrame(forecast_data["forecast"]["historical_data"])
    forecast = pd.DataFrame(forecast_data["forecast"]["forecast"])
    
    # Convert dates
    historical["date"] = pd.to_datetime(historical["date"])
    forecast["date"] = pd.to_datetime(forecast["date"])
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(historical["date"], historical["total_cost_usd"], 
             marker="o", label="Historical Cost", color="blue")
    
    # Plot forecast
    plt.plot(forecast["date"], forecast["forecasted_cost_usd"], 
             marker="o", linestyle="--", label="Forecast", color="red")
    
    # Plot prediction intervals if available
    if "lower_bound" in forecast.columns and "upper_bound" in forecast.columns:
        plt.fill_between(
            forecast["date"],
            forecast["lower_bound"],
            forecast["upper_bound"],
            alpha=0.2,
            color="red",
            label="Prediction Interval"
        )
    
    plt.title("BigQuery Cost Forecast")
    plt.xlabel("Date")
    plt.ylabel("Cost (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig("cost_forecast.png")
    logger.info("Cost forecast plot saved to cost_forecast.png")

def display_anomalies(anomalies_data):
    """Display detected anomalies.
    
    Args:
        anomalies_data: Anomaly detection data from API
    """
    anomalies = anomalies_data["anomalies"]
    
    # Display daily anomalies if available
    if "daily_anomalies" in anomalies and anomalies["daily_anomalies"]:
        print("\nDaily Cost Anomalies:")
        print("-" * 80)
        daily_anomalies = anomalies["daily_anomalies"]
        
        if "is_anomaly" in daily_anomalies:
            # Format for ML anomalies
            dates = daily_anomalies.get("dates", [])
            is_anomaly = daily_anomalies.get("is_anomaly", [])
            scores = daily_anomalies.get("anomaly_score", [])
            
            for i, (date, anomaly, score) in enumerate(zip(dates, is_anomaly, scores)):
                if anomaly:
                    print(f"Date: {date}, Anomaly Score: {score:.2f}")
        else:
            # Format for statistical anomalies
            for anomaly in anomalies["daily_anomalies"]:
                print(f"Date: {anomaly.get('date')}")
                print(f"  Actual Cost: ${anomaly.get('total_cost_usd', 0):.2f}")
                print(f"  Expected Cost: ${anomaly.get('expected_cost_usd', 0):.2f}")
                print(f"  Z-Score: {anomaly.get('z_score', 0):.2f}")
                print(f"  Percent Change: {anomaly.get('percent_change', 0):.1f}%")
                print()
    
    # Display user anomalies if available
    if "user_anomalies" in anomalies and anomalies["user_anomalies"]:
        print("\nUser Cost Anomalies:")
        print("-" * 80)
        for anomaly in anomalies["user_anomalies"]:
            print(f"User: {anomaly.get('user_email')}")
            print(f"  Team: {anomaly.get('team', 'Unknown')}")
            print(f"  Current Cost: ${anomaly.get('estimated_cost_usd_current', 0):.2f}")
            print(f"  Previous Cost: ${anomaly.get('estimated_cost_usd_previous', 0):.2f}")
            print(f"  Percent Change: {anomaly.get('percent_change', 0):.1f}%")
            print()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="BigQuery Cost Dashboard Client")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API base URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--use-ml", action="store_true", help="Use ML-enhanced anomaly detection")
    parser.add_argument("--output", default="dashboard_results.json", help="Output file for results")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Initialize client
    client = CostDashboardClient(base_url=args.api_url, api_key=args.api_key)
    
    try:
        # Get cost summary
        logger.info("Getting cost summary...")
        summary = client.get_cost_summary(args.project_id, args.days)
        
        # Get cost attribution data
        logger.info("Getting cost attribution...")
        attribution = client.get_cost_attribution(args.project_id, args.days)
        
        # Get cost trends
        logger.info("Getting cost trends...")
        trends = client.get_cost_trends(args.project_id, args.days * 3)  # 3x days for better trend visibility
        
        # Get cost anomalies
        logger.info("Detecting cost anomalies...")
        anomalies = client.get_anomalies(args.project_id, args.days, use_ml=args.use_ml)
        
        # Get cost forecast
        logger.info("Generating cost forecast...")
        forecast = client.get_forecast(args.project_id, args.days * 3, 14)  # 2 weeks forecast
        
        # Get cost alerts
        logger.info("Getting cost alerts...")
        alerts = client.get_alerts(args.project_id, args.days // 4)  # Recent alerts
        
        # Combine results
        results = {
            "summary": summary,
            "attribution": attribution,
            "trends": trends,
            "anomalies": anomalies,
            "forecast": forecast,
            "alerts": alerts,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save to file
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("Generating visualizations...")
            
            # Plot cost trends
            plot_cost_trends(trends)
            
            # Plot cost forecast
            plot_cost_forecast(forecast)
            
            # Display anomalies
            display_anomalies(anomalies)
        
        # Print summary
        total_cost = summary["summary"]["total_cost_usd"] if "summary" in summary else 0
        logger.info(f"Total BigQuery cost (last {args.days} days): ${total_cost:.2f}")
        
        anomaly_count = 0
        if "anomalies" in anomalies and "anomaly_counts" in anomalies["anomalies"]:
            anomaly_count = sum(anomalies["anomalies"]["anomaly_counts"].values())
        
        logger.info(f"Detected {anomaly_count} cost anomalies")
        
        alert_count = alerts.get("alert_count", 0)
        logger.info(f"Generated {alert_count} cost alerts")
        
    except requests.RequestException as e:
        logger.error(f"API request error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()