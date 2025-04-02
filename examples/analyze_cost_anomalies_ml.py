#!/usr/bin/env python3
"""Example script for analyzing BigQuery cost anomalies using machine learning."""

import argparse
import json
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigquerycostopt.src.ml.cost_anomaly_detection import (
    detect_anomalies_with_ml,
    TimeSeriesForecaster,
    MLCostAnomalyDetector,
    CostAttributionClusterer
)
from bigquerycostopt.src.analysis.cost_attribution import CostAttributionAnalyzer


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze BigQuery cost anomalies using machine learning"
    )
    parser.add_argument(
        "--project", required=True, help="GCP project ID"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of days of job history to analyze (default: 90)"
    )
    parser.add_argument(
        "--credentials", help="Path to service account credentials JSON file"
    )
    parser.add_argument(
        "--output", help="Output file path for results (default: stdout)"
    )
    parser.add_argument(
        "--model-path", help="Path to save/load trained ML models"
    )
    parser.add_argument(
        "--forecast-days", type=int, default=7,
        help="Number of days to forecast (default: 7)"
    )
    parser.add_argument(
        "--advanced", action="store_true",
        help="Run advanced analysis with all ML components"
    )
    args = parser.parse_args()

    print(f"Initializing ML cost analysis for project {args.project}...")
    
    if args.advanced:
        # Run all components separately for advanced analysis
        print(f"Running advanced ML analysis over {args.days} days...")
        
        # Initialize the cost attribution analyzer
        analyzer = CostAttributionAnalyzer(
            project_id=args.project,
            credentials_path=args.credentials
        )
        
        # Get cost attribution data
        print("Retrieving cost attribution data...")
        costs = analyzer.attribute_costs(args.days)
        
        # Initialize and train ML anomaly detector
        print("Training ML anomaly detector...")
        detector = MLCostAnomalyDetector()
        detector.train(costs['cost_by_day'])
        
        # Detect anomalies
        print("Detecting cost anomalies...")
        anomalies = detector.predict(costs['cost_by_day'])
        
        # Generate forecast
        print(f"Generating {args.forecast_days}-day forecast...")
        forecaster = TimeSeriesForecaster(analyzer)
        try:
            forecast = forecaster.forecast_daily_costs(
                training_days=args.days,
                forecast_days=args.forecast_days
            )
        except Exception as e:
            print(f"Could not generate forecast: {e}")
            forecast = {"error": str(e)}
        
        # Cluster users
        print("Clustering users by cost patterns...")
        user_costs = costs['cost_by_user']
        clusterer = CostAttributionClusterer()
        clusterer.train(user_costs)
        user_clusters = clusterer.predict(user_costs)
        
        # Combine results
        results = {
            'daily_anomalies': anomalies,
            'forecast': forecast,
            'user_clusters': user_clusters,
            'analysis_period_days': args.days,
            'generated_at': datetime.now().isoformat()
        }
        
        # Print summary
        print("\nML Analysis Summary:")
        print(f"- Analysis Period: {args.days} days")
        if 'is_anomaly' in anomalies:
            anomaly_count = sum(anomalies['is_anomaly'])
            print(f"- Detected Anomalies: {anomaly_count}")
        
        if 'forecast' in forecast:
            print(f"- Forecast Period: {args.forecast_days} days")
            total_forecast = sum(item['forecasted_cost_usd'] 
                               for item in forecast['forecast'])
            print(f"- Total Forecasted Cost: ${total_forecast:.2f}")
        
        if 'cluster_id' in user_clusters:
            cluster_counts = {}
            for cluster_id in user_clusters['cluster_id']:
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            
            print(f"- User Clusters: {len(set(user_clusters['cluster_id']))}")
            for cluster_id, count in cluster_counts.items():
                cluster_name = user_clusters['cluster_name'][user_clusters['cluster_id'].index(cluster_id)]
                print(f"  - {cluster_name}: {count} users")
        
    else:
        # Run simplified analysis with single function
        print(f"Running ML cost analysis over {args.days} days...")
        results = detect_anomalies_with_ml(
            project_id=args.project,
            days_back=args.days,
            model_path=args.model_path
        )
        
        # Print summary
        print("\nML Analysis Summary:")
        print(f"- Analysis Period: {args.days} days")
        
        # Anomalies
        anomalies = results.get('daily_anomalies', {})
        if 'is_anomaly' in anomalies:
            anomaly_count = sum(anomalies['is_anomaly'])
            print(f"- Detected Anomalies: {anomaly_count}")
            
            # Show specific anomalies
            if anomaly_count > 0 and 'dates' in anomalies:
                print("\nTop Anomalies:")
                for i, (is_anomaly, date, score) in enumerate(zip(
                    anomalies['is_anomaly'], 
                    anomalies['dates'], 
                    anomalies['anomaly_score']
                )):
                    if is_anomaly and i < 3:  # Show top 3
                        print(f"  - {date}: Severity {score:.2f}")
        
        # Forecast
        forecast = results.get('forecast', {})
        if 'forecast' in forecast:
            print(f"\nForecast Summary:")
            for i, day in enumerate(forecast['forecast'][:5]):  # Show 5 days
                print(f"  - {day['date']}: ${day['forecasted_cost_usd']:.2f} " +
                      f"(Range: ${day['lower_bound']:.2f} - ${day['upper_bound']:.2f})")
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()