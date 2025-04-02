#!/usr/bin/env python3
"""Example script for analyzing BigQuery cost attribution by user, team, and query patterns."""

import argparse
import json
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigquerycostopt.src.analysis.cost_attribution import (
    CostAttributionAnalyzer, 
    CostAnomalyDetector,
    get_cost_attribution_data,
    detect_cost_anomalies
)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze BigQuery cost attribution and detect cost anomalies"
    )
    parser.add_argument(
        "--project", required=True, help="GCP project ID"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days of job history to analyze (default: 30)"
    )
    parser.add_argument(
        "--credentials", help="Path to service account credentials JSON file"
    )
    parser.add_argument(
        "--output", help="Output file path for results (default: stdout)"
    )
    parser.add_argument(
        "--team-mapping", help="Path to JSON file with user to team mapping"
    )
    parser.add_argument(
        "--format", choices=["md", "html", "text", "json"], default="json",
        help="Output format: markdown, HTML, text, or JSON (default: json)"
    )
    parser.add_argument(
        "--mode", choices=["attribution", "anomalies", "both"], default="both",
        help="Analysis mode: cost attribution, anomaly detection, or both (default: both)"
    )
    parser.add_argument(
        "--min-cost", type=float, default=10.0,
        help="Minimum cost threshold for anomaly detection in USD (default: 10.0)"
    )
    args = parser.parse_args()

    # Initialize the cost attribution analyzer
    print(f"Initializing cost attribution analyzer for project {args.project}...")
    analyzer = CostAttributionAnalyzer(
        project_id=args.project,
        credentials_path=args.credentials
    )
    
    # Load team mapping if provided
    if args.team_mapping:
        print(f"Loading team mapping from {args.team_mapping}...")
        try:
            with open(args.team_mapping, 'r') as f:
                team_mapping = json.load(f)
            analyzer.set_team_mapping(team_mapping)
        except Exception as e:
            print(f"Error loading team mapping: {e}")
            print("Continuing without team mapping...")
    
    results = {}
    
    # Run analysis based on provided mode
    if args.mode in ["attribution", "both"]:
        print(f"Analyzing cost attribution over {args.days} days...")
        attribution_data = get_cost_attribution_data(args.project, args.days)
        results["attribution"] = attribution_data
        
        # Print summary to console
        summary = attribution_data["summary"]
        print("\nCost Attribution Summary:")
        print(f"- Total Cost (USD): ${summary['total_cost_usd']:.2f}")
        print(f"- Total Queries: {summary['total_queries']}")
        print(f"- Unique Users: {summary['unique_users']}")
        print(f"- Cost Per Query (USD): ${summary['cost_per_query_usd']:.4f}")
        
        if summary.get('top_teams'):
            print("\nTop Teams by Cost:")
            for team in summary['top_teams']:
                print(f"- {team['team']}: ${team['total_cost_usd']:.2f}")
    
    if args.mode in ["anomalies", "both"]:
        print(f"Detecting cost anomalies over {args.days} days...")
        
        # Initialize the anomaly detector
        anomaly_detector = CostAnomalyDetector(analyzer)
        
        # Detect daily cost anomalies
        daily_anomalies = anomaly_detector.detect_daily_cost_anomalies(
            days_back=args.days, 
            min_cost_usd=args.min_cost
        )
        
        # Detect user cost anomalies
        user_anomalies = anomaly_detector.detect_user_cost_anomalies(
            days_back=args.days // 2,  # Compare recent half with previous half
            comparison_days=args.days // 2
        )
        
        # Generate full anomaly report
        anomaly_report = anomaly_detector.generate_anomaly_report()
        results["anomalies"] = anomaly_report
        
        # Print anomaly summary to console
        anomaly_counts = anomaly_report.get("anomaly_counts", {})
        print("\nAnomaly Detection Summary:")
        print(f"- Daily Cost Anomalies: {anomaly_counts.get('daily', 0)}")
        print(f"- User Cost Anomalies: {anomaly_counts.get('user', 0)}")
        print(f"- Team Cost Anomalies: {anomaly_counts.get('team', 0)}")
        
        if daily_anomalies is not None and not daily_anomalies.empty:
            print("\nRecent Daily Cost Anomalies:")
            for _, anomaly in daily_anomalies.head(3).iterrows():
                print(f"- {anomaly['date'].strftime('%Y-%m-%d')}: " +
                      f"${anomaly['total_cost_usd']:.2f} " +
                      f"(expected: ${anomaly['expected_cost_usd']:.2f}, " +
                      f"{anomaly['percent_change']:.1f}% change)")
            
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == "json":
                json.dump(results, f, indent=2, default=str)
            else:
                # For future implementation of other formats
                f.write(json.dumps(results, indent=2, default=str))
        print(f"\nDetailed results saved to {args.output}")
    elif args.format == "json" and results:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()