#!/usr/bin/env python3
"""Example script for analyzing BigQuery queries for optimization opportunities."""

import argparse
import json
import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigquerycostopt.src.analysis.metadata import MetadataExtractor
from bigquerycostopt.src.analysis.query_optimizer import QueryOptimizer


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Analyze BigQuery queries for optimization opportunities.")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset ID")
    parser.add_argument("--table", help="Optional BigQuery table ID (if not specified, analyzes entire dataset)")
    parser.add_argument("--query", help="Optional query text to analyze")
    parser.add_argument("--days", type=int, default=30, help="Number of days of query history to analyze (default: 30)")
    parser.add_argument("--credentials", help="Path to service account credentials JSON file")
    parser.add_argument("--output", help="Output file path for recommendations (default: stdout)")
    parser.add_argument("--format", choices=["md", "html", "text", "json"], default="md", 
                        help="Output format: markdown, HTML, text, or JSON (default: md)")
    args = parser.parse_args()

    # Initialize the metadata extractor
    print(f"Initializing metadata extractor for project {args.project}...")
    metadata_extractor = MetadataExtractor(
        project_id=args.project,
        credentials_path=args.credentials
    )
    
    # Initialize the query optimizer
    print("Initializing query optimizer...")
    query_optimizer = QueryOptimizer(metadata_extractor=metadata_extractor)
    
    # Run analysis based on provided options
    if args.query:
        # Analyze a specific query
        print(f"Analyzing provided query text...")
        results = query_optimizer.analyze_query_text(args.query, args.dataset)
    elif args.table:
        # Analyze queries for a specific table
        print(f"Analyzing queries for table {args.project}.{args.dataset}.{args.table} over {args.days} days...")
        results = query_optimizer.analyze_table_queries(args.dataset, args.table, args.days)
    else:
        # Analyze all queries for the dataset
        print(f"Analyzing queries for dataset {args.project}.{args.dataset} over {args.days} days...")
        results = query_optimizer.analyze_dataset_queries(args.dataset, args.days)
    
    # Output results
    if args.format == "json":
        # Output as JSON
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed recommendations saved to {args.output}")
        else:
            print(json.dumps(results, indent=2, default=str))
    else:
        # Output as formatted report
        report = query_optimizer.generate_recommendations_report(results, args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nDetailed recommendations saved to {args.output}")
        else:
            print(report)
    
    # Print summary to console
    summary = results["summary"]
    print("\nAnalysis Summary:")
    print(f"- Total Queries Analyzed: {results['queries_analyzed']}")
    print(f"- Total Recommendations: {summary['total_recommendations']}")
    print(f"- Estimated Monthly Cost Savings: ${summary['estimated_monthly_cost_savings']:.2f}")
    print(f"- Estimated Annual Cost Savings: ${summary['estimated_annual_cost_savings']:.2f}")


if __name__ == "__main__":
    main()