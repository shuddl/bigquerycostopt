#!/usr/bin/env python3
"""
Example script for the BigQuery Recommendation Engine.

This script demonstrates how to:
1. Connect to a BigQuery project
2. Generate comprehensive optimization recommendations across storage, query, and schema dimensions
3. Create a detailed implementation plan
4. Format recommendations for different output types
5. Calculate ROI and prioritize recommendations

Usage:
    python generate_recommendations.py --project-id=your-project --dataset-id=your-dataset [--output-format=json]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path to allow running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bigquerycostopt.src.recommender.engine import RecommendationEngine
from bigquerycostopt.src.recommender.roi import ROICalculator


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive optimization recommendations for BigQuery datasets"
    )
    parser.add_argument(
        "--project-id", 
        required=True,
        help="GCP project ID containing the BigQuery dataset"
    )
    parser.add_argument(
        "--dataset-id", 
        required=True,
        help="BigQuery dataset ID to analyze"
    )
    parser.add_argument(
        "--credentials-path", 
        default=None,
        help="Path to GCP service account credentials JSON file"
    )
    parser.add_argument(
        "--days", 
        type=int,
        default=30,
        help="Number of days of query history to analyze (default: 30)"
    )
    parser.add_argument(
        "--min-table-size-gb", 
        type=float,
        default=1.0,
        help="Minimum table size in GB to analyze (default: 1 GB)"
    )
    parser.add_argument(
        "--include-storage", 
        action="store_true",
        default=True,
        help="Include storage optimization analysis"
    )
    parser.add_argument(
        "--include-query", 
        action="store_true",
        default=True,
        help="Include query optimization analysis"
    )
    parser.add_argument(
        "--include-schema", 
        action="store_true",
        default=True,
        help="Include schema optimization analysis"
    )
    parser.add_argument(
        "--output-format", 
        choices=["json", "bigquery", "dashboard", "implementation"],
        default="json",
        help="Output format for recommendations (default: json)"
    )
    parser.add_argument(
        "--output-file", 
        default=None,
        help="Output file path (default: recommendations_<dataset>_<timestamp>.<ext>)"
    )
    return parser


def save_output(data: Dict[str, Any], format_type: str, 
               output_file: Optional[str] = None, dataset_id: str = "unknown") -> str:
    """Save output to a file.
    
    Args:
        data: The data to save
        format_type: Output format (json, bigquery, dashboard, implementation)
        output_file: Output file path (if None, a default is used)
        dataset_id: Dataset ID for default filename
        
    Returns:
        Path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set file extension based on format
    ext = "json"
    
    # Use provided filename or generate default
    if not output_file:
        output_file = f"recommendations_{dataset_id}_{timestamp}.{ext}"
    
    # Convert to JSON
    content = json.dumps(data, indent=2, default=str)
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(content)
    
    return output_file


def main():
    """Run the recommendation engine."""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Extract parameters
    project_id = args.project_id
    dataset_id = args.dataset_id
    credentials_path = args.credentials_path
    days = args.days
    min_table_size_gb = args.min_table_size_gb
    include_storage = args.include_storage
    include_query = args.include_query
    include_schema = args.include_schema
    output_format = args.output_format
    output_file = args.output_file
    
    print(f"Generating optimization recommendations for dataset {project_id}.{dataset_id}")
    print(f"Analysis parameters:")
    print(f"  - Days of query history: {days}")
    print(f"  - Minimum table size: {min_table_size_gb} GB")
    print(f"  - Include storage optimizations: {include_storage}")
    print(f"  - Include query optimizations: {include_query}")
    print(f"  - Include schema optimizations: {include_schema}")
    
    try:
        # Initialize the recommendation engine
        print("Initializing recommendation engine...")
        engine = RecommendationEngine(
            project_id=project_id,
            credentials_path=credentials_path
        )
        
        # Generate recommendations
        print(f"Analyzing dataset {dataset_id}...")
        recommendations = engine.analyze_dataset(
            dataset_id=dataset_id,
            days=days,
            min_table_size_gb=min_table_size_gb,
            include_storage=include_storage,
            include_query=include_query,
            include_schema=include_schema
        )
        
        # Print summary
        summary = recommendations["summary"]
        total_recs = summary["total_recommendations"]
        total_savings = summary["savings_summary"]["total_annual_savings_usd"]
        high_priority = summary["priority_breakdown"]["high"]
        
        print("\nAnalysis Complete!")
        print(f"Found {total_recs} optimization recommendations")
        print(f"High priority recommendations: {high_priority}")
        print(f"Estimated annual savings: ${total_savings:.2f}")
        print(f"Implementation plan:")
        for phase in recommendations["implementation_plan"]["phases"]:
            phase_savings = phase["estimated_annual_savings_usd"]
            phase_steps = len(phase["steps"])
            print(f"  - {phase['name']}: {phase_steps} steps, ${phase_savings:.2f} annual savings")
        
        # Format output based on selected format
        if output_format == "json":
            output_data = recommendations
        elif output_format == "bigquery":
            output_data = engine.format_for_bigquery()
        elif output_format == "dashboard":
            output_data = engine.format_for_dashboard()
        elif output_format == "implementation":
            output_data = recommendations["implementation_plan"]
        
        # Save output
        if total_recs > 0:
            output_path = save_output(
                data=output_data,
                format_type=output_format,
                output_file=output_file,
                dataset_id=dataset_id
            )
            print(f"\nRecommendations saved to: {output_path}")
        else:
            print("\nNo optimization recommendations found.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()