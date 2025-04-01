#!/usr/bin/env python3
"""
Example script for the BigQuery Schema Optimizer module.

This script demonstrates how to use the SchemaOptimizer class to:
1. Connect to a BigQuery project
2. Analyze a dataset for schema optimization opportunities
3. Generate a report of recommendations
4. Save the recommendations to a file

Usage:
    python analyze_schema.py --project-id=your-project --dataset-id=your-dataset [--output-format=md]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path to allow running as standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bigquerycostopt.src.analysis.schema_optimizer import SchemaOptimizer
from bigquerycostopt.src.analysis.metadata import MetadataExtractor


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze BigQuery dataset schemas for optimization opportunities"
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
        "--min-table-size-gb", 
        type=float,
        default=1.0,
        help="Minimum table size in GB to analyze (default: 1 GB)"
    )
    parser.add_argument(
        "--output-format", 
        choices=["json", "md", "text", "html"],
        default="md",
        help="Output format for recommendations (default: md)"
    )
    parser.add_argument(
        "--output-file", 
        default=None,
        help="Output file path (default: schema_recommendations_<dataset>_<timestamp>.<ext>)"
    )
    return parser


def save_recommendations(recommendations: Dict[str, Any], 
                        output_format: str = "md", 
                        output_file: Optional[str] = None) -> str:
    """Save recommendations to a file.
    
    Args:
        recommendations: Schema optimization recommendations
        output_format: Output format (json, md, text, html)
        output_file: Output file path (if None, a default is used)
        
    Returns:
        Path to the saved file
    """
    dataset_id = recommendations["dataset_id"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set file extension based on format
    extensions = {
        "json": "json",
        "md": "md",
        "text": "txt",
        "html": "html"
    }
    ext = extensions.get(output_format, "txt")
    
    # Use provided filename or generate default
    if not output_file:
        output_file = f"schema_recommendations_{dataset_id}_{timestamp}.{ext}"
    
    # Create content based on format
    if output_format == "json":
        content = json.dumps(recommendations, indent=2, default=str)
    else:
        # For md, text, or html, use the SchemaOptimizer's report generator
        schema_optimizer = SchemaOptimizer(project_id=recommendations["project_id"])
        content = schema_optimizer.generate_recommendations_report(
            recommendations, format=output_format)
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(content)
    
    return output_file


def main():
    """Run schema optimization analysis."""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Extract parameters
    project_id = args.project_id
    dataset_id = args.dataset_id
    credentials_path = args.credentials_path
    min_table_size_gb = args.min_table_size_gb
    output_format = args.output_format
    output_file = args.output_file
    
    print(f"Analyzing schema optimizations for dataset {project_id}.{dataset_id}")
    print(f"Minimum table size: {min_table_size_gb} GB")
    
    try:
        # Initialize the MetadataExtractor
        print("Initializing metadata extractor...")
        metadata_extractor = MetadataExtractor(
            project_id=project_id,
            credentials_path=credentials_path
        )
        
        # Initialize the SchemaOptimizer
        print("Initializing schema optimizer...")
        schema_optimizer = SchemaOptimizer(
            metadata_extractor=metadata_extractor
        )
        
        # Analyze the dataset
        print(f"Analyzing schema optimizations for dataset {dataset_id}...")
        recommendations = schema_optimizer.analyze_dataset_schemas(
            dataset_id=dataset_id,
            min_table_size_gb=min_table_size_gb
        )
        
        # Print summary
        summary = recommendations["summary"]
        total_recs = summary["total_recommendations"]
        storage_savings_gb = summary["estimated_storage_savings_gb"]
        monthly_savings = summary["estimated_monthly_cost_savings"]
        savings_pct = summary["estimated_storage_savings_percentage"]
        
        print("\nAnalysis Complete!")
        print(f"Found {total_recs} schema optimization recommendations")
        print(f"Estimated storage savings: {storage_savings_gb:.2f} GB ({savings_pct:.1f}%)")
        print(f"Estimated monthly cost savings: ${monthly_savings:.2f}")
        
        # Save recommendations to file
        if total_recs > 0:
            output_path = save_recommendations(
                recommendations,
                output_format=output_format,
                output_file=output_file
            )
            print(f"\nRecommendations saved to: {output_path}")
        else:
            print("\nNo schema optimization recommendations found.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()