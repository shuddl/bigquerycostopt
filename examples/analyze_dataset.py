#!/usr/bin/env python3
"""Example script for analyzing a BigQuery dataset for cost optimization."""

import argparse
import json
import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigquerycostopt.src.analysis.metadata import MetadataExtractor
from bigquerycostopt.src.analysis.storage_optimizer import StorageOptimizer


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Analyze a BigQuery dataset for cost optimization opportunities.")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset ID")
    parser.add_argument("--credentials", help="Path to service account credentials JSON file")
    parser.add_argument("--output", help="Output file path for recommendations (default: stdout)")
    args = parser.parse_args()

    # Initialize the metadata extractor
    print(f"Initializing metadata extractor for project {args.project}...")
    metadata_extractor = MetadataExtractor(
        project_id=args.project,
        credentials_path=args.credentials
    )

    # Extract dataset metadata
    print(f"Extracting metadata for dataset {args.dataset}...")
    dataset_metadata = metadata_extractor.extract_dataset_metadata(args.dataset)
    
    # Print basic dataset stats
    table_count = dataset_metadata.get("table_count", 0)
    total_size_gb = dataset_metadata.get("total_size_gb", 0)
    
    print(f"Dataset {args.project}.{args.dataset} contains {table_count} tables, total size: {total_size_gb:.2f} GB")
    
    # Initialize the storage optimizer
    print("Analyzing storage optimization opportunities...")
    storage_optimizer = StorageOptimizer(metadata_extractor=metadata_extractor)
    
    # Generate storage recommendations
    recommendations = storage_optimizer.analyze_dataset(args.dataset)
    
    # Output results
    summary = recommendations["optimization_summary"]
    print("\nOptimization Summary:")
    print(f"- Total recommendations: {summary['total_recommendations']}")
    print(f"- Estimated monthly savings: ${summary['estimated_monthly_savings']:.2f}")
    print(f"- Estimated annual savings: ${summary['estimated_annual_savings']:.2f}")
    print(f"- Estimated storage reduction: {summary['estimated_size_reduction_gb']:.2f} GB ({summary['estimated_size_reduction_percentage']}%)")
    
    # Write detailed results to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"\nDetailed recommendations saved to {args.output}")
    else:
        print("\nTop 5 Recommendations:")
        for i, rec in enumerate(recommendations["recommendations"][:5], 1):
            print(f"{i}. Table: {rec['table_id']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Monthly savings: ${rec.get('estimated_monthly_savings', 0):.2f}")
            print(f"   Priority: {rec.get('priority', 'unknown')}")
            print()


if __name__ == "__main__":
    main()