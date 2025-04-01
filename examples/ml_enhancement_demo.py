#!/usr/bin/env python
"""
Demo script for the Machine Learning Enhancement Module.

This script demonstrates how to use the ML Enhancement Module to enhance
recommendations from the BigQuery Cost Intelligence Engine with ML-derived insights.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender.engine import RecommendationEngine
from src.analysis.storage_optimizer import StorageOptimizer
from src.analysis.schema_optimizer import SchemaOptimizer
from src.analysis.query_optimizer import QueryOptimizer
from src.ml.enhancer import MLEnhancementModule
from src.connectors.bigquery import BigQueryConnector


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo for ML-enhanced BigQuery cost recommendations."
    )
    
    parser.add_argument(
        "--project_id",
        required=True,
        help="GCP project ID to analyze"
    )
    
    parser.add_argument(
        "--dataset_id",
        required=True,
        help="BigQuery dataset ID to analyze"
    )
    
    parser.add_argument(
        "--credentials_path",
        help="Path to Google Cloud service account credentials JSON file"
    )
    
    parser.add_argument(
        "--model_dir",
        help="Directory for ML models (default: ./models)"
    )
    
    parser.add_argument(
        "--output_file",
        help="Path to save recommendations JSON (default: ./recommendations.json)"
    )
    
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Use pre-trained models if available"
    )
    
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()


def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """Run the ML enhancement demo."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Set default paths
    model_dir = args.model_dir or "./models"
    output_file = args.output_file or "./recommendations.json"
    
    # Initialize BigQuery connector
    connector = BigQueryConnector(
        project_id=args.project_id,
        credentials_path=args.credentials_path
    )
    
    # Initialize optimizer modules
    storage_optimizer = StorageOptimizer(connector)
    schema_optimizer = SchemaOptimizer(connector)
    query_optimizer = QueryOptimizer(connector)
    
    # Initialize recommendation engine
    recommendation_engine = RecommendationEngine(
        project_id=args.project_id,
        optimizers=[storage_optimizer, schema_optimizer, query_optimizer]
    )
    
    # Initialize ML enhancement module
    ml_module = MLEnhancementModule(
        project_id=args.project_id,
        model_dir=model_dir,
        credentials_path=args.credentials_path,
        use_pretrained=args.use_pretrained
    )
    
    print(f"Analyzing dataset {args.project_id}.{args.dataset_id}...")
    
    # Generate base recommendations
    try:
        recommendations, dataset_metadata = recommendation_engine.analyze_dataset(
            dataset_id=args.dataset_id
        )
        
        print(f"Generated {len(recommendations)} base recommendations")
        
        # Enhance recommendations with ML insights
        enhanced_recommendations = ml_module.enhance_recommendations(
            recommendations=recommendations,
            dataset_metadata=dataset_metadata
        )
        
        print(f"Enhanced {len(enhanced_recommendations)} recommendations with ML insights")
        
        # Generate ML insights report
        ml_report = ml_module.generate_ml_insights_report(enhanced_recommendations)
        
        # Save enhanced recommendations to file
        output = {
            "project_id": args.project_id,
            "dataset_id": args.dataset_id,
            "recommendations": enhanced_recommendations,
            "ml_report": ml_report
        }
        
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
            
        print(f"Saved enhanced recommendations to {output_file}")
        
        # Print summary
        print("\nRecommendation Summary:")
        print(f"  Total Recommendations: {len(enhanced_recommendations)}")
        print(f"  ML-Enhanced: {ml_report['ml_enhanced_count']} " +
             f"({ml_report['ml_enhanced_count']/len(enhanced_recommendations)*100:.1f}%)")
        print(f"  Detected Patterns: {len(ml_report['patterns'])}")
        print(f"  Anomalies: {ml_report['anomaly_count']}")
        
        # Print top recommendations
        print("\nTop 3 Recommendations by ML-enhanced Priority:")
        sorted_recs = sorted(enhanced_recommendations, 
                           key=lambda r: r.get("priority_score", 0), 
                           reverse=True)
        
        for i, rec in enumerate(sorted_recs[:3]):
            print(f"  {i+1}. {rec['recommendation_type'].upper()} - " +
                 f"{rec.get('target_table', '')} - " +
                 f"Priority: {rec.get('priority_score', 0):.1f}")
            
            if "ml_insights" in rec and "business_context" in rec["ml_insights"]:
                print(f"     Context: {rec['ml_insights']['business_context']}")
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()