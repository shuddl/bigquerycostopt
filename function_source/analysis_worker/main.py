"""
Analysis Worker Cloud Function for BigQuery Cost Intelligence Engine.

This function processes analysis requests published to a Pub/Sub topic.
"""

import base64
import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from google.cloud import bigquery
from google.cloud import storage

# Add the src directory to the Python path
import sys
sys.path.append('/workspace')

from bigquerycostopt.src.analysis.metadata import MetadataExtractor
from bigquerycostopt.src.analysis.query_optimizer import QueryOptimizer
from bigquerycostopt.src.analysis.schema_optimizer import SchemaOptimizer
from bigquerycostopt.src.analysis.storage_optimizer import StorageOptimizer
from bigquerycostopt.src.connectors.bigquery import BigQueryConnector
from bigquerycostopt.src.recommender.engine import RecommendationEngine
from bigquerycostopt.src.ml.enhancer import MLEnhancementModule

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
RESULTS_BUCKET = os.environ.get('RESULTS_BUCKET')
BIGQUERY_DATASET = os.environ.get('BIGQUERY_DATASET')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')

# Initialize clients
bq_client = bigquery.Client()
storage_client = storage.Client()

def process_analysis_request(event: Dict[str, Any], context: Any) -> None:
    """Cloud Function entry point for processing analysis requests.
    
    Args:
        event: Pub/Sub event payload
        context: Metadata for the event
    """
    try:
        # Parse Pub/Sub message
        if 'data' in event:
            message_data = base64.b64decode(event['data']).decode('utf-8')
            message = json.loads(message_data)
        else:
            raise ValueError("No data field in Pub/Sub message")
        
        logger.info(f"Received analysis request: {message}")
        
        # Extract message fields
        analysis_id = message.get('analysis_id')
        project_id = message.get('project_id')
        dataset_id = message.get('dataset_id')
        callback_url = message.get('callback_url')
        user_id = message.get('user_id', 'anonymous')
        
        if not all([analysis_id, project_id, dataset_id]):
            raise ValueError("Missing required fields in message")
        
        # Record start time
        start_time = time.time()
        
        # Update analysis status to "running"
        update_analysis_status(
            analysis_id,
            project_id,
            dataset_id,
            "running",
            {},
            user_id,
            callback_url
        )
        
        # Initialize connector
        connector = BigQueryConnector(project_id=project_id)
        
        # Initialize optimizer modules
        storage_optimizer = StorageOptimizer(connector)
        schema_optimizer = SchemaOptimizer(connector)
        query_optimizer = QueryOptimizer(connector)
        
        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine(
            project_id=project_id,
            optimizers=[storage_optimizer, schema_optimizer, query_optimizer]
        )
        
        # Initialize ML enhancement module (if available)
        try:
            ml_module = MLEnhancementModule(
                project_id=project_id,
                use_pretrained=True
            )
            use_ml_enhancement = True
        except Exception as e:
            logger.warning(f"Failed to initialize ML Enhancement Module: {e}")
            use_ml_enhancement = False
        
        # Run analysis
        logger.info(f"Starting analysis of {project_id}.{dataset_id}")
        
        try:
            recommendations, dataset_metadata = recommendation_engine.analyze_dataset(
                dataset_id=dataset_id
            )
            
            # Enhance recommendations with ML (if available)
            if use_ml_enhancement and recommendations:
                recommendations = ml_module.enhance_recommendations(
                    recommendations=recommendations,
                    dataset_metadata=dataset_metadata
                )
            
            # Prepare analysis summary
            summary = create_analysis_summary(dataset_metadata, recommendations)
            
            # Calculate duration
            duration_seconds = time.time() - start_time
            
            # Store results
            store_analysis_results(
                analysis_id,
                project_id,
                dataset_id,
                "completed",
                summary,
                recommendations,
                dataset_metadata,
                duration_seconds,
                user_id,
                callback_url,
                []
            )
            
            logger.info(f"Analysis completed successfully: {analysis_id}")
            
            # If callback URL is provided, send notification
            if callback_url:
                send_callback_notification(
                    callback_url,
                    {
                        "analysis_id": analysis_id,
                        "status": "completed",
                        "summary": summary
                    }
                )
        
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}", exc_info=True)
            
            # Store error results
            error_details = [
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "affected_resource": f"{project_id}.{dataset_id}"
                }
            ]
            
            store_analysis_results(
                analysis_id,
                project_id,
                dataset_id,
                "failed",
                {},
                [],
                {},
                time.time() - start_time,
                user_id,
                callback_url,
                error_details
            )
            
            # If callback URL is provided, send notification
            if callback_url:
                send_callback_notification(
                    callback_url,
                    {
                        "analysis_id": analysis_id,
                        "status": "failed",
                        "error": str(e)
                    }
                )
    
    except Exception as e:
        logger.error(f"Failed to process message: {e}", exc_info=True)
        raise

def update_analysis_status(
    analysis_id: str,
    project_id: str,
    dataset_id: str,
    status: str,
    summary: Dict[str, Any],
    user_id: str,
    callback_url: Optional[str]
) -> None:
    """Update the analysis status in BigQuery.
    
    Args:
        analysis_id: Unique identifier for the analysis
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        status: Analysis status
        summary: Analysis summary
        user_id: User ID
        callback_url: Callback URL
    """
    table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.analysis_results"
    
    rows_to_insert = [
        {
            "analysis_id": analysis_id,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "analysis_date": datetime.now().isoformat(),
            "status": status,
            "summary": summary,
            "tables_analyzed": [],
            "queries_analyzed": [],
            "analysis_errors": [],
            "user_id": user_id,
            "callback_url": callback_url
        }
    ]
    
    errors = bq_client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        logger.error(f"Error inserting rows: {errors}")
    else:
        logger.info(f"Updated analysis status to {status} for {analysis_id}")

def create_analysis_summary(
    dataset_metadata: Dict[str, Any],
    recommendations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create a summary of the analysis results.
    
    Args:
        dataset_metadata: Metadata about the dataset
        recommendations: List of recommendations
        
    Returns:
        Summary dictionary
    """
    total_tables = len(dataset_metadata.get("tables", []))
    
    total_bytes = sum(
        table.get("size_bytes", 0)
        for table in dataset_metadata.get("tables", [])
    )
    
    total_queries_analyzed = sum(
        len(table.get("query_patterns", []))
        for table in dataset_metadata.get("tables", [])
    )
    
    # Calculate estimated monthly cost (simplified)
    bytes_per_month = total_bytes
    query_bytes_per_month = sum(
        pattern.get("avg_bytes_processed", 0) * pattern.get("execution_count", 0)
        for table in dataset_metadata.get("tables", [])
        for pattern in table.get("query_patterns", [])
    )
    
    # Cost estimates: $5 per TB of storage, $5 per TB processed
    estimated_monthly_cost = (
        (bytes_per_month / 1e12) * 5 +  # Storage cost
        (query_bytes_per_month / 1e12) * 5  # Query cost
    )
    
    # Calculate potential savings from recommendations
    potential_savings = sum(
        rec.get("estimated_savings", {}).get("monthly", 0)
        for rec in recommendations
    )
    
    return {
        "total_tables": total_tables,
        "total_bytes": total_bytes,
        "total_queries_analyzed": total_queries_analyzed,
        "estimated_monthly_cost": estimated_monthly_cost,
        "potential_savings": potential_savings
    }

def store_analysis_results(
    analysis_id: str,
    project_id: str,
    dataset_id: str,
    status: str,
    summary: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    dataset_metadata: Dict[str, Any],
    duration_seconds: float,
    user_id: str,
    callback_url: Optional[str],
    errors: List[Dict[str, Any]]
) -> None:
    """Store analysis results in BigQuery and GCS.
    
    Args:
        analysis_id: Unique identifier for the analysis
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        status: Analysis status
        summary: Analysis summary
        recommendations: List of recommendations
        dataset_metadata: Dataset metadata
        duration_seconds: Analysis duration in seconds
        user_id: User ID
        callback_url: Callback URL
        errors: List of errors encountered
    """
    # Store in BigQuery
    table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.analysis_results"
    
    tables_analyzed = [
        {
            "table_id": table.get("table_id"),
            "bytes": table.get("size_bytes"),
            "row_count": table.get("num_rows"),
            "creation_time": table.get("creation_time"),
            "last_modified_time": table.get("last_modified_time"),
            "query_count_last_30d": len(table.get("query_patterns", [])),
            "is_partitioned": len(table.get("partitioning", [])) > 0,
            "is_clustered": len(table.get("clustering", [])) > 0
        }
        for table in dataset_metadata.get("tables", [])
    ]
    
    queries_analyzed = [
        {
            "query_hash": pattern.get("query_hash"),
            "execution_count": pattern.get("execution_count"),
            "avg_bytes_processed": pattern.get("avg_bytes_processed"),
            "avg_execution_time_ms": pattern.get("avg_execution_time_ms"),
            "total_slot_ms": pattern.get("total_slot_ms"),
            "tables_referenced": pattern.get("tables_referenced", [])
        }
        for table in dataset_metadata.get("tables", [])
        for pattern in table.get("query_patterns", [])
    ]
    
    rows_to_insert = [
        {
            "analysis_id": analysis_id,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "analysis_date": datetime.now().isoformat(),
            "status": status,
            "summary": summary,
            "tables_analyzed": tables_analyzed,
            "queries_analyzed": queries_analyzed,
            "analysis_errors": errors,
            "analysis_duration_seconds": duration_seconds,
            "user_id": user_id,
            "callback_url": callback_url
        }
    ]
    
    errors = bq_client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        logger.error(f"Error inserting analysis results: {errors}")
    else:
        logger.info(f"Stored analysis results in BigQuery for {analysis_id}")
    
    # Store recommendations in BigQuery if available
    if recommendations:
        recommendations_table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.recommendations"
        
        recommendation_rows = []
        for rec in recommendations:
            # Add analysis ID and timestamp to recommendation
            rec_copy = rec.copy()
            rec_copy["analysis_id"] = analysis_id
            rec_copy["creation_date"] = datetime.now().isoformat()
            rec_copy["status"] = "active"
            
            recommendation_rows.append(rec_copy)
        
        errors = bq_client.insert_rows_json(recommendations_table_id, recommendation_rows)
        if errors:
            logger.error(f"Error inserting recommendations: {errors}")
        else:
            logger.info(f"Stored {len(recommendations)} recommendations in BigQuery")
    
    # Store full analysis results in GCS
    if RESULTS_BUCKET:
        bucket = storage_client.bucket(RESULTS_BUCKET)
        
        # Store full analysis results
        results_blob = bucket.blob(f"analysis_results/{analysis_id}.json")
        results_blob.upload_from_string(
            json.dumps({
                "analysis_id": analysis_id,
                "project_id": project_id,
                "dataset_id": dataset_id,
                "analysis_date": datetime.now().isoformat(),
                "status": status,
                "summary": summary,
                "recommendations": recommendations,
                "dataset_metadata": dataset_metadata,
                "analysis_duration_seconds": duration_seconds,
                "user_id": user_id,
                "errors": errors
            }),
            content_type="application/json"
        )
        logger.info(f"Stored full analysis results in GCS: gs://{RESULTS_BUCKET}/analysis_results/{analysis_id}.json")

def send_callback_notification(url: str, data: Dict[str, Any]) -> None:
    """Send a notification to the callback URL.
    
    Args:
        url: Callback URL
        data: Data to send
    """
    import requests
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"Sent callback notification to {url}")
    except Exception as e:
        logger.error(f"Failed to send callback notification: {e}")