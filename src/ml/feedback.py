"""Feedback Collection System for ML Enhancement Module.

This module provides functionality for collecting and storing feedback on
implemented recommendations to improve future ML predictions.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
import pandas as pd
import numpy as np
import logging
import datetime
import json
import hashlib
from pathlib import Path
from google.cloud import bigquery
from google.cloud import storage

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class FeedbackCollector:
    """Feedback Collection System for ML-enhanced recommendations.
    
    This class provides functionality for collecting, storing, and managing
    feedback on implemented recommendations to improve future ML predictions.
    """
    
    def __init__(self,
                project_id: str,
                credentials_path: Optional[str] = None,
                feedback_store_path: Optional[str] = None):
        """Initialize the Feedback Collector.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to GCP service account credentials (optional)
            feedback_store_path: Path to store feedback data locally (optional)
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        
        # Set feedback store path
        if feedback_store_path:
            self.feedback_store_path = Path(feedback_store_path)
        else:
            # Default to a 'feedback' directory in the package
            self.feedback_store_path = Path(__file__).parent.parent.parent / "feedback"
            
        # Create feedback directory if it doesn't exist
        self.feedback_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize BigQuery client if credentials are provided
        try:
            if credentials_path:
                self.bq_client = bigquery.Client.from_service_account_json(credentials_path)
                self.storage_client = storage.Client.from_service_account_json(credentials_path)
            else:
                self.bq_client = bigquery.Client(project=project_id)
                self.storage_client = storage.Client(project=project_id)
                
            self.use_cloud_storage = True
            logger.info(f"Initialized GCP clients for project {project_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize GCP clients: {e}. Will use local storage only.")
            self.bq_client = None
            self.storage_client = None
            self.use_cloud_storage = False
    
    def store_feedback(self,
                     implemented_recommendations: List[Dict[str, Any]],
                     feedback_data: Dict[str, Any]) -> None:
        """Store feedback on implemented recommendations.
        
        Args:
            implemented_recommendations: List of recommendations that were implemented
            feedback_data: Feedback data about the implementations
        """
        logger.info(f"Storing feedback for {len(implemented_recommendations)} recommendations")
        
        # Generate a unique ID for this feedback submission
        feedback_id = hashlib.md5(json.dumps(feedback_data, sort_keys=True, default=str).encode()).hexdigest()
        
        # Add metadata to feedback data
        feedback_data["feedback_id"] = feedback_id
        feedback_data["timestamp"] = datetime.datetime.now().isoformat()
        feedback_data["project_id"] = self.project_id
        
        # Store feedback data locally
        self._store_local_feedback(feedback_id, implemented_recommendations, feedback_data)
        
        # Store feedback data in BigQuery if configured
        if self.use_cloud_storage and self.bq_client is not None:
            self._store_cloud_feedback(feedback_id, implemented_recommendations, feedback_data)
        
        logger.info(f"Feedback stored successfully with ID {feedback_id}")
    
    def _store_local_feedback(self,
                            feedback_id: str,
                            implemented_recommendations: List[Dict[str, Any]],
                            feedback_data: Dict[str, Any]) -> None:
        """Store feedback data locally.
        
        Args:
            feedback_id: Unique ID for this feedback submission
            implemented_recommendations: List of recommendations that were implemented
            feedback_data: Feedback data about the implementations
        """
        # Prepare data for storage
        storage_data = {
            "feedback_id": feedback_id,
            "feedback_data": feedback_data,
            "recommendations": implemented_recommendations,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Store as JSON file
        feedback_file = self.feedback_store_path / f"feedback_{feedback_id}.json"
        with open(feedback_file, "w") as f:
            json.dump(storage_data, f, default=str, indent=2)
            
        logger.info(f"Feedback data stored locally at {feedback_file}")
        
        # Update feedback index file
        self._update_feedback_index(feedback_id, feedback_data)
    
    def _update_feedback_index(self, feedback_id: str, feedback_data: Dict[str, Any]) -> None:
        """Update the feedback index file with new feedback entry.
        
        Args:
            feedback_id: Unique ID for this feedback submission
            feedback_data: Feedback data about the implementations
        """
        index_file = self.feedback_store_path / "feedback_index.json"
        
        # Load existing index if it exists
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    index = json.load(f)
            except Exception:
                index = {"feedback_entries": []}
        else:
            index = {"feedback_entries": []}
        
        # Create index entry
        entry = {
            "feedback_id": feedback_id,
            "timestamp": feedback_data.get("timestamp", datetime.datetime.now().isoformat()),
            "project_id": feedback_data.get("project_id", self.project_id),
            "dataset_id": feedback_data.get("dataset_id", ""),
            "recommendation_count": len(feedback_data.get("recommendations", {})),
            "feedback_source": feedback_data.get("feedback_source", "user"),
            "feedback_version": feedback_data.get("feedback_version", "1.0")
        }
        
        # Add to index
        index["feedback_entries"].append(entry)
        
        # Write updated index
        with open(index_file, "w") as f:
            json.dump(index, f, default=str, indent=2)
    
    def _store_cloud_feedback(self,
                            feedback_id: str,
                            implemented_recommendations: List[Dict[str, Any]],
                            feedback_data: Dict[str, Any]) -> None:
        """Store feedback data in BigQuery and GCS.
        
        Args:
            feedback_id: Unique ID for this feedback submission
            implemented_recommendations: List of recommendations that were implemented
            feedback_data: Feedback data about the implementations
        """
        try:
            # Store raw feedback data in GCS
            bucket_name = f"{self.project_id}-bqoptimizer-feedback"
            blob_name = f"feedback/{feedback_id}.json"
            
            # Check if bucket exists, create if not
            try:
                bucket = self.storage_client.get_bucket(bucket_name)
            except Exception:
                bucket = self.storage_client.create_bucket(bucket_name)
                logger.info(f"Created feedback storage bucket: {bucket_name}")
            
            # Upload feedback data to GCS
            blob = bucket.blob(blob_name)
            storage_data = {
                "feedback_id": feedback_id,
                "feedback_data": feedback_data,
                "recommendations": implemented_recommendations,
                "timestamp": datetime.datetime.now().isoformat()
            }
            blob.upload_from_string(json.dumps(storage_data, default=str))
            
            logger.info(f"Uploaded feedback data to GCS: gs://{bucket_name}/{blob_name}")
            
            # Store structured feedback data in BigQuery
            dataset_id = f"{self.project_id}.bqoptimizer"
            table_id = f"{dataset_id}.recommendation_feedback"
            
            # Check if dataset exists, create if not
            try:
                dataset = self.bq_client.get_dataset(dataset_id)
            except Exception:
                dataset = bigquery.Dataset(dataset_id)
                dataset = self.bq_client.create_dataset(dataset)
                logger.info(f"Created BigQuery dataset: {dataset_id}")
            
            # Check if table exists, create if not
            try:
                table = self.bq_client.get_table(table_id)
            except Exception:
                schema = [
                    bigquery.SchemaField("feedback_id", "STRING"),
                    bigquery.SchemaField("recommendation_id", "STRING"),
                    bigquery.SchemaField("recommendation_type", "STRING"),
                    bigquery.SchemaField("target_table", "STRING"),
                    bigquery.SchemaField("success", "BOOLEAN"),
                    bigquery.SchemaField("actual_cost_savings", "FLOAT"),
                    bigquery.SchemaField("expected_cost_savings", "FLOAT"),
                    bigquery.SchemaField("implementation_time_minutes", "INTEGER"),
                    bigquery.SchemaField("user_rating", "INTEGER"),
                    bigquery.SchemaField("complexity_rating", "INTEGER"),
                    bigquery.SchemaField("business_impact", "FLOAT"),
                    bigquery.SchemaField("business_impact_category", "STRING"),
                    bigquery.SchemaField("comments", "STRING"),
                    bigquery.SchemaField("feedback_timestamp", "TIMESTAMP"),
                    bigquery.SchemaField("feedback_source", "STRING")
                ]
                table = bigquery.Table(table_id, schema=schema)
                table = self.bq_client.create_table(table)
                logger.info(f"Created BigQuery table: {table_id}")
            
            # Prepare feedback rows
            rows = []
            for rec_id, rec_feedback in feedback_data.get("recommendations", {}).items():
                # Find the corresponding recommendation
                recommendation = None
                for rec in implemented_recommendations:
                    if rec.get("recommendation_id") == rec_id:
                        recommendation = rec
                        break
                
                if recommendation:
                    # Create row
                    row = {
                        "feedback_id": feedback_id,
                        "recommendation_id": rec_id,
                        "recommendation_type": recommendation.get("recommendation_type", "unknown"),
                        "target_table": recommendation.get("target_table", ""),
                        "success": rec_feedback.get("success", True),
                        "actual_cost_savings": rec_feedback.get("actual_cost_savings", 0.0),
                        "expected_cost_savings": recommendation.get("estimated_savings", {}).get("total", 0.0),
                        "implementation_time_minutes": rec_feedback.get("implementation_time_minutes", 0),
                        "user_rating": rec_feedback.get("user_rating", 0),
                        "complexity_rating": rec_feedback.get("complexity_rating", 3),
                        "business_impact": rec_feedback.get("business_impact", 0.0),
                        "business_impact_category": rec_feedback.get("business_impact_category", "unknown"),
                        "comments": rec_feedback.get("comments", ""),
                        "feedback_timestamp": datetime.datetime.now(),
                        "feedback_source": feedback_data.get("feedback_source", "user")
                    }
                    rows.append(row)
            
            # Insert feedback rows into BigQuery
            if rows:
                errors = self.bq_client.insert_rows_json(table_id, rows)
                if errors:
                    logger.warning(f"Errors inserting feedback rows: {errors}")
                else:
                    logger.info(f"Inserted {len(rows)} feedback rows into BigQuery")
                    
        except Exception as e:
            logger.warning(f"Failed to store feedback data in cloud: {e}")
    
    def get_feedback(self, 
                   feedback_id: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   recommendation_type: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve stored feedback data.
        
        Args:
            feedback_id: Specific feedback submission ID to retrieve
            start_date: Start date for filtering feedback (ISO format)
            end_date: End date for filtering feedback (ISO format)
            recommendation_type: Filter by recommendation type
            
        Returns:
            Dictionary containing feedback data
        """
        # Check for specific feedback ID
        if feedback_id:
            return self._get_feedback_by_id(feedback_id)
        
        # Load feedback index
        index_file = self.feedback_store_path / "feedback_index.json"
        if not index_file.exists():
            logger.warning("Feedback index not found")
            return {"feedback_entries": []}
            
        with open(index_file, "r") as f:
            index = json.load(f)
        
        # Filter by date if specified
        if start_date or end_date:
            filtered_entries = []
            
            start_dt = datetime.datetime.fromisoformat(start_date) if start_date else None
            end_dt = datetime.datetime.fromisoformat(end_date) if end_date else None
            
            for entry in index.get("feedback_entries", []):
                entry_dt = datetime.datetime.fromisoformat(entry["timestamp"])
                
                if start_dt and entry_dt < start_dt:
                    continue
                    
                if end_dt and entry_dt > end_dt:
                    continue
                    
                filtered_entries.append(entry)
                
            index["feedback_entries"] = filtered_entries
        
        # Load full feedback data for each entry
        feedback_data = []
        
        for entry in index.get("feedback_entries", []):
            entry_id = entry["feedback_id"]
            entry_data = self._get_feedback_by_id(entry_id)
            
            # Filter by recommendation type if specified
            if recommendation_type and entry_data:
                # Check if any recommendation matches the type
                recommendations = entry_data.get("recommendations", [])
                matching_recs = [r for r in recommendations if r.get("recommendation_type") == recommendation_type]
                
                if matching_recs:
                    feedback_data.append(entry_data)
            else:
                feedback_data.append(entry_data)
        
        return {"feedback_data": feedback_data}
    
    def _get_feedback_by_id(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific feedback submission by ID.
        
        Args:
            feedback_id: Feedback submission ID
            
        Returns:
            Dictionary containing feedback data or None if not found
        """
        feedback_file = self.feedback_store_path / f"feedback_{feedback_id}.json"
        
        if not feedback_file.exists():
            logger.warning(f"Feedback file not found for ID {feedback_id}")
            return None
            
        try:
            with open(feedback_file, "r") as f:
                feedback_data = json.load(f)
            
            return feedback_data
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            return None
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics on collected feedback.
        
        Returns:
            Dictionary containing feedback statistics
        """
        # Load feedback index
        index_file = self.feedback_store_path / "feedback_index.json"
        if not index_file.exists():
            logger.warning("Feedback index not found")
            return {"feedback_count": 0}
            
        with open(index_file, "r") as f:
            index = json.load(f)
        
        # Calculate basic statistics
        entries = index.get("feedback_entries", [])
        feedback_count = len(entries)
        
        if feedback_count == 0:
            return {"feedback_count": 0}
        
        # Get timestamps for first and last feedback
        timestamps = [datetime.datetime.fromisoformat(entry["timestamp"]) for entry in entries]
        first_feedback = min(timestamps).isoformat()
        last_feedback = max(timestamps).isoformat()
        
        # Count recommendations by type
        recommendation_counts = {}
        recommendation_success = {}
        total_recommendations = 0
        
        # Load all feedback to get detailed statistics
        for entry in entries:
            entry_id = entry["feedback_id"]
            entry_data = self._get_feedback_by_id(entry_id)
            
            if not entry_data:
                continue
                
            recommendations = entry_data.get("recommendations", [])
            feedback = entry_data.get("feedback_data", {}).get("recommendations", {})
            
            for rec in recommendations:
                rec_type = rec.get("recommendation_type", "unknown")
                rec_id = rec.get("recommendation_id", "")
                
                # Count by type
                if rec_type not in recommendation_counts:
                    recommendation_counts[rec_type] = 0
                    recommendation_success[rec_type] = {"success": 0, "failure": 0}
                    
                recommendation_counts[rec_type] += 1
                total_recommendations += 1
                
                # Track success/failure
                if rec_id in feedback:
                    if feedback[rec_id].get("success", True):
                        recommendation_success[rec_type]["success"] += 1
                    else:
                        recommendation_success[rec_type]["failure"] += 1
        
        # Calculate success rates
        success_rates = {}
        for rec_type, counts in recommendation_success.items():
            total = counts["success"] + counts["failure"]
            if total > 0:
                success_rates[rec_type] = counts["success"] / total
            else:
                success_rates[rec_type] = 0
        
        # Compile statistics
        statistics = {
            "feedback_count": feedback_count,
            "first_feedback": first_feedback,
            "last_feedback": last_feedback,
            "total_recommendations": total_recommendations,
            "recommendation_counts": recommendation_counts,
            "recommendation_success": recommendation_success,
            "success_rates": success_rates
        }
        
        return statistics