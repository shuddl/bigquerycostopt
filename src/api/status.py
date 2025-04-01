"""Analysis status tracking for the BigQuery Cost Intelligence Engine."""

from typing import Dict, Any, Optional
import json
import time
import requests
from enum import Enum
import threading

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class AnalysisStatus(str, Enum):
    """Enum representing analysis status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StatusTracker:
    """Tracks the status of BigQuery dataset analyses."""
    
    def __init__(self):
        """Initialize the status tracker."""
        self.analyses = {}  # In-memory storage of analysis status
        self._lock = threading.Lock()  # Thread safety for status updates
    
    def create_analysis(self, analysis_id: str, project_id: str, dataset_id: str,
                       callback_url: Optional[str] = None) -> Dict[str, Any]:
        """Create a new analysis status record.
        
        Args:
            analysis_id: Unique ID for the analysis
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            callback_url: Optional webhook URL for notification
            
        Returns:
            Dict containing the analysis status record
        """
        status = {
            "analysis_id": analysis_id,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "status": AnalysisStatus.PENDING,
            "progress": 0,
            "message": "Analysis queued",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "callback_url": callback_url,
            "stages": []
        }
        
        with self._lock:
            self.analyses[analysis_id] = status
        
        logger.info(f"Created analysis status for {analysis_id}")
        return status
    
    def get_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of an analysis.
        
        Args:
            analysis_id: Unique ID for the analysis
            
        Returns:
            Dict containing the analysis status or None if not found
        """
        with self._lock:
            return self.analyses.get(analysis_id)
    
    def update_status(self, analysis_id: str, status: AnalysisStatus = None,
                     progress: int = None, message: str = None,
                     stage: str = None, result: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Update the status of an analysis.
        
        Args:
            analysis_id: Unique ID for the analysis
            status: New analysis status
            progress: Progress percentage (0-100)
            message: Status message
            stage: Current processing stage
            result: Analysis result data (for completed analyses)
            
        Returns:
            Updated status record or None if analysis not found
        """
        with self._lock:
            if analysis_id not in self.analyses:
                logger.warning(f"Attempted to update unknown analysis {analysis_id}")
                return None
            
            analysis = self.analyses[analysis_id]
            
            # Update status fields
            if status is not None:
                analysis["status"] = status
            
            if progress is not None:
                analysis["progress"] = progress
            
            if message is not None:
                analysis["message"] = message
            
            if stage is not None:
                analysis["stages"].append({
                    "name": stage,
                    "timestamp": int(time.time())
                })
            
            if result is not None:
                analysis["result"] = result
            
            analysis["updated_at"] = int(time.time())
            
            # If completed or failed, send webhook if configured
            if status in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED) and analysis.get("callback_url"):
                self._send_webhook(analysis)
            
            return analysis
    
    def _send_webhook(self, analysis: Dict[str, Any]) -> None:
        """Send webhook notification for analysis status update.
        
        Args:
            analysis: Analysis status record
        """
        callback_url = analysis.get("callback_url")
        if not callback_url:
            return
        
        # Copy relevant fields for webhook payload
        payload = {
            "analysis_id": analysis["analysis_id"],
            "project_id": analysis["project_id"],
            "dataset_id": analysis["dataset_id"],
            "status": analysis["status"],
            "message": analysis["message"],
            "completed_at": analysis["updated_at"]
        }
        
        # Include result summary if available
        if "result" in analysis:
            payload["result_summary"] = {
                "total_recommendations": analysis["result"].get("total_recommendations", 0),
                "total_annual_savings_usd": analysis["result"].get("roi_summary", {}).get("total_annual_savings_usd", 0),
                "overall_roi": analysis["result"].get("roi_summary", {}).get("overall_roi", 0)
            }
        
        # Send webhook in a separate thread to avoid blocking
        threading.Thread(target=self._send_webhook_request, args=(callback_url, payload)).start()
    
    def _send_webhook_request(self, url: str, payload: Dict[str, Any]) -> None:
        """Send webhook HTTP request.
        
        Args:
            url: Webhook URL
            payload: Webhook payload
        """
        try:
            response = requests.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "BigQueryCostIntelligenceEngine/1.0"
                },
                timeout=10
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Successfully sent webhook to {url}")
            else:
                logger.warning(f"Webhook to {url} failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending webhook to {url}: {e}")


# Global status tracker instance
status_tracker = StatusTracker()


def get_status_tracker() -> StatusTracker:
    """Get the global status tracker instance.
    
    Returns:
        Global StatusTracker instance
    """
    return status_tracker
