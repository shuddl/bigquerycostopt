"""API endpoints for the BigQuery Cost Intelligence Engine."""

from flask import Flask, request, jsonify
from google.cloud import pubsub_v1
import json
import os
import uuid
import datetime

from ..utils.logging import setup_logger
from ..utils.security import validate_request
from .cost_dashboard import cost_dashboard_bp

app = Flask(__name__)
logger = setup_logger(__name__)

# Register blueprints
app.register_blueprint(cost_dashboard_bp)

# Initialize Pub/Sub publisher
publisher = pubsub_v1.PublisherClient()
project_id = os.environ.get("GCP_PROJECT_ID")
topic_name = os.environ.get("ANALYSIS_REQUEST_TOPIC")
topic_path = publisher.topic_path(project_id, topic_name)


@app.route("/api/v1/analyze", methods=["POST"])
def trigger_analysis():
    """Trigger a BigQuery dataset analysis.
    
    Expects JSON payload with:
        - project_id: GCP project ID containing the dataset
        - dataset_id: BigQuery dataset ID to analyze
        - callback_url: Optional webhook URL for completion notification
    """
    # Validate request
    if not validate_request(request):
        return jsonify({"error": "Unauthorized"}), 401
        
    # Parse request data
    try:
        data = request.get_json()
        required_fields = ["project_id", "dataset_id"]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
                
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Prepare message for Pub/Sub
        message = {
            "analysis_id": analysis_id,
            "project_id": data["project_id"],
            "dataset_id": data["dataset_id"],
            "callback_url": data.get("callback_url", None),
            "user_id": request.headers.get("X-User-ID", "anonymous"),
            "timestamp": str(datetime.datetime.utcnow())
        }
        
        # Publish message
        future = publisher.publish(
            topic_path, 
            data=json.dumps(message).encode("utf-8")
        )
        
        # Get published message ID
        message_id = future.result()
        logger.info(f"Published message {message_id} for analysis {analysis_id}")
        
        return jsonify({
            "analysis_id": analysis_id,
            "status": "pending",
            "message": "Analysis request submitted successfully"
        }), 202
        
    except Exception as e:
        logger.exception("Error processing analysis request")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/analysis/<analysis_id>", methods=["GET"])
def get_analysis_status(analysis_id):
    """Get the status of a previously submitted analysis."""
    # This would query the status from a database
    # For now, return a placeholder response
    return jsonify({
        "analysis_id": analysis_id,
        "status": "in_progress", 
        "progress": 45,
        "message": "Analyzing query patterns"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
