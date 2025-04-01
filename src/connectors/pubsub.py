"""Google Cloud Pub/Sub integration for BigQuery Cost Intelligence Engine."""

from google.cloud import pubsub_v1
from google.oauth2 import service_account
import json
import os
import time
from typing import Dict, Any, Optional, Callable

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class PubSubConnector:
    """Manages interactions with Google Cloud Pub/Sub."""
    
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        """Initialize Pub/Sub connector.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to service account credentials JSON file
        """
        self.project_id = project_id
        
        # Initialize publisher and subscriber clients
        try:
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.publisher = pubsub_v1.PublisherClient(credentials=credentials)
                self.subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
                logger.info(f"Initialized Pub/Sub clients with service account credentials")
            else:
                # Use default credentials
                self.publisher = pubsub_v1.PublisherClient()
                self.subscriber = pubsub_v1.SubscriberClient()
                logger.info(f"Initialized Pub/Sub clients with default credentials")
        except Exception as e:
            logger.error(f"Error initializing Pub/Sub clients: {e}")
            raise
    
    def publish_message(self, topic_name: str, message: Dict[str, Any], 
                       attributes: Optional[Dict[str, str]] = None) -> str:
        """Publish a message to a Pub/Sub topic.
        
        Args:
            topic_name: Name of the Pub/Sub topic
            message: Dictionary containing message data
            attributes: Optional message attributes
            
        Returns:
            Published message ID
        """
        try:
            # Create topic path
            topic_path = self.publisher.topic_path(self.project_id, topic_name)
            
            # Convert message to JSON string and encode as bytes
            message_data = json.dumps(message).encode("utf-8")
            
            # Publish message
            future = self.publisher.publish(topic_path, data=message_data, **attributes or {})
            message_id = future.result()
            
            logger.info(f"Published message {message_id} to {topic_path}")
            return message_id
            
        except Exception as e:
            logger.error(f"Error publishing message to {topic_name}: {e}")
            raise
    
    def create_subscription(self, topic_name: str, subscription_name: str) -> None:
        """Create a subscription if it doesn't exist.
        
        Args:
            topic_name: Name of the Pub/Sub topic
            subscription_name: Name of the subscription to create
        """
        try:
            topic_path = self.publisher.topic_path(self.project_id, topic_name)
            subscription_path = self.subscriber.subscription_path(self.project_id, subscription_name)
            
            # Check if subscription exists
            try:
                self.subscriber.get_subscription(subscription=subscription_path)
                logger.info(f"Subscription {subscription_name} already exists")
            except Exception:
                # Create subscription
                self.subscriber.create_subscription(name=subscription_path, topic=topic_path)
                logger.info(f"Created subscription {subscription_name} for topic {topic_name}")
                
        except Exception as e:
            logger.error(f"Error creating subscription {subscription_name}: {e}")
            raise
    
    def subscribe(self, subscription_name: str, callback: Callable, timeout: Optional[int] = None) -> None:
        """Subscribe to messages from a subscription with a callback.
        
        Args:
            subscription_name: Name of the subscription
            callback: Callback function to process messages
            timeout: Optional timeout in seconds
        """
        try:
            subscription_path = self.subscriber.subscription_path(self.project_id, subscription_name)
            
            def process_message(message):
                try:
                    # Parse message data
                    data = json.loads(message.data.decode("utf-8"))
                    
                    # Call user callback with message data and attributes
                    callback(data, message.attributes)
                    
                    # Acknowledge the message
                    message.ack()
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Don't acknowledge, so message will be redelivered
            
            # Start subscriber
            streaming_pull_future = self.subscriber.subscribe(
                subscription_path, process_message
            )
            logger.info(f"Listening for messages on {subscription_path}")
            
            # Wait for messages
            start_time = time.time()
            try:
                while True:
                    if timeout and time.time() - start_time > timeout:
                        streaming_pull_future.cancel()  # Trigger shutdown
                        break
                    time.sleep(1)  # Avoid tight loop
            except KeyboardInterrupt:
                logger.info("Subscription interrupted")
                
            # Wait for shutdown
            streaming_pull_future.result(timeout=30)
            
        except Exception as e:
            logger.error(f"Error in subscription: {e}")
            raise
    
    def close(self) -> None:
        """Close connections."""
        try:
            self.subscriber.close()
            logger.info("Closed Pub/Sub connections")
        except Exception as e:
            logger.error(f"Error closing Pub/Sub connections: {e}")
