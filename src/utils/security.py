"""Security utility functions for the BigQuery Cost Intelligence Engine."""

from flask import Request
import os
import hashlib
import hmac
import time
from typing import Optional

from .logging import setup_logger

logger = setup_logger(__name__)

# Default API key for development (this would be stored securely in production)
DEFAULT_API_KEY = "your-api-key-here"


def validate_request(request: Request) -> bool:
    """Validate an incoming API request.
    
    Args:
        request: Flask request object
        
    Returns:
        Boolean indicating if request is valid
    """
    # Get API key from environment variable or use default
    api_key = os.environ.get("API_KEY", DEFAULT_API_KEY)
    
    # Check for API key in header
    request_key = request.headers.get("X-API-Key")
    if not request_key:
        logger.warning("Missing API key header")
        return False
        
    # Check if key matches
    if not hmac.compare_digest(api_key, request_key):
        logger.warning("Invalid API key")
        return False
        
    # Check timestamp to prevent replay attacks
    timestamp = request.headers.get("X-Timestamp")
    if timestamp:
        try:
            request_time = int(timestamp)
            current_time = int(time.time())
            
            # Allow requests within 5 minutes of current time
            if abs(current_time - request_time) > 300:
                logger.warning("Request timestamp outside allowed window")
                return False
        except ValueError:
            logger.warning("Invalid timestamp format")
            return False
    
    return True


def generate_signature(data: str, secret: str) -> str:
    """Generate HMAC signature for request data.
    
    Args:
        data: The data to sign
        secret: The secret key to use for signing
        
    Returns:
        String containing the HMAC signature
    """
    signature = hmac.new(
        key=secret.encode(),
        msg=data.encode(),
        digestmod=hashlib.sha256
    ).hexdigest()
    
    return signature


def validate_signature(data: str, signature: str, secret: str) -> bool:
    """Validate that a signature matches the expected value.
    
    Args:
        data: The data that was signed
        signature: The signature to validate
        secret: The secret key used for signing
        
    Returns:
        Boolean indicating if signature is valid
    """
    expected_signature = generate_signature(data, secret)
    return hmac.compare_digest(expected_signature, signature)


def secure_logging(message: str, sensitive_data: bool = False) -> None:
    """Log messages with sensitive data handling.
    
    Args:
        message: The message to log
        sensitive_data: Whether the message contains sensitive data
    """
    if sensitive_data:
        # Hash any sensitive data before logging
        hashed_message = hashlib.sha256(message.encode()).hexdigest()[:8]
        logger.info(f"Sensitive data event [hash: {hashed_message}]")
    else:
        logger.info(message)
