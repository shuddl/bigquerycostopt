"""Authentication middleware for the BigQuery Cost Intelligence Engine API."""

from flask import request, Response, g
import functools
import os
import time
import hmac
import hashlib
from typing import Optional, Callable, Any

from ..utils.logging import setup_logger
from ..utils.security import validate_request, validate_signature

logger = setup_logger(__name__)


def auth_required(f: Callable) -> Callable:
    """Decorator for API endpoints that require authentication.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function that checks for valid authentication
    """
    @functools.wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        if not validate_request(request):
            return Response("Unauthorized", status=401)
        return f(*args, **kwargs)
    return decorated


def webhook_auth_required(f: Callable) -> Callable:
    """Decorator for webhook endpoints that require signature validation.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function that checks for valid webhook signature
    """
    @functools.wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        # Get secret from environment
        webhook_secret = os.environ.get("WEBHOOK_SECRET")
        if not webhook_secret:
            logger.warning("WEBHOOK_SECRET not configured")
            return Response("Server configuration error", status=500)
        
        # Get signature and timestamp from headers
        signature = request.headers.get("X-Signature")
        timestamp = request.headers.get("X-Timestamp")
        
        if not signature or not timestamp:
            logger.warning("Missing signature or timestamp headers")
            return Response("Unauthorized", status=401)
        
        # Check timestamp freshness (within 5 minutes)
        try:
            request_time = int(timestamp)
            current_time = int(time.time())
            
            if abs(current_time - request_time) > 300:
                logger.warning("Webhook timestamp outside allowed window")
                return Response("Request expired", status=401)
        except ValueError:
            logger.warning("Invalid timestamp format")
            return Response("Invalid timestamp", status=400)
        
        # Verify signature
        body = request.get_data(as_text=True)
        data_to_sign = f"{timestamp}:{body}"
        
        if not validate_signature(data_to_sign, signature, webhook_secret):
            logger.warning("Invalid webhook signature")
            return Response("Invalid signature", status=401)
        
        return f(*args, **kwargs)
    return decorated


def get_user_id() -> Optional[str]:
    """Get user ID from request.
    
    Returns:
        User ID from X-User-ID header or None if not present
    """
    return request.headers.get("X-User-ID")


def get_client_info() -> dict:
    """Get client information from request.
    
    Returns:
        Dict containing client information
    """
    return {
        "ip": request.remote_addr,
        "user_agent": request.headers.get("User-Agent", ""),
        "user_id": get_user_id(),
        "timestamp": int(time.time())
    }
