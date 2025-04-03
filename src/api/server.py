"""Server module for the BigQuery Cost Intelligence Engine API.

This module provides a unified entry point for running the API server
with either Flask or FastAPI backend, based on environment configuration.
"""

import os
import argparse
import importlib
from pathlib import Path
import sys

from ..utils.logging import setup_logger

logger = setup_logger(__name__)

def run_flask_server(host="0.0.0.0", port=8080, debug=False):
    """Run the Flask server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        debug: Whether to run in debug mode
    """
    from .endpoints import app
    logger.info(f"Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

def run_fastapi_server(host="0.0.0.0", port=8080, reload=False):
    """Run the FastAPI server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        reload: Whether to enable auto-reload on code changes
    """
    import uvicorn
    logger.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(
        "bigquerycostopt.src.api.fastapi_server:app",
        host=host,
        port=port,
        reload=reload
    )

def main():
    """Main entry point for running the API server."""
    parser = argparse.ArgumentParser(description="BigQuery Cost Intelligence Engine API Server")
    parser.add_argument(
        "--server-type",
        choices=["flask", "fastapi"],
        default=os.environ.get("API_SERVER_TYPE", "flask"),
        help="API server implementation to use (flask or fastapi)"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("API_HOST", "0.0.0.0"),
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("API_PORT", 8080)),
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("API_DEBUG", "").lower() in ["true", "1", "yes"],
        help="Enable debug mode"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check configuration without starting the server"
    )
    
    args = parser.parse_args()
    
    # Set up environment variables
    os.environ["API_SERVER_TYPE"] = args.server_type
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    os.environ["API_DEBUG"] = "true" if args.debug else "false"
    
    # If check-only, just validate configuration
    if args.check_only:
        logger.info(f"Server configuration check passed: {args.server_type} server on {args.host}:{args.port}")
        if args.server_type == "fastapi":
            required_packages = ["fastapi", "uvicorn", "pydantic", "starlette"]
            for package in required_packages:
                try:
                    importlib.import_module(package)
                    logger.info(f"✓ Required package found: {package}")
                except ImportError:
                    logger.error(f"✗ Required package missing: {package}")
        sys.exit(0)
            
    try:
        if args.server_type == "flask":
            run_flask_server(args.host, args.port, args.debug)
        else:
            run_fastapi_server(args.host, args.port, args.debug)
    except ImportError as e:
        if "fastapi" in str(e).lower() or "uvicorn" in str(e).lower():
            logger.error(f"Error starting FastAPI server: {e}")
            logger.error("Please install FastAPI dependencies: pip install fastapi uvicorn")
        else:
            logger.error(f"Error starting {args.server_type} server: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()