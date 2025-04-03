"""Test configuration for BigQuery Cost Intelligence Engine.

This module provides configuration for tests, handling environment-specific settings
and providing consistent test data across different environments.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Base test configuration with defaults
TEST_CONFIG = {
    # Project and dataset settings
    "project_id": "test-project",
    "dataset_id": "test_dataset",
    
    # Test settings
    "use_emulator": True,      # Use BigQuery emulator if True, else use mocks
    "test_data_path": str(Path(__file__).parent / "test_data"),
    "timeout_seconds": 30,
    "max_test_queries": 100,
    
    # Mocking settings
    "mock_storage": True,      # Mock Google Cloud Storage
    "mock_bigquery": True,     # Mock BigQuery (if use_emulator is False)
    
    # Performance test settings
    "perf_test_timeout": 300,  # 5 minutes
    "perf_test_datasets": [5000, 20000, 50000, 100000]
}

# Override with environment variables if present
def _load_from_env():
    """Load configuration from environment variables."""
    if os.environ.get("BQ_TEST_PROJECT_ID"):
        TEST_CONFIG["project_id"] = os.environ.get("BQ_TEST_PROJECT_ID")

    if os.environ.get("BQ_TEST_DATASET_ID"):
        TEST_CONFIG["dataset_id"] = os.environ.get("BQ_TEST_DATASET_ID")

    if os.environ.get("BQ_USE_EMULATOR") in ["0", "false", "False"]:
        TEST_CONFIG["use_emulator"] = False
        
    if os.environ.get("BQ_TEST_TIMEOUT"):
        TEST_CONFIG["timeout_seconds"] = int(os.environ.get("BQ_TEST_TIMEOUT"))
        
    if os.environ.get("BQ_PERF_TEST_TIMEOUT"):
        TEST_CONFIG["perf_test_timeout"] = int(os.environ.get("BQ_PERF_TEST_TIMEOUT"))
        
    # Use real BigQuery if credentials are available
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        TEST_CONFIG["mock_bigquery"] = False
        
    # Custom test data path
    if os.environ.get("BQ_TEST_DATA_PATH"):
        TEST_CONFIG["test_data_path"] = os.environ.get("BQ_TEST_DATA_PATH")

# Load environment variables on module import
_load_from_env()

def get_test_config() -> Dict[str, Any]:
    """Get test configuration with environment-specific settings.
    
    Returns:
        Dictionary containing test configuration
    """
    return TEST_CONFIG.copy()

def get_test_data_path(filename: Optional[str] = None) -> str:
    """Get path to test data file or directory.
    
    Args:
        filename: Optional filename within test data directory
        
    Returns:
        Absolute path to test data file or directory
    """
    data_path = Path(TEST_CONFIG["test_data_path"])
    if filename:
        return str(data_path / filename)
    return str(data_path)