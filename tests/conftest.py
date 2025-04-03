# filepath: /Users/spencerpro/BigQueryCostOpt/bigquerycostopt/tests/conftest.py
import sys
import os
from unittest.mock import MagicMock
from pathlib import Path
import pytest

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test configuration and mocks
from tests.test_config import get_test_config, get_test_data_path
from tests.bigquery_mocks import MockBigQueryClient

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return get_test_config()

@pytest.fixture
def mock_bigquery_client():
    """Provide a mock BigQuery client."""
    config = get_test_config()
    
    if config["use_emulator"]:
        # Use BigQuery emulator for tests
        try:
            from google.cloud.bigquery import Client
            os.environ["BIGQUERY_EMULATOR_HOST"] = "localhost:9050"
            client = Client(project=config["project_id"])
            return client
        except Exception:
            # Fall back to mock if emulator not available
            return create_mock_bigquery_client()
    else:
        # Use mock client
        return create_mock_bigquery_client()

def create_mock_bigquery_client():
    """Create a mock BigQuery client with test data."""
    config = get_test_config()
    client = MockBigQueryClient(project_id=config["project_id"])
    
    # Create test dataset with tables
    client.add_mock_table(
        dataset_id=config["dataset_id"],
        table_id="users",
        schema=[
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "STRING"},
            {"name": "email", "type": "STRING"},
            {"name": "created_at", "type": "TIMESTAMP"}
        ],
        num_rows=1000,
        num_bytes=1000000
    )
    
    client.add_mock_table(
        dataset_id=config["dataset_id"],
        table_id="orders",
        schema=[
            {"name": "id", "type": "INTEGER"},
            {"name": "user_id", "type": "INTEGER"},
            {"name": "amount", "type": "FLOAT"},
            {"name": "created_at", "type": "TIMESTAMP"}
        ],
        num_rows=5000,
        num_bytes=5000000
    )
    
    # Add mock query results for common patterns
    import pandas as pd
    import datetime
    
    # Mock query history
    query_history = pd.DataFrame({
        "query_id": ["q1", "q2", "q3"],
        "user_email": ["user1@example.com", "user2@example.com", "user1@example.com"],
        "query_text": [
            f"SELECT * FROM `{config['project_id']}.{config['dataset_id']}.users`",
            f"SELECT id, name FROM `{config['project_id']}.{config['dataset_id']}.users` WHERE id > 100",
            f"SELECT COUNT(*) FROM `{config['project_id']}.{config['dataset_id']}.orders`"
        ],
        "creation_time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "total_slot_ms": [5000, 2000, 1000],
        "total_bytes_processed": [1000000, 500000, 200000],
        "estimated_cost_usd": [0.005, 0.0025, 0.001]
    })
    
    client.set_query_result("INFORMATION_SCHEMA.JOBS", query_history)
    
    return client

@pytest.fixture
def test_data_path():
    """Provide path to test data directory."""
    return get_test_data_path()

@pytest.fixture
def mock_metadata_extractor():
    """Provide a mock metadata extractor."""
    metadata_extractor = MagicMock()
    config = get_test_config()
    metadata_extractor.project_id = config["project_id"]
    return metadata_extractor

@pytest.fixture
def mock_storage_client():
    """Provide a mock Google Cloud Storage client."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    return mock_client