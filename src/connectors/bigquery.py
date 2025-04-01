"""BigQuery connection manager."""

from google.cloud import bigquery
from google.oauth2 import service_account
import os
from typing import Optional, Dict, Any

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class BigQueryConnector:
    """Manages connections to BigQuery."""
    
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        """Initialize BigQuery connector.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to service account credentials JSON file
        """
        self.project_id = project_id
        
        # Initialize client
        try:
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.client = bigquery.Client(project=project_id, credentials=credentials)
                logger.info(f"Initialized BigQuery client with service account credentials")
            else:
                # Use default credentials
                self.client = bigquery.Client(project=project_id)
                logger.info(f"Initialized BigQuery client with default credentials")
        except Exception as e:
            logger.error(f"Error initializing BigQuery client: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> bigquery.table.RowIterator:
        """Execute a BigQuery SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)
            
        Returns:
            BigQuery query result
        """
        try:
            job_config = bigquery.QueryJobConfig()
            
            if params:
                job_config.query_parameters = [
                    bigquery.ScalarQueryParameter(k, self._get_param_type(v), v)
                    for k, v in params.items()
                ]
            
            return self.client.query(query, job_config=job_config)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def query_to_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None) -> "pd.DataFrame":
        """Execute a query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)
            
        Returns:
            pandas DataFrame with query results
        """
        try:
            query_job = self.execute_query(query, params)
            return query_job.to_dataframe()
        except Exception as e:
            logger.error(f"Error converting query to DataFrame: {e}")
            raise
    
    def get_dataset(self, dataset_id: str) -> bigquery.dataset.DatasetReference:
        """Get a reference to a BigQuery dataset.
        
        Args:
            dataset_id: BigQuery dataset ID
            
        Returns:
            Dataset reference
        """
        return self.client.dataset(dataset_id)
    
    def list_tables(self, dataset_id: str) -> list:
        """List all tables in a dataset.
        
        Args:
            dataset_id: BigQuery dataset ID
            
        Returns:
            List of table references
        """
        dataset_ref = self.get_dataset(dataset_id)
        return list(self.client.list_tables(dataset_ref))
    
    def get_table(self, dataset_id: str, table_id: str) -> bigquery.table.Table:
        """Get a BigQuery table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            Table object
        """
        table_ref = self.client.dataset(dataset_id).table(table_id)
        return self.client.get_table(table_ref)
    
    def _get_param_type(self, value: Any) -> str:
        """Get BigQuery parameter type from Python value.
        
        Args:
            value: Python value
            
        Returns:
            BigQuery parameter type string
        """
        if isinstance(value, str):
            return "STRING"
        elif isinstance(value, bool):
            return "BOOL"
        elif isinstance(value, int):
            return "INT64"
        elif isinstance(value, float):
            return "FLOAT64"
        else:
            return "STRING"
