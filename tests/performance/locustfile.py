"""
Locust performance testing file for BigQuery Cost Intelligence Engine API.
"""

from locust import HttpUser, task, between, tag
import json
import random

class BigQueryOptAPIUser(HttpUser):
    """User for testing BigQuery Cost Intelligence Engine API performance."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup before tests start."""
        # Use test API key
        self.client.headers = {
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json"
        }
        
        # Project ID to use in tests
        self.project_id = "test-project"
        
        # Dataset IDs to use in tests
        self.dataset_ids = ["test_dataset", "sales_data", "user_analytics"]
    
    @tag('health')
    @task(10)
    def get_health(self):
        """Test health endpoint."""
        self.client.get("/api/v1/health")
    
    @tag('dashboard')
    @task(5)
    def get_cost_summary(self):
        """Test cost dashboard summary endpoint."""
        with self.client.get(
            f"/api/v1/cost-dashboard/summary?project_id={self.project_id}&days=30",
            name="/api/v1/cost-dashboard/summary"
        ) as response:
            if response.status_code == 200:
                # Load the response to check data
                data = response.json()
                if "summary" not in data:
                    response.failure("Missing summary data in response")
    
    @tag('dashboard')
    @task(4)
    def get_cost_attribution(self):
        """Test cost attribution endpoint."""
        dimensions = random.choice([
            "user,team", 
            "user,team,pattern", 
            "day,table", 
            "user,team,pattern,day,table"
        ])
        
        with self.client.get(
            f"/api/v1/cost-dashboard/attribution?project_id={self.project_id}&days=30&dimensions={dimensions}",
            name="/api/v1/cost-dashboard/attribution"
        ) as response:
            if response.status_code == 200:
                # Load the response to check data
                data = response.json()
                if "attribution" not in data:
                    response.failure("Missing attribution data in response")
    
    @tag('dashboard')
    @task(3)
    def get_cost_trends(self):
        """Test cost trends endpoint."""
        granularity = random.choice(["day", "week", "month"])
        days = random.choice([30, 60, 90])
        
        with self.client.get(
            f"/api/v1/cost-dashboard/trends?project_id={self.project_id}&days={days}&granularity={granularity}",
            name="/api/v1/cost-dashboard/trends"
        ) as response:
            if response.status_code == 200:
                # Load the response to check data
                data = response.json()
                if "trends" not in data:
                    response.failure("Missing trends data in response")
    
    @tag('dashboard')
    @task(2)
    def get_anomalies(self):
        """Test anomalies endpoint."""
        use_ml = random.choice([True, False])
        
        with self.client.get(
            f"/api/v1/cost-dashboard/anomalies?project_id={self.project_id}&days=30&use_ml={str(use_ml).lower()}",
            name="/api/v1/cost-dashboard/anomalies"
        ) as response:
            if response.status_code == 200:
                # Load the response to check data
                data = response.json()
                if "anomalies" not in data:
                    response.failure("Missing anomalies data in response")
    
    @tag('dashboard')
    @task(2)
    def get_forecast(self):
        """Test forecast endpoint."""
        training_days = random.choice([30, 60, 90])
        forecast_days = random.choice([7, 14, 30])
        
        with self.client.get(
            f"/api/v1/cost-dashboard/forecast?project_id={self.project_id}&training_days={training_days}&forecast_days={forecast_days}",
            name="/api/v1/cost-dashboard/forecast"
        ) as response:
            if response.status_code == 200:
                # Load the response to check data
                data = response.json()
                if "forecast" not in data:
                    response.failure("Missing forecast data in response")
    
    @tag('analysis')
    @task(1)
    def trigger_analysis(self):
        """Test analysis trigger endpoint."""
        dataset_id = random.choice(self.dataset_ids)
        
        with self.client.post(
            "/api/v1/analyze",
            json={
                "project_id": self.project_id,
                "dataset_id": dataset_id
            },
            name="/api/v1/analyze"
        ) as response:
            if response.status_code == 200:
                # Load the response to check data
                data = response.json()
                if "analysis_id" not in data:
                    response.failure("Missing analysis_id in response")
                else:
                    # Store analysis ID for status check
                    self.analysis_id = data["analysis_id"]
    
    @tag('analysis')
    @task(1)
    def check_analysis_status(self):
        """Test analysis status endpoint."""
        if hasattr(self, 'analysis_id'):
            with self.client.get(
                f"/api/v1/analysis/{self.analysis_id}",
                name="/api/v1/analysis/{analysis_id}"
            ) as response:
                if response.status_code == 200:
                    # Load the response to check data
                    data = response.json()
                    if "status" not in data:
                        response.failure("Missing status in response")