"""Cost attribution module for BigQuery Cost Intelligence Engine.

This module provides functionality to analyze and attribute BigQuery costs
to specific teams, users, and query patterns. It also includes anomaly detection
to identify unusual spending patterns that may indicate inefficiencies.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from scipy import stats
from google.cloud import bigquery

from ..utils.logging import setup_logger
from ..connectors.bigquery import BigQueryConnector
from ..analysis.metadata import MetadataExtractor

logger = setup_logger(__name__)

# Constants
DEFAULT_ANALYSIS_PERIOD_DAYS = 30
ON_DEMAND_COST_PER_TB = 5.0  # $5 per TB for on-demand queries


class CostAttributionAnalyzer:
    """Analyzes and attributes BigQuery costs to teams, users, and query patterns."""
    
    def __init__(self, connector: Optional[BigQueryConnector] = None,
                 project_id: Optional[str] = None, 
                 credentials_path: Optional[str] = None):
        """Initialize the Cost Attribution Analyzer.
        
        Args:
            connector: Optional existing BigQueryConnector instance
            project_id: GCP project ID if connector not provided
            credentials_path: Path to service account credentials if connector not provided
        """
        if connector:
            self.connector = connector
            self.client = connector.client
            self.project_id = connector.project_id
        else:
            if not project_id:
                raise ValueError("Either connector or project_id must be provided")
            self.connector = BigQueryConnector(project_id, credentials_path)
            self.client = self.connector.client
            self.project_id = project_id
        
        logger.info(f"Initialized CostAttributionAnalyzer for project {self.project_id}")
        
        # Initialize metadata extractor for additional context
        self.metadata_extractor = MetadataExtractor(connector=self.connector)
        
        # Cache for storing results
        self._cache = {}
        
        # Team mapping (can be configured via set_team_mapping)
        self._team_mapping = {}
    
    def set_team_mapping(self, mapping: Dict[str, str]) -> None:
        """Set mapping from users to teams.
        
        Args:
            mapping: Dictionary mapping user emails to team names
        """
        self._team_mapping = mapping
        logger.info(f"Updated team mapping with {len(mapping)} entries")
    
    def get_user_team(self, user_email: str) -> str:
        """Get team name for a user.
        
        Args:
            user_email: User email
            
        Returns:
            Team name or "Unknown"
        """
        # Try exact match
        if user_email in self._team_mapping:
            return self._team_mapping[user_email]
        
        # Try domain match
        domain = user_email.split('@')[-1] if '@' in user_email else None
        if domain:
            domain_map = {k.split('@')[-1]: v for k, v in self._team_mapping.items() 
                         if '@' in k and k.startswith('*@')}
            if domain in domain_map:
                return domain_map[domain]
        
        return "Unknown"
    
    def get_job_history(self, days_back: int = DEFAULT_ANALYSIS_PERIOD_DAYS) -> pd.DataFrame:
        """Retrieve job history for the specified time period.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            DataFrame containing job history with cost attribution
        """
        cache_key = f"job_history_{days_back}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        start_date = datetime.now() - timedelta(days=days_back)
        
        # Query to get job history
        query = f"""
        SELECT 
            creation_time,
            user_email,
            job_id,
            job_type,
            query,
            state,
            total_bytes_processed,
            total_bytes_billed,
            total_slot_ms,
            cache_hit,
            statement_type,
            referenced_tables,
            destination_table,
            start_time,
            end_time,
            error_result
        FROM 
            `{self.project_id}.region-us.INFORMATION_SCHEMA.JOBS`
        WHERE 
            creation_time >= TIMESTAMP('{start_date.strftime('%Y-%m-%d')}')
            AND job_type = 'QUERY'
            AND state = 'DONE'
        ORDER BY 
            creation_time DESC
        """
        
        job_history = self.connector.query_to_dataframe(query)
        
        # Add cost estimation
        job_history['estimated_cost_usd'] = job_history['total_bytes_processed'].fillna(0) / 1e12 * ON_DEMAND_COST_PER_TB
        
        # Add duration in seconds
        job_history['duration_seconds'] = (job_history['end_time'] - job_history['start_time']).dt.total_seconds()
        
        # Add team attribution
        job_history['team'] = job_history['user_email'].apply(self.get_user_team)
        
        # Extract query patterns
        job_history['query_pattern'] = self._extract_query_patterns(job_history['query'].tolist())
        
        # Cache results
        self._cache[cache_key] = job_history
        
        return job_history
    
    def _extract_query_patterns(self, queries: List[str]) -> List[str]:
        """Extract common patterns from queries.
        
        Args:
            queries: List of SQL queries
            
        Returns:
            List of identified query patterns
        """
        patterns = []
        
        for query in queries:
            if not query:
                patterns.append("UNKNOWN")
                continue
                
            query = query.upper()
            
            if "CREATE TABLE" in query or "CREATE OR REPLACE TABLE" in query:
                patterns.append("TABLE_CREATION")
            elif "CREATE VIEW" in query or "CREATE OR REPLACE VIEW" in query:
                patterns.append("VIEW_CREATION")
            elif "INSERT INTO" in query:
                patterns.append("DATA_INSERTION")
            elif "DELETE FROM" in query:
                patterns.append("DATA_DELETION")
            elif "UPDATE" in query:
                patterns.append("DATA_UPDATE")
            elif "SELECT * FROM" in query:
                patterns.append("SELECT_STAR")
            elif "GROUP BY" in query and "ORDER BY" in query:
                patterns.append("AGGREGATION_WITH_SORTING")
            elif "JOIN" in query:
                if query.count("JOIN") > 3:
                    patterns.append("MULTI_JOIN")
                else:
                    patterns.append("JOIN_OPERATION")
            elif "WHERE" not in query and "FROM" in query:
                patterns.append("MISSING_WHERE_CLAUSE")
            elif "EXPORT DATA" in query or "EXTRACT" in query:
                patterns.append("DATA_EXPORT")
            elif "WITH" in query and "FROM" in query:
                patterns.append("CTE_QUERY")
            else:
                patterns.append("OTHER")
        
        return patterns
    
    def attribute_costs(self, days_back: int = DEFAULT_ANALYSIS_PERIOD_DAYS) -> Dict[str, pd.DataFrame]:
        """Attribute costs across different dimensions.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary of DataFrames with cost breakdowns by different dimensions
        """
        cache_key = f"attributed_costs_{days_back}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get job history
        job_history = self.get_job_history(days_back)
        
        # 1. Cost by user
        cost_by_user = job_history.groupby('user_email').agg({
            'estimated_cost_usd': 'sum',
            'total_bytes_processed': 'sum',
            'job_id': 'count',
            'duration_seconds': 'mean',
            'total_slot_ms': 'sum',
            'cache_hit': 'mean'
        }).reset_index()
        
        cost_by_user.columns = ['user_email', 'total_cost_usd', 'total_bytes_processed', 
                              'query_count', 'avg_duration', 'total_slot_ms', 'cache_hit_ratio']
        
        cost_by_user = cost_by_user.sort_values('total_cost_usd', ascending=False)
        
        # 2. Cost by team
        cost_by_team = job_history.groupby('team').agg({
            'estimated_cost_usd': 'sum',
            'total_bytes_processed': 'sum',
            'job_id': 'count',
            'user_email': pd.Series.nunique,
            'duration_seconds': 'mean',
            'total_slot_ms': 'sum'
        }).reset_index()
        
        cost_by_team.columns = ['team', 'total_cost_usd', 'total_bytes_processed', 
                              'query_count', 'unique_users', 'avg_duration', 'total_slot_ms']
        
        cost_by_team = cost_by_team.sort_values('total_cost_usd', ascending=False)
        
        # 3. Cost by query pattern
        cost_by_pattern = job_history.groupby('query_pattern').agg({
            'estimated_cost_usd': 'sum',
            'total_bytes_processed': 'sum',
            'job_id': 'count',
            'duration_seconds': 'mean',
            'total_slot_ms': 'sum'
        }).reset_index()
        
        cost_by_pattern.columns = ['query_pattern', 'total_cost_usd', 'total_bytes_processed', 
                                 'query_count', 'avg_duration', 'total_slot_ms']
        
        cost_by_pattern = cost_by_pattern.sort_values('total_cost_usd', ascending=False)
        
        # 4. Cost by day
        job_history['date'] = job_history['creation_time'].dt.date
        cost_by_day = job_history.groupby('date').agg({
            'estimated_cost_usd': 'sum',
            'total_bytes_processed': 'sum',
            'job_id': 'count',
            'user_email': pd.Series.nunique,
            'duration_seconds': 'mean',
            'total_slot_ms': 'sum'
        }).reset_index()
        
        cost_by_day.columns = ['date', 'total_cost_usd', 'total_bytes_processed', 
                             'query_count', 'unique_users', 'avg_duration', 'total_slot_ms']
        
        # 5. Cost by referenced table
        # Expand the referenced_tables column which contains arrays
        tables_data = []
        
        for _, row in job_history.iterrows():
            referenced_tables = row.get('referenced_tables')
            if referenced_tables:
                if isinstance(referenced_tables, str):
                    try:
                        tables = json.loads(referenced_tables.replace("'", "\""))
                    except:
                        tables = []
                else:
                    tables = referenced_tables
                
                for table in tables:
                    if isinstance(table, dict) and 'datasetId' in table and 'tableId' in table:
                        table_ref = f"{table.get('projectId', self.project_id)}.{table['datasetId']}.{table['tableId']}"
                        tables_data.append({
                            'table': table_ref,
                            'estimated_cost_usd': row['estimated_cost_usd'],
                            'total_bytes_processed': row['total_bytes_processed'],
                            'job_id': row['job_id'],
                            'user_email': row['user_email'],
                            'team': row['team']
                        })
        
        if tables_data:
            tables_df = pd.DataFrame(tables_data)
            cost_by_table = tables_df.groupby('table').agg({
                'estimated_cost_usd': 'sum',
                'total_bytes_processed': 'sum',
                'job_id': 'count',
                'user_email': pd.Series.nunique,
                'team': pd.Series.nunique
            }).reset_index()
            
            cost_by_table.columns = ['table', 'total_cost_usd', 'total_bytes_processed', 
                                  'query_count', 'unique_users', 'unique_teams']
            
            cost_by_table = cost_by_table.sort_values('total_cost_usd', ascending=False)
        else:
            cost_by_table = pd.DataFrame(columns=['table', 'total_cost_usd', 'total_bytes_processed', 
                                               'query_count', 'unique_users', 'unique_teams'])
        
        # Create result dictionary
        results = {
            'cost_by_user': cost_by_user,
            'cost_by_team': cost_by_team,
            'cost_by_pattern': cost_by_pattern,
            'cost_by_day': cost_by_day,
            'cost_by_table': cost_by_table
        }
        
        # Cache results
        self._cache[cache_key] = results
        
        return results
    
    def get_expensive_queries(self, days_back: int = DEFAULT_ANALYSIS_PERIOD_DAYS, 
                           limit: int = 100) -> pd.DataFrame:
        """Get the most expensive queries.
        
        Args:
            days_back: Number of days to analyze
            limit: Maximum number of queries to return
            
        Returns:
            DataFrame with the most expensive queries
        """
        job_history = self.get_job_history(days_back)
        
        # Sort by cost and get top queries
        expensive_queries = job_history.sort_values('estimated_cost_usd', ascending=False).head(limit)
        
        # Select relevant columns
        columns = ['job_id', 'user_email', 'team', 'creation_time', 'estimated_cost_usd', 
                 'total_bytes_processed', 'duration_seconds', 'query_pattern', 'query']
        
        return expensive_queries[columns]
    
    def get_cost_summary(self, days_back: int = DEFAULT_ANALYSIS_PERIOD_DAYS) -> Dict[str, Any]:
        """Get a summary of costs for the specified time period.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with cost summary metrics
        """
        # Get attributed costs
        costs = self.attribute_costs(days_back)
        
        # Calculate summary metrics
        total_cost = costs['cost_by_day']['total_cost_usd'].sum()
        total_bytes = costs['cost_by_day']['total_bytes_processed'].sum()
        total_queries = costs['cost_by_day']['query_count'].sum()
        unique_users = costs['cost_by_user'].shape[0]
        
        # Calculate daily averages
        daily_avg_cost = costs['cost_by_day']['total_cost_usd'].mean()
        daily_avg_queries = costs['cost_by_day']['query_count'].mean()
        
        # Get top contributors
        top_users = costs['cost_by_user'].head(5)[['user_email', 'total_cost_usd', 'query_count']].to_dict('records')
        top_teams = costs['cost_by_team'].head(5)[['team', 'total_cost_usd', 'query_count']].to_dict('records')
        top_patterns = costs['cost_by_pattern'].head(5)[['query_pattern', 'total_cost_usd', 'query_count']].to_dict('records')
        
        # Create summary dictionary
        summary = {
            'period_days': days_back,
            'total_cost_usd': total_cost,
            'total_bytes_processed': total_bytes,
            'total_queries': int(total_queries),
            'unique_users': int(unique_users),
            'daily_avg_cost_usd': daily_avg_cost,
            'daily_avg_queries': daily_avg_queries,
            'cost_per_query_usd': total_cost / total_queries if total_queries > 0 else 0,
            'cost_per_tb_processed': total_cost / (total_bytes / 1e12) if total_bytes > 0 else 0,
            'top_users': top_users,
            'top_teams': top_teams,
            'top_patterns': top_patterns
        }
        
        return summary
    
    def get_cost_trends(self, days_back: int = 90, 
                      granularity: str = 'day') -> pd.DataFrame:
        """Get cost trends over time.
        
        Args:
            days_back: Number of days to analyze
            granularity: Time granularity ('day', 'week', or 'month')
            
        Returns:
            DataFrame with cost trends
        """
        job_history = self.get_job_history(days_back)
        
        # Create date column based on granularity
        if granularity == 'day':
            job_history['period'] = job_history['creation_time'].dt.date
        elif granularity == 'week':
            job_history['period'] = job_history['creation_time'].dt.to_period('W').apply(lambda x: x.start_time.date())
        elif granularity == 'month':
            job_history['period'] = job_history['creation_time'].dt.to_period('M').apply(lambda x: x.start_time.date())
        else:
            raise ValueError("Granularity must be one of: 'day', 'week', 'month'")
        
        # Group by period
        trends = job_history.groupby('period').agg({
            'estimated_cost_usd': 'sum',
            'total_bytes_processed': 'sum',
            'job_id': 'count',
            'user_email': pd.Series.nunique,
            'team': pd.Series.nunique
        }).reset_index()
        
        trends.columns = ['period', 'total_cost_usd', 'total_bytes_processed', 
                         'query_count', 'unique_users', 'unique_teams']
        
        # Calculate moving averages
        window = 7 if granularity == 'day' else 4 if granularity == 'week' else 3
        trends['cost_ma'] = trends['total_cost_usd'].rolling(window=min(window, len(trends))).mean()
        
        return trends
    
    def compare_periods(self, current_days: int = 30, 
                      previous_days: int = 30) -> Dict[str, Any]:
        """Compare costs between two time periods.
        
        Args:
            current_days: Number of days in current period
            previous_days: Number of days in previous period
            
        Returns:
            Dictionary with period comparison metrics
        """
        # Get job history for both periods
        end_date = datetime.now()
        current_start = end_date - timedelta(days=current_days)
        previous_start = current_start - timedelta(days=previous_days)
        
        # Query for both periods
        query = f"""
        SELECT 
            creation_time,
            user_email,
            job_id,
            total_bytes_processed,
            total_slot_ms
        FROM 
            `{self.project_id}.region-us.INFORMATION_SCHEMA.JOBS`
        WHERE 
            creation_time >= TIMESTAMP('{previous_start.strftime('%Y-%m-%d')}')
            AND creation_time < TIMESTAMP('{end_date.strftime('%Y-%m-%d')}')
            AND job_type = 'QUERY'
            AND state = 'DONE'
        """
        
        df = self.connector.query_to_dataframe(query)
        
        # Calculate cost
        df['estimated_cost_usd'] = df['total_bytes_processed'].fillna(0) / 1e12 * ON_DEMAND_COST_PER_TB
        
        # Split into periods
        current_period = df[df['creation_time'] >= current_start]
        previous_period = df[(df['creation_time'] >= previous_start) & (df['creation_time'] < current_start)]
        
        # Calculate metrics for both periods
        current_metrics = {
            'total_cost_usd': current_period['estimated_cost_usd'].sum(),
            'total_bytes_processed': current_period['total_bytes_processed'].sum(),
            'query_count': len(current_period),
            'unique_users': current_period['user_email'].nunique(),
            'daily_cost_usd': current_period['estimated_cost_usd'].sum() / current_days
        }
        
        previous_metrics = {
            'total_cost_usd': previous_period['estimated_cost_usd'].sum(),
            'total_bytes_processed': previous_period['total_bytes_processed'].sum(),
            'query_count': len(previous_period),
            'unique_users': previous_period['user_email'].nunique(),
            'daily_cost_usd': previous_period['estimated_cost_usd'].sum() / previous_days
        }
        
        # Calculate changes
        changes = {}
        for key in current_metrics:
            if previous_metrics[key] > 0:
                changes[key] = (current_metrics[key] - previous_metrics[key]) / previous_metrics[key] * 100
            else:
                changes[key] = float('inf') if current_metrics[key] > 0 else 0
        
        return {
            'current_period_days': current_days,
            'current_period_start': current_start.strftime('%Y-%m-%d'),
            'current_period_end': end_date.strftime('%Y-%m-%d'),
            'previous_period_days': previous_days,
            'previous_period_start': previous_start.strftime('%Y-%m-%d'),
            'previous_period_end': current_start.strftime('%Y-%m-%d'),
            'current_metrics': current_metrics,
            'previous_metrics': previous_metrics,
            'percent_changes': changes
        }


class CostAnomalyDetector:
    """Detects anomalies in BigQuery costs and usage."""
    
    def __init__(self, attribution_analyzer: CostAttributionAnalyzer):
        """Initialize Cost Anomaly Detector.
        
        Args:
            attribution_analyzer: CostAttributionAnalyzer instance
        """
        self.attribution_analyzer = attribution_analyzer
        self.z_score_threshold = 3.0  # Z-score threshold for statistical anomalies
        
        # Cache for storing results
        self._cache = {}
    
    def detect_daily_cost_anomalies(self, days_back: int = 60, 
                                  min_cost_usd: float = 10.0) -> pd.DataFrame:
        """Detect anomalies in daily costs.
        
        Args:
            days_back: Number of days of history to analyze
            min_cost_usd: Minimum cost threshold to consider (to ignore very small costs)
            
        Returns:
            DataFrame with detected anomalies
        """
        cache_key = f"daily_anomalies_{days_back}_{min_cost_usd}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get daily costs
        costs = self.attribution_analyzer.attribute_costs(days_back)
        daily_costs = costs['cost_by_day'].copy()
        
        # Apply cost threshold
        daily_costs = daily_costs[daily_costs['total_cost_usd'] >= min_cost_usd]
        
        if len(daily_costs) < 7:  # Need at least a week of data
            return pd.DataFrame(columns=['date', 'total_cost_usd', 'expected_cost_usd', 
                                      'z_score', 'is_anomaly', 'percent_change'])
        
        # Calculate rolling mean and standard deviation
        window = min(7, len(daily_costs) - 1)  # Use a 7-day window if possible
        daily_costs = daily_costs.sort_values('date')
        daily_costs['rolling_mean'] = daily_costs['total_cost_usd'].rolling(window=window).mean()
        daily_costs['rolling_std'] = daily_costs['total_cost_usd'].rolling(window=window).std()
        
        # Calculate z-scores
        daily_costs['z_score'] = (daily_costs['total_cost_usd'] - daily_costs['rolling_mean']) / daily_costs['rolling_std'].replace(0, 1)
        
        # Flag anomalies
        daily_costs['is_anomaly'] = abs(daily_costs['z_score']) > self.z_score_threshold
        
        # Calculate percent change
        daily_costs['percent_change'] = (daily_costs['total_cost_usd'] - daily_costs['rolling_mean']) / daily_costs['rolling_mean'] * 100
        
        # Rename for clarity
        daily_costs = daily_costs.rename(columns={'rolling_mean': 'expected_cost_usd'})
        
        # Filter to anomalies and sort by severity
        anomalies = daily_costs[daily_costs['is_anomaly']].sort_values('date', ascending=False)
        
        # Cache results
        self._cache[cache_key] = anomalies
        
        return anomalies
    
    def detect_user_cost_anomalies(self, days_back: int = 30, 
                                 comparison_days: int = 30) -> pd.DataFrame:
        """Detect users with anomalous spending patterns.
        
        Args:
            days_back: Number of days in current period
            comparison_days: Number of days in comparison period
            
        Returns:
            DataFrame with users having anomalous costs
        """
        cache_key = f"user_anomalies_{days_back}_{comparison_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get job history for both periods
        end_date = datetime.now()
        current_start = end_date - timedelta(days=days_back)
        previous_start = current_start - timedelta(days=comparison_days)
        
        # Query for both periods
        query = f"""
        SELECT 
            creation_time,
            user_email,
            job_id,
            total_bytes_processed
        FROM 
            `{self.attribution_analyzer.project_id}.region-us.INFORMATION_SCHEMA.JOBS`
        WHERE 
            creation_time >= TIMESTAMP('{previous_start.strftime('%Y-%m-%d')}')
            AND creation_time < TIMESTAMP('{end_date.strftime('%Y-%m-%d')}')
            AND job_type = 'QUERY'
            AND state = 'DONE'
        """
        
        df = self.attribution_analyzer.connector.query_to_dataframe(query)
        
        # Calculate cost
        df['estimated_cost_usd'] = df['total_bytes_processed'].fillna(0) / 1e12 * ON_DEMAND_COST_PER_TB
        
        # Split into periods
        current_period = df[df['creation_time'] >= current_start]
        previous_period = df[(df['creation_time'] >= previous_start) & (df['creation_time'] < current_start)]
        
        # Calculate cost by user for both periods
        current_user_costs = current_period.groupby('user_email')['estimated_cost_usd'].sum().reset_index()
        previous_user_costs = previous_period.groupby('user_email')['estimated_cost_usd'].sum().reset_index()
        
        # Merge periods
        user_costs = pd.merge(current_user_costs, previous_user_costs, 
                           on='user_email', how='outer', suffixes=('_current', '_previous'))
        
        # Fill NaN values with 0
        user_costs = user_costs.fillna(0)
        
        # Calculate percent change
        user_costs['percent_change'] = 0.0
        mask = user_costs['estimated_cost_usd_previous'] > 0
        user_costs.loc[mask, 'percent_change'] = ((user_costs.loc[mask, 'estimated_cost_usd_current'] - 
                                                 user_costs.loc[mask, 'estimated_cost_usd_previous']) / 
                                                user_costs.loc[mask, 'estimated_cost_usd_previous'] * 100)
        
        # Calculate z-scores for percent change (only for users who spent in both periods)
        valid_users = user_costs[(user_costs['estimated_cost_usd_current'] > 0) & 
                               (user_costs['estimated_cost_usd_previous'] > 0)]
        
        if len(valid_users) > 2:  # Need at least a few users to calculate meaningful z-scores
            mean_pct_change = valid_users['percent_change'].mean()
            std_pct_change = valid_users['percent_change'].std()
            
            user_costs['z_score'] = 0.0
            user_costs.loc[valid_users.index, 'z_score'] = (valid_users['percent_change'] - 
                                                         mean_pct_change) / (std_pct_change if std_pct_change > 0 else 1)
            
            # Flag anomalies
            user_costs['is_anomaly'] = abs(user_costs['z_score']) > self.z_score_threshold
            
            # Add team attribution
            user_costs['team'] = user_costs['user_email'].apply(self.attribution_analyzer.get_user_team)
            
            # Filter to anomalies and sort by severity
            anomalies = user_costs[user_costs['is_anomaly']].sort_values('z_score', ascending=False)
        else:
            # Not enough data for statistical analysis
            user_costs['z_score'] = 0.0
            user_costs['is_anomaly'] = False
            user_costs['team'] = user_costs['user_email'].apply(self.attribution_analyzer.get_user_team)
            anomalies = pd.DataFrame(columns=user_costs.columns)
        
        # Cache results
        self._cache[cache_key] = anomalies
        
        return anomalies
    
    def detect_team_cost_anomalies(self, days_back: int = 30, 
                                 comparison_days: int = 30) -> pd.DataFrame:
        """Detect teams with anomalous spending patterns.
        
        Args:
            days_back: Number of days in current period
            comparison_days: Number of days in comparison period
            
        Returns:
            DataFrame with teams having anomalous costs
        """
        cache_key = f"team_anomalies_{days_back}_{comparison_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get job history for both periods
        end_date = datetime.now()
        current_start = end_date - timedelta(days=days_back)
        previous_start = current_start - timedelta(days=comparison_days)
        
        # Query for both periods
        query = f"""
        SELECT 
            creation_time,
            user_email,
            job_id,
            total_bytes_processed
        FROM 
            `{self.attribution_analyzer.project_id}.region-us.INFORMATION_SCHEMA.JOBS`
        WHERE 
            creation_time >= TIMESTAMP('{previous_start.strftime('%Y-%m-%d')}')
            AND creation_time < TIMESTAMP('{end_date.strftime('%Y-%m-%d')}')
            AND job_type = 'QUERY'
            AND state = 'DONE'
        """
        
        df = self.attribution_analyzer.connector.query_to_dataframe(query)
        
        # Calculate cost
        df['estimated_cost_usd'] = df['total_bytes_processed'].fillna(0) / 1e12 * ON_DEMAND_COST_PER_TB
        
        # Add team attribution
        df['team'] = df['user_email'].apply(self.attribution_analyzer.get_user_team)
        
        # Split into periods
        current_period = df[df['creation_time'] >= current_start]
        previous_period = df[(df['creation_time'] >= previous_start) & (df['creation_time'] < current_start)]
        
        # Calculate cost by team for both periods
        current_team_costs = current_period.groupby('team')['estimated_cost_usd'].sum().reset_index()
        previous_team_costs = previous_period.groupby('team')['estimated_cost_usd'].sum().reset_index()
        
        # Merge periods
        team_costs = pd.merge(current_team_costs, previous_team_costs, 
                           on='team', how='outer', suffixes=('_current', '_previous'))
        
        # Fill NaN values with 0
        team_costs = team_costs.fillna(0)
        
        # Calculate percent change
        team_costs['percent_change'] = 0.0
        mask = team_costs['estimated_cost_usd_previous'] > 0
        team_costs.loc[mask, 'percent_change'] = ((team_costs.loc[mask, 'estimated_cost_usd_current'] - 
                                                team_costs.loc[mask, 'estimated_cost_usd_previous']) / 
                                               team_costs.loc[mask, 'estimated_cost_usd_previous'] * 100)
        
        # Calculate z-scores for percent change (only for teams who spent in both periods)
        valid_teams = team_costs[(team_costs['estimated_cost_usd_current'] > 0) & 
                              (team_costs['estimated_cost_usd_previous'] > 0)]
        
        if len(valid_teams) > 2:  # Need at least a few teams to calculate meaningful z-scores
            mean_pct_change = valid_teams['percent_change'].mean()
            std_pct_change = valid_teams['percent_change'].std()
            
            team_costs['z_score'] = 0.0
            team_costs.loc[valid_teams.index, 'z_score'] = (valid_teams['percent_change'] - 
                                                         mean_pct_change) / (std_pct_change if std_pct_change > 0 else 1)
            
            # Flag anomalies
            team_costs['is_anomaly'] = abs(team_costs['z_score']) > self.z_score_threshold
            
            # Filter to anomalies and sort by severity
            anomalies = team_costs[team_costs['is_anomaly']].sort_values('z_score', ascending=False)
        else:
            # Not enough data for statistical analysis
            team_costs['z_score'] = 0.0
            team_costs['is_anomaly'] = False
            anomalies = pd.DataFrame(columns=team_costs.columns)
        
        # Cache results
        self._cache[cache_key] = anomalies
        
        return anomalies
    
    def detect_query_pattern_anomalies(self, days_back: int = 30, 
                                     comparison_days: int = 30) -> pd.DataFrame:
        """Detect query patterns with anomalous costs.
        
        Args:
            days_back: Number of days in current period
            comparison_days: Number of days in comparison period
            
        Returns:
            DataFrame with query patterns having anomalous costs
        """
        # Get current period job history
        current_jobs = self.attribution_analyzer.get_job_history(days_back)
        
        # Get previous period job history
        end_date = datetime.now() - timedelta(days=days_back)
        previous_start = end_date - timedelta(days=comparison_days)
        
        previous_query = f"""
        SELECT 
            creation_time,
            user_email,
            job_id,
            job_type,
            query,
            total_bytes_processed,
            total_slot_ms
        FROM 
            `{self.attribution_analyzer.project_id}.region-us.INFORMATION_SCHEMA.JOBS`
        WHERE 
            creation_time >= TIMESTAMP('{previous_start.strftime('%Y-%m-%d')}')
            AND creation_time < TIMESTAMP('{end_date.strftime('%Y-%m-%d')}')
            AND job_type = 'QUERY'
            AND state = 'DONE'
        """
        
        previous_jobs = self.attribution_analyzer.connector.query_to_dataframe(previous_query)
        
        # Calculate costs
        previous_jobs['estimated_cost_usd'] = previous_jobs['total_bytes_processed'].fillna(0) / 1e12 * ON_DEMAND_COST_PER_TB
        
        # Extract query patterns for previous period
        previous_jobs['query_pattern'] = self.attribution_analyzer._extract_query_patterns(previous_jobs['query'].tolist())
        
        # Calculate cost by pattern for both periods
        current_pattern_costs = current_jobs.groupby('query_pattern')['estimated_cost_usd'].sum().reset_index()
        previous_pattern_costs = previous_jobs.groupby('query_pattern')['estimated_cost_usd'].sum().reset_index()
        
        # Merge periods
        pattern_costs = pd.merge(current_pattern_costs, previous_pattern_costs, 
                              on='query_pattern', how='outer', suffixes=('_current', '_previous'))
        
        # Fill NaN values with 0
        pattern_costs = pattern_costs.fillna(0)
        
        # Calculate percent change
        pattern_costs['percent_change'] = 0.0
        mask = pattern_costs['estimated_cost_usd_previous'] > 0
        pattern_costs.loc[mask, 'percent_change'] = ((pattern_costs.loc[mask, 'estimated_cost_usd_current'] - 
                                                   pattern_costs.loc[mask, 'estimated_cost_usd_previous']) / 
                                                  pattern_costs.loc[mask, 'estimated_cost_usd_previous'] * 100)
        
        # Calculate z-scores for percent change
        valid_patterns = pattern_costs[(pattern_costs['estimated_cost_usd_current'] > 0) & 
                                    (pattern_costs['estimated_cost_usd_previous'] > 0)]
        
        if len(valid_patterns) > 2:
            mean_pct_change = valid_patterns['percent_change'].mean()
            std_pct_change = valid_patterns['percent_change'].std()
            
            pattern_costs['z_score'] = 0.0
            pattern_costs.loc[valid_patterns.index, 'z_score'] = (valid_patterns['percent_change'] - 
                                                               mean_pct_change) / (std_pct_change if std_pct_change > 0 else 1)
            
            # Flag anomalies
            pattern_costs['is_anomaly'] = abs(pattern_costs['z_score']) > self.z_score_threshold
            
            # Filter to anomalies and sort by severity
            anomalies = pattern_costs[pattern_costs['is_anomaly']].sort_values('z_score', ascending=False)
        else:
            pattern_costs['z_score'] = 0.0
            pattern_costs['is_anomaly'] = False
            anomalies = pd.DataFrame(columns=pattern_costs.columns)
        
        return anomalies
    
    def generate_anomaly_report(self, days_back: int = 30, 
                              comparison_days: int = 30) -> Dict[str, Any]:
        """Generate a comprehensive anomaly report.
        
        Args:
            days_back: Number of days for current period
            comparison_days: Number of days for comparison period
            
        Returns:
            Dictionary with different types of anomalies and summary information
        """
        # Detect various types of anomalies
        daily_anomalies = self.detect_daily_cost_anomalies(days_back + comparison_days)
        user_anomalies = self.detect_user_cost_anomalies(days_back, comparison_days)
        team_anomalies = self.detect_team_cost_anomalies(days_back, comparison_days)
        pattern_anomalies = self.detect_query_pattern_anomalies(days_back, comparison_days)
        
        # Calculate summary statistics
        total_anomalies = {
            'daily': len(daily_anomalies),
            'user': len(user_anomalies),
            'team': len(team_anomalies),
            'pattern': len(pattern_anomalies)
        }
        
        # Get cost summary for context
        cost_summary = self.attribution_analyzer.get_cost_summary(days_back)
        
        # Create report
        report = {
            'anomaly_counts': total_anomalies,
            'analysis_period_days': days_back,
            'comparison_period_days': comparison_days,
            'cost_summary': cost_summary,
            'daily_anomalies': daily_anomalies.to_dict('records') if not daily_anomalies.empty else [],
            'user_anomalies': user_anomalies.to_dict('records') if not user_anomalies.empty else [],
            'team_anomalies': team_anomalies.to_dict('records') if not team_anomalies.empty else [],
            'pattern_anomalies': pattern_anomalies.to_dict('records') if not pattern_anomalies.empty else [],
            'generated_at': datetime.now().isoformat()
        }
        
        return report


class CostAlertSystem:
    """Manages cost anomaly alerts and notifications."""
    
    def __init__(self, anomaly_detector: CostAnomalyDetector):
        """Initialize Cost Alert System.
        
        Args:
            anomaly_detector: CostAnomalyDetector instance
        """
        self.anomaly_detector = anomaly_detector
        self.alert_history = []
    
    def check_and_generate_alerts(self, days_back: int = 7, 
                                min_cost_increase_usd: float = 100.0) -> List[Dict[str, Any]]:
        """Check for anomalies and generate alerts.
        
        Args:
            days_back: Number of days to analyze
            min_cost_increase_usd: Minimum cost increase to trigger an alert
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Check for daily cost anomalies
        daily_anomalies = self.anomaly_detector.detect_daily_cost_anomalies(days_back)
        
        for _, anomaly in daily_anomalies.iterrows():
            cost_increase = anomaly['total_cost_usd'] - anomaly['expected_cost_usd']
            
            if cost_increase >= min_cost_increase_usd:
                alert = {
                    'type': 'daily_cost_spike',
                    'severity': 'high' if cost_increase >= min_cost_increase_usd * 2 else 'medium',
                    'date': anomaly['date'].strftime('%Y-%m-%d'),
                    'actual_cost_usd': float(anomaly['total_cost_usd']),
                    'expected_cost_usd': float(anomaly['expected_cost_usd']),
                    'cost_increase_usd': float(cost_increase),
                    'percent_increase': float(anomaly['percent_change']),
                    'z_score': float(anomaly['z_score']),
                    'message': f"Daily cost spike of ${cost_increase:.2f} ({anomaly['percent_change']:.1f}%) on {anomaly['date'].strftime('%Y-%m-%d')}",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        # Check for user anomalies
        user_anomalies = self.anomaly_detector.detect_user_cost_anomalies(days_back)
        
        for _, anomaly in user_anomalies.iterrows():
            cost_increase = anomaly['estimated_cost_usd_current'] - anomaly['estimated_cost_usd_previous']
            
            if cost_increase >= min_cost_increase_usd:
                alert = {
                    'type': 'user_cost_spike',
                    'severity': 'high' if cost_increase >= min_cost_increase_usd * 2 else 'medium',
                    'user_email': anomaly['user_email'],
                    'team': anomaly['team'],
                    'current_cost_usd': float(anomaly['estimated_cost_usd_current']),
                    'previous_cost_usd': float(anomaly['estimated_cost_usd_previous']),
                    'cost_increase_usd': float(cost_increase),
                    'percent_increase': float(anomaly['percent_change']),
                    'z_score': float(anomaly['z_score']),
                    'message': f"User cost spike: {anomaly['user_email']} increased by ${cost_increase:.2f} ({anomaly['percent_change']:.1f}%)",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        # Add alerts to history
        self.alert_history.extend(alerts)
        
        return alerts
    
    def get_alert_history(self, days_back: int = 30, 
                        include_dismissed: bool = False) -> List[Dict[str, Any]]:
        """Get alert history for the specified period.
        
        Args:
            days_back: Number of days to look back
            include_dismissed: Whether to include dismissed alerts
            
        Returns:
            List of historical alerts
        """
        cutoff = datetime.now() - timedelta(days=days_back)
        
        filtered_alerts = []
        for alert in self.alert_history:
            # Parse alert timestamp
            timestamp = datetime.fromisoformat(alert['timestamp'])
            
            # Filter by date
            if timestamp >= cutoff:
                # Filter by dismissed status
                if include_dismissed or not alert.get('dismissed', False):
                    filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert.
        
        Args:
            alert_id: Unique identifier for the alert
            
        Returns:
            Boolean indicating success
        """
        for alert in self.alert_history:
            if alert.get('id') == alert_id:
                alert['dismissed'] = True
                alert['dismissed_at'] = datetime.now().isoformat()
                return True
        
        return False


def get_cost_attribution_data(project_id: str, days_back: int = 30) -> Dict[str, Any]:
    """Get cost attribution data for a project.
    
    This function provides a simplified interface for getting cost attribution data
    without needing to instantiate the classes directly.
    
    Args:
        project_id: GCP project ID
        days_back: Number of days to analyze
        
    Returns:
        Dictionary with cost attribution data
    """
    analyzer = CostAttributionAnalyzer(project_id=project_id)
    costs = analyzer.attribute_costs(days_back)
    summary = analyzer.get_cost_summary(days_back)
    trends = analyzer.get_cost_trends(days_back)
    
    # Convert dataframes to dictionaries
    costs_dict = {k: v.to_dict('records') for k, v in costs.items()}
    trends_dict = trends.to_dict('records')
    
    return {
        'summary': summary,
        'attribution': costs_dict,
        'trends': trends_dict
    }


def detect_cost_anomalies(project_id: str, days_back: int = 30) -> Dict[str, Any]:
    """Detect cost anomalies for a project.
    
    This function provides a simplified interface for detecting cost anomalies
    without needing to instantiate the classes directly.
    
    Args:
        project_id: GCP project ID
        days_back: Number of days to analyze
        
    Returns:
        Dictionary with detected anomalies
    """
    analyzer = CostAttributionAnalyzer(project_id=project_id)
    detector = CostAnomalyDetector(analyzer)
    report = detector.generate_anomaly_report(days_back)
    
    return report