"""Query optimization analyzer for BigQuery datasets."""

from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Token, Function, Parenthesis
from sqlparse.tokens import Keyword, DML, Whitespace, Name, Punctuation, Wildcard
import logging

from ..utils.logging import setup_logger
from .metadata import MetadataExtractor

logger = setup_logger(__name__)

# Constants
DEFAULT_ANALYSIS_PERIOD_DAYS = 30
QUERY_COST_PER_TB = 5.0  # $5 per TB processed
MIN_QUERY_BYTES_TO_ANALYZE = 1 * (1024**3)  # Only analyze queries processing at least 1GB
MAX_QUERIES_TO_ANALYZE = 1000  # Limit for detailed analysis to manage memory usage


class QueryOptimizer:
    """Analyzer for BigQuery query optimization opportunities."""
    
    def __init__(self, metadata_extractor: Optional[MetadataExtractor] = None,
                project_id: Optional[str] = None,
                credentials_path: Optional[str] = None):
        """Initialize the query optimizer.
        
        Args:
            metadata_extractor: Optional existing MetadataExtractor instance
            project_id: GCP project ID if metadata_extractor not provided
            credentials_path: Path to service account credentials if metadata_extractor not provided
        """
        if metadata_extractor:
            self.metadata_extractor = metadata_extractor
            self.project_id = metadata_extractor.project_id
        else:
            if not project_id:
                raise ValueError("Either metadata_extractor or project_id must be provided")
            self.metadata_extractor = MetadataExtractor(project_id=project_id, credentials_path=credentials_path)
            self.project_id = project_id
        
        self.client = self.metadata_extractor.client
        self.connector = self.metadata_extractor.connector
        logger.info(f"Initialized QueryOptimizer for project {self.project_id}")
        
        # Cache for schema information
        self._schema_cache = {}
        # Cache for query parsing results
        self._query_parse_cache = {}
    
    def analyze_dataset_queries(self, dataset_id: str, days: int = DEFAULT_ANALYSIS_PERIOD_DAYS) -> Dict[str, Any]:
        """Analyze queries for a dataset and identify optimization opportunities.
        
        Args:
            dataset_id: BigQuery dataset ID
            days: Number of days of history to analyze
            
        Returns:
            Dict containing query optimization recommendations
        """
        logger.info(f"Analyzing query patterns for dataset {self.project_id}.{dataset_id} over {days} days")
        
        # Get dataset metadata to understand table structures
        dataset_metadata = self.metadata_extractor.extract_dataset_metadata(dataset_id)
        
        # Fetch query history for the dataset
        query_history = self._get_query_history(dataset_id, days)
        
        # Skip analysis if no queries found
        if not query_history or query_history.empty:
            logger.warning(f"No query history found for dataset {dataset_id} in the last {days} days")
            return {
                "dataset_id": dataset_id,
                "project_id": self.project_id,
                "queries_analyzed": 0,
                "total_bytes_processed": 0,
                "recommendations": [],
                "summary": {
                    "total_recommendations": 0,
                    "estimated_savings_bytes": 0,
                    "estimated_savings_percentage": 0,
                    "estimated_monthly_cost_savings": 0,
                    "estimated_annual_cost_savings": 0
                }
            }
        
        # Process the query history
        total_bytes_processed = query_history['total_bytes_processed'].sum()
        total_queries = len(query_history)
        
        logger.info(f"Found {total_queries} queries processing {total_bytes_processed/(1024**4):.2f} TB in the last {days} days")
        
        # Prepare tables information for reference
        tables_info = {table["table_id"]: table for table in dataset_metadata.get("tables", [])}
        
        # Analyze each query for optimization opportunities
        recommendations = []
        bytes_analyzed = 0
        
        # Focus on high-cost queries (sorted by bytes processed)
        high_cost_queries = query_history.sort_values('total_bytes_processed', ascending=False)
        
        if len(high_cost_queries) > MAX_QUERIES_TO_ANALYZE:
            logger.info(f"Limiting detailed analysis to the top {MAX_QUERIES_TO_ANALYZE} queries by bytes processed")
            high_cost_queries = high_cost_queries.head(MAX_QUERIES_TO_ANALYZE)
        
        for _, row in high_cost_queries.iterrows():
            query_text = row['query_text']
            bytes_processed = row['total_bytes_processed']
            
            # Skip small queries
            if bytes_processed < MIN_QUERY_BYTES_TO_ANALYZE:
                continue
                
            # Parse and analyze the query
            query_recs = self._analyze_single_query(
                query_text, 
                bytes_processed, 
                tables_info, 
                dataset_id,
                execution_count=row.get('execution_count', 1),
                user_email=row.get('user_email', 'unknown')
            )
            
            # Add recommendations
            recommendations.extend(query_recs)
            bytes_analyzed += bytes_processed
        
        # Calculate total savings
        total_savings_bytes = sum(rec.get("estimated_savings_bytes", 0) for rec in recommendations)
        savings_percentage = (total_savings_bytes / total_bytes_processed * 100) if total_bytes_processed > 0 else 0
        
        # Calculate cost savings based on $5 per TB processed
        monthly_savings = (total_savings_bytes / (1024**4)) * QUERY_COST_PER_TB
        annual_savings = monthly_savings * 12
        
        # Normalize recommendations to avoid duplicates and prioritize
        prioritized_recs = self._prioritize_recommendations(recommendations)
        
        # Build result
        result = {
            "dataset_id": dataset_id,
            "project_id": self.project_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "queries_analyzed": total_queries,
            "total_bytes_processed": total_bytes_processed,
            "bytes_analyzed": bytes_analyzed,
            "recommendations": prioritized_recs,
            "summary": {
                "total_recommendations": len(prioritized_recs),
                "total_patterns_detected": len(recommendations),
                "estimated_savings_bytes": total_savings_bytes,
                "estimated_savings_percentage": savings_percentage,
                "estimated_monthly_cost_savings": monthly_savings,
                "estimated_annual_cost_savings": annual_savings
            }
        }
        
        return result
    
    def analyze_table_queries(self, dataset_id: str, table_id: str, days: int = DEFAULT_ANALYSIS_PERIOD_DAYS) -> Dict[str, Any]:
        """Analyze queries for a specific table and identify optimization opportunities.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            days: Number of days of history to analyze
            
        Returns:
            Dict containing query optimization recommendations for the table
        """
        logger.info(f"Analyzing query patterns for table {self.project_id}.{dataset_id}.{table_id} over {days} days")
        
        # Get table metadata
        table_metadata = self.metadata_extractor.extract_table_metadata(dataset_id, table_id)
        
        # Fetch query history for the table
        query_history = self._get_table_query_history(dataset_id, table_id, days)
        
        # Skip analysis if no queries found
        if not query_history or query_history.empty:
            logger.warning(f"No query history found for table {table_id} in the last {days} days")
            return {
                "table_id": table_id,
                "dataset_id": dataset_id,
                "project_id": self.project_id,
                "queries_analyzed": 0,
                "total_bytes_processed": 0,
                "recommendations": [],
                "summary": {
                    "total_recommendations": 0,
                    "estimated_savings_bytes": 0,
                    "estimated_savings_percentage": 0,
                    "estimated_monthly_cost_savings": 0,
                    "estimated_annual_cost_savings": 0
                }
            }
        
        # Process the query history
        total_bytes_processed = query_history['total_bytes_processed'].sum()
        total_queries = len(query_history)
        
        logger.info(f"Found {total_queries} queries processing {total_bytes_processed/(1024**4):.2f} TB for table {table_id} in the last {days} days")
        
        # Prepare tables information for reference
        tables_info = {table_id: table_metadata}
        
        # Analyze each query for optimization opportunities
        recommendations = []
        bytes_analyzed = 0
        
        # Focus on high-cost queries (sorted by bytes processed)
        high_cost_queries = query_history.sort_values('total_bytes_processed', ascending=False)
        
        if len(high_cost_queries) > MAX_QUERIES_TO_ANALYZE:
            logger.info(f"Limiting detailed analysis to the top {MAX_QUERIES_TO_ANALYZE} queries by bytes processed")
            high_cost_queries = high_cost_queries.head(MAX_QUERIES_TO_ANALYZE)
        
        for _, row in high_cost_queries.iterrows():
            query_text = row['query_text']
            bytes_processed = row['total_bytes_processed']
            
            # Skip small queries
            if bytes_processed < MIN_QUERY_BYTES_TO_ANALYZE:
                continue
                
            # Parse and analyze the query
            query_recs = self._analyze_single_query(
                query_text, 
                bytes_processed, 
                tables_info, 
                dataset_id,
                execution_count=row.get('execution_count', 1),
                user_email=row.get('user_email', 'unknown')
            )
            
            # Add recommendations
            recommendations.extend(query_recs)
            bytes_analyzed += bytes_processed
        
        # Calculate total savings
        total_savings_bytes = sum(rec.get("estimated_savings_bytes", 0) for rec in recommendations)
        savings_percentage = (total_savings_bytes / total_bytes_processed * 100) if total_bytes_processed > 0 else 0
        
        # Calculate cost savings based on $5 per TB processed
        monthly_savings = (total_savings_bytes / (1024**4)) * QUERY_COST_PER_TB
        annual_savings = monthly_savings * 12
        
        # Normalize recommendations to avoid duplicates and prioritize
        prioritized_recs = self._prioritize_recommendations(recommendations)
        
        # Build result
        result = {
            "table_id": table_id,
            "dataset_id": dataset_id,
            "project_id": self.project_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "queries_analyzed": total_queries,
            "total_bytes_processed": total_bytes_processed,
            "bytes_analyzed": bytes_analyzed,
            "recommendations": prioritized_recs,
            "summary": {
                "total_recommendations": len(prioritized_recs),
                "total_patterns_detected": len(recommendations),
                "estimated_savings_bytes": total_savings_bytes,
                "estimated_savings_percentage": savings_percentage,
                "estimated_monthly_cost_savings": monthly_savings,
                "estimated_annual_cost_savings": annual_savings
            }
        }
        
        return result
    
    def analyze_query_text(self, query_text: str, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a single query text for optimization opportunities.
        
        Args:
            query_text: SQL query text to analyze
            dataset_id: Optional dataset ID for context (helps with schema lookups)
            
        Returns:
            Dict containing optimization recommendations
        """
        logger.info("Analyzing provided query text")
        
        # Guess the dataset if not provided
        if not dataset_id:
            # Try to extract dataset from query
            dataset_match = re.search(r'`?(\w+[-\w]*)\.(\w+[-\w]*)`?\.', query_text)
            if dataset_match and dataset_match.group(1) == self.project_id:
                dataset_id = dataset_match.group(2)
                logger.info(f"Extracted dataset ID from query: {dataset_id}")
            else:
                logger.warning("Could not determine dataset ID from query; schema information may be limited")
                
        # Get dataset metadata if possible
        tables_info = {}
        if dataset_id:
            try:
                dataset_metadata = self.metadata_extractor.extract_dataset_metadata(dataset_id)
                tables_info = {table["table_id"]: table for table in dataset_metadata.get("tables", [])}
            except Exception as e:
                logger.warning(f"Could not fetch dataset metadata: {e}")
        
        # Assume a reasonable byte processing amount for estimation purposes
        estimated_bytes = 10 * (1024**3)  # 10 GB as a default assumption
        
        # Analyze the query
        recommendations = self._analyze_single_query(
            query_text, 
            estimated_bytes, 
            tables_info, 
            dataset_id or "unknown"
        )
        
        # Calculate total savings
        total_savings_bytes = sum(rec.get("estimated_savings_bytes", 0) for rec in recommendations)
        savings_percentage = (total_savings_bytes / estimated_bytes * 100) if estimated_bytes > 0 else 0
        
        # Build result
        result = {
            "query_text": query_text,
            "dataset_id": dataset_id,
            "project_id": self.project_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "estimated_bytes_processed": estimated_bytes,
            "recommendations": recommendations,
            "summary": {
                "total_recommendations": len(recommendations),
                "estimated_savings_bytes": total_savings_bytes,
                "estimated_savings_percentage": savings_percentage,
                "estimated_cost_savings": (total_savings_bytes / (1024**4)) * QUERY_COST_PER_TB
            }
        }
        
        return result
    
    def _get_query_history(self, dataset_id: str, days: int) -> pd.DataFrame:
        """Fetch query history for a dataset from INFORMATION_SCHEMA.
        
        Args:
            dataset_id: BigQuery dataset ID
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame containing query history
        """
        query = f"""
        WITH query_stats AS (
            SELECT
                query_text,
                total_bytes_processed,
                total_slot_ms,
                user_email,
                creation_time,
                CONCAT(project_id, '.', dataset_id, '.', table_name) AS source_table,
                REGEXP_CONTAINS(LOWER(query_text), r'select\\s+\\*') AS has_select_star,
                REGEXP_CONTAINS(LOWER(query_text), r'where') AS has_where_clause,
                cache_hit
            FROM
                `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS J,
                UNNEST(referenced_tables) AS T
            WHERE
                creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
                AND job_type = 'QUERY'
                AND state = 'DONE'
                AND error_result IS NULL
                AND dataset_id = '{dataset_id}'
                AND statement_type = 'SELECT'
        ),
        query_groups AS (
            SELECT
                TRIM(REGEXP_REPLACE(query_text, r'[0-9]+', '?')) AS normalized_query,
                COUNT(*) AS execution_count,
                SUM(total_bytes_processed) AS total_bytes_processed,
                MAX(total_bytes_processed) AS max_bytes_processed,
                AVG(total_bytes_processed) AS avg_bytes_processed,
                MAX(creation_time) AS last_execution,
                ARRAY_AGG(DISTINCT user_email IGNORE NULLS) AS users,
                COUNTIF(has_select_star) > 0 AS has_select_star,
                COUNTIF(has_where_clause) = 0 AS missing_where_clause,
                COUNTIF(cache_hit) / COUNT(*) AS cache_hit_ratio,
                ANY_VALUE(query_text) AS query_text
            FROM
                query_stats
            WHERE
                total_bytes_processed > 0
            GROUP BY
                normalized_query
        )
        SELECT
            query_text,
            execution_count,
            total_bytes_processed,
            max_bytes_processed,
            avg_bytes_processed,
            last_execution,
            (SELECT STRING_AGG(u, ', ') FROM UNNEST(users) AS u) AS user_emails,
            has_select_star,
            missing_where_clause,
            cache_hit_ratio
        FROM
            query_groups
        ORDER BY
            total_bytes_processed DESC
        """
        
        try:
            df = self.connector.query_to_dataframe(query)
            return df
        except Exception as e:
            logger.warning(f"Error retrieving query history: {e}")
            return pd.DataFrame()
    
    def _get_table_query_history(self, dataset_id: str, table_id: str, days: int) -> pd.DataFrame:
        """Fetch query history for a specific table from INFORMATION_SCHEMA.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame containing query history
        """
        query = f"""
        WITH query_stats AS (
            SELECT
                query_text,
                total_bytes_processed,
                total_slot_ms,
                user_email,
                creation_time,
                REGEXP_CONTAINS(LOWER(query_text), r'select\\s+\\*') AS has_select_star,
                REGEXP_CONTAINS(LOWER(query_text), r'where') AS has_where_clause,
                cache_hit
            FROM
                `{self.project_id}.region-us`.INFORMATION_SCHEMA.JOBS J,
                UNNEST(referenced_tables) AS T
            WHERE
                creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) AND CURRENT_TIMESTAMP()
                AND job_type = 'QUERY'
                AND state = 'DONE'
                AND error_result IS NULL
                AND dataset_id = '{dataset_id}'
                AND table_name = '{table_id}'
                AND statement_type = 'SELECT'
        ),
        query_groups AS (
            SELECT
                TRIM(REGEXP_REPLACE(query_text, r'[0-9]+', '?')) AS normalized_query,
                COUNT(*) AS execution_count,
                SUM(total_bytes_processed) AS total_bytes_processed,
                MAX(total_bytes_processed) AS max_bytes_processed,
                AVG(total_bytes_processed) AS avg_bytes_processed,
                MAX(creation_time) AS last_execution,
                ARRAY_AGG(DISTINCT user_email IGNORE NULLS) AS users,
                COUNTIF(has_select_star) > 0 AS has_select_star,
                COUNTIF(has_where_clause) = 0 AS missing_where_clause,
                COUNTIF(cache_hit) / COUNT(*) AS cache_hit_ratio,
                ANY_VALUE(query_text) AS query_text
            FROM
                query_stats
            WHERE
                total_bytes_processed > 0
            GROUP BY
                normalized_query
        )
        SELECT
            query_text,
            execution_count,
            total_bytes_processed,
            max_bytes_processed,
            avg_bytes_processed,
            last_execution,
            (SELECT STRING_AGG(u, ', ') FROM UNNEST(users) AS u) AS user_emails,
            has_select_star,
            missing_where_clause,
            cache_hit_ratio
        FROM
            query_groups
        ORDER BY
            total_bytes_processed DESC
        """
        
        try:
            df = self.connector.query_to_dataframe(query)
            return df
        except Exception as e:
            logger.warning(f"Error retrieving table query history: {e}")
            return pd.DataFrame()
    
    def _analyze_single_query(self, query_text: str, bytes_processed: int, 
                            tables_info: Dict[str, Dict[str, Any]], dataset_id: str,
                            execution_count: int = 1, user_email: str = 'unknown') -> List[Dict[str, Any]]:
        """Analyze a single query for optimization opportunities.
        
        Args:
            query_text: SQL query text
            bytes_processed: Bytes processed by the query
            tables_info: Dict mapping table IDs to their metadata
            dataset_id: BigQuery dataset ID
            execution_count: How many times the query has been executed
            user_email: Email of the user who ran the query
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Parse the query
        try:
            # Use cache to avoid repeated parsing
            if query_text in self._query_parse_cache:
                parsed_query = self._query_parse_cache[query_text]
            else:
                parsed_query = sqlparse.parse(query_text)
                if parsed_query:
                    self._query_parse_cache[query_text] = parsed_query
                else:
                    logger.warning(f"Failed to parse query: {query_text[:100]}...")
                    return recommendations
                    
            # Skip non-SELECT statements
            stmt = parsed_query[0]
            if stmt.get_type() != 'SELECT':
                return recommendations
            
            # Extract query components
            query_info = self._extract_query_components(stmt, query_text)
            
            # Check for various optimization opportunities
            select_star_rec = self._check_select_star(query_info, tables_info, bytes_processed, dataset_id)
            if select_star_rec:
                recommendations.append(select_star_rec)
                
            missing_partition_filter_rec = self._check_missing_partition_filter(query_info, tables_info, bytes_processed, dataset_id)
            if missing_partition_filter_rec:
                recommendations.append(missing_partition_filter_rec)
                
            inefficient_join_recs = self._check_inefficient_joins(query_info, tables_info, bytes_processed, dataset_id)
            recommendations.extend(inefficient_join_recs)
            
            subquery_recs = self._check_inefficient_subqueries(query_info, tables_info, bytes_processed, dataset_id)
            recommendations.extend(subquery_recs)
            
            aggregation_recs = self._check_missing_aggregations(query_info, tables_info, bytes_processed, dataset_id)
            recommendations.extend(aggregation_recs)
            
            # Add query metadata to all recommendations
            for rec in recommendations:
                rec.update({
                    "query_text": query_text,
                    "bytes_processed": bytes_processed,
                    "execution_count": execution_count,
                    "user_email": user_email
                })
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error analyzing query: {e}")
            return recommendations
    
    def _extract_query_components(self, parsed_query, query_text: str) -> Dict[str, Any]:
        """Extract key components from a parsed SQL query.
        
        Args:
            parsed_query: SQLParse parsed query object
            query_text: Original query text
            
        Returns:
            Dict with query components
        """
        # Initialize result
        components = {
            "select_items": [],
            "from_tables": [],
            "join_conditions": [],
            "where_conditions": [],
            "group_by_items": [],
            "order_by_items": [],
            "limit": None,
            "has_select_star": False,
            "has_where_clause": False,
            "has_group_by": False,
            "has_order_by": False,
            "has_limit": False,
            "subqueries": []
        }
        
        # Extract SELECT items
        select_idx = None
        for i, token in enumerate(parsed_query.tokens):
            if token.ttype is DML and token.value.upper() == 'SELECT':
                select_idx = i
                break
                
        if select_idx is not None and select_idx + 1 < len(parsed_query.tokens):
            # Look for the items in the SELECT clause
            next_token = parsed_query.tokens[select_idx + 1]
            if next_token.ttype is Whitespace:
                next_token = parsed_query.tokens[select_idx + 2] if select_idx + 2 < len(parsed_query.tokens) else None
            
            if next_token:
                # Check for SELECT *
                if isinstance(next_token, IdentifierList):
                    for identifier in next_token.get_identifiers():
                        if identifier.value == '*':
                            components["has_select_star"] = True
                        components["select_items"].append(str(identifier).strip())
                elif isinstance(next_token, Identifier):
                    if next_token.value == '*':
                        components["has_select_star"] = True
                    components["select_items"].append(str(next_token).strip())
                elif str(next_token).strip() == '*':
                    components["has_select_star"] = True
                    components["select_items"].append('*')
        
        # Find FROM clause tables
        from_seen = False
        join_seen = False
        where_seen = False
        group_by_seen = False
        order_by_seen = False
        
        for token in parsed_query.tokens:
            # FROM clause
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
                continue
                
            if from_seen and not join_seen and not where_seen:
                if isinstance(token, Identifier):
                    components["from_tables"].append(str(token).strip())
                    from_seen = False
                elif isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        components["from_tables"].append(str(identifier).strip())
                    from_seen = False
                    
            # JOIN clause
            if token.ttype is Keyword and token.value.upper() in ('JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'FULL JOIN', 'CROSS JOIN'):
                join_seen = True
                
                # Extract the table being joined
                table_token = None
                for t in token.parent.tokens[token.parent.tokens.index(token)+1:]:
                    if not t.ttype is Whitespace and not t.is_whitespace:
                        table_token = t
                        break
                        
                if table_token and isinstance(table_token, Identifier):
                    components["from_tables"].append(str(table_token).strip())
                    
                # Look for JOIN conditions
                on_token = None
                for t in token.parent.tokens[token.parent.tokens.index(token):]:
                    if t.ttype is Keyword and t.value.upper() == 'ON':
                        on_token = t
                        break
                        
                if on_token:
                    condition_tokens = []
                    for t in token.parent.tokens[token.parent.tokens.index(on_token)+1:]:
                        if t.ttype is Keyword and t.value.upper() in ('JOIN', 'WHERE', 'GROUP', 'ORDER', 'LIMIT'):
                            break
                        condition_tokens.append(t)
                    
                    join_condition = ''.join(str(t) for t in condition_tokens).strip()
                    if join_condition:
                        components["join_conditions"].append(join_condition)
            
            # WHERE clause
            if token.ttype is Keyword and token.value.upper() == 'WHERE':
                where_seen = True
                join_seen = False
                components["has_where_clause"] = True
                
                # Extract WHERE conditions
                condition_tokens = []
                for t in token.parent.tokens[token.parent.tokens.index(token)+1:]:
                    if t.ttype is Keyword and t.value.upper() in ('GROUP', 'ORDER', 'LIMIT'):
                        break
                    condition_tokens.append(t)
                
                where_condition = ''.join(str(t) for t in condition_tokens).strip()
                if where_condition:
                    components["where_conditions"].append(where_condition)
            
            # GROUP BY clause
            if token.ttype is Keyword and token.value.upper() == 'GROUP':
                group_by_seen = True
                where_seen = False
                components["has_group_by"] = True
                
                # Look for the BY part
                by_token = None
                for t in token.parent.tokens[token.parent.tokens.index(token)+1:]:
                    if t.ttype is Keyword and t.value.upper() == 'BY':
                        by_token = t
                        break
                
                if by_token:
                    groupby_tokens = []
                    for t in token.parent.tokens[token.parent.tokens.index(by_token)+1:]:
                        if t.ttype is Keyword and t.value.upper() in ('ORDER', 'LIMIT'):
                            break
                        groupby_tokens.append(t)
                    
                    groupby_text = ''.join(str(t) for t in groupby_tokens).strip()
                    if groupby_text:
                        # Split by commas to get individual group by items
                        for item in groupby_text.split(','):
                            components["group_by_items"].append(item.strip())
            
            # ORDER BY clause
            if token.ttype is Keyword and (token.value.upper() == 'ORDER' or (token.value.upper() == 'ORDER BY')):
                order_by_seen = True
                group_by_seen = False
                components["has_order_by"] = True
                
                # Handle 'ORDER BY' as single token or 'ORDER' followed by 'BY'
                if token.value.upper() == 'ORDER':
                    # Look for the BY part
                    by_token = None
                    for t in token.parent.tokens[token.parent.tokens.index(token)+1:]:
                        if t.ttype is Keyword and t.value.upper() == 'BY':
                            by_token = t
                            break
                    
                    if by_token:
                        start_idx = token.parent.tokens.index(by_token) + 1
                    else:
                        continue
                else:
                    # 'ORDER BY' as a single token
                    start_idx = token.parent.tokens.index(token) + 1
                
                # Extract ORDER BY items
                orderby_tokens = []
                for t in token.parent.tokens[start_idx:]:
                    if t.ttype is Keyword and t.value.upper() == 'LIMIT':
                        break
                    orderby_tokens.append(t)
                
                orderby_text = ''.join(str(t) for t in orderby_tokens).strip()
                if orderby_text:
                    # Split by commas to get individual order by items
                    for item in orderby_text.split(','):
                        components["order_by_items"].append(item.strip())
            
            # LIMIT clause
            if token.ttype is Keyword and token.value.upper() == 'LIMIT':
                components["has_limit"] = True
                
                # Extract the LIMIT value
                limit_tokens = []
                for t in token.parent.tokens[token.parent.tokens.index(token)+1:]:
                    if t.ttype is Keyword:
                        break
                    limit_tokens.append(t)
                
                limit_text = ''.join(str(t) for t in limit_tokens).strip()
                if limit_text:
                    try:
                        components["limit"] = int(limit_text)
                    except ValueError:
                        components["limit"] = limit_text
            
            # Look for subqueries
            if isinstance(token, Parenthesis):
                # Check if this might be a subquery by looking for SELECT keyword
                subquery_text = str(token).strip()
                if 'SELECT' in subquery_text.upper():
                    components["subqueries"].append(subquery_text)
        
        return components
    
    def _check_select_star(self, query_info: Dict[str, Any], tables_info: Dict[str, Dict[str, Any]], 
                         bytes_processed: int, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Check for SELECT * anti-pattern and suggest column pruning.
        
        Args:
            query_info: Extracted query components
            tables_info: Dict mapping table IDs to their metadata
            bytes_processed: Bytes processed by the query
            dataset_id: BigQuery dataset ID
            
        Returns:
            Dict with recommendation if applicable, otherwise None
        """
        if not query_info["has_select_star"]:
            return None
            
        # Check if we have any matching tables and their schemas
        matched_tables = []
        for table_ref in query_info["from_tables"]:
            # Extract table name from references like 'project.dataset.table' or 'table'
            table_parts = table_ref.replace('`', '').split('.')
            table_name = table_parts[-1]
            
            if table_name in tables_info:
                matched_tables.append((table_name, tables_info[table_name]))
        
        if not matched_tables:
            return None
            
        # Create recommendation for column pruning
        optimized_query = None
        relevant_columns = []
        
        for table_name, table_info in matched_tables:
            if "schema" in table_info:
                # Get columns from the table's schema
                if not relevant_columns:
                    # First table, include all non-nested columns as candidates
                    relevant_columns = [
                        field["name"] for field in table_info["schema"]
                        if "type" in field and field["type"] not in ("STRUCT", "ARRAY")
                    ]
                    
        # Check if we could determine appropriate columns
        if not relevant_columns:
            return None
        
        # Limit to a reasonable number (first 10)
        display_columns = relevant_columns[:10]
        if len(relevant_columns) > 10:
            display_columns.append("... [additional columns truncated]")
        
        # Estimate savings: We can assume 50% reduction if we only select needed columns
        # This is a conservative estimate as most queries only need a subset of columns
        estimated_savings_bytes = int(bytes_processed * 0.5)
        
        # Generate optimized query
        try:
            original_query_text = query_info.get("query_text", "")
            optimized_query = original_query_text.replace("SELECT *", f"SELECT {', '.join(display_columns[:5])}")
        except:
            optimized_query = f"SELECT {', '.join(display_columns[:5])} FROM ... [rest of your query]"
        
        # Create the recommendation
        recommendation = {
            "type": "select_star",
            "description": "Replace SELECT * with specific columns",
            "rationale": f"The query is reading all columns from the table, but may only need a subset. By selecting only the required columns, you can reduce the amount of data processed and improve query performance.",
            "estimated_savings_bytes": estimated_savings_bytes,
            "estimated_savings_percentage": 50.0,  # Conservative estimate
            "estimated_cost_savings": (estimated_savings_bytes / (1024**4)) * QUERY_COST_PER_TB,
            "implementation_difficulty": "low",
            "before": "SELECT * FROM ...",
            "after": f"SELECT {', '.join(display_columns[:5])} FROM ...",
            "improvement": "Selects only necessary columns instead of all columns",
            "available_columns": relevant_columns,
            "optimized_query": optimized_query,
            "priority_score": 80  # High priority due to ease of implementation and common occurrence
        }
        
        return recommendation
    
    def _check_missing_partition_filter(self, query_info: Dict[str, Any], tables_info: Dict[str, Dict[str, Any]], 
                                     bytes_processed: int, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Check for missing partition filters in the query.
        
        Args:
            query_info: Extracted query components
            tables_info: Dict mapping table IDs to their metadata
            bytes_processed: Bytes processed by the query
            dataset_id: BigQuery dataset ID
            
        Returns:
            Dict with recommendation if applicable, otherwise None
        """
        # Check if we have any partitioned tables in the query
        partitioned_tables = []
        for table_ref in query_info["from_tables"]:
            # Extract table name from references like 'project.dataset.table' or 'table'
            table_parts = table_ref.replace('`', '').split('.')
            table_name = table_parts[-1]
            
            if table_name in tables_info and "partitioning" in tables_info[table_name] and tables_info[table_name]["partitioning"]:
                partitioned_tables.append((table_name, tables_info[table_name]))
        
        if not partitioned_tables:
            return None
            
        # Check if partition fields are used in WHERE conditions
        where_conditions = ' '.join(query_info["where_conditions"]).lower()
        missing_filters = []
        
        for table_name, table_info in partitioned_tables:
            partition_field = table_info["partitioning"].get("field")
            
            if partition_field and partition_field.lower() not in where_conditions:
                missing_filters.append((table_name, partition_field, table_info))
        
        if not missing_filters:
            return None
            
        # Create recommendation for adding partition filters
        table_name, partition_field, table_info = missing_filters[0]
        partition_type = table_info["partitioning"].get("type", "")
        
        # Suggest appropriate filter based on partition type
        suggested_filter = ""
        if partition_type in ("DAY", "MONTH", "YEAR") or "DATE" in str(table_info["partitioning"]).upper():
            # Time-based partitioning
            suggested_filter = f"{partition_field} >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)"
        else:
            # Integer range partitioning
            suggested_filter = f"{partition_field} BETWEEN 1000 AND 2000"  # Generic example
        
        # Estimate savings: partition filters can typically reduce data by 90%+
        # Be conservative and assume 80% reduction
        estimated_savings_bytes = int(bytes_processed * 0.8)
        
        # Generate optimized query example
        optimized_query = None
        if query_info["has_where_clause"]:
            # Add to existing WHERE clause
            optimized_query = f"... WHERE ... AND {suggested_filter} ..."
        else:
            # Add new WHERE clause
            optimized_query = f"... FROM {table_name} WHERE {suggested_filter} ..."
        
        # Create the recommendation
        recommendation = {
            "type": "missing_partition_filter",
            "description": f"Add partition filter on {partition_field}",
            "rationale": f"The query accesses partitioned table '{table_name}' without a filter on the partition column '{partition_field}'. Adding a filter will significantly reduce the amount of data scanned and improve query performance.",
            "estimated_savings_bytes": estimated_savings_bytes,
            "estimated_savings_percentage": 80.0,
            "estimated_cost_savings": (estimated_savings_bytes / (1024**4)) * QUERY_COST_PER_TB,
            "implementation_difficulty": "low",
            "before": f"... FROM {table_name} ...",
            "after": f"... FROM {table_name} WHERE {suggested_filter} ...",
            "improvement": "Restricts query to specific partitions instead of scanning the entire table",
            "partition_field": partition_field,
            "partition_type": partition_type,
            "suggested_filter": suggested_filter,
            "optimized_query": optimized_query,
            "priority_score": 90  # Very high priority due to large potential savings
        }
        
        return recommendation
    
    def _check_inefficient_joins(self, query_info: Dict[str, Any], tables_info: Dict[str, Dict[str, Any]], 
                               bytes_processed: int, dataset_id: str) -> List[Dict[str, Any]]:
        """Check for inefficient JOIN conditions in the query.
        
        Args:
            query_info: Extracted query components
            tables_info: Dict mapping table IDs to their metadata
            bytes_processed: Bytes processed by the query
            dataset_id: BigQuery dataset ID
            
        Returns:
            List of recommendations for JOIN optimizations
        """
        recommendations = []
        
        # Check if we have JOINs but no JOIN conditions (potential Cartesian product)
        if len(query_info["from_tables"]) > 1 and not query_info["join_conditions"]:
            # This could be a CROSS JOIN or missing JOIN conditions
            
            # Estimate savings: JOIN conditions can typically reduce data by 90%+
            # when avoiding a Cartesian product
            estimated_savings_bytes = int(bytes_processed * 0.9)
            
            recommendation = {
                "type": "cartesian_join",
                "description": "Add JOIN conditions to avoid Cartesian product",
                "rationale": "The query appears to join tables without explicit JOIN conditions, which may result in a Cartesian product (every row from first table joined with every row from second table). This is highly inefficient and can cause exponential data growth.",
                "estimated_savings_bytes": estimated_savings_bytes,
                "estimated_savings_percentage": 90.0,
                "estimated_cost_savings": (estimated_savings_bytes / (1024**4)) * QUERY_COST_PER_TB,
                "implementation_difficulty": "medium",
                "before": f"... FROM {', '.join(query_info['from_tables'])} ...",
                "after": f"... FROM {query_info['from_tables'][0]} JOIN {query_info['from_tables'][1]} ON {query_info['from_tables'][0]}.id = {query_info['from_tables'][1]}.id ...",
                "improvement": "Adds JOIN conditions to limit the combined result set to only rows that match the condition",
                "tables_joined": query_info["from_tables"],
                "priority_score": 95  # Extremely high priority due to exponential growth with Cartesian products
            }
            
            recommendations.append(recommendation)
        
        # Check join conditions - if tables are joined on non-indexed or high-cardinality columns
        # This would require schema information, which we may have for some tables
        if query_info["join_conditions"]:
            join_conditions = ' '.join(query_info["join_conditions"]).lower()
            
            # Look for potential joins on non-optimal columns
            for table_name, table_info in tables_info.items():
                # Skip tables not in the FROM clause
                if not any(table_name.lower() in table_ref.lower() for table_ref in query_info["from_tables"]):
                    continue
                    
                if "schema" in table_info and "clustering" in table_info and table_info["clustering"]:
                    # Table is clustered, check if clustering fields are used in joins
                    clustering_fields = table_info["clustering"].get("fields", [])
                    
                    # If any clustering field is not used in joins, suggest it
                    for field in clustering_fields:
                        field_pattern = r'\b' + re.escape(field.lower()) + r'\b'
                        if not re.search(field_pattern, join_conditions):
                            # Estimate savings: using clustering fields can improve JOIN performance by 50%
                            estimated_savings_bytes = int(bytes_processed * 0.5)
                            
                            recommendation = {
                                "type": "join_on_clustered_column",
                                "description": f"Use clustered column '{field}' in JOIN conditions",
                                "rationale": f"Table '{table_name}' is clustered on column '{field}' but this column is not used in JOIN conditions. Using clustered columns in JOINs can significantly improve performance by reducing data shuffling.",
                                "estimated_savings_bytes": estimated_savings_bytes,
                                "estimated_savings_percentage": 50.0,
                                "estimated_cost_savings": (estimated_savings_bytes / (1024**4)) * QUERY_COST_PER_TB,
                                "implementation_difficulty": "medium",
                                "before": "... JOIN ... ON other_condition ...",
                                "after": f"... JOIN ... ON {table_name}.{field} = other_table.{field} ...",
                                "improvement": "Uses clustered columns for JOINs which reduces data shuffling and improves performance",
                                "table_name": table_name,
                                "clustered_field": field,
                                "priority_score": 75  # High priority but requires schema changes potentially
                            }
                            
                            recommendations.append(recommendation)
        
        return recommendations
    
    def _check_inefficient_subqueries(self, query_info: Dict[str, Any], tables_info: Dict[str, Dict[str, Any]], 
                                    bytes_processed: int, dataset_id: str) -> List[Dict[str, Any]]:
        """Check for inefficient subqueries that could be optimized.
        
        Args:
            query_info: Extracted query components
            tables_info: Dict mapping table IDs to their metadata
            bytes_processed: Bytes processed by the query
            dataset_id: BigQuery dataset ID
            
        Returns:
            List of recommendations for subquery optimizations
        """
        recommendations = []
        
        # Check for correlated subqueries (indicated by references to outer query tables)
        subqueries = query_info.get("subqueries", [])
        if not subqueries:
            return recommendations
            
        # Look for subqueries in the SELECT clause (potential candidates for JOINs)
        select_items = query_info.get("select_items", [])
        subquery_in_select = any("select" in item.lower() for item in select_items)
        
        if subquery_in_select:
            # Estimate savings: converting subqueries to JOINs can save 30-50%
            estimated_savings_bytes = int(bytes_processed * 0.4)
            
            recommendation = {
                "type": "subquery_to_join",
                "description": "Convert subquery to JOIN",
                "rationale": "The query uses a subquery in the SELECT clause, which can be inefficient. Converting subqueries to JOINs often improves performance by allowing the query engine to optimize the execution plan better.",
                "estimated_savings_bytes": estimated_savings_bytes,
                "estimated_savings_percentage": 40.0,
                "estimated_cost_savings": (estimated_savings_bytes / (1024**4)) * QUERY_COST_PER_TB,
                "implementation_difficulty": "medium",
                "before": "SELECT col1, (SELECT value FROM table2 WHERE table2.id = table1.id) FROM table1",
                "after": "SELECT col1, table2.value FROM table1 JOIN table2 ON table2.id = table1.id",
                "improvement": "Converts a potentially inefficient subquery to an explicit JOIN",
                "priority_score": 70  # Medium-high priority, requires careful rewriting
            }
            
            recommendations.append(recommendation)
        
        # Look for repeated subqueries that could be common table expressions (CTEs)
        if len(subqueries) > 1:
            # Check for similar subqueries
            normalized_subqueries = [re.sub(r'\b\d+\b', '?', sq.lower()) for sq in subqueries]
            
            if len(set(normalized_subqueries)) < len(normalized_subqueries):
                # There are potential duplicate subqueries
                estimated_savings_bytes = int(bytes_processed * 0.3)
                
                recommendation = {
                    "type": "repeated_subquery_to_cte",
                    "description": "Use CTEs for repeated subqueries",
                    "rationale": "The query appears to have repeated or similar subqueries. Using Common Table Expressions (CTEs) with WITH clause can improve readability and performance by computing the result once and reusing it.",
                    "estimated_savings_bytes": estimated_savings_bytes,
                    "estimated_savings_percentage": 30.0,
                    "estimated_cost_savings": (estimated_savings_bytes / (1024**4)) * QUERY_COST_PER_TB,
                    "implementation_difficulty": "medium",
                    "before": "SELECT * FROM (subquery) WHERE ... (same subquery) ...",
                    "after": "WITH cte AS (subquery) SELECT * FROM cte WHERE ... cte ...",
                    "improvement": "Computes the subquery result once and reuses it with a CTE",
                    "priority_score": 65  # Medium priority, good optimization with moderate impact
                }
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _check_missing_aggregations(self, query_info: Dict[str, Any], tables_info: Dict[str, Dict[str, Any]], 
                                  bytes_processed: int, dataset_id: str) -> List[Dict[str, Any]]:
        """Check for queries that could benefit from pre-aggregation or earlier filtering.
        
        Args:
            query_info: Extracted query components
            tables_info: Dict mapping table IDs to their metadata
            bytes_processed: Bytes processed by the query
            dataset_id: BigQuery dataset ID
            
        Returns:
            List of recommendations for aggregation and filtering optimizations
        """
        recommendations = []
        
        # Check if query has GROUP BY and aggregates a large dataset
        if query_info["has_group_by"] and bytes_processed > 5 * (1024**3):  # > 5GB
            # May benefit from materialized view
            estimated_savings_bytes = int(bytes_processed * 0.7)  # Materialized views can save 70%+ for repeated queries
            
            recommendation = {
                "type": "materialized_view",
                "description": "Create materialized view for common aggregation",
                "rationale": "The query performs aggregation on a large dataset. If this aggregation is run frequently, creating a materialized view can significantly reduce query costs and improve performance.",
                "estimated_savings_bytes": estimated_savings_bytes,
                "estimated_savings_percentage": 70.0,
                "estimated_cost_savings": (estimated_savings_bytes / (1024**4)) * QUERY_COST_PER_TB,
                "implementation_difficulty": "medium",
                "before": "SELECT field1, SUM(value) FROM table GROUP BY field1",
                "after": "CREATE MATERIALIZED VIEW dataset.view AS SELECT field1, SUM(value) AS total FROM table GROUP BY field1",
                "improvement": "Pre-computes and stores the aggregation results, avoiding repeated computation",
                "priority_score": 75  # High priority for frequently run queries
            }
            
            recommendations.append(recommendation)
        
        # Check for queries without aggregations that return large result sets
        if not query_info["has_group_by"] and not query_info["has_limit"]:
            # This query might return a large result set without aggregation or limits
            # Suggest adding LIMIT
            estimated_savings_bytes = int(bytes_processed * 0.5)  # Adding LIMIT can save 50%+ in many cases
            
            recommendation = {
                "type": "missing_limit",
                "description": "Add LIMIT clause to reduce result set size",
                "rationale": "The query doesn't limit the number of results returned, which could lead to large result sets being processed and transferred. If you don't need all results, adding a LIMIT clause can reduce costs.",
                "estimated_savings_bytes": estimated_savings_bytes,
                "estimated_savings_percentage": 50.0,
                "estimated_cost_savings": (estimated_savings_bytes / (1024**4)) * QUERY_COST_PER_TB,
                "implementation_difficulty": "low",
                "before": "SELECT fields FROM table WHERE condition",
                "after": "SELECT fields FROM table WHERE condition LIMIT 1000",
                "improvement": "Limits the result set size, reducing processing and data transfer costs",
                "priority_score": 60  # Medium priority, easy to implement but may change functionality
            }
            
            recommendations.append(recommendation)
            
        # Check if ORDER BY is used without LIMIT
        if query_info["has_order_by"] and not query_info["has_limit"]:
            # Ordering without limiting is often unnecessary overhead
            estimated_savings_bytes = int(bytes_processed * 0.2)  # Removing unnecessary ORDER BY can save 20%
            
            recommendation = {
                "type": "unnecessary_order_by",
                "description": "Remove ORDER BY or add LIMIT",
                "rationale": "The query uses ORDER BY without LIMIT, which means all results must be sorted even though the entire sorted result set is returned. This is inefficient unless sorting is specifically needed by the client.",
                "estimated_savings_bytes": estimated_savings_bytes,
                "estimated_savings_percentage": 20.0,
                "estimated_cost_savings": (estimated_savings_bytes / (1024**4)) * QUERY_COST_PER_TB,
                "implementation_difficulty": "low",
                "before": "SELECT fields FROM table ORDER BY field",
                "after": "SELECT fields FROM table -- Remove ORDER BY if not needed\n-- OR: SELECT fields FROM table ORDER BY field LIMIT 1000",
                "improvement": "Avoids unnecessary sorting of the entire result set",
                "priority_score": 55  # Medium priority, easy to implement but may be intentional
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize and deduplicate recommendations.
        
        Args:
            recommendations: List of optimization recommendations
            
        Returns:
            Deduplicated and prioritized list of recommendations
        """
        if not recommendations:
            return []
            
        # Group similar recommendations by type and query hash
        grouped_recs = {}
        for rec in recommendations:
            rec_type = rec["type"]
            # Create a simplified hash of the query text for grouping similar queries
            query_text = rec.get("query_text", "")
            query_hash = self._get_query_hash(query_text)
            
            key = f"{rec_type}_{query_hash}"
            
            if key not in grouped_recs:
                grouped_recs[key] = []
            grouped_recs[key].append(rec)
            
        # Merge similar recommendations and calculate aggregate savings
        merged_recs = []
        for _, group in grouped_recs.items():
            if not group:
                continue
                
            # Use the first recommendation as the base
            base_rec = group[0].copy()
            
            if len(group) > 1:
                # Multiple similar recommendations, calculate aggregate savings
                total_bytes = sum(r.get("bytes_processed", 0) for r in group)
                estimated_savings = sum(r.get("estimated_savings_bytes", 0) for r in group)
                execution_count = sum(r.get("execution_count", 1) for r in group)
                
                # Update metrics
                base_rec["bytes_processed"] = total_bytes
                base_rec["estimated_savings_bytes"] = estimated_savings
                base_rec["estimated_cost_savings"] = (estimated_savings / (1024**4)) * QUERY_COST_PER_TB
                base_rec["execution_count"] = execution_count
                base_rec["similar_queries_count"] = len(group)
                
                # Include example queries
                base_rec["example_queries"] = [r.get("query_text", "")[:200] + "..." for r in group[:3]]
                
                # Update priority based on aggregate impact
                base_rec["priority_score"] += min(20, len(group))  # Increase priority based on frequency
            
            merged_recs.append(base_rec)
            
        # Sort by priority score and estimated savings
        sorted_recs = sorted(
            merged_recs, 
            key=lambda r: (r.get("priority_score", 0), r.get("estimated_savings_bytes", 0)), 
            reverse=True
        )
        
        return sorted_recs
    
    def _get_query_hash(self, query_text: str) -> str:
        """Generate a simplified hash for a query to group similar queries.
        
        Args:
            query_text: SQL query text
            
        Returns:
            Simplified query hash
        """
        # Normalize the query:
        # 1. Convert to lowercase
        # 2. Replace literal values with placeholders
        # 3. Remove whitespace
        normalized = query_text.lower()
        # Replace numeric literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        # Replace string literals
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Use only the first 200 chars for the hash to group similar queries
        return hashlib.md5(normalized[:200].encode()).hexdigest()
    
    def generate_recommendations_report(self, recommendations: Dict[str, Any], format: str = 'md') -> str:
        """Generate a formatted report of query optimization recommendations.
        
        Args:
            recommendations: Recommendations from analyze_dataset_queries
            format: Output format ('md' for Markdown, 'html', or 'text')
            
        Returns:
            Formatted report as a string
        """
        dataset_id = recommendations.get("dataset_id", "unknown")
        total_recs = recommendations.get("summary", {}).get("total_recommendations", 0)
        monthly_savings = recommendations.get("summary", {}).get("estimated_monthly_cost_savings", 0)
        annual_savings = recommendations.get("summary", {}).get("estimated_annual_cost_savings", 0)
        
        if format == 'md':
            # Markdown format
            report = [
                f"# Query Optimization Recommendations: {dataset_id}",
                "",
                "## Summary",
                "",
                f"- **Total Recommendations:** {total_recs}",
                f"- **Estimated Monthly Savings:** ${monthly_savings:.2f}",
                f"- **Estimated Annual Savings:** ${annual_savings:.2f}",
                f"- **Total Queries Analyzed:** {recommendations.get('queries_analyzed', 0)}",
                "",
                "## Top Recommendations",
                ""
            ]
            
            # Add each recommendation
            for i, rec in enumerate(recommendations.get("recommendations", [])[:10], 1):
                rec_type = rec.get("type", "unknown").replace("_", " ").title()
                description = rec.get("description", "")
                savings = rec.get("estimated_cost_savings", 0)
                
                report.append(f"### {i}. {rec_type}: {description}")
                report.append("")
                report.append(f"**Estimated Savings:** ${savings:.2f}")
                report.append("")
                report.append(f"**Rationale:** {rec.get('rationale', '')}")
                report.append("")
                report.append("```sql")
                report.append(f"-- Before:")
                report.append(rec.get("before", ""))
                report.append("")
                report.append(f"-- After:")
                report.append(rec.get("after", ""))
                report.append("```")
                report.append("")
                
            return "\n".join(report)
            
        elif format == 'html':
            # HTML format (simplified example)
            report = [
                f"<h1>Query Optimization Recommendations: {dataset_id}</h1>",
                "<h2>Summary</h2>",
                "<ul>",
                f"<li><strong>Total Recommendations:</strong> {total_recs}</li>",
                f"<li><strong>Estimated Monthly Savings:</strong> ${monthly_savings:.2f}</li>",
                f"<li><strong>Estimated Annual Savings:</strong> ${annual_savings:.2f}</li>",
                "</ul>",
                "<h2>Top Recommendations</h2>"
            ]
            
            # Add each recommendation
            for i, rec in enumerate(recommendations.get("recommendations", [])[:10], 1):
                rec_type = rec.get("type", "unknown").replace("_", " ").title()
                description = rec.get("description", "")
                savings = rec.get("estimated_cost_savings", 0)
                
                report.append(f"<h3>{i}. {rec_type}: {description}</h3>")
                report.append(f"<p><strong>Estimated Savings:</strong> ${savings:.2f}</p>")
                report.append(f"<p><strong>Rationale:</strong> {rec.get('rationale', '')}</p>")
                report.append("<pre><code>")
                report.append(f"-- Before:\n{rec.get('before', '')}\n")
                report.append(f"-- After:\n{rec.get('after', '')}")
                report.append("</code></pre>")
                
            return "\n".join(report)
            
        else:
            # Plain text format
            report = [
                f"QUERY OPTIMIZATION RECOMMENDATIONS: {dataset_id}",
                "=" * 50,
                "",
                "SUMMARY:",
                f"- Total Recommendations: {total_recs}",
                f"- Estimated Monthly Savings: ${monthly_savings:.2f}",
                f"- Estimated Annual Savings: ${annual_savings:.2f}",
                "",
                "TOP RECOMMENDATIONS:",
                ""
            ]
            
            # Add each recommendation
            for i, rec in enumerate(recommendations.get("recommendations", [])[:10], 1):
                rec_type = rec.get("type", "unknown").replace("_", " ").title()
                description = rec.get("description", "")
                savings = rec.get("estimated_cost_savings", 0)
                
                report.append(f"{i}. {rec_type}: {description}")
                report.append(f"   Estimated Savings: ${savings:.2f}")
                report.append(f"   Rationale: {rec.get('rationale', '')}")
                report.append("")
                report.append("   Before:")
                report.append(f"   {rec.get('before', '')}")
                report.append("")
                report.append("   After:")
                report.append(f"   {rec.get('after', '')}")
                report.append("-" * 50)
                
            return "\n".join(report)