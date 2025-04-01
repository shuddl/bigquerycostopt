#!/usr/bin/env python
"""
Load testing script for BigQuery Cost Intelligence Engine API.

This script performs load testing on the API service to verify performance
under various load conditions.
"""

import argparse
import json
import time
import uuid
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple

import requests
from tqdm import tqdm

# Default test settings
DEFAULT_URL = "https://bqcostopt-api-dev-xxxxxxx-uc.a.run.app"
DEFAULT_API_KEY = "test-api-key"
DEFAULT_THREADS = 5
DEFAULT_REQUESTS = 100
DEFAULT_RAMP_UP = 10

# Test scenarios
SCENARIOS = {
    "health_check": {
        "method": "GET",
        "endpoint": "/api/v1/health",
        "body": None
    },
    "list_recommendations": {
        "method": "GET",
        "endpoint": "/api/v1/recommendations?limit=20&offset=0",
        "body": None
    },
    "get_recommendation": {
        "method": "GET",
        "endpoint": "/api/v1/recommendations/{recommendation_id}",
        "body": None,
        "requires_id": True
    },
    "analyze_dataset": {
        "method": "POST",
        "endpoint": "/api/v1/analyze",
        "body": {
            "project_id": "test-project",
            "dataset_id": "test_dataset"
        }
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load testing for BigQuery Cost Intelligence Engine API"
    )
    
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"API base URL (default: {DEFAULT_URL})"
    )
    
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for authentication"
    )
    
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="health_check",
        help="Test scenario to run (default: health_check)"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of concurrent threads (default: {DEFAULT_THREADS})"
    )
    
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help=f"Total number of requests to send (default: {DEFAULT_REQUESTS})"
    )
    
    parser.add_argument(
        "--ramp-up",
        type=int,
        default=DEFAULT_RAMP_UP,
        help=f"Ramp-up period in seconds (default: {DEFAULT_RAMP_UP}s)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for test results (JSON)"
    )
    
    return parser.parse_args()

def get_recommendation_ids(base_url: str, api_key: str, count: int = 10) -> List[str]:
    """Get a list of recommendation IDs for testing.
    
    Args:
        base_url: API base URL
        api_key: API key for authentication
        count: Number of IDs to retrieve
        
    Returns:
        List of recommendation IDs
    """
    try:
        response = requests.get(
            f"{base_url}/api/v1/recommendations?limit={count}",
            headers={"X-API-Key": api_key}
        )
        response.raise_for_status()
        data = response.json()
        return [rec["recommendation_id"] for rec in data.get("recommendations", [])]
    except Exception:
        # Return mock IDs if real ones can't be retrieved
        return [f"mock-rec-{i}" for i in range(count)]

def send_request(
    base_url: str,
    api_key: str,
    scenario: Dict[str, Any],
    recommendation_ids: List[str] = None
) -> Tuple[int, float]:
    """Send a single API request.
    
    Args:
        base_url: API base URL
        api_key: API key for authentication
        scenario: Test scenario configuration
        recommendation_ids: List of recommendation IDs (for scenarios requiring IDs)
        
    Returns:
        Tuple of (status_code, response_time)
    """
    method = scenario["method"]
    endpoint = scenario["endpoint"]
    body = scenario["body"]
    
    # Replace placeholders in endpoint
    if "{recommendation_id}" in endpoint and recommendation_ids:
        rec_id = random.choice(recommendation_ids)
        endpoint = endpoint.replace("{recommendation_id}", rec_id)
    
    # Prepare request
    url = f"{base_url}{endpoint}"
    headers = {"X-API-Key": api_key}
    
    if body:
        # Make a copy of the body to avoid modifying the original
        body = body.copy()
        
        # Add unique ID to avoid caching
        if method == "POST":
            body["request_id"] = str(uuid.uuid4())
    
    # Send request and measure time
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=10
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        status_code = response.status_code
    except Exception as e:
        print(f"Request error: {e}")
        status_code = 0
    
    response_time = time.time() - start_time
    
    return status_code, response_time

def run_test_scenario(
    base_url: str,
    api_key: str,
    scenario_name: str,
    num_threads: int,
    num_requests: int,
    ramp_up: int
) -> Dict[str, Any]:
    """Run a test scenario.
    
    Args:
        base_url: API base URL
        api_key: API key for authentication
        scenario_name: Name of the scenario to run
        num_threads: Number of concurrent threads
        num_requests: Total number of requests to send
        ramp_up: Ramp-up period in seconds
        
    Returns:
        Dictionary of test results
    """
    print(f"\nRunning scenario: {scenario_name}")
    
    scenario = SCENARIOS[scenario_name]
    recommendation_ids = None
    
    # Get recommendation IDs if needed
    if scenario.get("requires_id", False):
        print("Retrieving recommendation IDs...")
        recommendation_ids = get_recommendation_ids(base_url, api_key)
    
    # Calculate delay between thread starts
    thread_delay = ramp_up / num_threads if num_threads > 1 else 0
    requests_per_thread = num_requests // num_threads
    
    # Initialize result tracking
    results = {
        "scenario": scenario_name,
        "total_requests": num_requests,
        "successful_requests": 0,
        "failed_requests": 0,
        "status_codes": {},
        "response_times": [],
        "min_time": 0,
        "max_time": 0,
        "avg_time": 0,
        "median_time": 0,
        "p95_time": 0
    }
    
    def worker(thread_id: int) -> List[Tuple[int, float]]:
        """Worker function for thread pool.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            List of (status_code, response_time) tuples
        """
        # Delay thread start based on ID
        time.sleep(thread_delay * thread_id)
        
        thread_results = []
        for _ in range(requests_per_thread):
            status, resp_time = send_request(
                base_url,
                api_key,
                scenario,
                recommendation_ids
            )
            thread_results.append((status, resp_time))
            
            # Small random delay between requests
            time.sleep(random.uniform(0.05, 0.2))
        
        return thread_results
    
    # Execute requests using thread pool
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        
        all_results = []
        for future in tqdm(futures, desc="Threads", total=num_threads):
            thread_results = future.result()
            all_results.extend(thread_results)
    
    end_time = time.time()
    
    # Process results
    response_times = [r[1] for r in all_results]
    
    for status, resp_time in all_results:
        if status >= 200 and status < 300:
            results["successful_requests"] += 1
        else:
            results["failed_requests"] += 1
        
        status_str = str(status)
        if status_str in results["status_codes"]:
            results["status_codes"][status_str] += 1
        else:
            results["status_codes"][status_str] = 1
    
    # Calculate statistics
    if response_times:
        results["response_times"] = response_times
        results["min_time"] = min(response_times)
        results["max_time"] = max(response_times)
        results["avg_time"] = statistics.mean(response_times)
        results["median_time"] = statistics.median(response_times)
        results["p95_time"] = sorted(response_times)[int(len(response_times) * 0.95)]
    
    results["total_duration"] = end_time - start_time
    results["requests_per_second"] = len(all_results) / results["total_duration"]
    
    # Print summary
    print(f"\nResults for {scenario_name}:")
    print(f"Total requests: {len(all_results)}")
    print(f"Successful requests: {results['successful_requests']} ({results['successful_requests']/len(all_results)*100:.1f}%)")
    print(f"Failed requests: {results['failed_requests']} ({results['failed_requests']/len(all_results)*100:.1f}%)")
    print(f"Status codes: {results['status_codes']}")
    print(f"Response time (min/avg/median/95%/max): {results['min_time']:.3f}s / {results['avg_time']:.3f}s / {results['median_time']:.3f}s / {results['p95_time']:.3f}s / {results['max_time']:.3f}s")
    print(f"Throughput: {results['requests_per_second']:.1f} requests/second")
    
    return results

def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate URL format
    if not args.url.startswith(("http://", "https://")):
        args.url = f"https://{args.url}"
    
    print(f"Load testing API at: {args.url}")
    print(f"Threads: {args.threads}, Requests: {args.requests}, Ramp-up: {args.ramp_up}s")
    
    # Run requested scenarios
    results = {}
    
    if args.scenario == "all":
        for scenario_name in SCENARIOS:
            results[scenario_name] = run_test_scenario(
                args.url,
                args.api_key,
                scenario_name,
                args.threads,
                args.requests,
                args.ramp_up
            )
    else:
        results[args.scenario] = run_test_scenario(
            args.url,
            args.api_key,
            args.scenario,
            args.threads,
            args.requests,
            args.ramp_up
        )
    
    # Save results if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "url": args.url,
                    "threads": args.threads,
                    "requests": args.requests,
                    "ramp_up": args.ramp_up,
                    "results": results
                },
                f,
                indent=2
            )
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()