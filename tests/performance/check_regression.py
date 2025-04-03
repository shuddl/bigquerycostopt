#!/usr/bin/env python
"""
Performance regression checker for BigQuery Cost Intelligence Engine.

This script compares current performance test results with a baseline
to detect performance regressions.
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading file {file_path}: {e}")
        return {}

def compare_performance(current: Dict[str, Any], baseline: Dict[str, Any], threshold: float) -> bool:
    """Compare current performance with baseline.
    
    Args:
        current: Current performance data
        baseline: Baseline performance data
        threshold: Regression threshold percentage
        
    Returns:
        True if performance is acceptable, False if regression detected
    """
    if not current or not baseline:
        print("Missing current or baseline data")
        return False
    
    if "summary" not in current or "summary" not in baseline:
        print("Missing summary data in current or baseline")
        return False
    
    regression_detected = False
    
    # Compare for each dataset
    for dataset_id, dataset_data in current["summary"].items():
        if dataset_id not in baseline["summary"]:
            print(f"Dataset {dataset_id} not found in baseline, skipping")
            continue
        
        baseline_data = baseline["summary"][dataset_id]
        
        # Check if large dataset (100K+ records)
        if dataset_data["size"] >= 100000:
            current_time = dataset_data["average_times"]["total"]
            baseline_time = baseline_data["average_times"]["total"]
            
            # Calculate percentage difference
            diff_pct = ((current_time - baseline_time) / baseline_time) * 100
            
            print(f"Dataset {dataset_id} (size: {dataset_data['size']})")
            print(f"  Current: {current_time:.2f}s, Baseline: {baseline_time:.2f}s")
            print(f"  Difference: {diff_pct:.2f}%")
            
            # Check if exceeds threshold
            if diff_pct > threshold:
                print(f"  REGRESSION DETECTED: {diff_pct:.2f}% increase exceeds {threshold}% threshold")
                regression_detected = True
            else:
                print(f"  Performance OK")
            
            # Check if 4-minute requirement still met
            if current_time > 240:
                print(f"  WARNING: Dataset processing time {current_time:.2f}s exceeds 4-minute requirement")
                regression_detected = True
    
    return not regression_detected

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check for performance regression")
    parser.add_argument("--current", required=True, help="Path to current performance results")
    parser.add_argument("--baseline", help="Path to baseline performance results")
    parser.add_argument("--threshold", type=float, default=10.0, help="Regression threshold percentage")
    args = parser.parse_args()
    
    # Load current results
    current_data = load_json_file(args.current)
    
    # Load or create baseline
    if args.baseline:
        baseline_data = load_json_file(args.baseline)
    else:
        # Look for baseline file
        baseline_path = Path("tests/performance/baseline_results.json")
        if baseline_path.exists():
            baseline_data = load_json_file(str(baseline_path))
        else:
            print("Baseline not provided and no baseline file found")
            print("Creating new baseline from current results")
            baseline_data = current_data
            
            # Save as new baseline
            os.makedirs(baseline_path.parent, exist_ok=True)
            with open(baseline_path, "w") as f:
                json.dump(current_data, f, indent=2)
    
    # Compare performance
    if compare_performance(current_data, baseline_data, args.threshold):
        print("Performance check passed")
        sys.exit(0)
    else:
        print("Performance check failed - regression detected")
        sys.exit(1)

if __name__ == "__main__":
    main()