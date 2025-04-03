#!/usr/bin/env python3
"""
Verify that all dependencies are correctly installed and the system is properly integrated.
This script performs a comprehensive check of the BigQuery Cost Intelligence Engine installation.
"""

import importlib
import sys
import os
import pkg_resources

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_dependency(module_name, min_version=None):
    """Check if a dependency is installed and meets minimum version requirements."""
    try:
        module = importlib.import_module(module_name)
        if module_name == 'google.cloud.bigquery':
            actual_module = 'google-cloud-bigquery'
        elif module_name == 'google.cloud.storage':
            actual_module = 'google-cloud-storage'
        elif module_name == 'google.cloud.pubsub':
            actual_module = 'google-cloud-pubsub'
        elif module_name == 'sklearn':
            actual_module = 'scikit-learn'
        else:
            actual_module = module_name.split('.')[0]
        
        try:
            version = pkg_resources.get_distribution(actual_module).version
            if min_version and pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                return f"⚠️  {module_name} version {version} is installed but {min_version} is required"
            return f"✅ {module_name} {version} is installed"
        except pkg_resources.DistributionNotFound:
            return f"✅ {module_name} is installed (version unknown)"
    except ImportError as e:
        return f"❌ {module_name} is not installed: {e}"

def check_project_module(module_path):
    """Check if a project module can be imported."""
    try:
        importlib.import_module(module_path)
        return f"✅ Project module {module_path} is accessible"
    except ImportError as e:
        return f"❌ Project module {module_path} is not accessible: {e}"

def run_basic_tests():
    """Run basic tests to verify functionality."""
    from bigquerycostopt.src.analysis.cost_attribution import CostAttributionAnalyzer
    from bigquerycostopt.src.ml.cost_anomaly_detection import MLCostAnomalyDetector
    
    print("\n🧪 Running basic functional tests...")
    
    # Test CostAttributionAnalyzer initialization
    try:
        # Using dummy project_id, this won't actually connect to BigQuery
        analyzer = CostAttributionAnalyzer(project_id="test-project")
        print("✅ CostAttributionAnalyzer initialized successfully")
    except Exception as e:
        print(f"❌ CostAttributionAnalyzer initialization failed: {e}")
    
    # Test MLCostAnomalyDetector initialization
    try:
        detector = MLCostAnomalyDetector()
        print("✅ MLCostAnomalyDetector initialized successfully")
    except Exception as e:
        print(f"❌ MLCostAnomalyDetector initialization failed: {e}")

def main():
    """Main function to verify installation."""
    print("🔍 Verifying BigQuery Cost Intelligence Engine Installation\n")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    if pkg_resources.parse_version(python_version) < pkg_resources.parse_version("3.9"):
        print("⚠️  Warning: Python 3.9 or higher is recommended\n")
    else:
        print("✅ Python version is 3.9 or higher\n")
    
    # Core dependencies
    print("📦 Checking core dependencies:")
    core_deps = [
        ("pandas", "1.3.0"),
        ("numpy", "1.20.0"),
        ("google.cloud.bigquery", "2.30.0"),
        ("google.cloud.storage", "2.0.0"),
        ("google.cloud.pubsub", "2.8.0"),
        ("flask", "2.0.0"),
        ("requests", "2.25.0"),
        ("joblib", "1.1.0"),
        ("tqdm", "4.62.0"),
        ("matplotlib", "3.4.0")
    ]
    
    for dep, version in core_deps:
        print(check_dependency(dep, version))
    
    # ML dependencies
    print("\n📊 Checking ML dependencies:")
    ml_deps = [
        ("scipy", "1.7.0"),
        ("sklearn", "1.0.0"),
        ("statsmodels", "0.13.0")
    ]
    
    for dep, version in ml_deps:
        print(check_dependency(dep, version))
    
    # Optional dependencies
    print("\n🔍 Checking optional dependencies:")
    optional_deps = [
        ("fastapi", "0.68.0"),
        ("pydantic", "1.8.0"),
        ("uvicorn", "0.15.0"),
        ("prophet", "1.1.0"),
        ("seaborn", "0.11.0")
    ]
    
    for dep, version in optional_deps:
        print(check_dependency(dep, version))
    
    # Project modules
    print("\n🧩 Checking project modules:")
    project_modules = [
        "bigquerycostopt.src.analysis.metadata",
        "bigquerycostopt.src.analysis.query_optimizer",
        "bigquerycostopt.src.analysis.schema_optimizer",
        "bigquerycostopt.src.analysis.storage_optimizer",
        "bigquerycostopt.src.analysis.cost_attribution",  # New module
        "bigquerycostopt.src.ml.models",
        "bigquerycostopt.src.ml.cost_anomaly_detection",  # New module
        "bigquerycostopt.src.recommender.engine",
        "bigquerycostopt.src.recommender.roi",
        "bigquerycostopt.src.utils.logging"
    ]
    
    for module in project_modules:
        print(check_project_module(module))
    
    # Run basic tests
    try:
        run_basic_tests()
    except Exception as e:
        print(f"\n❌ Basic tests failed: {e}")
    
    print("\n✨ Verification complete")

if __name__ == "__main__":
    main()