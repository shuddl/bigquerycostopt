"""
Tests for verifying that all dependencies are correctly installed and modules are properly integrated.
"""

import importlib
import unittest


class DependencyIntegrationTest(unittest.TestCase):
    """Tests for verifying that all required dependencies are installed and modules can be imported."""

    def test_core_dependencies(self):
        """Test that all core dependencies can be imported."""
        # Split into required and optional core dependencies
        required_modules = [
            'pandas',
            'numpy',
            'google.cloud.bigquery',
            'requests',
            'joblib',
            'tqdm'
        ]
        
        optional_modules = [
            'google.cloud.storage',
            'google.cloud.pubsub',
            'flask',
            'matplotlib'
        ]
        
        # Test required modules - these should cause test failures if not available
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                self.fail(f"Failed to import required module {module}: {e}")
                
        # Test optional modules - just log if not available
        for module in optional_modules:
            try:
                importlib.import_module(module)
                print(f"Optional core module {module} is available")
            except ImportError as e:
                print(f"Optional core module {module} is not available: {e}")
    
    def test_ml_dependencies(self):
        """Test that all ML dependencies can be imported."""
        # All ML modules are optional, as ML features are an enhancement
        ml_modules = [
            'scipy',
            'sklearn',
            'statsmodels'
        ]
        
        for module in ml_modules:
            try:
                importlib.import_module(module)
                print(f"ML module {module} is available")
            except ImportError as e:
                print(f"ML module {module} is not available: {e} (this is optional)")
    
    def test_optional_dependencies(self):
        """Test optional dependencies if they are installed."""
        optional_modules = [
            'fastapi',
            'pydantic',
            'seaborn'
        ]
        
        for module in optional_modules:
            try:
                importlib.import_module(module)
                print(f"Optional module {module} is available")
            except ImportError:
                print(f"Optional module {module} is not installed")
    
    def test_prophet_availability(self):
        """Test if Prophet is available, but don't fail if it's not."""
        try:
            import prophet
            print("Prophet is available")
        except ImportError:
            print("Prophet is not available - this is optional for advanced time series forecasting")
    
    def test_project_modules(self):
        """Test that all project modules can be imported."""
        # Split into required and ML/optional modules
        required_modules = [
            'src.analysis.metadata',
            'src.analysis.query_optimizer',
            'src.analysis.schema_optimizer',
            'src.analysis.storage_optimizer',
            'src.analysis.cost_attribution',  # New module
            'src.recommender.engine',
            'src.recommender.roi',
            'src.utils.logging'
        ]
        
        ml_modules = [
            'src.ml.models',
            'src.ml.cost_anomaly_detection',  # New module
        ]
        
        # Test required core modules
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                self.fail(f"Failed to import required project module {module}: {e}")
                
        # Test ML modules (which are optional)
        for module in ml_modules:
            try:
                importlib.import_module(module)
                print(f"ML project module {module} is available")
            except ImportError as e:
                print(f"ML project module {module} is not available: {e} (this is optional)")


if __name__ == '__main__':
    unittest.main()