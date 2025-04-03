# Dependency Management Guide

## Overview

The BigQuery Cost Intelligence Engine has several dependencies with specific version requirements. This guide helps you understand these dependencies and resolve common conflicts that may arise during installation or upgrades.

## Core Dependencies

The system is built on the following core dependencies:

| Package | Min Version | Purpose |
|---------|-------------|---------|
| google-cloud-bigquery | 2.30.0 | BigQuery API interaction |
| pandas | 1.3.0 | Data manipulation and analysis |
| numpy | 1.20.0 | Numerical computations |
| flask/fastapi | 2.0.0/0.68.0 | API framework |
| scikit-learn | 1.0.0 | Machine learning algorithms |

## ML Component Dependencies

The ML-enhanced features for anomaly detection require additional dependencies:

| Package | Min Version | Purpose |
|---------|-------------|---------|
| scipy | 1.7.0 | Scientific computing |
| statsmodels | 0.13.0 | Time series analysis |
| prophet | 1.1.0 | Time series forecasting |
| joblib | 1.1.0 | Model persistence |

## Known Dependency Conflicts

### Prophet Installation Issues

The Prophet package can be challenging to install due to its dependencies on Stan:

**Problem**: Prophet installation fails with PyStan errors  
**Solution**: Install a compatible version of PyStan first, then install Prophet without dependencies:

```bash
pip install "pystan<3.0.0"
pip install prophet --no-deps
pip install numpy pandas holidays
```

### Numpy/Pandas Version Conflicts

**Problem**: Different components may require different numpy/pandas versions  
**Solution**: Specify a compatible range and always install numpy first:

```bash
pip install "numpy>=1.20.0,<1.24.0"
pip install "pandas>=1.3.0,<1.5.0"
```

### Tensorflow Compatibility

If you extend the ML components to use TensorFlow:

**Problem**: TensorFlow has specific dependency requirements  
**Solution**: Install TensorFlow in a separate virtual environment or specify compatible versions:

```bash
pip install "tensorflow>=2.8.0"
pip install "numpy>=1.20.0,<1.24.0"  # TensorFlow requires specific numpy versions
```

## Installation Methods

### Method 1: Standard Installation

For basic usage without ML components:

```bash
pip install -e .
```

### Method 2: Full Installation with ML Components

```bash
pip install -e .
pip install -r requirements_ml.txt
```

### Method 3: Development Installation

```bash
./setup_dev.sh
```

### Method 4: Docker Installation

Build and run the Docker container:

```bash
docker build -t bigquerycostopt .
docker run -p 8080:8080 -e API_SERVER=fastapi bigquerycostopt
```

## Troubleshooting

### Dependency Resolution Errors

If you encounter dependency resolution errors:

1. Install dependencies in order from most fundamental to most specific
2. Use `pip install --no-deps` for problematic packages
3. Manually specify compatible versions for conflicting packages

### Import Errors After Installation

If you encounter import errors:

1. Check the installed versions: `pip list | grep <package_name>`
2. Verify Python path includes the installation directory
3. Ensure all dependencies are installed correctly

### Docker Build Errors

If Docker build fails due to dependency issues:

1. Update the Dockerfile to install problematic packages separately
2. Add specific OS-level dependencies that may be required
3. Use multi-stage builds to separate dependency installation from runtime

## Best Practices

1. Use virtual environments for isolation
2. Pin versions for production deployments
3. Regularly update dependencies for security patches
4. Test dependency changes in a development environment first
5. Document any specific version requirements in your deployment documentation