#!/bin/bash
# Development environment setup script

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 before continuing."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing project dependencies..."
pip install -e .
pip install pytest pytest-cov black flake8

# Verify installation
echo "Verifying installation..."
python --version
pytest --version

echo "Development environment setup complete."
echo "To activate this environment in the future, run: source venv/bin/activate"