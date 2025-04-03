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

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing project dependencies..."
pip install -e ".[dev,test]"

# Ensure required packages are installed
echo "Ensuring required test packages are installed..."
pip install tqdm joblib scikit-learn matplotlib

# Install ML-specific dependencies
echo "Installing ML dependencies..."
pip install -r requirements_ml.txt

# Handle potential conflicts with special attention to Prophet dependencies
echo "Handling dependency conflicts..."
# Prophet requires specific pystan version
pip install "pystan<3.0.0"
pip install prophet --no-deps
pip install numpy pandas holidays

# Verify installation
echo "Verifying installation..."
python --version
pytest --version

echo "Development environment setup complete."
echo "To activate this environment in the future, run: source venv/bin/activate"