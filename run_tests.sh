#!/bin/bash
# Test runner script

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Virtual environment not activated. Activating now..."
    source venv/bin/activate
fi

# Run tests
echo "Running tests..."

# Check if specific tests are requested
if [ $# -gt 0 ]; then
    python -m pytest tests/"$@" -v
else
    # Run unit tests first
    echo "Running unit tests..."
    python -m pytest tests/unit -v
    
    # Run integration tests if unit tests pass
    if [ $? -eq 0 ]; then
        echo "Running integration tests..."
        python -m pytest tests/integration -v
    fi
fi

# Check exit code
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Please check the output above."
fi