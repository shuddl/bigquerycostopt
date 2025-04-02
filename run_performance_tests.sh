#!/bin/bash
# Performance test runner script

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Virtual environment not activated. Activating now..."
    source venv/bin/activate
fi

# Ensure required packages are installed
pip install -q tqdm

# Run performance test
echo "Running performance tests..."
python tests/performance/load_test.py "$@"

# Check exit code
if [ $? -eq 0 ]; then
    echo "Performance tests completed successfully!"
else
    echo "Performance tests failed. Please check the output above."
    exit 1
fi