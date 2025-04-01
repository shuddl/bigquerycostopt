#!/bin/bash
# Test runner script

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Virtual environment not activated. Activating now..."
    source venv/bin/activate
fi

# Run tests
echo "Running tests..."
python -m pytest tests/"$@" -v

# Check exit code
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Please check the output above."
fi