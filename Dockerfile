FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY setup.py requirements_ml.txt README.md /app/
COPY src /app/src/

# Install core dependencies
RUN pip install --no-cache-dir -e .

# Install ML-specific dependencies with special handling for Prophet
RUN pip install --no-cache-dir "pystan<3.0.0" && \
    pip install --no-cache-dir prophet --no-deps && \
    pip install --no-cache-dir pandas numpy holidays

# Copy the rest of the application
COPY bigquerycostopt /app/bigquerycostopt/
COPY examples /app/examples/
COPY docs /app/docs/

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Choose between FastAPI and Flask based on environment variable
RUN echo '#!/bin/bash\n\
if [ "$API_SERVER" = "fastapi" ]; then\n\
  echo "Starting FastAPI server..."\n\
  exec uvicorn bigquerycostopt.src.api.fastapi_server:app --host 0.0.0.0 --port $PORT\n\
else\n\
  echo "Starting Flask server..."\n\
  exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 bigquerycostopt.src.api.server:main\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Run the API server
CMD ["/app/start.sh"]
