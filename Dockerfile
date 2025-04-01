FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY setup.py /app/
COPY README.md /app/
COPY bigquerycostopt /app/bigquerycostopt/

RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the API server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 bigquerycostopt.src.api.endpoints:app
