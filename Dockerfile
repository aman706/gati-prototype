# Dockerfile - Python 3.14+ compatible
# Updated for latest dependencies and build optimizations

FROM python:3.14-slim

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies needed by scikit-survival and numeric libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install with optimizations
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy repo
COPY . /app

# Create models and data directories
RUN mkdir -p /app/models /app/data

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501', timeout=5)" || exit 1

# Run Streamlit app with optimizations
CMD ["streamlit", "run", "app_gati.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--logger.level", "info", \
     "--client.showErrorDetails", "false"]
