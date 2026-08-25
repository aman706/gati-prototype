# Dockerfile

FROM python:3.10-slim

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

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy repo
COPY . /app

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "app_survival.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
