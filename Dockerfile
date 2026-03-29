# Dockerfile for ANNS Thesis Experiments
# ======================================
# This Dockerfile provides a reproducible environment for running
# HNSW and early termination algorithm benchmarks.

FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42

# Set number of threads for reproducibility
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Build C++ modules
RUN mkdir -p hnsw_cpp/src/build \
    && cd hnsw_cpp/src/build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" .. \
    && cmake --build . -j$(nproc) \
    && cd ../../..

RUN mkdir -p hnswDarth_cpp/src/build \
    && cd hnswDarth_cpp/src/build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" .. \
    && cmake --build . -j$(nproc) \
    && cd ../../..

RUN mkdir -p hnsw_pip_cpp/src/build \
    && cd hnsw_pip_cpp/src/build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" .. \
    && cmake --build . -j$(nproc) \
    && cd ../../..

RUN mkdir -p hnsw_adaef_cpp/src/build \
    && cd hnsw_adaef_cpp/src/build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" .. \
    && cmake --build . -j$(nproc) \
    && cd ../../..

# Download sample datasets
RUN python scripts/download_siftsmall.py || true

# Create output directories
RUN mkdir -p results_csv plot_results

# Set environment variables for experiments
ENV EXPERIMENT_SEED=42
ENV EXPERIMENT_OUTPUT_DIR=/app/results_csv

# Default command - run quick test
CMD ["python", "run_experiments.py", "--quick-test"]
