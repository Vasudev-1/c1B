FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code first
COPY . .

# Create _init_.py files to make src a proper Python package
RUN touch /app/src/__init__.py

# Install Python dependencies with proper versions
RUN pip install --no-cache-dir \
    pymupdf==1.23.14 \
    sentence-transformers==2.2.2 \
    rake-nltk==1.0.6 \
    nltk==3.8.1 \
    tqdm==4.66.1 \
    numpy==1.24.3 \
    torch==2.1.0 \
    transformers==4.35.0 \
    scikit-learn==1.3.0 \
    scipy==1.11.0 \
    pandas==2.1.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    psutil==5.9.0

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create necessary directories
RUN mkdir -p /tmp/hackathon_cache

# Set environment variables
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

# Debug: List contents to verify structure
RUN echo "=== App directory contents ===" && \
    find /app -type f -name "*.py" | head -20 && \
    echo "=== Challenge_1b contents ===" && \
    ls -la /app/Challenge_1b/ 2>/dev/null || echo "Challenge_1b not found" && \
    echo "=== Source structure ===" && \
    ls -la /app/src/ 2>/dev/null || echo "src not found"

# Default command
CMD ["python", "-m", "src.__main__", "--base", "Challenge_1b", "--debug"]