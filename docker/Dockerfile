# syntax=docker/dockerfile:1
# HFRI-NKUA Corpus Platform + AthDGC v3.0.0
# Production-ready Docker image

FROM python:3.11-slim as builder

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc \
      libxml2-dev \
      libxslt1-dev \
      tesseract-ocr \
      tesseract-ocr-eng \
      ca-certificates \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements_full.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --user -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN groupadd -r athdgc && useradd -m -r -g athdgc athdgc && \
    chown -R athdgc:athdgc /app

# Runtime stage
FROM python:3.11-slim
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY --chown=athdgc:athdgc . .

# Runtime system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libxml2 \
      libxslt1.1 \
      tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Download spaCy models (optional)
# RUN python -m spacy download en_core_web_sm

# Non-root user
USER athdgc

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

# Start application
CMD ["python", "professional_dashboard.py"]
