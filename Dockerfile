FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download Stanza models
RUN python -c "import stanza; stanza.download('grc', verbose=False)"
RUN python -c "import stanza; stanza.download('en', verbose=False)"

# Copy application files
COPY unified_corpus_platform.py .
COPY web_dashboard.py .

# Create data directories
RUN mkdir -p data/raw data/parsed data/annotated

# Expose port for web dashboard
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DB_PATH=/app/data/corpus_platform.db

# Default command (can be overridden)
CMD ["python", "web_dashboard.py"]
