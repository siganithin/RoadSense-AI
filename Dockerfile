# ── RoadSense AI — Docker Image ──────────────────────────────────────────────
FROM python:3.11.9-slim

LABEL maintainer="RoadSense AI Team"
LABEL description="Road Damage Detection System — Streamlit App"
LABEL version="2.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs results reports/figures

# Expose Streamlit port — HF Spaces requires port 7860
EXPOSE 7860

# Run Streamlit app on port 7860 (required by HF Spaces)
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
