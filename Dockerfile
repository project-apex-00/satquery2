FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
# libgdal-dev/gdal-bin/libgeos-dev/libproj-dev are a safety net: modern
# rasterio wheels bundle their own GDAL and normally don't need these, but if
# no matching wheel exists for this platform, pip silently falls back to
# building from source -- which fails outright without these, rather than
# quietly shipping a broken rasterio import.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fail the BUILD, loudly, if rasterio can't actually be imported -- instead
# of silently shipping an image where GeoTIFF preview/analysis breaks only
# when a real user uploads a real file (rasterio_available=false at runtime).
RUN python -c "import rasterio; print('rasterio OK:', rasterio.__version__, rasterio.gdal_version())"

# Copy application source code
COPY . .

# Expose default port (7860 for Hugging Face Spaces, or fallback for Docker)
EXPOSE 7860

# Support PORT environment variable (Railway/Render) with fallback to 7860 (Hugging Face Spaces) or 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
