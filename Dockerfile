FROM python:3.11-slim

# System tools for Docling
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create mount points
RUN mkdir -p /app/documents /app/qdrant_db

COPY app/ /app/

CMD ["python", "main.py"]