# Stage 1: Build React UI
FROM node:18-alpine AS frontend-builder
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ .
RUN npm run build

# Stage 2: Build Python Backend
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (for NER extraction)
RUN python -m spacy download en_core_web_sm || echo "spaCy model download failed — NER will be skipped"

# Copy application code
COPY . .

# Copy built React UI from Stage 1
COPY --from=frontend-builder /app/ui/dist ./ui/dist

# Create data directories
RUN mkdir -p data/uploads data/faiss_index data/bm25_index data/feedback

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health', timeout=5); exit(0 if r.status_code == 200 else 1)" \
    || exit 1

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
