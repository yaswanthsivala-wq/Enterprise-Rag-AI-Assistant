FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is required at runtime by faiss-cpu and onnxruntime (fastembed).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user; pre-create writable data dirs.
RUN useradd -m appuser \
    && mkdir -p /app/uploads /app/vector_store \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)" || exit 1

# Production WSGI server. One worker keeps the embedding model in a single
# process (memory); threads provide concurrency. Tune via WEB_CONCURRENCY.
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-8080} --workers ${WEB_CONCURRENCY:-1} --threads 4 --timeout 180 app:app"]
