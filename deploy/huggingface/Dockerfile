# Dockerfile for Hugging Face Spaces (SDK: docker)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    UPLOAD_FOLDER=/tmp/uploads \
    VECTOR_DB=/tmp/vector_store \
    HF_HOME=/tmp/hf \
    FASTEMBED_CACHE_PATH=/tmp/fastembed_cache

# libgomp1 is needed at runtime by faiss-cpu and onnxruntime (fastembed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# One worker keeps the embedding model in a single process; long timeout lets the
# model download on the first request.
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:7860 --workers 1 --threads 4 --timeout 300 app:app"]
