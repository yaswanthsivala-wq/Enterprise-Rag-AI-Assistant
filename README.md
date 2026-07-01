# 🤖 Enterprise RAG AI Assistant

**▶️ Live demo:** https://stark218-rag-assistant.hf.space

A Retrieval-Augmented Generation chatbot: upload PDFs, ask questions, get answers
grounded in your documents with source citations. Built with Flask, LangChain,
FAISS, and FastEmbed, with a pluggable LLM provider (Groq by default, IBM
watsonx.ai optional), containerized with Docker and deployed on Hugging Face Spaces.

## Screenshots

> Add screenshots here: drop image files in a `docs/` folder and reference them, e.g.
> `![Chat UI](docs/screenshot.png)`

## Architecture

Browser -> Flask (`app.py`) -> RAG pipeline (`rag_pipeline.py`) -> FastEmbed (local
embeddings) + FAISS (local vector store) for retrieval, and an LLM provider for
generation. Embeddings run locally, so the only credential you need is one LLM
API key. Chat history is kept per browser session.

## Requirements

* Python 3.11
* One free LLM API key (see below)

## Get a free API key (Groq)

Groq's API is free with **no credit card required**.

1. Go to https://console.groq.com and sign up with your email (or Google/GitHub).
2. In the left sidebar, open **API Keys**.
3. Click **Create API Key**, name it, and **copy it immediately** (it is shown once). It starts with `gsk_`.
4. Put it in your `.env` as `GROQ_API_KEY`.

Default model is `llama-3.3-70b-versatile`; set `GROQ_MODEL=llama-3.1-8b-instant`
for faster responses / higher rate limits.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then paste your GROQ_API_KEY
```

### Environment variables

| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | yes | Free key from console.groq.com (starts with `gsk_`) |
| `FLASK_SECRET_KEY` | yes (prod) | Signs session cookies. A random key is used if unset (sessions reset on restart). |
| `GROQ_MODEL` | no | default `llama-3.3-70b-versatile` |
| `LLM_PROVIDER` | no | `groq` (default) or `watsonx` |
| `UPLOAD_FOLDER` | no | default `uploads` (use an absolute path for a mounted volume) |
| `VECTOR_DB` | no | default `vector_store` |
| `MAX_UPLOAD_MB` | no | default `25` |
| `WEB_CONCURRENCY` | no | gunicorn workers, default `1` |
| `PORT` | no | default `8080` |

## Run it

```bash
python app.py
```

Then open http://127.0.0.1:8080 in your browser, upload a PDF, and ask a question.

Production server (what the container runs):

```bash
gunicorn -b 0.0.0.0:8080 --workers 1 --threads 4 --timeout 180 app:app
```

Check it's healthy: `curl http://127.0.0.1:8080/health` should show
`"llm_configured": true` once your key is set.

## Test

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Using IBM watsonx instead (optional)

```bash
pip install -r requirements-watsonx.txt
# in .env:
#   LLM_PROVIDER=watsonx
#   IBM_API_KEY=...   IBM_URL=...   IBM_PROJECT_ID=...
```

## Docker

```bash
docker build -t enterprise-rag-assistant .
docker run -p 8080:8080 --env-file .env enterprise-rag-assistant
```

## Deploy

Uploads and the FAISS index are written to disk, so a deployment needs a
persistent volume (single instance) or an external vector store (multi-instance).

* **Fly.io** (recommended): `fly launch --no-deploy`, `fly volumes create rag_data --size 1`, `fly secrets set GROQ_API_KEY=... FLASK_SECRET_KEY=...`, `fly deploy`.
* **Render**: push the repo (reads `render.yaml`); set `GROQ_API_KEY` in the dashboard.
* **Railway**: detects the `Dockerfile`/`Procfile`; add a volume and set `GROQ_API_KEY` + `FLASK_SECRET_KEY`.

## Notes & limitations

* Per-session chat history is in-process. For multiple gunicorn workers or
  instances, back it with a shared store (e.g. Redis).
* No user authentication yet -- put it behind your own auth/proxy before exposing
  sensitive documents.
* FastEmbed downloads its embedding model on first use, so the first request
  after a cold start is slower.
