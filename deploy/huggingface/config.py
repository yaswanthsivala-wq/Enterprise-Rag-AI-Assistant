"""Centralized configuration.

All environment-driven settings live here so the rest of the app never reads
os.environ directly. New canonical (UPPERCASE) names are preferred, with the
original mixed-case names accepted as a fallback so existing deployments keep
working.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get(*names, default=None):
    """Return the first non-empty env var among ``names``."""
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


# --- Storage (override with absolute paths when mounting a cloud volume) -----
UPLOAD_FOLDER = _get("UPLOAD_FOLDER", default="uploads")
VECTOR_DB = _get("VECTOR_DB", default="vector_store")

# --- Flask -------------------------------------------------------------------
# Canonical: FLASK_SECRET_KEY. Legacy fallback: Flask_App_Secret_Key.
SECRET_KEY = _get("FLASK_SECRET_KEY", "Flask_App_Secret_Key")

# --- LLM provider ------------------------------------------------------------
# "groq" (free, no credit card) is the default. "watsonx" is also supported
# (requires `pip install -r requirements-watsonx.txt`).
LLM_PROVIDER = _get("LLM_PROVIDER", default="groq").lower()

# Groq (https://console.groq.com -> API Keys). Free tier, key starts with gsk_.
GROQ_API_KEY = _get("GROQ_API_KEY")
GROQ_MODEL = _get("GROQ_MODEL", default="llama-3.3-70b-versatile")

# IBM watsonx.ai (optional alternative provider)
IBM_API_KEY = _get("IBM_API_KEY")
IBM_URL = _get("IBM_URL")
IBM_PROJECT_ID = _get("IBM_PROJECT_ID", "IBM_Project_id")
MODEL_ID = _get("IBM_MODEL_ID", default="mistralai/mistral-small-3-1-24b-instruct-2503")

# --- Generation + retrieval tunables ----------------------------------------
TEMPERATURE = float(_get("LLM_TEMPERATURE", "IBM_TEMPERATURE", default="0.3"))
MAX_NEW_TOKENS = int(_get("LLM_MAX_TOKENS", "IBM_MAX_NEW_TOKENS", default="400"))
RETRIEVAL_K = int(_get("RETRIEVAL_K", default="4"))
CHUNK_SIZE = int(_get("CHUNK_SIZE", default="500"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", default="50"))

# --- Limits ------------------------------------------------------------------
MAX_CONTENT_LENGTH = int(_get("MAX_UPLOAD_MB", default="25")) * 1024 * 1024
MAX_HISTORY_TURNS = int(_get("MAX_HISTORY_TURNS", default="20"))


def missing_llm_vars():
    """Return required env vars for the selected provider that are not set."""
    if LLM_PROVIDER == "groq":
        return [] if GROQ_API_KEY else ["GROQ_API_KEY"]
    if LLM_PROVIDER == "watsonx":
        missing = []
        if not IBM_API_KEY:
            missing.append("IBM_API_KEY")
        if not IBM_URL:
            missing.append("IBM_URL")
        if not IBM_PROJECT_ID:
            missing.append("IBM_PROJECT_ID")
        return missing
    return [f"unknown LLM_PROVIDER '{LLM_PROVIDER}' (use 'groq' or 'watsonx')"]


def llm_config_present():
    return not missing_llm_vars()
