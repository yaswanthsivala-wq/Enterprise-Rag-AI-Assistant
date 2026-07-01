"""Retrieval-Augmented Generation pipeline.

Heavy dependencies (FastEmbed, FAISS, ibm-watsonx-ai) are imported lazily inside
functions, and the embedding model / LLM client are created on first use rather
than at import time. This means:
  * importing this module never crashes due to missing credentials, and
  * the process starts fast and only pays the model-download / connection cost
    when a request actually needs it.
"""

import os
import logging
import threading

import config

logger = logging.getLogger(__name__)

_embeddings = None
_groq_client = None
_watsonx_model = None
_lock = threading.Lock()

_SYSTEM_PROMPT = (
    "You are an enterprise AI assistant. Answer using the provided document "
    "context and conversation history. If the answer is not in the context, "
    "say you don't know rather than guessing."
)


def _get_embeddings():
    """Lazily build and cache the FastEmbed embeddings model."""
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                from langchain_community.embeddings.fastembed import (
                    FastEmbedEmbeddings,
                )

                logger.info("Initializing FastEmbed embeddings (first use)")
                _embeddings = FastEmbedEmbeddings()
    return _embeddings


def _get_groq_client():
    """Lazily build and cache the Groq client."""
    global _groq_client
    if _groq_client is None:
        with _lock:
            if _groq_client is None:
                from groq import Groq

                logger.info("Initializing Groq client (first use)")
                _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_watsonx_model():
    """Lazily build and cache the watsonx model client (optional provider)."""
    global _watsonx_model
    if _watsonx_model is None:
        with _lock:
            if _watsonx_model is None:
                from ibm_watsonx_ai import Credentials
                from ibm_watsonx_ai.foundation_models import ModelInference

                logger.info("Initializing watsonx model '%s' (first use)", config.MODEL_ID)
                _watsonx_model = ModelInference(
                    model_id=config.MODEL_ID,
                    credentials=Credentials(
                        api_key=config.IBM_API_KEY,
                        url=config.IBM_URL,
                    ),
                    project_id=config.IBM_PROJECT_ID,
                    params={
                        "temperature": config.TEMPERATURE,
                        "max_new_tokens": config.MAX_NEW_TOKENS,
                    },
                )
    return _watsonx_model


def _generate_text(prompt):
    """Dispatch generation to the configured provider.

    Raises RuntimeError with a clear message if the provider is not configured,
    so the caller can surface exactly which env var is missing.
    """
    missing = config.missing_llm_vars()
    if missing:
        raise RuntimeError("LLM is not configured. Missing: " + ", ".join(missing))

    if config.LLM_PROVIDER == "groq":
        client = _get_groq_client()
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_NEW_TOKENS,
        )
        return completion.choices[0].message.content

    if config.LLM_PROVIDER == "watsonx":
        model = _get_watsonx_model()
        return _extract_generated_text(model.generate_text(prompt=prompt))

    raise RuntimeError(f"Unsupported LLM_PROVIDER '{config.LLM_PROVIDER}'")


def extract_text_from_pdf(pdf_path):
    """Extract all selectable text from a PDF using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def chunk_text(text):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def _vector_store_exists():
    return os.path.exists(os.path.join(config.VECTOR_DB, "index.faiss"))


def load_and_embed_pdfs(pdf_paths):
    """Embed the given PDFs and ADD them to the vector store.

    Unlike the original implementation, this appends to an existing index rather
    than overwriting it, so uploading a second batch does not discard the first.
    Returns the number of new chunks added.
    """
    from langchain_community.vectorstores import FAISS

    embeddings = _get_embeddings()

    all_chunks, metadata = [], []
    for pdf_path in pdf_paths:
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            logger.warning("No extractable text in %s (scanned PDF?)", pdf_path)
            continue
        filename = os.path.basename(pdf_path)
        for i, chunk in enumerate(chunk_text(raw_text)):
            all_chunks.append(chunk)
            metadata.append({"source": filename, "chunk": i + 1})

    if not all_chunks:
        return 0

    if _vector_store_exists():
        store = FAISS.load_local(
            config.VECTOR_DB, embeddings, allow_dangerous_deserialization=True
        )
        store.add_texts(all_chunks, metadatas=metadata)
    else:
        store = FAISS.from_texts(all_chunks, embeddings, metadatas=metadata)

    store.save_local(config.VECTOR_DB)
    return len(all_chunks)


def _extract_generated_text(response):
    """Normalize watsonx generate_text output.

    ModelInference.generate_text returns a plain string by default, but can
    return a dict when raw_response options are used. Handle both defensively.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        results = response.get("results")
        if results and isinstance(results, list):
            return results[0].get("generated_text", "")
    return str(response)


def answer_question(question, chat_history):
    if not _vector_store_exists():
        return ("No documents uploaded yet.", [])

    from langchain_community.vectorstores import FAISS

    embeddings = _get_embeddings()
    store = FAISS.load_local(
        config.VECTOR_DB, embeddings, allow_dangerous_deserialization=True
    )
    docs = store.similarity_search(question, k=config.RETRIEVAL_K)

    context = "\n\n".join(d.page_content for d in docs)
    sources = [
        f"{d.metadata.get('source', '?')} (Chunk {d.metadata.get('chunk', '?')})"
        for d in docs
    ]

    history_text = ""
    for item in chat_history[-5:]:
        history_text += (
            f"\nPrevious Question:\n{item['question']}\n"
            f"\nPrevious Answer:\n{item['answer']}\n"
        )

    prompt = f"""You are an enterprise AI assistant. Use the previous conversation and the retrieved document context to answer the current question accurately. If the answer is not contained in the context, say you don't know.

Conversation History:
{history_text}

Document Context:
{context}

Current Question:
{question}

Answer:
"""

    answer = _generate_text(prompt)
    return answer, sources
