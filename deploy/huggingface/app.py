"""Flask entrypoint for the Enterprise RAG assistant."""

import os
import uuid
import shutil
import logging
import threading

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge, HTTPException

import config
from rag_pipeline import load_and_embed_pdfs, answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("rag_app")

app = Flask(__name__)

# Support both layouts: templates/index.html (local/dev) and index.html at the
# app root (e.g. a flat file upload to Hugging Face Spaces).
_HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(_HERE, "templates", "index.html")):
    app.template_folder = "."
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

if config.SECRET_KEY:
    app.secret_key = config.SECRET_KEY
else:
    app.secret_key = os.urandom(32)
    logger.warning(
        "FLASK_SECRET_KEY not set; using an ephemeral key. Sessions reset on restart."
    )

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Per-session chat history, isolated by browser session id.
# NOTE: this is in-process only. For multi-worker / multi-instance deployments,
# back this with a shared store (e.g. Redis). The default gunicorn config runs a
# single worker so a single process owns all sessions.
_histories = {}
_hist_lock = threading.Lock()


def _session_id():
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    return sid


def _session_paths(sid):
    """Each browser session (chat) gets its own upload folder and FAISS index,
    so documents uploaded in one chat never leak into another chat's answers."""
    upload_dir = os.path.join(config.UPLOAD_FOLDER, sid)
    vector_dir = os.path.join(config.VECTOR_DB, sid)
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir, vector_dir


def _get_history(sid):
    with _hist_lock:
        return list(_histories.get(sid, []))


def _append_history(sid, question, answer):
    with _hist_lock:
        hist = _histories.setdefault(sid, [])
        hist.append({"question": question, "answer": answer})
        if len(hist) > config.MAX_HISTORY_TURNS:
            del hist[: len(hist) - config.MAX_HISTORY_TURNS]


@app.after_request
def _security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    # Allow the Hugging Face Spaces preview to embed this app in its iframe,
    # while still blocking other sites (clickjacking protection). frame-ancestors
    # is the modern replacement for X-Frame-Options and supports a whitelist.
    resp.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://huggingface.co https://*.hf.space"
    )
    return resp


@app.errorhandler(RequestEntityTooLarge)
def _too_large(_e):
    mb = config.MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify({"error": f"Upload too large. Limit is {mb} MB."}), 413


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "provider": config.LLM_PROVIDER,
            "llm_configured": config.llm_config_present(),
        }
    )


def _is_pdf(filename):
    return filename.lower().endswith(".pdf")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        files = [f for f in request.files.getlist("files") if f and f.filename]
        if not files:
            return jsonify({"error": "No files selected"}), 400

        if any(not _is_pdf(f.filename) for f in files):
            return jsonify({"error": "Only .pdf files are accepted."}), 400

        sid = _session_id()
        upload_dir, vector_dir = _session_paths(sid)

        file_paths = []
        for f in files:
            filename = secure_filename(f.filename)
            if not filename:
                continue
            path = os.path.join(upload_dir, filename)
            f.save(path)
            file_paths.append(path)

        chunks = load_and_embed_pdfs(file_paths, vector_db_path=vector_dir)
        if chunks == 0:
            return jsonify(
                {"message": "No extractable text found (are these scanned PDFs?)."}
            )
        return jsonify({"message": f"Processed successfully with {chunks} chunks."})
    except HTTPException:
        raise
    except Exception:
        logger.exception("Upload failed")
        return jsonify({"error": "Failed to process the uploaded file(s)."}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Question is required."}), 400

        sid = _session_id()
        _, vector_dir = _session_paths(sid)
        answer, sources = answer_question(
            question, _get_history(sid), vector_db_path=vector_dir
        )
        _append_history(sid, question, answer)
        return jsonify({"answer": answer, "sources": sources})
    except HTTPException:
        raise
    except RuntimeError as e:
        # Configuration problems (e.g. watsonx not set up). The message is safe
        # to surface and helps the operator fix their env.
        logger.warning("Ask failed (configuration): %s", e)
        return jsonify({"error": str(e)}), 503
    except Exception:
        logger.exception("Ask failed")
        return jsonify({"error": "Failed to generate a response."}), 500


@app.route("/documents")
def documents():
    try:
        sid = _session_id()
        upload_dir, _ = _session_paths(sid)
        return jsonify(sorted(os.listdir(upload_dir)))
    except FileNotFoundError:
        return jsonify([])
    except Exception:
        logger.exception("Listing documents failed")
        return jsonify([]), 500


@app.route("/clear-memory", methods=["POST"])
def clear_memory():
    """Used by the 'New chat' button: wipes this session's chat history AND its
    uploaded documents/vector store, so the next chat starts from a clean slate
    and only ever answers from whatever gets uploaded in it."""
    sid = _session_id()
    with _hist_lock:
        _histories.pop(sid, None)

    upload_dir, vector_dir = _session_paths(sid)
    shutil.rmtree(upload_dir, ignore_errors=True)
    shutil.rmtree(vector_dir, ignore_errors=True)
    os.makedirs(upload_dir, exist_ok=True)

    return jsonify({"message": "Chat memory cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
