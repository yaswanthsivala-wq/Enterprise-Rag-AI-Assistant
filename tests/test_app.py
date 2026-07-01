import io


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["provider"] == "groq"
    assert body["llm_configured"] is False


def test_home_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<html" in r.data.lower()


def test_security_headers(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    # Embeddable by Hugging Face, but frame-ancestors still restricts others.
    assert "huggingface.co" in r.headers["Content-Security-Policy"]
    assert "frame-ancestors" in r.headers["Content-Security-Policy"]


def test_documents_empty(client):
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.get_json() == []


def test_upload_no_file(client):
    r = client.post("/upload", data={})
    assert r.status_code == 400
    assert "No files" in r.get_json()["error"]


def test_upload_rejects_non_pdf(client):
    data = {"files": (io.BytesIO(b"hello"), "notes.txt")}
    r = client.post("/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert ".pdf" in r.get_json()["error"]


def test_upload_success(client, appmod, monkeypatch):
    monkeypatch.setattr(appmod, "load_and_embed_pdfs", lambda paths: 7)
    data = {"files": (io.BytesIO(b"%PDF-1.4 fake"), "doc.pdf")}
    r = client.post("/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert "7 chunks" in r.get_json()["message"]


def test_ask_requires_question(client):
    r = client.post("/ask", json={})
    assert r.status_code == 400


def test_ask_success(client, appmod, monkeypatch):
    monkeypatch.setattr(appmod, "answer_question", lambda q, h: ("an answer", ["s.pdf (Chunk 1)"]))
    r = client.post("/ask", json={"question": "what is x?"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["answer"] == "an answer"
    assert body["sources"] == ["s.pdf (Chunk 1)"]


def test_error_response_is_sanitized(client, appmod, monkeypatch):
    def boom(q, h):
        raise ValueError("internal path /etc/secret leaked")

    monkeypatch.setattr(appmod, "answer_question", boom)
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 500
    body = r.get_json()
    assert "secret" not in body["error"].lower()
    assert body["error"] == "Failed to generate a response."


def test_session_isolation_and_history(appmod, monkeypatch):
    seen = []

    def capture(q, h):
        seen.append(list(h))
        return ("ok", [])

    monkeypatch.setattr(appmod, "answer_question", capture)
    appmod.app.config.update(TESTING=True)

    a = appmod.app.test_client()
    b = appmod.app.test_client()

    a.post("/ask", json={"question": "a1"})
    a.post("/ask", json={"question": "a2"})  # should see 1 prior turn
    b.post("/ask", json={"question": "b1"})  # separate session: sees 0 prior

    assert len(seen[0]) == 0          # a1: empty
    assert len(seen[1]) == 1          # a2: one prior (a1)
    assert seen[1][0]["question"] == "a1"
    assert len(seen[2]) == 0          # b1: isolated from a


def test_clear_memory_resets_history(appmod, monkeypatch):
    seen = []
    monkeypatch.setattr(appmod, "answer_question", lambda q, h: (seen.append(list(h)) or ("ok", [])))
    appmod.app.config.update(TESTING=True)
    c = appmod.app.test_client()

    c.post("/ask", json={"question": "q1"})
    c.post("/clear-memory")
    c.post("/ask", json={"question": "q2"})
    assert len(seen[-1]) == 0  # history cleared


def test_upload_too_large(client, appmod):
    appmod.app.config["MAX_CONTENT_LENGTH"] = 10  # 10 bytes
    data = {"files": (io.BytesIO(b"x" * 5000), "big.pdf")}
    r = client.post("/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 413
    appmod.app.config["MAX_CONTENT_LENGTH"] = None  # reset for other tests
