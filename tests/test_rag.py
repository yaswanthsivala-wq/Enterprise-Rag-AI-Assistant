import rag_pipeline as rp
import config


def test_chunk_text_splits_long_text():
    text = "sentence. " * 500
    chunks = rp.chunk_text(text)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_extract_generated_text_handles_str():
    assert rp._extract_generated_text("hello") == "hello"


def test_extract_generated_text_handles_dict():
    resp = {"results": [{"generated_text": "world"}]}
    assert rp._extract_generated_text(resp) == "world"


def test_extract_generated_text_fallback():
    assert rp._extract_generated_text(123) == "123"


def test_answer_question_without_store(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VECTOR_DB", str(tmp_path / "missing"))
    answer, sources = rp.answer_question("anything", [])
    assert answer == "No documents uploaded yet."
    assert sources == []


def test_missing_llm_vars_groq_default():
    # conftest leaves GROQ_API_KEY unset and provider defaults to groq.
    assert config.LLM_PROVIDER == "groq"
    assert config.missing_llm_vars() == ["GROQ_API_KEY"]
    assert config.llm_config_present() is False


def test_generate_raises_when_unconfigured():
    import pytest
    with pytest.raises(RuntimeError) as exc:
        rp._generate_text("hello")
    assert "not configured" in str(exc.value).lower()


def test_generate_with_groq(monkeypatch):
    class _Msg:
        content = "grounded answer"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            assert kwargs["model"] == config.GROQ_MODEL
            assert kwargs["messages"][-1]["content"] == "PROMPT"
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeGroq:
        chat = _Chat()

    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(rp, "_get_groq_client", lambda: _FakeGroq())
    assert rp._generate_text("PROMPT") == "grounded answer"
