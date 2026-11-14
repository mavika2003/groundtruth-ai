import pytest

from groundtruth.config import settings
from groundtruth.providers import embed, generate


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_gemini_embedding_uses_task_type_and_normalizes(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            calls.append((url, headers, json))
            return FakeResponse({"embedding": {"values": [3.0, 4.0]}})

    monkeypatch.setattr("groundtruth.providers.httpx.Client", FakeClient)
    monkeypatch.setattr(settings, "model_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "embedding_dimensions", 2)
    assert embed(["a"], purpose="document")[0] == pytest.approx([0.6, 0.8])
    assert embed(["b"], purpose="query")[0] == pytest.approx([0.6, 0.8])
    assert calls[0][2]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert calls[1][2]["taskType"] == "RETRIEVAL_QUERY"
    assert calls[0][2]["outputDimensionality"] == 2
    assert calls[0][1]["x-goog-api-key"] == "test-key"


def test_gemini_generation_uses_compatible_endpoint(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            calls.append((url, headers, json))
            return FakeResponse({"choices": [{"message": {"content": "Yes [doc:0]"}}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 3}})

    monkeypatch.setattr("groundtruth.providers.httpx.Client", FakeClient)
    monkeypatch.setattr(settings, "model_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "chat_model", "gemini-3.8-flash")
    assert generate("Question?", [{"chunk_id": "doc:0", "passage": "Yes"}]) == ("Yes [doc:0]", ["doc:0"], 10, 3)
    assert calls[0][0].endswith("/v1beta/openai/chat/completions")
    assert calls[0][1]["Authorization"] == "Bearer test-key"
