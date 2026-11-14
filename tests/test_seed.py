from types import SimpleNamespace

import pytest

from groundtruth import benchmark


class FakeSession:
    def __init__(self):
        self.documents = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, model, document_id):
        return self.documents.get(document_id)


def test_seed_resumes_without_reembedding_existing_documents(monkeypatch):
    session = FakeSession()
    writes = []

    def fake_ingest(_, document_id, title, source, audience, content):
        writes.append(document_id)
        session.documents[document_id] = SimpleNamespace(source=source)

    monkeypatch.setattr(benchmark, "SessionLocal", lambda: session)
    monkeypatch.setattr(benchmark, "ingest_text", fake_ingest)

    assert benchmark.seed(0) == {"ingested": 90, "skipped": 0}
    assert benchmark.seed(0) == {"ingested": 0, "skipped": 90}
    assert len(writes) == 90


def test_seed_does_not_overwrite_another_source(monkeypatch):
    session = FakeSession()
    session.documents["travel"] = SimpleNamespace(source="user-upload")
    monkeypatch.setattr(benchmark, "SessionLocal", lambda: session)
    with pytest.raises(RuntimeError, match="already used by another source"):
        benchmark.seed(0)
