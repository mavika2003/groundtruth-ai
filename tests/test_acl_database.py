import uuid

import pytest
from groundtruth.db import Chunk, Document, SessionLocal, init_db
from groundtruth.ingest import ingest_text
from groundtruth.retrieval import retrieve
from groundtruth.security import Principal, authorized_chunks


def test_restricted_chunk_never_enters_retrieval():
    try:
        init_db()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    prefix = str(uuid.uuid4())
    public_id, secret_id = f"{prefix}-public", f"{prefix}-secret"
    with SessionLocal() as session:
        session.add_all([
            Document(id=public_id, title="Public", source="test", audience=["employee"]),
            Document(id=secret_id, title="Secret", source="test", audience=["admin"]),
        ])
        vector = [1.0] + [0.0] * 1535
        session.add_all([
            Chunk(id=f"{public_id}:0", document_id=public_id, ordinal=0, passage="Travel requires manager approval", embedding=vector),
            Chunk(id=f"{secret_id}:0", document_id=secret_id, ordinal=0, passage="Travel requires secret approval", embedding=vector),
        ])
        session.flush()
        employee = Principal("employee", frozenset({"employee"}))
        visible = session.scalars(authorized_chunks(session, employee)).all()
        assert f"{public_id}:0" in [chunk.id for chunk in visible]
        assert f"{secret_id}:0" not in [chunk.id for chunk in visible]
        for mode in ("dense", "bm25", "hybrid"):
            hits = retrieve(session, employee, "travel approval", vector, mode)
            assert all(hit.document_id != secret_id for hit in hits)
        session.rollback()


def test_reingest_replaces_old_chunks(monkeypatch):
    try:
        init_db()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    monkeypatch.setattr("groundtruth.ingest.embed", lambda texts: [[1.0] + [0.0] * 1535 for _ in texts])
    document_id = str(uuid.uuid4())
    with SessionLocal() as session:
        ingest_text(session, document_id, "Policy", "test", ["employee"], "Old deadline is Monday.")
        ingest_text(session, document_id, "Policy", "test", ["employee"], "New deadline is Friday.")
        session.refresh(session.get(Document, document_id))
        chunks = session.get(Document, document_id).chunks
        assert len(chunks) == 1
        assert chunks[0].passage == "New deadline is Friday."
        assert session.get(Document, document_id).version == 2
        session.delete(session.get(Document, document_id))
        session.commit()
