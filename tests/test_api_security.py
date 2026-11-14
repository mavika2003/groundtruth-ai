import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from groundtruth.api import app
from groundtruth.config import settings
from groundtruth.db import Chunk, Document, SessionLocal, init_db


def test_demo_token_is_employee_only_and_opt_in(monkeypatch):
    from fastapi import HTTPException
    from groundtruth.api import demo_token

    monkeypatch.setattr(settings, "public_demo_enabled", False)
    with pytest.raises(HTTPException) as error:
        demo_token()
    assert error.value.status_code == 404

    monkeypatch.setattr(settings, "public_demo_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", "integration-test-secret-32-characters-long")
    payload = demo_token()
    claims = jwt.decode(payload["token"], settings.jwt_secret, algorithms=["HS256"])
    assert claims["sub"] == "public-demo"
    assert claims["groups"] == ["employee"]


def test_document_catalog_respects_acl(monkeypatch):
    try:
        init_db()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    monkeypatch.setattr(settings, "jwt_secret", "integration-test-secret-32-characters-long")
    public_id, secret_id = str(uuid.uuid4()), str(uuid.uuid4())
    with SessionLocal() as session:
        session.add_all([Document(id=public_id, title="Visible", source="test", audience=["employee"]),
                         Document(id=secret_id, title="Hidden", source="test", audience=["admin"])])
        session.commit()
    token = jwt.encode({"sub": "catalog-user", "groups": ["employee"],
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=5)}, settings.jwt_secret, algorithm="HS256")
    try:
        with TestClient(app) as client:
            response = client.get("/documents", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200, response.text
        ids = {item["document_id"] for item in response.json()["documents"]}
        assert public_id in ids
        assert secret_id not in ids
    finally:
        with SessionLocal() as session:
            session.delete(session.get(Document, public_id))
            session.delete(session.get(Document, secret_id))
            session.commit()


def test_generator_receives_only_authorized_passages(monkeypatch):
    try:
        init_db()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    monkeypatch.setattr(settings, "jwt_secret", "integration-test-secret-32-characters-long")
    monkeypatch.setattr("groundtruth.api.embed", lambda _, **kwargs: [[1.0] + [0.0] * 1535])
    prefix = str(uuid.uuid4())
    public_id, secret_id = f"{prefix}-public", f"{prefix}-secret"
    vector = [1.0] + [0.0] * 1535
    with SessionLocal() as session:
        session.add_all([Document(id=public_id, title="Public", source="test", audience=["employee"]),
                         Document(id=secret_id, title="Secret", source="test", audience=["admin"])])
        session.add_all([Chunk(id=f"{public_id}:0", document_id=public_id, ordinal=0, passage="Travel requires manager approval", embedding=vector),
                         Chunk(id=f"{secret_id}:0", document_id=secret_id, ordinal=0, passage="Travel requires secret approval", embedding=vector)])
        session.commit()
    observed = []

    def fake_generate(question, passages):
        observed.extend(passages)
        return f"Manager approval [{public_id}:0]", [f"{public_id}:0"], 100, 20

    monkeypatch.setattr("groundtruth.api.generate", fake_generate)
    token = jwt.encode({"sub": "test-user", "groups": ["employee"], "exp": datetime.now(timezone.utc) + timedelta(minutes=5)}, settings.jwt_secret, algorithm="HS256")
    try:
        with TestClient(app) as client:
            response = client.post("/query", json={"question": "travel approval", "mode": "hybrid"}, headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "answered"
        assert public_id in {item["document_id"] for item in observed}
        assert secret_id not in {item["document_id"] for item in observed}
        assert {item["document_id"] for item in response.json()["citations"]} == {public_id}
    finally:
        with SessionLocal() as session:
            session.delete(session.get(Document, public_id))
            session.delete(session.get(Document, secret_id))
            session.commit()
