import secrets
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import jwt
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from groundtruth.config import settings
from groundtruth.db import Chunk, Document, QueryTrace, SessionLocal, init_db
from groundtruth.ingest import extract_file, ingest_text
from groundtruth.providers import embed, generate
from groundtruth.retrieval import evidence_sufficient, retrieve
from groundtruth.security import Principal, authorized_documents, principal_from_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider = TracerProvider(resource=Resource.create({"service.name": "groundtruth-api"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces")))
        trace.set_tracer_provider(provider)
    yield


app = FastAPI(title="GroundTruth AI", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()], allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type", "X-Admin-Token"])


@app.exception_handler(httpx.HTTPStatusError)
def provider_http_error(_request, exc: httpx.HTTPStatusError):
    if exc.response.status_code == 429:
        return JSONResponse(status_code=429, content={"detail": "Model provider rate limit reached. Try again later."})
    return JSONResponse(status_code=502, content={"detail": f"Model provider request failed ({exc.response.status_code})."})


@app.exception_handler(httpx.TimeoutException)
def provider_timeout(_request, _exc: httpx.TimeoutException):
    return JSONResponse(status_code=504, content={"detail": "Model provider timed out. Try again later."})


def db_session():
    with SessionLocal() as session:
        yield session


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    if not settings.admin_token or not x_admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(403, "Admin token required")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid"


class TextDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=255)
    audience: list[str] = Field(default_factory=list)
    content: str = Field(min_length=1, max_length=500_000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/demo")
def demo_token():
    if not settings.public_demo_enabled or len(settings.jwt_secret) < 32:
        raise HTTPException(404, "Demo access is not enabled")
    token = jwt.encode({"sub": "public-demo", "groups": ["employee"],
                        "exp": datetime.now(timezone.utc) + timedelta(hours=1)}, settings.jwt_secret, algorithm="HS256")
    return {"token": token, "expires_in_seconds": 3600, "daily_query_limit": settings.demo_daily_limit}


@app.post("/documents/text", dependencies=[Depends(require_admin)])
def add_text(request: TextDocumentRequest, session: Session = Depends(db_session)):
    count = ingest_text(session, request.document_id, request.title, "api", request.audience, request.content)
    return {"document_id": request.document_id, "chunks": count}


@app.post("/documents/file", dependencies=[Depends(require_admin)])
async def add_file(file: UploadFile, document_id: str, audience: str = "", session: Session = Depends(db_session)):
    if file.size and file.size > 10_000_000:
        raise HTTPException(413, "File exceeds 10 MB")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".md", ".txt"}:
        raise HTTPException(400, "Unsupported file type")
    data = await file.read(10_000_001)
    if len(data) > 10_000_000:
        raise HTTPException(413, "File exceeds 10 MB")
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(data)
        content = extract_file(path)
    count = ingest_text(session, document_id, file.filename or document_id, "file", [group.strip() for group in audience.split(",") if group.strip()], content)
    return {"document_id": document_id, "chunks": count}


@app.get("/documents")
def list_documents(principal: Principal = Depends(principal_from_token), session: Session = Depends(db_session)):
    permitted = authorized_documents(principal).subquery()
    rows = session.execute(select(permitted.c.id, permitted.c.title, permitted.c.source,
                                  permitted.c.version, func.count(Chunk.id).label("chunks"))
                           .outerjoin(Chunk, Chunk.document_id == permitted.c.id)
                           .group_by(permitted.c.id, permitted.c.title, permitted.c.source, permitted.c.version)
                           .order_by(permitted.c.title)).all()
    return {"documents": [{"document_id": row.id, "title": row.title, "source": row.source,
                            "version": row.version, "chunks": row.chunks} for row in rows]}


@app.post("/query")
def query(request: QueryRequest, principal: Principal = Depends(principal_from_token), session: Session = Depends(db_session)):
    if principal.subject == "public-demo":
        if not settings.public_demo_enabled:
            raise HTTPException(403, "Demo access is disabled")
        # Hold a transaction-scoped lock through the query so concurrent demo requests share one daily cap.
        session.execute(text("SELECT pg_advisory_xact_lock(61724519031)"))
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        used = session.scalar(select(func.count()).select_from(QueryTrace).where(
            QueryTrace.subject == "public-demo", QueryTrace.created_at >= today))
        if used >= settings.demo_daily_limit:
            session.rollback()
            raise HTTPException(429, "Public demo query limit reached for today")
    start = time.perf_counter()
    trace_id = str(uuid.uuid4())
    status = "error"
    answer = ""
    citations: list[dict] = []
    hits = []
    input_tokens = output_tokens = 0
    tracer = trace.get_tracer("groundtruth.query")
    try:
        with tracer.start_as_current_span("embedding") as span:
            span.set_attribute("model", settings.embedding_model)
            vector = embed([request.question], purpose="query")[0]
        with tracer.start_as_current_span("retrieval") as span:
            span.set_attribute("mode", request.mode)
            hits = retrieve(session, principal, request.question, vector, request.mode)
            span.set_attribute("retrieved_count", len(hits))
        if not evidence_sufficient(request.question, hits):
            status = "insufficient_evidence"
            answer = "I cannot determine the answer from the documents you can access."
        else:
            passage_map = {hit.chunk_id: hit for hit in hits}
            with tracer.start_as_current_span("generation") as span:
                span.set_attribute("model", settings.chat_model)
                span.set_attribute("prompt_version", "grounded-v1")
                answer, cited_ids, input_tokens, output_tokens = generate(request.question, [hit.__dict__ for hit in hits])
                span.set_attribute("input_tokens", input_tokens)
                span.set_attribute("output_tokens", output_tokens)
            citations = [{"chunk_id": key, "document_id": passage_map[key].document_id, "passage": passage_map[key].passage}
                         for key in cited_ids if key in passage_map]
            if "cannot determine" in answer.lower():
                status = "insufficient_evidence"
                citations = []
            elif not citations:
                status = "ungrounded"
                answer = "I cannot verify an answer with citations from the available documents."
            else:
                status = "answered"
        return {"trace_id": trace_id, "status": status, "answer": answer, "citations": citations,
                "retrieved": [hit.__dict__ for hit in hits], "model_version": settings.chat_model, "prompt_version": "grounded-v1"}
    finally:
        elapsed = round((time.perf_counter() - start) * 1000)
        cost = (input_tokens * settings.input_cost_per_million + output_tokens * settings.output_cost_per_million) / 1_000_000
        session.add(QueryTrace(id=trace_id, subject=principal.subject, mode=request.mode, status=status,
                               prompt_version="grounded-v1", model_version=settings.chat_model, latency_ms=elapsed,
                               input_tokens=input_tokens, output_tokens=output_tokens, estimated_cost_usd=cost,
                               retrieved_ids=[hit.chunk_id for hit in hits]))
        session.commit()


@app.get("/metrics", dependencies=[Depends(require_admin)])
def metrics(session: Session = Depends(db_session)):
    traces = session.scalars(select(QueryTrace).order_by(QueryTrace.created_at.desc()).limit(2000)).all()
    by_mode: dict[str, list[QueryTrace]] = defaultdict(list)
    for item in traces:
        by_mode[item.mode].append(item)
    result = []
    for mode, items in sorted(by_mode.items()):
        latencies = sorted(item.latency_ms for item in items)
        successes = sum(item.status == "answered" for item in items)
        result.append({"mode": mode, "queries": len(items), "answer_rate": successes / len(items),
                       "p95_latency_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
                       "cost_per_answer_usd": sum(item.estimated_cost_usd for item in items) / successes if successes else None,
                       "retrieval_failures": sum(item.status == "insufficient_evidence" for item in items)})
    return {"sample_size": len(traces), "modes": result}
