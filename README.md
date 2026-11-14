# GroundTruth AI

An enterprise RAG workbench with document-level access control, four retrieval modes, grounded citations, traces, and an ablation benchmark. This is a **portfolio reference implementation**, not a claim of measured production performance.

For a public portfolio demo, see [the deployment guide](docs/deployment.md). GitHub Pages alone cannot host the Python API or PostgreSQL database.

## What is implemented

- FastAPI ingestion for PDF, Markdown, text, and a SQLAlchemy `SELECT` adapter for structured SQL sources.
- PostgreSQL + pgvector dense search, BM25, reciprocal-rank-fused hybrid search, and an ONNX cross-encoder rerank mode.
- JWT group claims and a document ACL predicate in the retrieval SQL query. Unauthorized passages are never passed to the generator.
- Evidence gate, citation ID validation, prompt/model version fields, query traces, latency, token usage, and configurable cost estimates.
- 150 labeled synthetic evaluation cases across answerable, unanswerable, conflicting-document, and restricted-document scenarios.
- Next.js query workbench and operational metrics table, Docker Compose, Phoenix OTLP export, and GitHub Actions.
- Permission-filtered source catalog and an optional, daily-capped employee-only demo token for portfolio visitors.

## Run

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or configure Gemini below), a random `JWT_SECRET` (32+ characters), and a separate random `ADMIN_TOKEN`.
2. Run `docker compose up --build`.
3. Open [the workbench](http://localhost:3000), [API docs](http://localhost:8000/docs), and [Phoenix](http://localhost:6006).
4. Create a demo user token with `python -m groundtruth.token --sub demo --groups employee` from an environment with `JWT_SECRET` set. Paste it into the UI.
5. Seed the synthetic corpus with `docker compose exec api python -m groundtruth.benchmark seed`.
6. Run an ablation mode: `docker compose exec api python -m groundtruth.benchmark evaluate --mode dense --output /app/dense-results.json`. Repeat for `bm25`, `hybrid`, and `hybrid_rerank`. The result JSON includes aggregate metrics and each case's failure details.
7. Copy results to the host with `docker compose cp api:/app/dense-results.json ./dense-results.json` (and repeat per mode), then load them in the UI's ablation explorer.
8. For optional LLM-judged metrics, install `pip install -e '.[eval]'` and run `python -m groundtruth.ragas_eval dense-results.json dense-ragas.json` outside the container. This makes additional evaluator model calls.

The first rerank call downloads its cross-encoder weights, so allow network access and expect a cold-start delay. Live inference requires an API key. No benchmark scores are checked in or invented.

Seeding is resumable: existing synthetic document IDs are skipped instead of embedded again. It waits one second between new documents by default; use `seed --delay 2` for a slower run. If Gemini returns 429, check the rate-limit details in Google AI Studio and retry after the applicable minute or daily quota resets. Each successfully ingested document is committed before the next one.

## Input data and Gemini

The default seed is **synthetic**, not a copy of real company documents. `groundtruth/benchmark.py` defines 30 short policy statements (travel, expenses, security, retention, and more). Seeding creates 90 documents: each policy, a conflicting legacy variant, and an admin-only confidential variant. The 150 template-derived questions are evaluation inputs, not source documents. Nothing scans a folder automatically.

For your own data, use the admin-protected `POST /documents/file` for text-based PDFs, Markdown, or `.txt` (10 MB max), or `POST /documents/text` for raw content. The SQLAlchemy `ingest_sql_rows` adapter is available in Python for structured database rows, but has no HTTP endpoint. Scanned PDFs need OCR before upload. Set `audience` to the groups permitted to read a document; an empty audience makes it readable by any authenticated user.

To use a Gemini API key, set these in your gitignored `.env` before starting the API and before seeding:

```dotenv
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
EMBEDDING_MODEL=gemini-embedding-001
CHAT_MODEL=gemini-3.5-flash-lite
EMBEDDING_DIMENSIONS=1536
```

Keep the key out of chat and commits. Gemini embeddings use Google's native embedding endpoint and 1536 dimensions; generation uses Google's OpenAI-compatible chat endpoint. Do not mix vectors from different embedding models in one database: use a fresh database and reseed when switching providers. The default cost rates are illustrative and should be changed to your actual Gemini pricing. The optional Ragas path is currently configured for OpenAI; the main query and benchmark paths support Gemini.

The publication-ready study structure is in [`docs/ablation-study.md`](docs/ablation-study.md). Fill its table from measured runs, then add a short human-reviewed failure analysis before citing improvements on a resume.

## Metrics and limits

The benchmark reports Recall@5, MRR, citation ID validity, exact expected-answer containment, p95 latency, permission leaks, and estimated cost per successful case. **Citation ID validity is not semantic citation correctness**, and exact containment is only a proxy for grounded-answer accuracy. The optional Ragas pass measures faithfulness, context precision, context recall, and response relevancy for answered cases. Human review is still needed before making quality claims. The 150 cases are template-derived synthetic policy scenarios, not a human-curated enterprise dataset.

BM25 currently loads authorized chunks into memory for transparent scoring. A large deployment needs an indexed lexical engine and tenant-scale performance tests. The JWT uses a local shared secret for demonstration; production deployments should verify an external identity provider's issuer, audience, keys, and expiry. Admin token and UI CORS are configured for local development. The SQL adapter accepts a caller-constructed SQLAlchemy `SELECT`, never raw user SQL.

## Architecture

`Document -> chunks + embeddings -> ACL-filtered dense/BM25 -> RRF -> optional cross-encoder -> evidence gate -> LLM -> validated citations -> trace`

Prompts are versioned as `grounded-v1`. Traces store IDs and metrics, not question text. The default per-million token rates are illustrative; update them in `.env` to match your provider's current pricing. Changing embedding dimensions after table creation requires a database migration.
