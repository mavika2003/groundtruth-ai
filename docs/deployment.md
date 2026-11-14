# Deploying GroundTruth AI as a portfolio demo

This is a hobby deployment, not an enterprise production environment. The recommended split is **Vercel (Next.js UI) + Render Free (FastAPI) + Neon Free (PostgreSQL/pgvector)**. Gemini remains the external model provider. GitHub Pages can serve a static frontend, but it cannot run FastAPI, PostgreSQL, or embeddings; it would still need the same hosted backend and database.

Check the providers' current terms and limits before deployment: [Vercel Hobby](https://vercel.com/docs/plans/hobby), [Render Free](https://render.com/docs/free), and [Neon Free](https://neon.com/pricing). Render Free sleeps after inactivity and may take about a minute to wake. Its free PostgreSQL expires after 30 days, so use Neon for the database. A cold-start reranker can be too memory-intensive for Render's free service; use Dense, BM25, or Hybrid for the public demo.

## 1. Push the repository

Create a GitHub repository and push this project. The project currently has no Git remote configured. Check that `.env`, `.venv`, `web/node_modules`, and build output are not staged. Never put the Gemini key or Neon URL in GitHub source files.

## 2. Create the database in Neon

Create a Neon Free project and copy its **direct** Postgres connection string. For this Python app, change only the URL scheme from `postgresql://` to `postgresql+psycopg://`. Keep its host, credentials, database, and `sslmode=require` query parameter. The API creates the `vector` extension and tables on startup. Do not use the local Docker database URL for Render.

## 3. Deploy the FastAPI service on Render

Create a Render **Web Service** from the GitHub repository, choose the **Docker** runtime, and use the repository-root `Dockerfile`. Choose the Free plan and set health check path `/health`. The image reads Render's `PORT` setting automatically.

Set these Render environment variables in the dashboard, not in the repository:

| Variable | Value |
| --- | --- |
| `DATABASE_URL` | Neon direct URL with `postgresql+psycopg://` scheme |
| `MODEL_PROVIDER` | `gemini` |
| `GEMINI_API_KEY` | Your key |
| `EMBEDDING_MODEL` | `gemini-embedding-001` |
| `EMBEDDING_DIMENSIONS` | `1536` |
| `CHAT_MODEL` | `gemini-3.5-flash-lite` |
| `JWT_SECRET` | A new random 64+ character secret |
| `ADMIN_TOKEN` | A different random 64+ character secret |
| `PUBLIC_DEMO_ENABLED` | `true` only for a synthetic-data-only demo |
| `DEMO_DAILY_LIMIT` | `20` or lower |
| `CORS_ORIGINS` | Your exact Vercel URL, e.g. `https://groundtruth-ai.vercel.app` |

Leave `OTEL_EXPORTER_OTLP_ENDPOINT` unset unless you deploy a real collector. Do not expose an admin token in the frontend. The public demo endpoint issues employee-only JWTs and caps demo queries by UTC day. The cap is a basic portfolio safeguard, **not** a billing guarantee; keep an eye on Gemini usage and quotas.

Once Render deploys, visit `https://YOUR-SERVICE.onrender.com/health` and expect `{"status":"ok"}`.

## 4. Deploy the UI on Vercel

Import the same GitHub repository into Vercel. Set **Root Directory** to `web` and use the Next.js framework preset. Add `NEXT_PUBLIC_API_URL=https://YOUR-SERVICE.onrender.com` as a production environment variable. Deploy, then update Render's `CORS_ORIGINS` to the exact Vercel production URL and redeploy Render. Preview URLs need separate CORS entries if you want them to call the API.

## 5. Seed the hosted database

The local Docker volume is separate from Neon; the hosted database starts empty. From your local project directory, activate the Python environment and temporarily point the seed CLI at Neon. In zsh, this prompt keeps the database URL out of shell history:

```zsh
read -rs 'DATABASE_URL?Paste Neon SQLAlchemy URL: '
echo
export DATABASE_URL
.venv/bin/python -m groundtruth.benchmark seed --delay 2
unset DATABASE_URL
```

The URL you paste must start with `postgresql+psycopg://`. The local `.env` still supplies your Gemini key. The seed is resumable after a 429; rerunning skips documents already stored. It creates 90 synthetic documents with employee and admin ACLs. Do not seed real confidential documents into a public-demo database.

## 6. Verify the public flow

Open the Vercel URL, click **Use demo access**, ask a sample question, and inspect the cited passage. Open **Sources** to see the employee-visible corpus. For a restricted query such as `What is the confidential exception code for travel?`, confirm no `*-secret` document is retrieved. The Evaluation metrics view still requires the private admin token. No benchmark result is displayed until you run evaluations and load their output JSON.

## What a resume can honestly claim

You can claim a deployed permission-aware RAG demo after the public flow above works. Do not claim continuous evaluation, production-grade identity, semantic citation verification, or published ablation improvements until those are implemented and measured. The 150 questions are template-derived synthetic cases, not a human-curated enterprise benchmark.
