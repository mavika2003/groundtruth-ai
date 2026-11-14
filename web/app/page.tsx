"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity, ArrowRight, BookOpenText, Check, ChevronRight, CircleAlert,
  Database, FileSearch2, Gauge, KeyRound, Layers3, LockKeyhole,
  Search, ShieldCheck, Sparkles,
} from "lucide-react";
import BenchmarkView from "./benchmark-view";

type Tab = "query" | "sources" | "evaluation";
type Mode = "dense" | "bm25" | "hybrid" | "hybrid_rerank";
type Citation = { chunk_id: string; document_id: string; passage: string };
type Hit = Citation & { score: number };
type QueryResult = {
  trace_id: string; status: string; answer: string; citations: Citation[];
  retrieved: Hit[]; model_version: string; prompt_version: string;
};
type Document = { document_id: string; title: string; source: string; version: number; chunks: number };
type ModeMetric = {
  mode: string; queries: number; answer_rate: number; p95_latency_ms: number;
  cost_per_answer_usd: number | null; retrieval_failures: number;
};

const API = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
const MODES: { id: Mode; label: string }[] = [
  { id: "dense", label: "Dense" },
  { id: "bm25", label: "BM25" },
  { id: "hybrid", label: "Hybrid" },
  { id: "hybrid_rerank", label: "Reranked" },
];
const PROMPTS = [
  "When must expense reports be submitted?",
  "What is the requirement for security incidents?",
  "What is the standard service availability target?",
  "What is the confidential exception code for travel?",
];

async function responseData(response: Response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("query");
  const [token, setToken] = useState("");
  const [adminToken, setAdminToken] = useState("");
  const [demoActive, setDemoActive] = useState(false);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [sourceSearch, setSourceSearch] = useState("");
  const [metrics, setMetrics] = useState<ModeMetric[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [sourcesBusy, setSourcesBusy] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API}/health`, { signal: controller.signal })
      .then(response => setApiOnline(response.ok))
      .catch(() => { if (!controller.signal.aborted) setApiOnline(false); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (tab !== "sources" || !token) return;
    const controller = new AbortController();
    setSourcesBusy(true);
    fetch(`${API}/documents`, { headers: { Authorization: `Bearer ${token}` }, signal: controller.signal })
      .then(responseData)
      .then(data => { setDocuments(data.documents); setError(""); })
      .catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Sources unavailable"); })
      .finally(() => { if (!controller.signal.aborted) setSourcesBusy(false); });
    return () => controller.abort();
  }, [tab, token]);

  const filteredDocuments = useMemo(() => documents.filter(item =>
    `${item.title} ${item.document_id} ${item.source}`.toLowerCase().includes(sourceSearch.toLowerCase())),
  [documents, sourceSearch]);

  function changeTab(next: Tab) {
    setTab(next);
    setError("");
  }

  async function enableDemo() {
    setBusy(true); setError("");
    try {
      const data = await responseData(await fetch(`${API}/auth/demo`, { method: "POST" }));
      setToken(data.token);
      setDemoActive(true);
      setApiOnline(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Demo access unavailable");
    } finally { setBusy(false); }
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true); setError(""); setResult(null);
    try {
      const data = await responseData(await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ question, mode }),
      }));
      setResult(data);
      setApiOnline(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Query failed");
    } finally { setBusy(false); }
  }

  async function loadMetrics() {
    setBusy(true); setError("");
    try {
      const data = await responseData(await fetch(`${API}/metrics`, { headers: { "X-Admin-Token": adminToken } }));
      setMetrics(data.modes);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Metrics unavailable");
    } finally { setBusy(false); }
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-symbol"><Layers3 size={21} strokeWidth={2.2}/></div><div><strong>GroundTruth</strong><span>AI</span></div></div>
      <div className="nav-heading">WORKSPACE</div>
      <nav aria-label="Main navigation" className="primary-nav">
        <button className={tab === "query" ? "nav-link active" : "nav-link"} onClick={() => changeTab("query")}><Search size={18}/><span>Query workbench</span><ChevronRight className="nav-chevron" size={15}/></button>
        <button className={tab === "sources" ? "nav-link active" : "nav-link"} onClick={() => changeTab("sources")}><BookOpenText size={18}/><span>Sources</span><ChevronRight className="nav-chevron" size={15}/></button>
        <button className={tab === "evaluation" ? "nav-link active" : "nav-link"} onClick={() => changeTab("evaluation")}><Gauge size={18}/><span>Evaluation</span><ChevronRight className="nav-chevron" size={15}/></button>
      </nav>
      <div className="sidebar-footer"><div className="sidebar-footer-icon"><ShieldCheck size={18}/></div><div><strong>Permission-aware</strong><span>Enterprise retrieval lab</span></div></div>
    </aside>

    <main className="main" id="main-content">
      <header className="topbar"><div className="crumb">GroundTruth <span>/</span> <strong>{tab === "query" ? "Query workbench" : tab === "sources" ? "Sources" : "Evaluation"}</strong></div><div className="topbar-right"><span className={`connection-dot ${apiOnline === false ? "offline" : ""}`}/>{apiOnline === null ? "Connecting" : apiOnline ? "API connected" : "API unavailable"}</div></header>
      <div className="workspace">
        {tab === "query" && <>
          <div className="page-intro"><div><p className="eyebrow">RETRIEVAL WORKBENCH / 01</p><h1>Ask with evidence.</h1><p className="page-subtitle">Every answer is tied to the passages your access level permits.</p></div><div className="intro-mark"><FileSearch2 size={27}/></div></div>
          <div className="context-strip"><div><span className="context-index">01</span><span>Choose a retrieval mode</span></div><div><span className="context-index">02</span><span>Inspect the answer</span></div><div><span className="context-index">03</span><span>Verify the sources</span></div></div>
          <div className="workbench-grid">
            <section className="input-panel" aria-label="Query controls">
              <div className="panel-heading"><span>QUERY SETUP</span><KeyRound size={16}/></div>
              <div className="access-row"><div><span className="field-label">Access token</span><span className="field-help">Employee or admin JWT</span></div><button className="text-action" type="button" onClick={enableDemo} disabled={busy}>{demoActive ? <><Check size={14}/> Demo active</> : <><Sparkles size={14}/> Use demo access</>}</button></div>
              <input className="text-input" aria-label="Access token" type="password" value={token} onChange={event => { setToken(event.target.value); setDemoActive(false); }} placeholder="Paste your JWT or use demo access" autoComplete="off"/>
              <form onSubmit={ask}>
                <label className="field-label question-label" htmlFor="question">Question</label>
                <textarea id="question" value={question} onChange={event => setQuestion(event.target.value)} placeholder="Ask a question about the knowledge base..." required minLength={3} maxLength={1000}/>
                <div className="preset-list" aria-label="Sample questions">{PROMPTS.map(prompt => <button type="button" key={prompt} onClick={() => setQuestion(prompt)}>{prompt}<ArrowRight size={13}/></button>)}</div>
                <div className="field-label mode-label">Retrieval mode</div>
                <div className="mode-control" role="group" aria-label="Retrieval mode">{MODES.map(item => <button type="button" key={item.id} className={mode === item.id ? "mode-button selected" : "mode-button"} aria-pressed={mode === item.id} onClick={() => setMode(item.id)}>{item.label}</button>)}</div>
                <button className="primary-button run-button" disabled={busy || !question.trim() || !token} type="submit">{busy ? "Running query" : "Run query"}<ArrowRight size={17}/></button>
              </form>
            </section>
            <section className="result-panel" aria-live="polite" aria-label="Answer and evidence">
              <div className="panel-heading"><span>ANSWER & EVIDENCE</span>{result && <span className={`status-tag ${result.status}`}>{result.status.replaceAll("_", " ")}</span>}</div>
              {error && <div className="inline-error"><CircleAlert size={17}/>{error}</div>}
              {busy && !result && <div className="result-loading"><div className="loading-line long"/><div className="loading-line"/><div className="loading-line medium"/><div className="loading-rule"/><div className="loading-line medium"/><div className="loading-line short"/></div>}
              {!result && !busy && !error && <div className="empty-result"><div className="empty-icon"><Database size={25}/></div><strong>Ready when you are.</strong><p>Run a query to see its answer, citations, and retrieved passages.</p></div>}
              {result && <div className="answer-content"><div className="answer-kicker">GROUNDED RESPONSE</div><p className="answer-text">{result.answer}</p><div className="answer-meta"><span>TRACE {result.trace_id.slice(0, 8)}</span><span>{result.model_version}</span><span>{result.prompt_version}</span></div>
                <div className="evidence-heading"><h2>Cited passages</h2><span>{result.citations.length}</span></div>
                {result.citations.length ? result.citations.map(citation => <article className="citation-row" key={citation.chunk_id}><div><strong>{citation.document_id}</strong><code>{citation.chunk_id}</code></div><p>{citation.passage}</p></article>) : <p className="quiet-note">No verified citations for this response.</p>}
                <div className="evidence-heading"><h2>Retrieved chunks</h2><span>{result.retrieved.length}</span></div>
                <div className="retrieved-list">{result.retrieved.map((hit, index) => <div key={hit.chunk_id}><span>{String(index + 1).padStart(2, "0")}</span><strong>{hit.document_id}</strong><code>{hit.score.toFixed(3)}</code></div>)}</div>
              </div>}
            </section>
          </div>
        </>}

        {tab === "sources" && <>
          <div className="page-intro"><div><p className="eyebrow">KNOWLEDGE BASE / 02</p><h1>Accessible sources.</h1><p className="page-subtitle">Documents visible to the current identity, filtered before retrieval.</p></div><div className="intro-mark"><BookOpenText size={27}/></div></div>
          <div className="source-toolbar"><div className="source-count"><strong>{documents.length}</strong><span>accessible documents</span></div><label className="search-field"><Search size={16}/><input aria-label="Search sources" value={sourceSearch} onChange={event => setSourceSearch(event.target.value)} placeholder="Search document titles or IDs"/></label></div>
          {!token && <div className="source-prompt"><LockKeyhole size={22}/><div><strong>Access required</strong><p>Use demo access in the Query workbench or paste your JWT there to browse documents.</p></div><button onClick={() => changeTab("query")}>Open workbench <ArrowRight size={15}/></button></div>}
          {error && <div className="inline-error"><CircleAlert size={17}/>{error}</div>}
          {token && <div className="sources-table-wrap"><table className="sources-table"><thead><tr><th>Document</th><th>Source</th><th>Version</th><th>Chunks</th></tr></thead><tbody>{sourcesBusy ? <tr><td colSpan={4} className="table-empty">Loading sources...</td></tr> : filteredDocuments.length ? filteredDocuments.map(item => <tr key={item.document_id}><td><strong>{item.title}</strong><small>{item.document_id}</small></td><td><span className="source-type">{item.source}</span></td><td>{item.version}</td><td>{item.chunks}</td></tr>) : <tr><td colSpan={4} className="table-empty">No matching documents.</td></tr>}</tbody></table></div>}
        </>}

        {tab === "evaluation" && <>
          <div className="page-intro"><div><p className="eyebrow">EVALUATION / 03</p><h1>Measure the system.</h1><p className="page-subtitle">Operational traces and offline retrieval comparisons in one place.</p></div><div className="intro-mark"><Activity size={27}/></div></div>
          <div className="section-heading"><div><h2>Live query metrics</h2><p>Aggregated from the most recent 2,000 stored query traces.</p></div></div>
          <div className="metrics-controls"><div className="admin-token-field"><span className="field-label">Admin API secret</span><span className="field-help">Paste `ADMIN_TOKEN` from the API env, not a JWT from access token</span><input className="text-input" aria-label="Admin token" type="password" placeholder="ADMIN_TOKEN value" value={adminToken} onChange={event => setAdminToken(event.target.value)}/></div><button className="primary-button" onClick={loadMetrics} disabled={busy || !adminToken}>Load metrics <ArrowRight size={16}/></button></div>
          {error && <div className="inline-error"><CircleAlert size={17}/>{error}</div>}
          <div className="metrics-table-wrap"><table><thead><tr><th>Mode</th><th>Queries</th><th>Answer rate</th><th>p95 latency</th><th>Cost / answer</th><th>Retrieval failures</th></tr></thead><tbody>{metrics.length ? metrics.map(item => <tr key={item.mode}><td><strong>{item.mode.replaceAll("_", " + ")}</strong></td><td>{item.queries}</td><td>{(item.answer_rate * 100).toFixed(1)}%</td><td>{item.p95_latency_ms} ms</td><td>{item.cost_per_answer_usd == null ? "—" : `$${item.cost_per_answer_usd.toFixed(5)}`}</td><td>{item.retrieval_failures}</td></tr>) : <tr><td colSpan={6} className="table-empty">Enter the admin token to load live metrics.</td></tr>}</tbody></table></div>
          <BenchmarkView/>
        </>}
      </div>
    </main>
  </div>;
}
