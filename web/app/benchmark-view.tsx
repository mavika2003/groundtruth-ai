"use client";

import { useState } from "react";
import { Upload } from "lucide-react";

type Case = { id: string; type: string; question: string; expected: string | null; answer: string; retrieved_document_ids: string[]; grounded_answer_correct: boolean; latency_ms: number; estimated_cost_usd: number };
type Result = { metrics: { mode: string; cases: number; recall_at_5: number; mrr: number; citation_correctness: number; grounded_answer_accuracy: number; permission_leaks: number; p95_latency_ms: number; cost_per_success_usd: number }; cases: Case[] };

export default function BenchmarkView() {
  const [results, setResults] = useState<Record<string, Result>>({});
  const [baseline, setBaseline] = useState("dense");
  const [candidate, setCandidate] = useState("hybrid_rerank");
  const [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState("");

  async function loadFiles(files: FileList | null) {
    if (!files) return;
    const next = { ...results };
    try {
      for (const file of Array.from(files)) {
        const parsed = JSON.parse(await file.text()) as Result;
        if (!parsed.metrics?.mode || !Array.isArray(parsed.cases)) throw new Error(`${file.name} is not a benchmark result`);
        next[parsed.metrics.mode] = parsed;
      }
      setResults(next); setError("");
      if (!next[baseline]) setBaseline(Object.keys(next)[0]);
      if (!next[candidate]) setCandidate(Object.keys(next)[Object.keys(next).length - 1]);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not read results"); }
  }

  const left = results[baseline], right = results[candidate];
  const rows = left?.cases.filter(item => filter === "all" || item.type === filter) || [];
  const active = selected ? rows.find(item => item.id === selected) : rows[0];
  const comparison = right?.cases.find(item => item.id === active?.id);
  const fmt = (value: number | undefined, digits = 1) => value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;

  return <section className="benchmark-section">
    <div className="benchmark-heading"><div><h2>Ablation explorer</h2><p>Load result JSON files from two or more benchmark runs to compare failures and tradeoffs.</p></div><label className="upload"><Upload size={16}/> Load results<input type="file" accept=".json,application/json" multiple onChange={event => loadFiles(event.target.files)}/></label></div>
    {error && <div className="error">{error}</div>}
    {Object.keys(results).length === 0 ? <div className="benchmark-empty">Run the benchmark modes, then load their JSON files here. No scores are fabricated.</div> : <>
      <div className="compare-selectors"><label>BASELINE<select value={baseline} onChange={event => setBaseline(event.target.value)}>{Object.keys(results).map(mode => <option key={mode}>{mode}</option>)}</select></label><span>vs</span><label>CANDIDATE<select value={candidate} onChange={event => setCandidate(event.target.value)}>{Object.keys(results).map(mode => <option key={mode}>{mode}</option>)}</select></label></div>
      <div className="comparison-table"><table><thead><tr><th>Measure</th><th>{baseline}</th><th>{candidate}</th></tr></thead><tbody>
        <tr><td>Recall@5</td><td>{fmt(left?.metrics.recall_at_5)}</td><td>{fmt(right?.metrics.recall_at_5)}</td></tr>
        <tr><td>MRR</td><td>{left?.metrics.mrr.toFixed(3) || "—"}</td><td>{right?.metrics.mrr.toFixed(3) || "—"}</td></tr>
        <tr><td>Citation ID validity</td><td>{fmt(left?.metrics.citation_correctness)}</td><td>{fmt(right?.metrics.citation_correctness)}</td></tr>
        <tr><td>Answer proxy accuracy</td><td>{fmt(left?.metrics.grounded_answer_accuracy)}</td><td>{fmt(right?.metrics.grounded_answer_accuracy)}</td></tr>
        <tr><td>p95 latency</td><td>{left?.metrics.p95_latency_ms ?? "—"} ms</td><td>{right?.metrics.p95_latency_ms ?? "—"} ms</td></tr>
        <tr><td>Cost / successful case</td><td>{left ? `$${left.metrics.cost_per_success_usd.toFixed(5)}` : "—"}</td><td>{right ? `$${right.metrics.cost_per_success_usd.toFixed(5)}` : "—"}</td></tr>
        <tr><td>Permission leaks</td><td>{left?.metrics.permission_leaks ?? "—"}</td><td>{right?.metrics.permission_leaks ?? "—"}</td></tr>
      </tbody></table></div>
      <div className="case-head"><h3>Case inspection</h3><select aria-label="Filter case type" value={filter} onChange={event => { setFilter(event.target.value); setSelected(null); }}><option value="all">All cases</option><option value="answerable">Answerable</option><option value="unanswerable">Unanswerable</option><option value="conflicting_document">Conflicting</option><option value="permission_restricted">Restricted</option></select></div>
      <div className="case-inspector"><div className="case-list">{rows.map(item => { const other = right?.cases.find(value => value.id === item.id); return <button className={active?.id === item.id ? "case active" : "case"} key={item.id} onClick={() => setSelected(item.id)}><span>{item.id}</span><small>{!item.grounded_answer_correct && other?.grounded_answer_correct ? "Fixed by candidate" : item.grounded_answer_correct ? "Baseline passed" : "Still failing"}</small></button>; })}</div><div className="case-detail">{active && <><span className="case-type">{active.type.replaceAll("_", " ")}</span><h4>{active.question}</h4><div className="answer-pair"><div><strong>{baseline}</strong><p>{active.answer || "No answer"}</p><small>{active.retrieved_document_ids.join(", ") || "No retrieved documents"}</small></div><div><strong>{candidate}</strong><p>{comparison?.answer || "No answer"}</p><small>{comparison?.retrieved_document_ids.join(", ") || "No retrieved documents"}</small></div></div><div className="case-foot">Expected: {active.expected || "Abstain"} · Latency: {active.latency_ms} / {comparison?.latency_ms ?? "—"} ms</div></>}</div></div>
    </>}
  </section>;
}
