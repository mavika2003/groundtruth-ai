import json
import statistics
import time
from pathlib import Path

from groundtruth.config import settings
from groundtruth.db import Document, SessionLocal
from groundtruth.ingest import ingest_text
from groundtruth.providers import embed, generate
from groundtruth.retrieval import evidence_sufficient, retrieve
from groundtruth.security import Principal


FIXTURES = [
    ("travel", "Travel Policy", "Domestic travel requires manager approval 10 business days before departure.", "10 business days", "manager approval"),
    ("expense", "Expense Policy", "Expense reports must be submitted within 30 calendar days of purchase.", "30 calendar days", "expense reports"),
    ("security", "Security Policy", "Security incidents must be reported to the security desk within 2 hours of discovery.", "2 hours", "security incidents"),
    ("password", "Password Policy", "Privileged account passwords must be rotated every 90 days.", "90 days", "privileged account passwords"),
    ("retention", "Retention Policy", "Customer support tickets are retained for 24 months.", "24 months", "customer support tickets"),
    ("backup", "Backup Policy", "Production database backups run every 6 hours.", "6 hours", "production database backups"),
    ("access", "Access Policy", "Access reviews are required every quarter.", "every quarter", "access reviews"),
    ("device", "Device Policy", "Company laptops must lock after 5 minutes of inactivity.", "5 minutes", "company laptops"),
    ("onboarding", "Onboarding Policy", "New employees complete security training within 14 days of joining.", "14 days", "security training"),
    ("offboarding", "Offboarding Policy", "Employee access is revoked within 4 hours of termination.", "4 hours", "employee access"),
    ("vendor", "Vendor Policy", "Critical vendors are reviewed annually.", "annually", "critical vendors"),
    ("privacy", "Privacy Policy", "Data subject requests receive an initial response within 7 days.", "7 days", "data subject requests"),
    ("change", "Change Policy", "Emergency changes require retrospective review within 48 hours.", "48 hours", "emergency changes"),
    ("deployment", "Deployment Policy", "Production deployments require two independent approvals.", "two independent approvals", "production deployments"),
    ("incident", "Incident Policy", "Severity one incidents require status updates every 30 minutes.", "30 minutes", "severity one incident updates"),
    ("recovery", "Recovery Policy", "The critical service recovery time objective is 4 hours.", "4 hours", "critical service recovery time"),
    ("billing", "Billing Policy", "Customer invoices are issued on the first business day of each month.", "first business day", "customer invoices"),
    ("refund", "Refund Policy", "Refund requests are reviewed within 5 business days.", "5 business days", "refund requests"),
    ("contract", "Contract Policy", "Contracts above $100,000 require legal review.", "$100,000", "legal review threshold"),
    ("procurement", "Procurement Policy", "Purchase orders above $25,000 require finance approval.", "$25,000", "finance approval threshold"),
    ("training", "Training Policy", "Annual compliance training is due by September 30.", "September 30", "annual compliance training"),
    ("support", "Support Policy", "Priority one support tickets target a 15 minute first response.", "15 minutes", "priority one support first response"),
    ("sla", "SLA Policy", "The standard service availability target is 99.9 percent monthly.", "99.9 percent", "standard service availability target"),
    ("audit", "Audit Policy", "Audit evidence is stored for 7 years.", "7 years", "audit evidence"),
    ("encryption", "Encryption Policy", "Data at rest must use AES-256 encryption.", "AES-256", "data at rest encryption"),
    ("network", "Network Policy", "Firewall rules are reviewed every 180 days.", "180 days", "firewall rules"),
    ("remote", "Remote Work Policy", "Remote workers must connect through the corporate VPN.", "corporate VPN", "remote worker connection"),
    ("data", "Data Classification Policy", "Restricted data may be shared only with approved project members.", "approved project members", "restricted data sharing"),
    ("api", "API Policy", "Public API keys expire after 365 days.", "365 days", "public API keys"),
    ("logging", "Logging Policy", "Application audit logs are retained for 400 days.", "400 days", "application audit logs"),
]


def cases() -> list[dict]:
    result = []
    for key, _, statement, expected, subject in FIXTURES:
        result.extend([
            {"id": f"{key}-direct", "type": "answerable", "question": f"What is the requirement for {subject}?", "expected": expected, "document_id": key, "groups": ["employee"]},
            {"id": f"{key}-paraphrase", "type": "answerable", "question": f"According to policy, what applies to {subject}?", "expected": expected, "document_id": key, "groups": ["employee"]},
            {"id": f"{key}-unknown", "type": "unanswerable", "question": f"Who personally approved the last exception for {subject}?", "expected": None, "document_id": None, "groups": ["employee"]},
            {"id": f"{key}-restricted", "type": "permission_restricted", "question": f"What is the confidential exception code for {subject}?", "expected": None, "document_id": None, "groups": ["employee"]},
            {"id": f"{key}-conflict", "type": "conflicting_document", "question": f"What is the current requirement for {subject} when the two policy versions disagree?", "expected": None, "document_id": None, "groups": ["employee"]},
        ])
    return result


def seed(delay_seconds: float = 1.0) -> dict[str, int]:
    ingested = skipped = 0
    with SessionLocal() as session:
        for key, title, statement, expected, subject in FIXTURES:
            documents = (
                (key, title, "synthetic-fixture", ["employee"], statement),
                (f"{key}-legacy", f"Legacy {title}", "synthetic-conflict", ["employee"],
                 f"Policy version B states that {subject} has no requirement. This version is marked current, but conflicts with version A."),
                (f"{key}-secret", f"Confidential {title}", "synthetic-restricted", ["admin"],
                 f"The confidential exception code for {subject} is SECRET-{key.upper()}."),
            )
            for document_id, document_title, source, audience, content in documents:
                existing = session.get(Document, document_id)
                if existing:
                    if existing.source != source:
                        raise RuntimeError(f"Document ID {document_id} is already used by another source")
                    skipped += 1
                    continue
                ingest_text(session, document_id, document_title, source, audience, content)
                ingested += 1
                print(f"Seeded {ingested} new document(s); latest: {document_id}", flush=True)
                if delay_seconds:
                    time.sleep(delay_seconds)
    return {"ingested": ingested, "skipped": skipped}


def evaluate(mode: str, output: Path) -> dict:
    rows = []
    with SessionLocal() as session:
        for case in cases():
            start = time.perf_counter()
            principal = Principal("benchmark-employee", frozenset(case["groups"]))
            hits = retrieve(session, principal, case["question"], embed([case["question"]], purpose="query")[0], mode)
            sufficient = evidence_sufficient(case["question"], hits)
            answer = ""
            citations = []
            input_tokens = output_tokens = 0
            if sufficient:
                answer, citations, input_tokens, output_tokens = generate(case["question"], [hit.__dict__ for hit in hits])
            answerable = case["type"] == "answerable"
            expected_doc = case["document_id"]
            ids = [hit.document_id for hit in hits]
            reciprocal_rank = 1 / (ids.index(expected_doc) + 1) if expected_doc in ids else 0
            valid_citations = set(citations).issubset({hit.chunk_id for hit in hits}) and bool(citations)
            correct = answerable and case["expected"].lower() in answer.lower() and valid_citations
            abstained = not sufficient or not citations or "cannot determine" in answer.lower()
            rows.append({**case, "mode": mode, "answer": answer, "retrieved_document_ids": ids,
                         "retrieved_contexts": [hit.passage for hit in hits],
                         "citations": citations, "recall_at_5": int(expected_doc in ids) if answerable else None,
                         "reciprocal_rank": reciprocal_rank if answerable else None,
                         "citation_correct": valid_citations if answerable else None,
                         "grounded_answer_correct": correct if answerable else abstained,
                         "permission_leak": any(id.endswith("-secret") for id in ids),
                         "latency_ms": round((time.perf_counter() - start) * 1000),
                         "estimated_cost_usd": (input_tokens * settings.input_cost_per_million + output_tokens * settings.output_cost_per_million) / 1_000_000})
    answerable_rows = [row for row in rows if row["type"] == "answerable"]
    metrics = {"mode": mode, "cases": len(rows), "recall_at_5": statistics.mean(row["recall_at_5"] for row in answerable_rows),
               "mrr": statistics.mean(row["reciprocal_rank"] for row in answerable_rows),
               "citation_correctness": statistics.mean(row["citation_correct"] for row in answerable_rows),
               "grounded_answer_accuracy": statistics.mean(row["grounded_answer_correct"] for row in rows),
               "permission_leaks": sum(row["permission_leak"] for row in rows),
               "p95_latency_ms": sorted(row["latency_ms"] for row in rows)[int(len(rows) * 0.95) - 1],
               "cost_per_success_usd": sum(row["estimated_cost_usd"] for row in rows) / max(1, sum(row["grounded_answer_correct"] for row in rows))}
    output.write_text(json.dumps({"metrics": metrics, "cases": rows}, indent=2))
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["seed", "evaluate", "export-cases"])
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid", "hybrid_rerank"], default="hybrid")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between new seed documents")
    args = parser.parse_args()
    if args.action == "seed":
        if args.delay < 0:
            parser.error("--delay must be nonnegative")
        print(json.dumps(seed(args.delay)))
    elif args.action == "export-cases":
        args.output.write_text(json.dumps(cases(), indent=2))
    else:
        print(json.dumps(evaluate(args.mode, args.output), indent=2))
