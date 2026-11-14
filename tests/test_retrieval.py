from types import SimpleNamespace

from groundtruth.benchmark import cases
from groundtruth.ingest import split_passages
from groundtruth.retrieval import Hit, bm25_rank, evidence_sufficient, reciprocal_rank_fusion
from groundtruth.security import is_allowed


def test_benchmark_has_150_labeled_cases():
    dataset = cases()
    assert len(dataset) == 150
    assert len({item["id"] for item in dataset}) == 150
    assert {item["type"] for item in dataset} == {"answerable", "unanswerable", "permission_restricted", "conflicting_document"}


def test_document_acl():
    assert is_allowed([], frozenset())
    assert is_allowed(["finance"], frozenset({"finance"}))
    assert not is_allowed(["finance"], frozenset({"employee"}))


def test_bm25_and_fusion():
    chunks = [SimpleNamespace(id="a", document_id="a", passage="Access reviews happen every quarter"),
              SimpleNamespace(id="b", document_id="b", passage="Travel requires manager approval")]
    lexical = bm25_rank("manager approval travel", chunks)
    assert lexical[0].chunk_id == "b"
    dense = [Hit("a", "a", chunks[0].passage, 0.8), Hit("b", "b", chunks[1].passage, 0.7)]
    assert reciprocal_rank_fusion([dense, lexical])[0].chunk_id == "b"


def test_evidence_gate_and_chunking():
    assert not evidence_sufficient("Who approved the last exception?", [])
    assert evidence_sufficient("What applies to travel manager approval?", [Hit("a", "a", "Travel requires manager approval", 0.9)])
    assert split_passages(" ".join(str(n) for n in range(190))).__len__() == 2
    assert len(split_passages(" ".join(str(n) for n in range(180)))) == 1
