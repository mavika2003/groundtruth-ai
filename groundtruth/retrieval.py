from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Plus
from sqlalchemy.orm import Session

from groundtruth.db import Chunk
from groundtruth.security import Principal, authorized_chunks


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    document_id: str
    passage: str
    score: float


def tokenize(value: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", value.lower())


def reciprocal_rank_fusion(rankings: list[list[Hit]], limit: int = 5, k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            hits[hit.chunk_id] = hit
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0) + 1 / (k + rank)
    return [Hit(hits[key].chunk_id, hits[key].document_id, hits[key].passage, score)
            for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def bm25_rank(question: str, chunks: list[Chunk], limit: int = 20) -> list[Hit]:
    if not chunks:
        return []
    corpus = [tokenize(chunk.passage) for chunk in chunks]
    query_terms = set(tokenize(question))
    model = BM25Plus(corpus)
    scores = model.get_scores(list(query_terms))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [Hit(chunk.id, chunk.document_id, chunk.passage, float(score))
            for chunk, score in ranked if query_terms.intersection(tokenize(chunk.passage))][:limit]


def retrieve(session: Session, principal: Principal, question: str, query_embedding: list[float], mode: str, limit: int = 5) -> list[Hit]:
    authorized = authorized_chunks(session, principal)
    distance = Chunk.embedding.cosine_distance(query_embedding)
    dense_rows = session.execute(authorized.add_columns(distance.label("distance")).order_by(distance).limit(20)).all()
    dense = [Hit(chunk.id, chunk.document_id, chunk.passage, 1 - float(value)) for chunk, value in dense_rows]
    # BM25 is calculated only over authorized passages; this reference implementation favors auditability over corpus-scale indexing.
    lexical_rows = session.execute(authorized).scalars().all() if mode != "dense" else []
    lexical = bm25_rank(question, lexical_rows)
    if mode == "dense":
        return dense[:limit]
    if mode == "bm25":
        return lexical[:limit]
    fused = reciprocal_rank_fusion([dense, lexical], limit=20)
    if mode == "hybrid":
        return fused[:limit]
    if mode == "hybrid_rerank":
        scores = list(reranker().rerank(question, [hit.passage for hit in fused]))
        return [Hit(hit.chunk_id, hit.document_id, hit.passage, float(score))
                for hit, score in sorted(zip(fused, scores), key=lambda pair: pair[1], reverse=True)[:limit]]
    raise ValueError(f"Unknown retrieval mode: {mode}")


@lru_cache(maxsize=1)
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


def terms_overlap(question_terms: set[str], passage_tokens: set[str]) -> float:
    matched = 0
    for term in question_terms:
        if term in passage_tokens:
            matched += 1
            continue
        if len(term) > 3 and term.endswith("s") and term[:-1] in passage_tokens:
            matched += 1
            continue
        if not term.endswith("s") and f"{term}s" in passage_tokens:
            matched += 1
            continue
        stem = term[:6] if len(term) >= 6 else term
        if len(stem) >= 4 and any(
            token.startswith(stem) or stem.startswith(token[: min(len(token), 6)])
            for token in passage_tokens
            if len(token) >= 4
        ):
            matched += 1
    return matched / len(question_terms)


def evidence_sufficient(question: str, hits: list[Hit]) -> bool:
    if not hits:
        return False
    stopwords = {"what", "when", "where", "which", "who", "does", "the", "for", "from", "with", "and", "are", "can", "how", "is", "a", "an", "of", "to", "in", "on"}
    terms = set(tokenize(question)) - stopwords
    if not terms:
        return False
    return any(terms_overlap(terms, set(tokenize(hit.passage))) >= 0.35 for hit in hits)
