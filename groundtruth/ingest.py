import hashlib
from pathlib import Path

from pypdf import PdfReader
from sqlalchemy import delete
from sqlalchemy.orm import Session

from groundtruth.db import Chunk, Document
from groundtruth.providers import embed


def split_passages(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    words = text.split()
    passages = []
    for start in range(0, len(words), max_words - overlap):
        end = min(start + max_words, len(words))
        passages.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return passages


def extract_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    raise ValueError("Only PDF, Markdown, and text files are supported")


def ingest_text(session: Session, document_id: str, title: str, source: str, audience: list[str], content: str) -> int:
    if not document_id or len(document_id) > 128 or not content.strip():
        raise ValueError("Valid document ID and nonempty content required")
    passages = split_passages(content)
    vectors = embed(passages)
    document = session.get(Document, document_id)
    if document:
        document.version += 1
        session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        session.expire(document, ["chunks"])
    else:
        document = Document(id=document_id, title=title, source=source, audience=audience)
        session.add(document)
    document.title, document.source, document.audience = title, source, audience
    for ordinal, (passage, vector) in enumerate(zip(passages, vectors)):
        digest = hashlib.sha256(passage.encode()).hexdigest()[:12]
        document.chunks.append(Chunk(id=f"{document_id}:{ordinal}:{digest}", ordinal=ordinal, passage=passage, embedding=vector))
    session.commit()
    return len(passages)


def ingest_sql_rows(session: Session, source_session: Session, query, *, id_column: str, text_columns: list[str], title_column: str, audience: list[str]) -> int:
    """Ingest rows from a caller-supplied SQLAlchemy SELECT, not untrusted SQL strings."""
    count = 0
    for row in source_session.execute(query).mappings():
        content = "\n".join(f"{column}: {row[column]}" for column in text_columns)
        ingest_text(session, str(row[id_column]), str(row[title_column]), "sql", audience, content)
        count += 1
    return count
