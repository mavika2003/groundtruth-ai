from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from groundtruth.config import settings
from groundtruth.db import Chunk, Document


@dataclass(frozen=True)
class Principal:
    subject: str
    groups: frozenset[str]


def principal_from_token(authorization: str | None = Header(default=None)) -> Principal:
    if not settings.jwt_secret or not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Bearer token required")
    try:
        claims = jwt.decode(authorization[7:], settings.jwt_secret, algorithms=["HS256"], options={"require": ["sub", "exp"]})
        groups = claims.get("groups", [])
        if not isinstance(groups, list) or not all(isinstance(group, str) for group in groups):
            raise ValueError("invalid groups")
        return Principal(str(claims["sub"]), frozenset(groups))
    except (jwt.PyJWTError, ValueError) as exc:
        raise HTTPException(401, "Invalid token") from exc


def authorized_chunks(session: Session, principal: Principal):
    # The permission predicate is part of the DB query, before any passage is ranked or sent to an LLM.
    audience = Document.audience
    allowed = or_(audience == [], *[audience.contains([group]) for group in principal.groups])
    return select(Chunk).join(Document).where(allowed)


def authorized_documents(principal: Principal):
    audience = Document.audience
    return select(Document).where(or_(audience == [], *[audience.contains([group]) for group in principal.groups]))


def is_allowed(audience: list[str], groups: frozenset[str]) -> bool:
    return not audience or bool(set(audience) & groups)
