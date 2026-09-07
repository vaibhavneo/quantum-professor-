"""Book retrieval that works where the second_brain gateway does not.

WHY THIS EXISTS

    tutor.py reaches the gateway by importing it from a sibling workspace on
    the author's Mac. In the deployed container that path does not exist, so
    every answer came back with

        {"available": false,
         "reason": "ModuleNotFoundError: No module named 'second_brain'"}

    and the tutor answered from the model's own knowledge with no book behind
    it. It degraded silently - nothing errored, the answers simply stopped
    being grounded - which is why it went unnoticed while the shelf grew to
    64 books the app could not quote.

    This module is the fallback: a self-contained index built at first use
    from a corpus bundle committed to the repo.

    Measured: 50,202 chunks from 60 books, 19.5 MB gzipped. 0.6s to
    decompress, 1.6s to build the FTS5 index, then queries in under a
    millisecond. About two seconds, once, on the first question a container
    serves.

IT MIRRORS THE GATEWAY RATHER THAN INVENTING A SECOND SCORER

    The query construction, the stopword list, the BM25 ordering and the
    sign-flipped score below are copied from second_brain/fts.py on purpose.
    Two retrieval paths that rank differently would mean the same question
    gets different evidence locally and in production, and no way to tell
    which was right - the same "two answers to one question" problem this
    project keeps removing.

    Copied rather than imported because the whole point is that the gateway
    is unavailable here. The copy is small, and the reason it exists is
    written down.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

BUNDLE = Path(__file__).parent / "data" / "quantum_corpus.json.gz"
CORPUS_ID = "quantum-books"

# Copied verbatim from second_brain/fts.py — see the module docstring.
_WORD = re.compile(r"[A-Za-z0-9_]+")
_STOP = frozenset(
    "the a an and or of to in for on with is are was were be been being this "
    "that it as by from at what which who how why when where do does did can "
    "could would should will shall may might must i we you they he she my our "
    "your their its about into than then there here also very more most such "
    "some any all no not but if so because between over under".split()
)

_lock = threading.Lock()
_con: Optional[sqlite3.Connection] = None
_state: Dict[str, Any] = {"built": False, "reason": None, "chunks": 0, "books": 0}


def to_match_query(question: str, max_terms: int = 12) -> str:
    """Free text to a safe FTS5 expression.

    OR rather than AND: a long question ANDed together matches nothing, and
    BM25 already rewards documents carrying more of the rare terms. Measured
    here before the OR was in place - "renormalisation field theory" returned
    zero hits against a corpus containing four QFT texts.
    """
    seen, terms = set(), []
    for w in _WORD.findall(question.lower()):
        if len(w) < 3 or w in _STOP or w in seen:
            continue
        seen.add(w)
        terms.append(w)
        if len(terms) >= max_terms:
            break
    return " OR ".join(terms)


def available() -> bool:
    return BUNDLE.exists()


def status() -> Dict[str, Any]:
    return {"bundle": str(BUNDLE), "present": BUNDLE.exists(), **_state}


def _build() -> Optional[sqlite3.Connection]:
    """Load the bundle into an in-memory FTS5 index. Once per process."""
    global _con
    if _con is not None:
        return _con
    if not BUNDLE.exists():
        _state.update(built=False, reason=f"corpus bundle missing: {BUNDLE.name}")
        return None
    try:
        with gzip.open(BUNDLE, "rt", encoding="utf-8") as f:
            chunks = json.load(f)["chunks"]
        con = sqlite3.connect(":memory:", check_same_thread=False)
        # UNINDEXED on everything except text: the other columns are payload,
        # and indexing them lets a book's own title match its every chunk.
        con.execute("CREATE VIRTUAL TABLE chunks USING fts5("
                    "chunk_id UNINDEXED, source UNINDEXED, text, "
                    "page_start UNINDEXED, title UNINDEXED)")
        con.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?)",
            [(c.get("chunk_id"), c.get("source"), c.get("text"),
              str(c.get("page_start") or ""), c.get("title") or "")
             for c in chunks])
        con.commit()
        books = {os.path.basename(str(c.get("source", ""))) for c in chunks}
        _state.update(built=True, reason=None, chunks=len(chunks), books=len(books))
        _con = con
        return _con
    except Exception as exc:                 # a corrupt bundle must not 500
        _state.update(built=False, reason=f"{type(exc).__name__}: {exc}")
        return None


def search(question: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """BM25-ranked chunks, in the shape the gateway returns them.

    Empty list rather than an exception on any failure: a missing or corrupt
    bundle should cost grounding, never the answer.
    """
    match = to_match_query(question)
    if not match:
        return []
    with _lock:
        con = _build()
    if con is None:
        return []
    try:
        rows = con.execute(
            "SELECT text, source, title, page_start, bm25(chunks) AS s "
            "FROM chunks WHERE chunks MATCH ? ORDER BY s LIMIT ?",
            (match, top_k)).fetchall()
    except sqlite3.OperationalError:
        return []                            # malformed query → no hits, never a 500
    # bm25() is negative and better-is-lower. Flipped so callers treat this
    # score exactly as they treat the gateway's.
    return [{"text": text, "source": source, "corpus": [CORPUS_ID],
             "raw_score": round(-score, 4), "chunk_id": None,
             "author": None, "title": title, "chapter": None,
             "page_start": page_start or None, "page_end": None}
            for text, source, title, page_start, score in rows]


def retrieve(question: str, corpora=None, top_k: int = 6) -> Dict[str, Any]:
    """gateway.retrieve()-compatible wrapper, so the caller needs no branch."""
    return {"hits": search(question, top_k=top_k), "corpora": [CORPUS_ID]}
