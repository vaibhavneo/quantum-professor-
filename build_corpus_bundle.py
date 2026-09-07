"""Build the shippable book corpus for deployment.

WHY THIS EXISTS

    Retrieval works locally by importing the second_brain gateway from a
    sibling workspace on this Mac. On Railway that path does not exist, so
    the deployed app returned

        {"available": false,
         "reason": "ModuleNotFoundError: No module named 'second_brain'"}

    and answered from the model's own knowledge with no book behind it. The
    64-book shelf was recommending texts the app could not quote.

    It degraded silently, which is why it went unnoticed: nothing errors, the
    answers just stop being grounded.

WHAT IT PRODUCES

    data/quantum_corpus.json.gz — the quantum-relevant slice of the
    desk-physics corpus, slimmed to the fields retrieval actually reads.

    Measured: 50,606 chunks from 61 books, 20 MB gzipped. The container
    decompresses it in 0.6s and builds a SQLite FTS5 index in 1.6s, so the
    whole cost is about two seconds of startup and queries then run in
    under a millisecond. No volume, no second service, no network call.

WHY A SLICE AND NOT THE WHOLE SHELF

    desk-physics is 165,423 chunks across 166 books, including astrophysics,
    semiconductors and particle physics. This app teaches quantum mechanics.
    Shipping the rest would cost eight times the size to make the retrieval
    worse - unrelated books with shared vocabulary crowding out the relevant
    ones is a failure this codebase has already seen once, which is why
    tutor.py gives desk-quantum-computing its own small quota.

WHAT IS DELIBERATELY EXCLUDED

    Solutions manuals. A worked answer key ranks well on any question phrased
    like a problem, and then the tutor explains from the answer rather than
    from the physics. The Griffiths solutions manual contributed 404 chunks
    and surfaced top-ranked on a conceptual query during testing.

Run:  python3 build_corpus_bundle.py
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

SOURCE = Path(os.getenv(
    "DESK_PHYSICS_CHUNKS",
    "/Users/vaibhavgupta/Desktop/Project Agentic AI/Agentic-AI"
    "/memory/corpora/desk-physics/chunks.json"))
OUT = Path(__file__).parent / "data" / "quantum_corpus.json.gz"

# Folders whose books this app actually teaches from.
KEEP_DIRS = ("Quantum Mechanics/", "Applied Quantum Physics/")

# Substrings that mark a file as an answer key rather than a text.
EXCLUDE = ("Solution Manual", "Solutions Manual", "Instructor Solutions")

# Only the fields retrieval reads. The full chunk carries author/chapter/
# page_end too; dropping them saves nothing worth the ambiguity of a
# half-populated record.
FIELDS = ("chunk_id", "source", "text", "page_start", "title")


def _wanted(source: str) -> bool:
    if not any(d in source for d in KEEP_DIRS):
        return False
    return not any(x.lower() in source.lower() for x in EXCLUDE)


def _shelf_levels(sources) -> dict:
    """Map each corpus filename to the curated level of its shelf entry.

    The shelf already records whether a book is basics / intermediate /
    advanced. Retrieval could not use that because the corpus keys on a
    filename ("Quantum Mechanics. Concepts and Applications (Nouredine
    Zettili) (z-library...).pdf") and the shelf keys on a title. This is the
    join, computed once at build time and shipped, so nothing pays for fuzzy
    matching at query time.

    Measured: 58 of 60 corpus files match a shelf entry. The two that do not
    are the non-quantum titles deliberately kept off the shelf, so an unmapped
    file is a signal rather than a failure - it gets no level and no
    adjustment.
    """
    import difflib
    import re as _re
    try:
        from .library import BOOKS
    except ImportError:
        from library import BOOKS

    def norm(text: str) -> str:
        text = text.lower()
        text = _re.sub(r"\(z-library.*|\(1lib.*|\.pdf$", " ", text)
        return _re.sub(r"[^a-z0-9 ]", " ", text)

    shelf = {b.id: (norm(f"{b.title} {b.authors}"), b.level) for b in BOOKS.values()}
    out = {}
    for name in sources:
        n = norm(name)
        best, score = None, 0.0
        for bid, (text, _lvl) in shelf.items():
            r = difflib.SequenceMatcher(None, n, text).ratio()
            if r > score:
                best, score = bid, r
        if score >= 0.60:
            out[name] = shelf[best][1]
    return out


def build() -> dict:
    if not SOURCE.exists():
        raise SystemExit(
            f"source corpus not found: {SOURCE}\n"
            "Set DESK_PHYSICS_CHUNKS, or run the desk-physics ingest first.")
    chunks = json.loads(SOURCE.read_text())["chunks"]
    kept = [{k: c[k] for k in FIELDS if k in c}
            for c in chunks if _wanted(str(c.get("source", "")))]
    dropped = {os.path.basename(str(c.get("source", "")))
               for c in chunks
               if any(d in str(c.get("source", "")) for d in KEEP_DIRS)
               and not _wanted(str(c.get("source", "")))}
    books = {os.path.basename(str(c["source"])) for c in kept}
    levels = _shelf_levels(books)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump({"chunks": kept, "levels": levels}, f, ensure_ascii=False)
    return {"chunks": len(kept), "books": len(books),
            "levelled": len(levels),
            "mb": round(OUT.stat().st_size / 1e6, 1),
            "excluded_files": sorted(dropped)}


if __name__ == "__main__":
    r = build()
    print(f"{OUT.name}: {r['chunks']:,} chunks from {r['books']} books "
          f"({r['levelled']} with a curated level), {r['mb']} MB")
    for f in r["excluded_files"]:
        print(f"  excluded: {f[:74]}")
