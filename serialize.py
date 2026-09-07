"""Dict-serialization helpers for quantum_prof data types."""
from __future__ import annotations

try:
    from .problems import problems_for_topic
except ImportError:
    from problems import problems_for_topic

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ingest import SearchResult
    from .library import Book, Topic
    from .problems import Problem


def topic_to_dict(topic: "Topic") -> dict:
    return {
        "id": topic.id,
        "title": topic.title,
        "level": topic.level,
        "prerequisites": list(topic.prerequisites),
        "key_concepts": list(topic.key_concepts),
        "key_equations": list(topic.key_equations),
        "intuition": topic.intuition,
        "book_refs": list(topic.book_refs),
        # DERIVED FROM THE PROBLEM BANK, not from the hand-maintained list.
        #
        # Topic.problem_ids was a second copy of this relationship and it had
        # drifted completely: all 14 topics that declared ids declared ones
        # that resolve to nothing ("duality-p1", "bohr-p1") while the bank
        # holds "wpd-1", "bohr-1". Nothing broke, because problems_for_topic()
        # matches on Problem.topic_id instead - so the API shipped 14 topics
        # worth of ids pointing at nothing, and problem() would raise on any
        # of them.
        #
        # One source of truth: the bank. The field stays on the dataclass for
        # callers that construct Topics directly, but it is no longer what the
        # API reports.
        "problem_ids": [p.id for p in problems_for_topic(topic.id)],
    }


def book_to_dict(book: "Book") -> dict:
    return {
        "id": book.id,
        "title": book.title,
        # SPLIT ON THE SEPARATOR, not into characters.
        #
        # `Book.authors` is a str (library.py). list() on a str splats it into
        # single characters, so the API returned
        #     ["D","a","v","i","d"," ","J",".", ...]
        # and web/app.js renders `authors.join(", ")`, putting
        # "D, a, v, i, d,  , J, ." on every book card. All 13 books, and now
        # all 64.
        "authors": [a.strip() for a in book.authors.split(",") if a.strip()],
        "level": book.level,
        "topics": list(book.topics),
        "note": book.note,
        "why_read": book.why_read,
    }


def problem_to_dict(problem: "Problem") -> dict:
    return {
        "id": problem.id,
        "topic_id": problem.topic_id,
        "difficulty": problem.difficulty,
        "stem": problem.stem,
        "hint": problem.hint,
        "answer_latex": problem.answer_latex,
    }


def search_result_to_dict(result: "SearchResult") -> dict:
    return {
        "file": result.file,
        "score": result.score,
        "excerpt": result.excerpt,
    }
