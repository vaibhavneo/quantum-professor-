"""Refresh the DATA blob embedded in web/standalone.html from the Python modules.

standalone.html carries its own copy of the curriculum so it works as a bare
file with no server. That copy was hand-pasted once and then drifted: the
curriculum grew to 41 topics while the page still advertised 32, so the three
statistical-mechanics topics were reachable through /api/ask but invisible in
the Topics tab. Run this after any change to library.py or problems.py.

    python3 build_standalone.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from library import TOPICS, BOOKS
from problems import PROBLEMS
from serialize import topic_to_dict, book_to_dict, problem_to_dict

PAGE = Path(__file__).parent / "web" / "standalone.html"
# The blob is emitted as a single line, which is what makes this a safe
# line-oriented substitution rather than a brace-matching exercise.
PATTERN = re.compile(r"^const DATA = \{.*\};$", re.M)


def build() -> str:
    problems = PROBLEMS.values() if isinstance(PROBLEMS, dict) else PROBLEMS
    return "const DATA = " + json.dumps({
        "topics": [topic_to_dict(t) for t in TOPICS.values()],
        "books": [book_to_dict(b) for b in BOOKS.values()],
        "problems": [problem_to_dict(p) for p in problems],
    }, ensure_ascii=True) + ";"


def main() -> int:
    text = PAGE.read_text(encoding="utf-8")
    if not PATTERN.search(text):
        print("could not find the DATA line in standalone.html — not touching it")
        return 1
    # A callable replacement is used literally — no backslash-escape pass, which
    # matters because the blob is dense with LaTeX like \frac and \hbar.
    new = PATTERN.sub(lambda _: build(), text, count=1)
    PAGE.write_text(new, encoding="utf-8")
    problems = PROBLEMS.values() if isinstance(PROBLEMS, dict) else PROBLEMS
    print(f"standalone.html refreshed: {len(TOPICS)} topics, "
          f"{len(BOOKS)} books, {len(list(problems))} problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
