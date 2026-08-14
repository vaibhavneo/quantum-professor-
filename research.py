"""Research capabilities: primary literature, symbolic derivation, a notebook.

Deliberately a sibling copy of ai_brain/research.py rather than a shared
import: the two apps live in separate repositories and deploy independently,
so a cross-repo import would break this one the moment it is deployed alone.
The duplication is the cheaper failure mode.


A teacher explains what is settled. A research partner needs three things the
teaching pipeline does not have:

  literature  — arXiv, because textbooks are two to five years behind by
                construction and research lives in preprints. Results carry
                their publication date so recency is visible, not assumed.
  derivation  — sympy, so algebra is *checked* rather than asserted. This
                extends the discipline already used for the physics solvers:
                the model may reason about a derivation, but the symbolic
                engine decides whether the steps actually hold.
  notebook    — a durable place for open questions, findings and
                contradictions. Without it every session restarts from zero
                and the system can accumulate nothing.

Everything here is keyless and stdlib + sympy, so it runs wherever the rest
of the brain runs.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "memory" / "research_notebook.json"
ARXIV_API = "https://export.arxiv.org/api/query"

# Words that carry no discriminating power in an arXiv full-text search.
_ARXIV_STOP = frozenset("""
the a an and or of to in for on with is are was were be been being this that it
as by from at what which who how why when where do does did can could would
should will shall may might must i we you they my our your their its about into
than then there here also very more most such some any all no not but if so
because between over under actually determine determines determined give me
complete entire understand explain compare difference relationship using use
used get make know tell show help need want think good best better
""".split())

# Generic technical filler — real terms, but they match half of arXiv.
_ARXIV_WEAK = frozenset("""
model models method methods approach approaches system systems data learning
network networks training train problem problems result results work paper study
analysis performance based new novel large small deep neural
""".split())


# ── primary literature ────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 6,
                 sort: str = "relevance") -> dict:
    """Search arXiv. `sort` is 'relevance' or 'submittedDate' (newest first).

    Note the https scheme: the http endpoint answers 301 and a naive client
    silently gets nothing.
    """
    # Stopwords must go before the AND. Feeding a raw question in ANDs its
    # opening words together — an observed query became
    # `what AND actually AND determines AND the` and returned nothing, while
    # the synthesis stage went on to describe "the primary literature" from
    # zero papers.
    terms = [t for t in re.findall(r"[A-Za-z0-9\-]{3,}", query.lower())
             if t not in _ARXIV_STOP]
    if not terms:
        return {"ok": False, "error": "no searchable terms after stopwords",
                "papers": []}
    # Rare words first: a distinctive term like "lora" discriminates, "model"
    # does not. Longer tokens and hyphenated compounds are the better bet.
    terms.sort(key=lambda t: (t in _ARXIV_WEAK, -len(t)))
    picked = terms[:4]
    q = " AND ".join(f"all:{t}" for t in picked)
    url = (f"{ARXIV_API}?search_query={urllib.parse.quote(q)}"
           f"&max_results={max_results}&sortBy={sort}&sortOrder=descending")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "AIBrain/1.0 (personal research tool)"})
        with urllib.request.urlopen(req, timeout=25) as r:
            xml = r.read().decode("utf-8", "ignore")
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "papers": []}

    papers = []
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        def grab(tag):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", entry, re.S)
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        link = re.search(r'<id>(.*?)</id>', entry, re.S)
        papers.append({
            "title": grab("title"),
            "summary": grab("summary")[:1100],
            "published": grab("published")[:10],
            "updated": grab("updated")[:10],
            "authors": re.findall(r"<name>(.*?)</name>", entry)[:4],
            "url": link.group(1).strip() if link else "",
        })
    if not papers:
        return {"ok": True, "papers": [], "query": q,
                "note": "arXiv returned no matches for these terms"}
    newest = max(p["published"] for p in papers)
    return {"ok": True, "papers": papers, "query": q, "newest": newest,
            "note": "primary literature — dates shown so recency is explicit"}


# ── symbolic derivation ───────────────────────────────────────────────────

def _sympy():
    import sympy as sp
    return sp


def check_identity(lhs: str, rhs: str, assumptions: str = "real") -> dict:
    """Decide whether lhs == rhs symbolically.

    The point is that the *engine* decides, not the model. A claimed
    simplification that is merely plausible fails here.
    """
    try:
        sp = _sympy()
        syms = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lhs + " " + rhs)))
        ns = {}
        for s in syms:
            if hasattr(sp, s) and s in ("pi", "E", "I", "oo"):
                ns[s] = getattr(sp, s)
            elif s not in dir(sp):
                ns[s] = sp.Symbol(s, **({assumptions: True} if assumptions else {}))
        L = sp.sympify(lhs, locals=ns)
        R = sp.sympify(rhs, locals=ns)
        diff = sp.simplify(L - R)
        equal = bool(diff == 0)
        return {"ok": True, "equal": equal, "lhs": str(L), "rhs": str(R),
                "difference": str(diff),
                "verdict": "identity holds" if equal
                           else "NOT an identity — difference is non-zero"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def derive(expression: str, operation: str, wrt: str = "x",
           **kw) -> dict:
    """Run one symbolic operation and return the result plus a LaTeX form.

    operation: simplify | expand | factor | diff | integrate | solve | limit | series
    """
    try:
        sp = _sympy()
        syms = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)))
        ns = {s: sp.Symbol(s, real=True) for s in syms if s not in dir(sp)}
        x = ns.get(wrt) or sp.Symbol(wrt, real=True)
        ns.setdefault(wrt, x)
        e = sp.sympify(expression, locals=ns)
        ops = {
            "simplify": lambda: sp.simplify(e),
            "expand": lambda: sp.expand(e),
            "factor": lambda: sp.factor(e),
            "diff": lambda: sp.diff(e, x),
            "integrate": lambda: sp.integrate(e, x),
            "solve": lambda: sp.solve(e, x),
            "limit": lambda: sp.limit(e, x, sp.sympify(kw.get("to", 0))),
            "series": lambda: sp.series(e, x, 0, int(kw.get("order", 5))),
        }
        if operation not in ops:
            return {"ok": False, "error": f"unknown operation {operation!r}",
                    "available": sorted(ops)}
        out = ops[operation]()
        return {"ok": True, "operation": operation, "input": str(e),
                "result": str(out), "latex": sp.latex(out)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ── research notebook ─────────────────────────────────────────────────────

def _load() -> dict:
    try:
        return json.loads(NOTEBOOK_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"threads": {}}


def _save(nb: dict) -> None:
    try:
        NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    except OSError:
        pass          # the notebook is a convenience; never fail a run over it


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(thread: str, *, finding: str = "", question: str = "",
           contradiction: str = "", next_step: str = "",
           sources: list | None = None) -> dict:
    """Append to a research thread. Entries are only ever added, so the trail
    of what was believed when stays intact."""
    nb = _load()
    t = nb["threads"].setdefault(thread, {
        "created": _now(), "findings": [], "open_questions": [],
        "contradictions": [], "next_steps": [], "entries": 0})
    stamp = _now()
    if finding:
        t["findings"].append({"at": stamp, "text": finding, "sources": sources or []})
    if question:
        t["open_questions"].append({"at": stamp, "text": question, "status": "open"})
    if contradiction:
        t["contradictions"].append({"at": stamp, "text": contradiction,
                                    "sources": sources or []})
    if next_step:
        t["next_steps"].append({"at": stamp, "text": next_step, "status": "proposed"})
    t["entries"] += 1
    t["updated"] = stamp
    _save(nb)
    return t


def thread(name: str) -> dict:
    return _load()["threads"].get(name, {})


def threads() -> list[dict]:
    nb = _load()
    return [{"thread": k,
             "updated": v.get("updated", v.get("created", "")),
             "findings": len(v.get("findings", [])),
             "open_questions": sum(1 for q in v.get("open_questions", [])
                                   if q.get("status") == "open"),
             "contradictions": len(v.get("contradictions", [])),
             "next_steps": len(v.get("next_steps", []))}
            for k, v in sorted(nb["threads"].items(),
                               key=lambda kv: kv[1].get("updated", ""), reverse=True)]


def brief(name: str, max_items: int = 6) -> str:
    """Compact prior context for a thread, to prime the next run."""
    t = thread(name)
    if not t:
        return ""
    bits = []
    if t.get("findings"):
        bits.append("Established so far:\n" + "\n".join(
            f"  - {f['text']}" for f in t["findings"][-max_items:]))
    openq = [q for q in t.get("open_questions", []) if q.get("status") == "open"]
    if openq:
        bits.append("Still open:\n" + "\n".join(f"  - {q['text']}" for q in openq[-max_items:]))
    if t.get("contradictions"):
        bits.append("Unresolved contradictions:\n" + "\n".join(
            f"  - {c['text']}" for c in t["contradictions"][-max_items:]))
    return "\n\n".join(bits)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "threads"
    if cmd == "threads":
        for r in threads():
            print(f"  {r['thread']:<44s} findings={r['findings']:<3d} "
                  f"open={r['open_questions']:<3d} next={r['next_steps']}")
    elif cmd == "brief":
        print(brief(sys.argv[2]))
    elif cmd == "arxiv":
        r = search_arxiv(" ".join(sys.argv[2:]))
        for p in r.get("papers", []):
            print(f"  {p['published']}  {p['title'][:74]}")
