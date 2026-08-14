"""Quantum Professor — the nine-stage flow.

    USER → UNDERSTAND → ROUTER → { CURRICULUM | BOOKS | SOLVER | ARXIV | SYMPY }
         → EVIDENCE (verify/compare) → REASONING → PROFESSOR
         → VALIDATION → ANSWER

Replaces the old match → retrieve → compute → adapt → compose flow, which had
no understanding, routing, evidence or validation stage at all.

Two things are specific to this app rather than inherited from the AI Brain:

  Multi-topic curriculum. The old matcher chose exactly one topic, so a
  question spanning general relativity, special relativity and particle
  physics was answered from the Dirac equation alone. Retrieval now takes the
  top several topics and unions their concepts and equations, which is the
  only way a synthesis question can be answered at all.

  Deterministic numbers. physics.py solvers and sympy both produce values the
  model is forbidden to restate. The model may reason *about* a derivation;
  the solver or the symbolic engine decides what the number or identity
  actually is.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

# Importable both as a package member and as a flat script, matching
# web_server.py: local runs use the package form, the deployment runs flat.
try:
    from . import research as R
    from .library import TOPICS
    from .tutor import (DEPTH_DIRECTIVE, MODE_DIRECTIVE, MODEL_DEEP, MODEL_FAST,
                        _api_key, _familiarity, compute_for, match_topic,
                        record_visit, retrieve_evidence)
except ImportError:
    import research as R
    from library import TOPICS
    from tutor import (DEPTH_DIRECTIVE, MODE_DIRECTIVE, MODEL_DEEP, MODEL_FAST,
                       _api_key, _familiarity, compute_for, match_topic,
                       record_visit, retrieve_evidence)

STAGE_PLAN = {
    "understand": {"intro": (MODEL_FAST, 3000),  "intermediate": (MODEL_FAST, 3000),
                   "advanced": (MODEL_FAST, 4000)},
    "evidence":   {"intro": None,                "intermediate": (MODEL_FAST, 6000),
                   "advanced": (MODEL_FAST, 8000)},
    "reasoning":  {"intro": None,                "intermediate": (MODEL_FAST, 10000),
                   "advanced": (MODEL_DEEP, 18000)},
    "professor":  {"intro": (MODEL_FAST, 10000), "intermediate": (MODEL_FAST, 16000),
                   "advanced": (MODEL_DEEP, 24000)},
    "validation": {"intro": None,                "intermediate": (MODEL_FAST, 5000),
                   "advanced": (MODEL_FAST, 6000)},
}


def plan_for(stage, depth):
    return STAGE_PLAN.get(stage, {}).get(depth, STAGE_PLAN.get(stage, {}).get("intermediate"))


class Budget:
    def __init__(self):
        self.calls = self.prompt = self.completion = self.reasoning = 0
        self.by_stage: dict = {}

    def record(self, stage, model, usage, secs):
        self.calls += 1
        p = getattr(usage, "prompt_tokens", 0) or 0
        c = getattr(usage, "completion_tokens", 0) or 0
        r = getattr(getattr(usage, "completion_tokens_details", None),
                    "reasoning_tokens", 0) or 0
        self.prompt += p; self.completion += c; self.reasoning += r
        self.by_stage[stage] = {"model": model, "completion": c, "reasoning": r,
                                "secs": round(secs, 1)}

    def summary(self):
        return {"llm_calls": self.calls, "prompt_tokens": self.prompt,
                "completion_tokens": self.completion, "reasoning_tokens": self.reasoning,
                "total_tokens": self.prompt + self.completion, "by_stage": self.by_stage}


def _call(client, stage, depth, system, user, budget, force=None):
    """One call, retrying once if the reasoning chain eats the whole budget.

    deepseek-v4 reasons before answering and those tokens count against
    max_tokens, so a budget sized for the prose alone can come back empty with
    finish_reason "length". Guarding here means every stage inherits it.
    """
    spec = force or plan_for(stage, depth)
    if spec is None:
        return ""
    model, max_tokens = spec

    def once(limit):
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=model, max_tokens=limit,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        budget.record(stage, model, resp.usage, time.monotonic() - t0)
        return (resp.choices[0].message.content or "").strip()

    text = once(max_tokens)
    if not text:
        budget.by_stage.setdefault(stage, {})["retried"] = f"empty at {max_tokens}"
        text = once(max_tokens * 2)
    return text


def _json_from(text, fallback):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return fallback
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return fallback


# ── multi-topic curriculum retrieval ──────────────────────────────────────

def match_topics(question: str, k: int = 4):
    """Top-k curriculum topics, not one.

    match_topic() returns a single best topic plus 'related' it never actually
    uses. A question spanning several areas needs all of them in the prompt.
    """
    primary, related = match_topic(question)
    picked = ([primary] if primary else []) + [t for t in related if t is not primary]
    return picked[:k]


def curriculum_block(topics) -> str:
    if not topics:
        return ""
    out = []
    for t in topics:
        out.append(
            f"[C:{t.id}] {t.title} ({t.level})\n"
            f"  intuition: {t.intuition}\n"
            f"  key concepts: {', '.join(t.key_concepts)}\n"
            f"  key equations (LaTeX): {' ; '.join(t.key_equations)}")
    return "\n\n".join(out)


# ── stage 2: understanding ────────────────────────────────────────────────

_UNDERSTAND_SYS = """You classify a physics question for a retrieval pipeline. \
Reply with ONLY JSON:

{"intent":"explain|derive|compute|compare|research",
 "topics":["3-6 physics terms to search on"],
 "needs_literature":true|false,  // true for "current/recent/state of the art"
                                 // or a research-level open question
 "needs_symbolic":true|false,    // true if an identity or derivation should be
                                 // checked algebraically
 "identity":"lhs = rhs to verify, or empty",
 "restate":"one sentence restating what is being asked"}"""


def understand(question, depth, client, budget):
    u = _json_from(_call(client, "understand", depth, _UNDERSTAND_SYS,
                         f"QUESTION: {question}", budget), {})
    return {
        "intent": u.get("intent", "explain"),
        "topics": u.get("topics") or re.findall(r"[a-z][a-z\-]{3,}", question.lower())[:6],
        "needs_literature": bool(u.get("needs_literature")),
        "needs_symbolic": bool(u.get("needs_symbolic")),
        "identity": (u.get("identity") or "").strip(),
        "restate": u.get("restate") or question,
    }


# ── stage 3: router ───────────────────────────────────────────────────────

def route(u, topics, solver_probe, depth):
    """Deterministic. Routing is policy; spending a model call to re-derive a
    rule the code already knows is the waste this design avoids."""
    r = {
        "curriculum": bool(topics),
        "books": True,
        "solver": bool(solver_probe and solver_probe.get("ran")),
        "arxiv": bool(u["needs_literature"]) and depth != "intro",
        "sympy": bool(u["needs_symbolic"] and u["identity"]) and depth != "intro",
    }
    r["why"] = [
        f"curriculum: {len(topics)} topic(s) matched" if topics else "curriculum: no topic matched",
        "books: always — the physics shelf is the primary source",
        ("solver: the question carries numeric parameters" if r["solver"]
         else "solver: nothing to compute from this question"),
        ("arxiv: research-level or time-sensitive" if r["arxiv"]
         else "arxiv: skipped at intro depth" if u["needs_literature"]
         else "arxiv: not a literature question"),
        ("sympy: an identity was offered to check" if r["sympy"]
         else "sympy: skipped at intro depth" if u["needs_symbolic"]
         else "sympy: nothing symbolic to verify"),
    ]
    return r


# ── stage 5: evidence engine ──────────────────────────────────────────────

_EVIDENCE_SYS = """You are the evidence stage. You do NOT answer the question. \
Assess the material and reply with ONLY JSON:

{"usable":["tags that genuinely bear on the question"],
 "off_topic":["tags that merely share vocabulary"],
 "agreements":["what two or more sources independently support"],
 "conflicts":["where sources disagree, naming tags"],
 "gaps":["what the question needs that no source provides"],
 "covered_by_curriculum":true|false,
 "confidence":"high|medium|low"}

Set covered_by_curriculum false when the question is outside what the [C:] \
topics actually teach — saying so plainly is more useful than stretching a \
loosely-related topic to fit."""


def evidence_engine(question, topics, book_ev, papers, computed, symbolic,
                    depth, client, budget):
    if plan_for("evidence", depth) is None:
        return {"skipped": True, "usable": [c["tag"] for c in book_ev.get("kept", [])],
                "off_topic": [], "agreements": [], "conflicts": [], "gaps": [],
                "covered_by_curriculum": bool(topics), "confidence": "unassessed"}
    parts = [curriculum_block(topics)] if topics else []
    parts += [f"[{c['tag']}] BOOK · {c['source']}\n{c['text'][:800]}"
              for c in book_ev.get("kept", [])]
    parts += [f"[A{i}] ARXIV {p['published']} · {p['title']}\n{p['summary'][:700]}"
              for i, p in enumerate(papers, 1)]
    if computed and computed.get("ran"):
        parts.append(f"[T1] SOLVER · {computed['result'].get('formula','')} → "
                     f"{ {k: v for k, v in computed['result'].items() if k != 'formula'} }")
    if symbolic and symbolic.get("ok"):
        parts.append(f"[X1] SYMBOLIC · {symbolic.get('verdict')}")
    out = _json_from(_call(client, "evidence", depth, _EVIDENCE_SYS,
                           f"QUESTION: {question}\n\nMATERIAL:\n" + "\n\n".join(parts),
                           budget), {})
    out.setdefault("usable", [c["tag"] for c in book_ev.get("kept", [])])
    for k in ("off_topic", "agreements", "conflicts", "gaps"):
        out.setdefault(k, [])
    out.setdefault("covered_by_curriculum", bool(topics))
    out.setdefault("confidence", "medium")
    out["skipped"] = False
    return out


# ── stage 6: reasoning ────────────────────────────────────────────────────

_REASON_SYS = """You are the reasoning stage. Do NOT write the final answer and \
do not address the reader. Produce a terse analytical skeleton the teaching \
stage will expand:

- the physical mechanism or argument, in logical order
- which tag supports each step, or "unsupported"
- where the topics connect to each other, if several are in play
- the subtlety or limiting case a careful student should notice

Bullets, not prose."""


def reasoning_engine(question, u, topics, book_ev, papers, computed, symbolic,
                     assessment, depth, client, budget):
    if plan_for("reasoning", depth) is None:
        return {"skipped": True, "text": ""}
    src = "\n\n".join(
        ([curriculum_block(topics)] if topics else []) +
        [f"[{c['tag']}] {c['source']}\n{c['text'][:800]}" for c in book_ev.get("kept", [])] +
        [f"[A{i}] {p['title']}: {p['summary'][:500]}" for i, p in enumerate(papers, 1)])
    if computed and computed.get("ran"):
        src += f"\n\n[T1] computed: {computed['result']}"
    if symbolic and symbolic.get("ok"):
        src += f"\n\n[X1] symbolic: {symbolic.get('verdict')}"
    return {"skipped": False,
            "text": _call(client, "reasoning", depth, _REASON_SYS,
                          f"QUESTION: {question}\nRESTATED: {u['restate']}\n\n"
                          f"ASSESSMENT: {assessment.get('gaps')} | "
                          f"covered={assessment.get('covered_by_curriculum')}\n\n"
                          f"MATERIAL:\n{src}", budget)}


# ── stage 7: professor ────────────────────────────────────────────────────

_PROF_SYS = """You are a physics tutor in the style of Feynman: build the \
intuition first, then formalise it.

Cite inline by tag — [C:topic-id] curriculum, [S#] a book, [A#] an arXiv \
paper, [T1] a solver value, [X1] a symbolic check. Never invent a tag.

NEVER state a numeric result of your own. Arithmetic has been done by the \
solver and shown separately; refer to the computed value in words. The same \
applies to any identity checked symbolically — if [X1] says a claimed identity \
does not hold, you must not assert it.

If the evidence says the question is not covered by the curriculum, open by \
saying so in one sentence, then answer from the books and general knowledge \
and make clear which is which. Use LaTeX for mathematics."""


def professor_engine(question, u, topics, book_ev, papers, computed, symbolic,
                     assessment, reasoning, mode, depth, mastery, client, budget):
    parts = ([curriculum_block(topics)] if topics else [])
    parts += [f"[{c['tag']}] ({c['source']} — {c['shelf'] if 'shelf' in c else 'physics'})\n{c['text'][:1000]}"
              for c in book_ev.get("kept", [])]
    parts += [f"[A{i}] ({p['published']} · {p['title']})\n{p['summary'][:800]}"
              for i, p in enumerate(papers, 1)]
    if computed and computed.get("ran"):
        parts.append(f"[T1] SOLVER — formula {computed['result'].get('formula','')}, "
                     f"inputs {computed['inputs']}, result "
                     f"{ {k: v for k, v in computed['result'].items() if k != 'formula'} }. "
                     f"Refer to it in words; do NOT restate the digits.")
    if symbolic and symbolic.get("ok"):
        parts.append(f"[X1] SYMBOLIC CHECK — {symbolic['verdict']} "
                     f"(difference: {symbolic.get('difference')})")
    src = "\n\n".join(parts) or "(no sources — say so, then answer from general knowledge)"

    fam = _familiarity(mastery)
    hist = {"new": "First time on this topic.",
            "revisited": "Seen once or twice before — go a level deeper than an introduction.",
            "familiar": "Worked several times — skip the introductory framing entirely."}[fam]
    extra = ""
    if not assessment.get("skipped"):
        extra = (f"\nEVIDENCE: covered_by_curriculum={assessment.get('covered_by_curriculum')} · "
                 f"conflicts={assessment.get('conflicts')} · gaps={assessment.get('gaps')} · "
                 f"confidence={assessment.get('confidence')}")
    if reasoning.get("text"):
        extra += f"\n\nREASONING SKELETON (expand, do not repeat verbatim):\n{reasoning['text']}"

    return _call(client, "professor", depth, _PROF_SYS,
                 f"QUESTION: {question}\n\n"
                 f"HOW TO ANSWER: {MODE_DIRECTIVE.get(mode, MODE_DIRECTIVE['explain'])}\n"
                 f"WHO YOU ARE ANSWERING: {DEPTH_DIRECTIVE.get(depth, DEPTH_DIRECTIVE['intermediate'])}\n"
                 f"LEARNER HISTORY: {hist}{extra}\n\nSOURCES:\n{src}\n\nWrite the answer now.",
                 budget)


# ── stage 8: validation ───────────────────────────────────────────────────

_VALIDATE_SYS = """You are the validation stage. Check the ANSWER against the \
SOURCES. Reply with ONLY JSON:

{"unsupported_claims":["claims presented as fact that no source backs"],
 "contradicts_sources":["claims conflicting with a source, naming the tag"],
 "restated_computed_numbers":["any figure the answer states that should have
   come from [T1] but was written out instead — include rounded or
   spelled-out restatements, not just exact digits"],
 "verdict":"pass|caution|fail",
 "note":"one sentence for the reader, or empty"}

Judge sourcing honesty, not coverage. Content the answer itself flags as
unsupported, or openly attributes to general knowledge, is CORRECT behaviour
and must not lower the verdict — an answer that says "the curriculum does not
cover this" is doing its job. Reserve "fail" for claims presented as sourced
that are not, for contradictions of a supplied source, or for restating a
computed number instead of referring to it."""


def validation(question, prose, book_ev, papers, computed, topics, depth, client, budget):
    offered = {c["tag"] for c in book_ev.get("kept", [])}
    offered |= {f"A{i}" for i in range(1, len(papers) + 1)}
    offered |= {f"C:{t.id}" for t in topics}
    if computed and computed.get("ran"):
        offered.add("T1")
    cited = set(re.findall(r"\[([A-Za-z]+:?[\w\-]*)\]", prose))
    cited = {c for c in cited if c in offered or re.match(r"^[SATX]\d+$", c)}
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", prose) if len(s.strip()) > 40]
    out = {"cited": sorted(cited), "offered": sorted(offered),
           "fabricated_tags": sorted(c for c in cited if c not in offered),
           "sentences": len(sentences),
           "uncited_sentences": sum(1 for s in sentences if not re.search(r"\[[A-Za-z]", s))}
    if plan_for("validation", depth) is None:
        out.update(verdict="pass" if not out["fabricated_tags"] else "caution",
                   semantic_skipped=True, note="intro depth — structural checks only")
        return out
    src = "\n\n".join([f"[{c['tag']}] {c['text'][:600]}" for c in book_ev.get("kept", [])] +
                      [f"[A{i}] {p['summary'][:500]}" for i, p in enumerate(papers, 1)])
    if computed and computed.get("ran"):
        src += f"\n\n[T1] {computed['result']}"
    sem = _json_from(_call(client, "validation", depth, _VALIDATE_SYS,
                           f"QUESTION: {question}\n\nSOURCES:\n{src or '(none)'}\n\n"
                           f"ANSWER:\n{prose[:6000]}", budget),
                     {"verdict": "pass", "note": ""})
    out.update(semantic_skipped=False,
               unsupported_claims=sem.get("unsupported_claims", []),
               contradicts_sources=sem.get("contradicts_sources", []),
               restated_computed_numbers=sem.get("restated_computed_numbers", []),
               note=sem.get("note", ""),
               verdict=("caution" if out["fabricated_tags"] else sem.get("verdict", "pass")))
    return out


# ── the pipeline ──────────────────────────────────────────────────────────

def run(question: str, mode: str = "explain",
        depth: str = "intermediate") -> Iterator[tuple[str, dict]]:
    question = (question or "").strip()
    if not question:
        yield "error", {"message": "empty question"}
        return
    key = _api_key()
    if not key:
        yield "error", {"message": "No DEEPSEEK_API_KEY found"}
        return
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com",
                    timeout=300.0, max_retries=1)
    budget = Budget()
    t_start = time.monotonic()

    yield "understand", {"msg": "Reading the question…"}
    u = understand(question, depth, client, budget)
    yield "understand", {"msg": f"{u['intent']} · {', '.join(u['topics'][:4])}",
                         "understanding": u}

    topics = match_topics(question, k=4)
    probe = compute_for(question, topics[0].id if topics else None)
    r = route(u, topics, probe, depth)
    yield "route", {"msg": " + ".join(k for k in
                                      ("curriculum", "books", "solver", "arxiv", "sympy")
                                      if r[k]),
                    "routing": r,
                    "topics": [{"id": t.id, "title": t.title, "level": t.level}
                               for t in topics]}

    yield "gather", {"msg": "Querying sources…"}
    t0 = time.monotonic()
    query = question + " " + " ".join(u["topics"][:4])
    papers, symbolic = [], None
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_books = pool.submit(retrieve_evidence, query)
        f_arx = pool.submit(R.search_arxiv, query, 5) if r["arxiv"] else None
        if r["sympy"] and "=" in u["identity"]:
            lhs, _, rhs = u["identity"].partition("=")
            symbolic = R.check_identity(lhs.strip(), rhs.strip())
        book_ev = f_books.result()
        if f_arx:
            papers = (f_arx.result() or {}).get("papers", [])
    computed = probe if (probe and probe.get("ran")) else None
    yield "gather", {"msg": (f"{len(topics)} topic(s), {len(book_ev.get('kept', []))} passage(s)"
                             + (f", {len(papers)} paper(s)" if papers else "")
                             + (", 1 computed" if computed else "")
                             + (", 1 symbolic check" if symbolic and symbolic.get("ok") else "")
                             + f" · {int((time.monotonic()-t0)*1000)}ms"),
                     "evidence": book_ev, "papers": papers,
                     "computed": computed, "symbolic": symbolic}

    yield "evidence", {"msg": "Verifying and comparing…"}
    assessment = evidence_engine(question, topics, book_ev, papers, computed,
                                 symbolic, depth, client, budget)
    yield "evidence", {"msg": ("skipped (intro)" if assessment.get("skipped") else
                               f"{len(assessment['usable'])} usable · "
                               f"curriculum covers it: {assessment.get('covered_by_curriculum')} · "
                               f"confidence {assessment.get('confidence')}"),
                       "assessment": assessment}

    yield "reasoning", {"msg": "Building the argument…"}
    reasoning = reasoning_engine(question, u, topics, book_ev, papers, computed,
                                 symbolic, assessment, depth, client, budget)
    yield "reasoning", {"msg": ("folded into teaching (intro)" if reasoning.get("skipped")
                                else f"{len(reasoning['text'].split())} words of skeleton"),
                        "reasoning": reasoning}

    mastery = record_visit(topics[0].id if topics else None, mode, depth)
    box: dict = {}

    def _teach():
        try:
            box["prose"] = professor_engine(question, u, topics, book_ev, papers,
                                            computed, symbolic, assessment, reasoning,
                                            mode, depth, mastery, client, budget)
        except Exception as exc:
            box["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=_teach, daemon=True)
    t_p = time.monotonic()
    yield "professor", {"msg": "Teaching…", "elapsed_s": 0}
    worker.start()
    while worker.is_alive():
        worker.join(timeout=5.0)
        if worker.is_alive():
            yield "professor", {"msg": f"Teaching… {int(time.monotonic()-t_p)}s",
                                "elapsed_s": int(time.monotonic() - t_p)}
    if box.get("error"):
        yield "error", {"message": box["error"]}
        return
    prose = (box.get("prose") or "").strip()
    if not prose:
        yield "error", {"message": "the teaching stage returned nothing"}
        return
    yield "professor", {"msg": f"Taught in {int(time.monotonic()-t_p)}s",
                        "elapsed_s": int(time.monotonic() - t_p)}

    yield "validation", {"msg": "Checking the answer against its sources…"}
    checks = validation(question, prose, book_ev, papers, computed, topics,
                        depth, client, budget)
    yield "validation", {"msg": (f"{checks['verdict']} · {checks['uncited_sentences']}"
                                 f"/{checks['sentences']} uncited"
                                 + (f" · {len(checks['fabricated_tags'])} fabricated"
                                    if checks["fabricated_tags"] else "")),
                         "validation": checks}

    yield "done", {
        "question": question, "mode": mode, "depth": depth,
        "prose": prose, "understanding": u, "routing": r,
        "topics": [{"id": t.id, "title": t.title, "level": t.level} for t in topics],
        "evidence": book_ev, "papers": papers, "computed": computed,
        "symbolic": symbolic, "assessment": assessment, "reasoning": reasoning,
        "validation": checks, "mastery": mastery, "familiarity": _familiarity(mastery),
        "budget": budget.summary(), "elapsed_s": int(time.monotonic() - t_start),
        "honesty": {
            "numbers_are_computed_not_generated": bool(computed),
            "identity_checked_symbolically": bool(symbolic and symbolic.get("ok")),
            "covered_by_curriculum": assessment.get("covered_by_curriculum"),
            "note": ("[C:] curriculum · [S#] your books · [A#] arXiv · [T1] solver · "
                     "[X1] symbolic check. Untagged sentences are the model's synthesis."),
        },
    }


def answer(question: str, mode: str = "explain", depth: str = "intermediate") -> dict:
    last: dict = {}
    for stage, payload in run(question, mode, depth):
        if stage in ("done", "error"):
            last = {"stage": stage, **payload}
    return last
