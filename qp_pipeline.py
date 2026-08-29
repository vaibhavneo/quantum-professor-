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
    from .evidence_pack import EvidencePack, build_evidence_pack
    from .library import TOPICS
    from .llm_errors import ProviderError, classify_llm_error, is_llm_sdk_error
    from .model_gateway import get_gateway
    from .verification import verify_derivation
    from .deterministic_execution import execute_deterministically
    from .tutor import (DEPTH_DIRECTIVE, MODE_DIRECTIVE, MODEL_DEEP, MODEL_FAST,
                        MIN_TOPIC_SCORE, _api_key, _apply_secondary_floor, _familiarity,
                        _score_topics, compute_for, record_visit, retrieve_evidence,
                        suggest_related)
except ImportError:
    import research as R
    from evidence_pack import EvidencePack, build_evidence_pack
    from library import TOPICS
    from llm_errors import ProviderError, classify_llm_error, is_llm_sdk_error
    from model_gateway import get_gateway
    from verification import verify_derivation
    from deterministic_execution import execute_deterministically
    from tutor import (DEPTH_DIRECTIVE, MODE_DIRECTIVE, MODEL_DEEP, MODEL_FAST,
                       MIN_TOPIC_SCORE, _api_key, _apply_secondary_floor, _familiarity,
                       _score_topics, compute_for, record_visit, retrieve_evidence,
                       suggest_related)

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


# Shared ordinal scale for the UI's depth dropdown and the question's own
# inferred difficulty (understand(), below) - route() takes the harder of
# the two rather than trusting the UI depth alone.
_DEPTH_RANK = {"intro": 0, "intermediate": 1, "advanced": 2}


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
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=limit,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
        except Exception as exc:
            if is_llm_sdk_error(exc):
                kind, status, msg = classify_llm_error(exc)
                raise ProviderError(kind, status, msg, exc) from exc
            raise
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
    """Top-k curriculum topics that each individually clear the relevance
    floor - not one best guess plus whatever ranked 2nd-4th regardless of
    score. match_topic()'s own 'related' list used to do exactly that (no
    floor at all on positions 2-4), which is how a topic that only shares
    one coincidental word with the question could still ride along into the
    evidence a real question never asked for.

    Also applies the secondary-match ratio floor: a candidate beyond the top
    one must retain a meaningful fraction of the top match's own score, so a
    strong primary match (e.g. hydrogen-atom) can't drag in a same-ballpark-
    but-unrelated topic (e.g. harmonic-oscillator) just because both cleared
    the same absolute floor independently.
    """
    scored = _apply_secondary_floor(_score_topics(question))
    return [t for score, t in scored[:k] if score >= MIN_TOPIC_SCORE]


def _topic_block(t) -> str:
    return (f"[C:{t.id}] {t.title} ({t.level})\n"
            f"  intuition: {t.intuition}\n"
            f"  key concepts: {', '.join(t.key_concepts)}\n"
            f"  key equations (LaTeX): {' ; '.join(t.key_equations)}")


def curriculum_block(topics, sides=None) -> str:
    if sides:
        parts = []
        for side in sides:
            side_topics = side.get("topics") or []
            body = ("\n\n".join(_topic_block(t) for t in side_topics) if side_topics
                    else "(no curriculum topic covers this side)")
            parts.append(f"COMPARISON SIDE: {side['label']}\n{body}")
        return "\n\n".join(parts)
    if not topics:
        return ""
    return "\n\n".join(_topic_block(t) for t in topics)


def _evidence_rank(ev: dict) -> int:
    return {"none": 0, "weak": 1, "usable": 2}.get(ev.get("evidence_strength", "none"), 0)


def _diversify_evidence(kept: list, max_per_source: int = 3) -> list:
    """Caps how many chunks one single book can contribute, preserving
    relative rank order otherwise. Without this, a top-K cutoff can fill
    every slot from one book's table of contents or index pages purely
    because that book scores highest on the query's vocabulary, crowding
    out corroborating evidence from anywhere else - "diversity", not more
    results, is the actual fix, so this reorders/drops, it never widens K.
    """
    counts: dict = {}
    out = []
    for item in kept:
        src = item.get("source", "")
        if counts.get(src, 0) >= max_per_source:
            continue
        counts[src] = counts.get(src, 0) + 1
        out.append(item)
    return out


_EQUATION_MARKERS_RE = re.compile(r"[=∫∑∂ℏ≈≤≥±→]|\\[a-zA-Z]+|\$")


def _rank_for_intent(kept: list, equations_required: bool) -> list:
    """When the question actually asked for mathematics, break ties toward
    passages that are denser in equations rather than pure prose - the raw
    TF-IDF score alone doesn't know the difference between a worked
    derivation and a paragraph that merely mentions the same terms."""
    if not equations_required:
        return kept
    return sorted(kept, key=lambda c: (-len(_EQUATION_MARKERS_RE.findall(c.get("text", ""))),
                                       -c.get("raw_score", 0)))


def _retrieve_with_topic_context(query_base: str, topics: list, equations_required: bool = False):
    """Widens the TF-IDF query with real, curriculum-grounded vocabulary
    instead of searching on the raw question alone - this is what "domain
    matching" and "prerequisite relationships" mean as retrieval techniques,
    not just as labels attached after the fact. Returns (evidence, linked_topic_ids).

    If the best-matched topic's own key concepts still come back weak, one
    extra attempt widens further using that topic's first prerequisite -
    the foundational material a weak-evidence topic builds on is often where
    the actual explanatory passages live. The final kept list is capped for
    per-source diversity and, when mathematics was requested, re-ranked
    toward equation-dense passages - retrieval intelligence applied on top
    of the existing TF-IDF index, not a bigger top-K pulled from it.
    """
    if not topics:
        ev = retrieve_evidence(query_base)
        kept = _rank_for_intent(_diversify_evidence(ev.get("kept", [])), equations_required)
        return {**ev, "kept": kept}, []
    best = topics[0]
    query = f"{query_base} {best.title} {' '.join(best.key_concepts[:3])} {domain_for(topics)}"
    ev = retrieve_evidence(query)
    linked = [best.id]
    if _evidence_rank(ev) < 2 and best.prerequisites:
        prereq = TOPICS.get(best.prerequisites[0])
        if prereq:
            widened = retrieve_evidence(f"{query} {prereq.title}")
            if _evidence_rank(widened) > _evidence_rank(ev):
                ev, linked = widened, linked + [prereq.id]
    kept = _rank_for_intent(_diversify_evidence(ev.get("kept", [])), equations_required)
    return {**ev, "kept": kept}, linked


def gather_comparison_sides(targets: list[str], top_k_per_side: int = 5,
                            equations_required: bool = False):
    """Retrieves curriculum + book evidence for each comparison target
    independently, so one side's vocabulary never leaks into the other's
    retrieval - the actual mechanism that lets "classical mechanics and
    quantum mechanics" be scored as one compound string and match neither
    side honestly. Tags are renumbered sequentially across sides (each
    side's retrieve_evidence() independently starts at S1) so citations
    stay unambiguous.
    """
    merged_kept, merged_rejected, sides_meta = [], [], []
    n = 1
    for label in targets:
        topics = match_topics(label, k=top_k_per_side)
        ev, linked = _retrieve_with_topic_context(label, topics, equations_required)
        for item in ev.get("kept", []):
            item = dict(item, tag=f"S{n}", side=label, linked_topics=linked)
            n += 1
            merged_kept.append(item)
        for item in ev.get("rejected", []):
            merged_rejected.append(dict(item, side=label))
        sides_meta.append({
            "label": label, "topics": topics,
            "covered_by_curriculum": bool(topics),
            "evidence_strength": ev.get("evidence_strength", "none"),
            "domain": domain_for(topics),
        })
    seen, all_topics = set(), []
    for side in sides_meta:
        for t in side["topics"]:
            if t.id not in seen:
                seen.add(t.id)
                all_topics.append(t)
    strengths = {s["evidence_strength"] for s in sides_meta}
    overall_strength = ("usable" if "usable" in strengths
                        else "weak" if "weak" in strengths else "none")
    merged_book_ev = {"available": True, "kept": merged_kept, "rejected": merged_rejected,
                      "evidence_strength": overall_strength}
    return merged_book_ev, all_topics, sides_meta


# Coarse subject label for the handful of curriculum topics that reach beyond
# core quantum mechanics - not a second classification path, just a grouping
# over whichever topics match_topics() already found.
_DOMAIN_OVERRIDES = {
    "special-relativity": "relativity", "general-relativity": "relativity",
    "standard-model": "particle-physics", "qft-fundamentals": "particle-physics",
    "astrophysics-stars": "astrophysics", "cosmology": "astrophysics",
    "quantum-information": "quantum-computing", "decoherence": "quantum-computing",
    "quantum-optics": "quantum-computing",
    "lagrangian-hamiltonian-mechanics": "classical-mechanics",
}


def domain_for(topics) -> str:
    if not topics:
        return "general-physics"
    from collections import Counter
    return Counter(_DOMAIN_OVERRIDES.get(t.id, "quantum-mechanics")
                   for t in topics).most_common(1)[0][0]


def subdomain_for(topics) -> str:
    return topics[0].title if topics else "uncovered"


def concepts_for(topics, limit: int = 8) -> list[str]:
    """Real, curriculum-grounded physics concepts - the matched topics' own
    key_concepts, not a re-derivation of the free-text search terms already
    sitting in "topics". Distinct field, distinct source of truth."""
    seen, out = set(), []
    for t in topics:
        for c in t.key_concepts:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out[:limit]


# ── deterministic question understanding (no LLM call) ─────────────────────

def _clean_target(s: str) -> str:
    s = s.strip().strip(" .,;:!?")
    return re.sub(r"^(the|a|an)\s+", "", s, flags=re.I).strip()


def _borrow_shared_head_noun(parts: list[str]) -> list[str]:
    """"Newtonian, Lagrangian and Hamiltonian mechanics" naively splits into
    ["Newtonian", "Lagrangian", "Hamiltonian mechanics"] - the bare-word
    parts silently lose the noun the group shares. Borrow the trailing
    qualifier from the last multi-word part onto every bare single-word
    part, so retrieval for the short side searches "Newtonian mechanics",
    not the far more generic "Newtonian"."""
    multi = [p for p in parts if len(p.split()) > 1]
    if not multi:
        return parts
    trailing = " ".join(multi[-1].split()[1:])
    return [p if len(p.split()) > 1 else f"{p} {trailing}".strip() for p in parts]


def _split_n_way(body: str) -> list[str]:
    """Splits "A, B and C" / "A, B, C" / "A vs B vs C" into parts - a
    comparison isn't always binary ("compare Newtonian, Lagrangian and
    Hamiltonian mechanics" names three)."""
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+|\s+(?:vs\.?|versus)\s+", body, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


# Common everyday phrasings for the same curriculum subject - "quantum
# physics" and "quantum mechanics" mean the same thing to a person asking,
# but only one of them is the vocabulary the curriculum and retrieval index
# actually use. Applied only to extracted comparison targets, never to the
# raw question shown back to the user elsewhere.
_DOMAIN_SYNONYMS = {
    "quantum physics": "quantum mechanics", "quantum theory": "quantum mechanics",
    "classical physics": "classical mechanics", "newtonian physics": "newtonian mechanics",
}


def _normalize_domain_term(term: str) -> str:
    return _DOMAIN_SYNONYMS.get(term.lower().strip(), term)


_SENTENCE_SPLIT_RE = re.compile(r"[^.?!]+[.?!]?")

_COMPARISON_PATTERNS = [
    re.compile(r"\bi\s+(?:actually\s+)?meant\s+(.+?)[\?\.\!]?$", re.I),
    re.compile(r"\bdifference[s]?\s+between\s+(.+?)[\?\.\!]?$", re.I),
    re.compile(r"\bhow\s+(?:does|do)\s+(.+?)\s+differ[s]?\s+from\s+(.+?)[\?\.\!]?$", re.I),
    re.compile(r"\bcompar(?:e|ing|ison)(?:\s+of)?\s+(.+?)[\?\.\!]?$", re.I),
    re.compile(r"(.+?)\s+compared\s+to\s+(.+?)[\?\.\!]?$", re.I),
    re.compile(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)[\?\.\!]?$", re.I),
]


_TRAILING_MANNER_RE = re.compile(
    r"\s+(?:mathematically|conceptually|physically|numerically|graphically|"
    r"qualitatively|quantitatively|briefly|rigorously|in\s+detail)\.?$", re.I)


def _extract_from_sentence(sentence: str) -> list[str]:
    for pat in _COMPARISON_PATTERNS:
        m = pat.search(sentence)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        # A trailing manner adverb ("...mathematically") describes HOW to
        # compare, not a fourth thing being compared - it always lands in
        # the last captured group regardless of how many groups matched, so
        # strip it there before splitting, not after (splitting first would
        # weld it onto only the last target via the shared-head-noun borrow).
        groups[-1] = _TRAILING_MANNER_RE.sub("", groups[-1]).strip()
        parts = _split_n_way(groups[0]) if len(groups) == 1 else list(groups)
        parts = [_clean_target(p) for p in parts]
        parts = [p for p in parts if p]
        if len(parts) >= 2 and len({p.lower() for p in parts}) == len(parts):
            return _borrow_shared_head_noun(parts)
    return []


def extract_comparison_targets(question: str) -> list[str]:
    """Deterministic backstop for "what is being compared" - tried before
    any LLM call, and used as-is when the LLM is unavailable. Scoped to one
    sentence at a time so a trailing "Give me the mathematical explanation."
    can never get swallowed into the comparison's last target.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.findall(question) if s.strip()]
    for sentence in sentences or [question]:
        targets = _extract_from_sentence(sentence)
        if targets:
            return [_normalize_domain_term(t) for t in targets]
    return []


_MATH_RIGOROUS_KEYWORDS = ("derive", "derivation", "prove", "rigorously", "show that")
# Deliberately excludes bare "equation" - "explain the Schrodinger equation"
# names a topic, it doesn't ask for mathematical treatment, and the two read
# identically to a substring check. Only phrasing that's actually a REQUEST
# for equations counts.
_MATH_STANDARD_KEYWORDS = ("mathematically", "mathematical", "formally", "commutator",
                          "operator form", "in terms of")
_MATH_STANDARD_PHRASES = ("write the equation", "write down the equation",
                          "show the equation", "give the equation",
                          "what is the equation for", "in equation form",
                          "governing equation")


def detect_math_requirements(question: str) -> tuple[bool, str]:
    q = question.lower()
    if any(k in q for k in _MATH_RIGOROUS_KEYWORDS):
        return True, "rigorous"
    if any(k in q for k in _MATH_STANDARD_KEYWORDS) or any(p in q for p in _MATH_STANDARD_PHRASES):
        return True, "standard"
    return False, "none"


# "difficulty" (understand()'s existing field) is how advanced the QUESTION
# itself reads; "requested depth" is how much thoroughness the person asked
# for in the ANSWER - a graduate-level question can still want "just the
# short version", and an intro question can still ask to go deep.
_DEPTH_DEEP_KEYWORDS = ("in depth", "in-depth", "thoroughly", "comprehensive", "detailed",
                       "in detail", "deep dive", "exhaustive", "fully explain")
_DEPTH_SHALLOW_KEYWORDS = ("briefly", "quick overview", "in short", "concisely",
                          "short answer", "quickly", "tl;dr", "summary")


def detect_requested_depth(question: str) -> str:
    q = question.lower()
    if any(k in q for k in _DEPTH_DEEP_KEYWORDS):
        return "deep"
    if any(k in q for k in _DEPTH_SHALLOW_KEYWORDS):
        return "shallow"
    return "standard"


_CORRECTION_PATTERNS = [
    re.compile(r"\bi\s+(?:actually\s+)?meant\b", re.I),
    re.compile(r"^\s*correction[:\-]", re.I),
    re.compile(r"\bsorry,?\s+i\s+meant\b", re.I),
    re.compile(r"\binstead\s+of\b", re.I),
    re.compile(r"\bnot\s+.+?\s+but\s+.+", re.I),
]


def detect_correction(question: str) -> bool:
    return any(p.search(question) for p in _CORRECTION_PATTERNS)


_CONSTRAINT_PATTERNS = [
    (re.compile(r"\bwithout (?:using )?calculus\b|\bno calculus\b", re.I), "no calculus"),
    (re.compile(r"\bwithout (?:using )?(?:any )?equations\b", re.I), "no equations"),
    (re.compile(r"\bno (?:math|maths|mathematics)\b", re.I), "no math"),
    (re.compile(r"\bfor a (?:complete )?beginner\b", re.I), "for a beginner"),
    (re.compile(r"\bfor a high school(?:er| student)\b", re.I), "for a high schooler"),
    (re.compile(r"\beli5\b|\bexplain like i'?m five\b", re.I), "explain like I'm five"),
    (re.compile(r"\bin (?:one|a single) (?:paragraph|sentence)\b", re.I), "in one paragraph/sentence"),
    (re.compile(r"\bbriefly\b|\bin a few (?:sentences|words)\b|\bkeep it short\b", re.I), "keep it brief"),
    (re.compile(r"\bin (?:simple|plain) terms\b|\bin layman'?s terms\b", re.I), "in simple/plain terms"),
    (re.compile(r"\bno jargon\b", re.I), "no jargon"),
    (re.compile(r"\bstep[- ]by[- ]step\b", re.I), "step-by-step"),
]


def detect_explicit_constraints(question: str) -> list[str]:
    """Deterministic backstop for constraints the question states outright -
    "no calculus", "for a beginner", "briefly", etc. Purely additive: an
    unmatched constraint just never surfaces here, it isn't invented."""
    found = []
    for pattern, label in _CONSTRAINT_PATTERNS:
        if pattern.search(question) and label not in found:
            found.append(label)
    return found


def deterministic_understand(question: str, prior: dict | None = None) -> dict:
    """The offline-capable substitute for understand() - same return shape,
    computed entirely from cheap heuristics, zero LLM cost. Used as the
    baseline before any LLM call is attempted, and as the whole of "understand"
    when the LLM is unavailable."""
    comparison_targets = extract_comparison_targets(question)
    equations_required, math_depth = detect_math_requirements(question)
    is_correction = detect_correction(question)
    if is_correction and prior:
        if not comparison_targets and prior.get("comparison_targets"):
            comparison_targets = prior["comparison_targets"]
        corrected_from = prior.get("restate")
    else:
        corrected_from = None
    if comparison_targets:
        intent = "compare"
    elif equations_required:
        intent = "derive"
    else:
        intent = "explain"
    topics = re.findall(r"[a-z][a-z\-]{3,}", question.lower())[:6]
    if is_correction and prior and prior.get("topics") and not comparison_targets:
        topics = list(dict.fromkeys(topics + prior["topics"]))[:6]
    return {
        "intent": intent, "topics": topics, "needs_literature": False,
        "needs_symbolic": False, "identity": "", "difficulty": "intermediate",
        "restate": question, "comparison_targets": comparison_targets,
        "equations_required": equations_required, "math_depth": math_depth,
        "requested_depth": detect_requested_depth(question),
        "explicit_constraints": detect_explicit_constraints(question),
        "is_correction": is_correction, "corrected_from": corrected_from,
    }


def backfill_deterministic_fields(u: dict, question: str, prior: dict | None = None) -> dict:
    """Fills genuine gaps with deterministic heuristics - never overrides a
    value the LLM bothered to supply. Also the only source for is_correction/
    corrected_from, which no LLM call is ever asked about."""
    det = deterministic_understand(question, prior)
    u["comparison_targets"] = u.get("comparison_targets") or det["comparison_targets"]
    if u.get("equations_required") is None:
        u["equations_required"] = det["equations_required"]
    u["math_depth"] = u.get("math_depth") or det["math_depth"]
    u["requested_depth"] = u.get("requested_depth") or det["requested_depth"]
    u["explicit_constraints"] = u.get("explicit_constraints") or det["explicit_constraints"]
    u["is_correction"] = det["is_correction"]
    u["corrected_from"] = det["corrected_from"]
    if det["is_correction"] and prior and prior.get("topics") and not u.get("topics"):
        u["topics"] = prior["topics"]
    return u


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
 "difficulty":"intro|intermediate|advanced",  // the QUESTION's own apparent
                                              // level - independent of
                                              // whatever depth the UI is set to
 "comparison_targets":["the two (or more) things being compared"] or [],
 "equations_required":true|false,   // does answering well require showing
                                    // equations/derivations, not just prose
 "math_depth":"none|standard|rigorous",
 "requested_depth":"shallow|standard|deep",  // how thorough the ANSWER should
                                             // be - distinct from difficulty,
                                             // which is how advanced the
                                             // QUESTION itself reads
 "explicit_constraints":["any constraint the question itself states, e.g. "
                        "'no calculus', 'in one paragraph', 'for a beginner'"],
 "restate":"one sentence restating what is being asked"}"""


def understand(question, depth, client, budget, prior=None):
    prior_line = f"\n\nPRIOR TURN (for corrections/continuity): {prior}" if prior else ""
    u = _json_from(_call(client, "understand", depth, _UNDERSTAND_SYS,
                         f"QUESTION: {question}{prior_line}", budget), {})
    result = {
        "intent": u.get("intent", "explain"),
        "topics": u.get("topics") or re.findall(r"[a-z][a-z\-]{3,}", question.lower())[:6],
        "needs_literature": bool(u.get("needs_literature")),
        "needs_symbolic": bool(u.get("needs_symbolic")),
        "identity": (u.get("identity") or "").strip(),
        "difficulty": u.get("difficulty") if u.get("difficulty") in _DEPTH_RANK else "intermediate",
        "restate": u.get("restate") or question,
        "comparison_targets": u.get("comparison_targets") or None,
        "equations_required": u.get("equations_required") if isinstance(u.get("equations_required"), bool) else None,
        "math_depth": u.get("math_depth") if u.get("math_depth") in ("none", "standard", "rigorous") else None,
        "requested_depth": u.get("requested_depth") if u.get("requested_depth") in
                           ("shallow", "standard", "deep") else None,
        "explicit_constraints": u.get("explicit_constraints") or None,
    }
    return backfill_deterministic_fields(result, question, prior)


# ── stage 3: router ───────────────────────────────────────────────────────

def route(u, topics, solver_probe, depth):
    """Deterministic. Routing is policy; spending a model call to re-derive a
    rule the code already knows is the waste this design avoids.

    "Skipped at intro depth" used to mean the UI's depth dropdown alone; now
    it means neither the UI depth NOR the question's own inferred difficulty
    called for it — a graduate-level question still gets literature/symbolic
    search even when the UI is set to intro.
    """
    eff = max(_DEPTH_RANK.get(depth, 1), _DEPTH_RANK.get(u.get("difficulty", "intermediate"), 1))
    r = {
        "curriculum": bool(topics),
        "books": True,
        "solver": bool(solver_probe and solver_probe.get("ran")),
        "arxiv": bool(u["needs_literature"]) and eff > 0,
        "sympy": bool(u["needs_symbolic"] and u["identity"]) and eff > 0,
    }
    r["why"] = [
        f"curriculum: {len(topics)} topic(s) matched" if topics else "curriculum: no topic matched",
        "books: always — the physics shelf is the primary source",
        ("solver: the question carries numeric parameters" if r["solver"]
         else "solver: nothing to compute from this question"),
        ("arxiv: research-level or time-sensitive" if r["arxiv"]
         else "arxiv: skipped — both UI depth and question read as intro" if u["needs_literature"]
         else "arxiv: not a literature question"),
        ("sympy: an identity was offered to check" if r["sympy"]
         else "sympy: skipped — both UI depth and question read as intro" if u["needs_symbolic"]
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
                    depth, client, budget, sides=None):
    if plan_for("evidence", depth) is None:
        return {"skipped": True, "usable": [c["tag"] for c in book_ev.get("kept", [])],
                "off_topic": [], "agreements": [], "conflicts": [], "gaps": [],
                "covered_by_curriculum": bool(topics), "confidence": "unassessed"}
    parts = [curriculum_block(topics, sides)] if (topics or sides) else []
    parts += [f"[{c['tag']}] BOOK" + (f" (side: {c['side']})" if c.get("side") else "")
              + f" · {c['source']}\n{c['text'][:800]}"
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


def _off_topic_tags(assessment: dict) -> set[str]:
    """Tags the evidence stage judged off-topic - S#/A# only.

    A deny-list, not an allow-list: excluding only what was explicitly
    flagged avoids silently dropping a good source the model simply forgot
    to re-list in "usable" (a JSON-omission slip, not a relevance verdict).
    Curriculum ([C:]) and computed ([T1]/[X1]) tags are deliberately left
    alone - see the evidence stage's own docstring for why.
    """
    return set(assessment.get("off_topic") or [])


# ── stage 6: derivation plan & physical interpretation ─────────────────────
# (previously a single generic "reasoning" skeleton; folded into one call
# per the cost-control decision - Mathematical Objects above this stage are
# extracted deterministically, so this call only has to plan the derivation
# and interpret it, not also hunt for which equations are even in play.)

_REASON_SYS = """You are the Derivation Plan & Physical Interpretation stage. \
Do NOT write the final answer and do not address the reader - this feeds the \
teaching stage, which expands it into prose. Reply in labeled parts:

GIVEN / FIND / ASSUMPTIONS
- only when GIVENS, FIND, or ASSUMPTIONS are supplied below: restate them \
plainly - they are already extracted for you, do not re-derive or second-\
guess them
- omit this section entirely when none are supplied (a purely conceptual \
question) - never invent a given or an assumption that wasn't handed to you

DERIVATION PLAN
- the starting point (a named equation or definition, tagged) and the \
assumptions it rests on
- each algebraic or logical step in order, one per line, noting which tag \
supports it, or "unsupported"
- when a numeric GIVEN was supplied, show the explicit substitution step - \
never jump straight from the symbolic formula to a bare stated number
- the result the steps arrive at
- if the supplied MATHEMATICAL OBJECTS don't actually support a full \
derivation, say plainly which step can't be justified from what's given - \
never invent a step to complete one

PHYSICAL INTERPRETATION
- what the mathematical result actually means physically, in plain language
- where the topics connect to each other, if several are in play
- the subtlety or limiting case a careful student should notice

Bullets, not prose, in every part."""


def reasoning_engine(question, u, topics, book_ev, papers, computed, symbolic,
                     assessment, depth, client, budget, sides=None, pack=None):
    if plan_for("reasoning", depth) is None:
        return {"skipped": True, "text": "", "derivation_plan": "", "physical_interpretation": ""}
    excluded = _off_topic_tags(assessment)
    src = "\n\n".join(
        ([curriculum_block(topics, sides)] if (topics or sides) else []) +
        [f"[{c['tag']}]" + (f" (side: {c['side']})" if c.get("side") else "")
         + f" {c['source']}\n{c['text'][:800]}" for c in book_ev.get("kept", [])
         if c["tag"] not in excluded] +
        [f"[A{i}] {p['title']}: {p['summary'][:500]}" for i, p in enumerate(papers, 1)
         if f"A{i}" not in excluded])
    if computed and computed.get("ran"):
        src += f"\n\n[T1] computed: {computed['result']}"
    if symbolic and symbolic.get("ok"):
        src += f"\n\n[X1] symbolic: {symbolic.get('verdict')}"
    if pack is not None and pack.prerequisite_concepts:
        src += f"\n\nPREREQUISITE CONCEPTS: {', '.join(pack.prerequisite_concepts[:5])}"
    if pack is not None and pack.mathematical_objects:
        src += "\n\nMATHEMATICAL OBJECTS:\n" + "\n".join(
            f"[{o['tag']}] {o['name']}: {o['expression']}" for o in pack.mathematical_objects)
    if pack is not None and pack.givens:
        src += "\n\nGIVEN: " + "; ".join(pack.givens)
    if pack is not None and pack.unknowns:
        src += "\nFIND: " + "; ".join(pack.unknowns)
    if pack is not None and pack.assumptions:
        src += "\nSTANDING ASSUMPTIONS: " + "; ".join(pack.assumptions)
    if pack is not None and pack.strategy:
        src += f"\nSOLUTION STRATEGY: {pack.strategy}"
    text = _call(client, "reasoning", depth, _REASON_SYS,
                f"QUESTION: {question}\nRESTATED: {u['restate']}\n\n"
                f"ASSESSMENT: {assessment.get('gaps')} | "
                f"covered={assessment.get('covered_by_curriculum')}\n\n"
                f"MATERIAL:\n{src}", budget)
    derivation_plan, _, physical_interpretation = text.partition("PHYSICAL INTERPRETATION")
    derivation_plan = derivation_plan.replace("DERIVATION PLAN", "", 1).strip()
    return {"skipped": False, "text": text, "derivation_plan": derivation_plan,
           "physical_interpretation": physical_interpretation.strip()}


# ── stage 7: professor ────────────────────────────────────────────────────

_PROF_SYS = """You are a physics tutor in the style of Feynman: build the \
intuition first, then formalise it.

Cite inline by tag — [C:topic-id] curriculum, [S#] a book, [A#] an arXiv \
paper, [T1] a solver value, [X1] a symbolic check. Never invent a tag.

NEVER state a numeric result of your own. Arithmetic has been done by the \
solver and shown separately; refer to the computed value in words. The same \
applies to any identity checked symbolically — if [X1] says a claimed identity \
does not hold, you must not assert it.

Answer the question. Lead with the physics, never with an apology or an \
inventory of what you lack. The reader came for the connection between ideas, \
not for a report on your retrieval.

Sourcing is shown by your tags, so it needs no preamble. Use every source you \
were given — the [C:] curriculum topics are real material, and their concepts \
and equations are yours to build on and connect even when no book passage was \
retrieved. Reason across them: joining two topics the curriculum teaches \
separately is exactly the work expected here.

When something genuinely falls outside everything supplied, note it in one \
short clause at the point where it arises — "the curriculum does not cover the \
Standard Model, so this part is from general physics" — and carry on. Never \
open with it, never dwell on it, and never let it displace the explanation. \
Use LaTeX for mathematics.

Keep the epistemic status of each claim recoverable from how you write it, \
not by labeling every sentence: a cited claim ([C:]/[S#]/[A#]) is source-\
grounded; a computed or symbolically-checked one ([T1]/[X1]) is exact; \
anything else you say is your own reasoning connecting them, and it should \
read that way rather than borrowing a citation's authority. If sources \
disagree, say what each one claims rather than silently picking a side. If \
the evidence is too thin to derive something rigorously, say what step you \
can support and name the gap - a partial derivation the reader can trust \
beats a complete one you invented.

A MATRIX OPERATION EXECUTED, UNIT CONVERSION EXECUTED, or DIFFERENTIAL \
EQUATION SOLVED line, when present, is a real computation the pipeline \
already ran (like [T1]/[X1]) - refer to it in words, never restate the raw \
matrix/equation or recompute it yourself.

When a VERIFICATION result is supplied, it is a deterministic check the \
pipeline already ran, not your opinion - treat it as ground truth. State \
plainly whether the math was verified mathematically, partially verified, \
not independently verified, or failed verification outright, and if a check \
failed, use the given correction rather than the derivation's original \
(wrong) value. A failed or partially verified result is not something to \
soften - say so as plainly as a verified one."""


_STRUCTURE_DIRECTIVE = (
    "Where they genuinely apply, structure the answer with these sections, in "
    "this order, omitting any that don't apply to this question: Direct Answer, "
    "Given & Find, Intuition, Mathematical Formulation, Assumptions, Derivation, "
    "Comparison, Worked Example, Key Takeaway. Given & Find applies only to a "
    "genuine multi-step numeric problem (real given values and a real quantity "
    "to compute) - state them exactly as supplied, never invented. Do not force "
    "an empty section to exist just to fill the template. Knowledge-Base Sources "
    "and Related Concepts are appended automatically after your answer from real "
    "data - do not write your own versions of those two sections.")


def _has_math_support(pack) -> bool:
    """Whether the evidence actually backs a derivation, not just whether
    one was asked for - a topic with real key_equations, a solver/symbolic
    result, or an equation-dense passage all count; plain prose doesn't."""
    if pack is None:
        return True  # no pack to check against; don't second-guess the request
    if pack.computed or pack.symbolic:
        return True
    if any(t.key_equations for t in pack.topics):
        return True
    return any(_EQUATION_MARKERS_RE.search(p.get("text", "")) for p in pack.passages)


def structure_directive(u: dict, pack=None) -> str:
    """Deterministic - never a new LLM call, just different text into the
    existing single professor call. A plain explain-intent question with no
    math/comparison signal keeps today's flowing prose; a derive/compare/
    math-flagged one gets asked for the structured contract instead."""
    if u.get("intent") not in ("derive", "compare") and not u.get("equations_required") \
       and not u.get("comparison_targets"):
        return ""
    if u.get("equations_required") and not _has_math_support(pack):
        return (_STRUCTURE_DIRECTIVE + "\n\nThe evidence supplied doesn't include a worked "
               "derivation for this - build the Mathematical Formulation section from what "
               "the sources actually state (definitions, named relationships), and say "
               "plainly that a full step-by-step derivation isn't supported by what was "
               "retrieved, rather than inventing one.")
    return _STRUCTURE_DIRECTIVE


def comparison_structure_hint(sides) -> str:
    """A deterministic, per-side concept progression built only from real
    curriculum key_concepts (never invented physics) - gives the professor a
    concrete skeleton grounded in actual data to compare against, rather
    than leaving "how do these two things actually line up, structurally"
    entirely to the model's own unaided judgement.
    """
    if not sides or len(sides) < 2:
        return ""
    lines = []
    for side in sides:
        topics = side.get("topics") or []
        if topics:
            concepts = concepts_for(topics, limit=6)
            body = " → ".join(concepts) if concepts else "(topic matched, but it lists no key concepts)"
        else:
            body = ("not covered by the curriculum - draw only on the retrieved "
                    "book evidence and general physics knowledge for this side")
        lines.append(f"- {side['label']}: {body}")
    return ("SUGGESTED PROGRESSION PER SIDE (from the curriculum's own concept "
           "ordering where available - expand and connect these, don't just list "
           "them):\n" + "\n".join(lines))


def knowledge_trailer(topics, book_ev, excluded) -> str:
    """Deterministic, never LLM-written, so it can never fabricate a graph
    edge or a source that wasn't really offered - extends the existing
    "never let the LLM state a number the solver computed" philosophy to
    "never let the LLM enumerate graph edges the code already knows."
    """
    lines = []
    for t in topics:
        if t.prerequisites:
            names = ", ".join(TOPICS[p].title for p in t.prerequisites if p in TOPICS)
            if names:
                lines.append(f"- **{t.title}** builds on: {names}")
        unlocks = [o.title for o in TOPICS.values() if t.id in o.prerequisites]
        if unlocks:
            lines.append(f"- **{t.title}** leads toward: {', '.join(unlocks)}")
    kept = [c for c in book_ev.get("kept", []) if c["tag"] not in excluded]
    kept_tags = sorted({c["tag"] for c in kept})
    # Which curriculum topic's vocabulary actually helped surface which book
    # tag - a real, traceable link between the curriculum graph and specific
    # retrieved evidence, not just two parallel lists shown side by side.
    by_topic: dict = {}
    for c in kept:
        for tid in c.get("linked_topics") or []:
            by_topic.setdefault(tid, []).append(c["tag"])
    links = [f"- **{TOPICS[tid].title}** surfaced: {', '.join(tags)}"
            for tid, tags in by_topic.items() if tid in TOPICS]
    if not lines and not kept_tags:
        return ""
    out = []
    if kept_tags:
        out.append(f"**Knowledge-Base Sources**: {', '.join(kept_tags)}")
    if links:
        out.append("**Curriculum → Sources**\n" + "\n".join(links))
    if lines:
        out.append("**Related Concepts**\n" + "\n".join(lines))
    return "\n\n".join(out)


def professor_engine(question, u, topics, book_ev, papers, computed, symbolic,
                     assessment, reasoning, mode, depth, mastery, client, budget,
                     sides=None, pack=None, execution=None, verification=None):
    excluded = _off_topic_tags(assessment)
    parts = ([curriculum_block(topics, sides)] if (topics or sides) else [])
    parts += [f"[{c['tag']}] ({c['source']} — {c['shelf'] if 'shelf' in c else 'physics'}"
              + (f" — side: {c['side']}" if c.get("side") else "") + f")\n{c['text'][:1000]}"
              for c in book_ev.get("kept", []) if c["tag"] not in excluded]
    parts += [f"[A{i}] ({p['published']} · {p['title']})\n{p['summary'][:800]}"
              for i, p in enumerate(papers, 1) if f"A{i}" not in excluded]
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
    if pack is not None:
        # Reasoning from the consolidated pack, not just the raw citation
        # dump above: agreements and prerequisite scaffolding were being
        # computed and then silently dropped before reaching this prompt.
        if pack.supporting_evidence:
            extra += f"\nSUPPORTING (independently agreed by multiple sources): {pack.supporting_evidence}"
        if pack.prerequisite_concepts:
            extra += f"\nPREREQUISITE CONCEPTS (may help scaffold the explanation): {', '.join(pack.prerequisite_concepts[:5])}"
    if reasoning.get("text"):
        extra += f"\n\nDERIVATION PLAN & PHYSICAL INTERPRETATION (expand, do not repeat verbatim):\n{reasoning['text']}"
    # Numerical calculation and symbolic algebra are already covered above
    # via [T1]/[X1] - only the two capabilities not otherwise surfaced
    # (matrix operations, unit conversion) get their own explicit mention.
    if execution is not None and execution.get("matrix_operations"):
        mo = execution["matrix_operations"]
        extra += (f"\n\nMATRIX OPERATION EXECUTED ({mo['kind']}): the derivation's claim "
                 f"{mo['claim']!r} was checked by direct matrix computation, which gives "
                 f"{mo['computed']}. Refer to this in words; do not restate the matrix.")
    if execution is not None and execution.get("unit_conversions"):
        conv_strs = [f"{c['from_value']} {c['from_unit']} = {c['to_value']} {c['to_unit']}"
                    for c in execution["unit_conversions"]]
        extra += f"\n\nUNIT CONVERSION EXECUTED: {'; '.join(conv_strs)}"
    if execution is not None and execution.get("differential_equation"):
        de = execution["differential_equation"]
        extra += (f"\n\nDIFFERENTIAL EQUATION SOLVED: {de['equation']} has general solution "
                 f"{de['solution']} (solved directly by sympy). Refer to this in words; do "
                 f"not restate the equation.")
    if verification is not None:
        extra += (f"\n\nVERIFICATION (deterministic checks, not your own judgement): "
                 f"status={verification.status} · passed={[p['check'] for p in verification.passed]} · "
                 f"failed={[f['check'] for f in verification.failed]}")
        if verification.corrections:
            extra += f" · corrections={verification.corrections}"
        extra += ("\nState this status plainly using one of these exact phrases where the answer "
                 "makes a mathematical claim: \"verified mathematically\" (status=verified_mathematically), "
                 "\"partially verified\" (status=partially_verified), \"not independently verified\" "
                 "(status=not_independently_verified), or \"verification failed\" "
                 "(status=failed - a check actively found the claimed result wrong, with nothing "
                 "else corroborating it). If any check failed, use the correction given - never "
                 "present the failed value as correct.")

    structure = structure_directive(u, pack)
    structure_line = f"\nSTRUCTURE: {structure}" if structure else ""
    hint = comparison_structure_hint(sides)
    hint_line = f"\n\n{hint}" if hint else ""
    requested_depth = u.get("requested_depth", "standard")
    depth_line = ""
    if requested_depth == "shallow":
        depth_line = ("\nLENGTH: The question explicitly asked to keep this brief - a short, "
                      "direct answer beats a thorough one here. Don't pad it out.")
    elif requested_depth == "deep":
        depth_line = ("\nLENGTH: The question explicitly asked to go in depth - be thorough, "
                      "don't cut the explanation short to save space.")

    return _call(client, "professor", depth, _PROF_SYS,
                 f"QUESTION: {question}\n\n"
                 f"HOW TO ANSWER: {MODE_DIRECTIVE.get(mode, MODE_DIRECTIVE['explain'])}\n"
                 f"WHO YOU ARE ANSWERING: {DEPTH_DIRECTIVE.get(depth, DEPTH_DIRECTIVE['intermediate'])}{depth_line}\n"
                 f"LEARNER HISTORY: {hist}{extra}{structure_line}{hint_line}\n\n"
                 f"SOURCES:\n{src}\n\nWrite the answer now.",
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
cover this" is doing its job.

A [C:] source is a short SUMMARY of a topic the course teaches — its intuition,
its key concepts, its key equations. It is not the full text of that topic.
Standard textbook development of material the summary names — deriving a listed
equation, working a standard example, explaining a listed concept in depth — is
the expected elaboration of that source, NOT an unsupported claim. Teaching
requires saying far more than the summary line contains. Only call something
unsupported when it belongs to no listed topic at all AND is presented as if it
came from one.

Do not expect a citation on every sentence. Prose that develops an already-cited
point needs no tag of its own; demanding one would make ordinary exposition look
dishonest.

Reserve "fail" for claims presented as sourced that are not, for contradictions
of a supplied source, or for restating a computed number instead of referring to
it. If the only issue is that the answer says more than the summaries do, that
is "pass"."""


def validation(question, prose, book_ev, papers, computed, topics, depth, client, budget,
              sides=None):
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
    # The curriculum has to be in here. C: tags are offered as citable above,
    # but the validator used to receive only books and papers — so it looked
    # for [C:quantum-statistical-mechanics], failed to find it, and reported
    # correctly-cited curriculum material as unsupported. Any answer built on
    # the curriculum was guaranteed to fail a check it could never pass.
    src = "\n\n".join(([curriculum_block(topics, sides)] if (topics or sides) else []) +
                      [f"[{c['tag']}] {c['text'][:600]}" for c in book_ev.get("kept", [])] +
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

def offline_synthesis(question, u, topics, sides, book_ev, computed,
                      insufficient=False, degraded=False, pack=None) -> str:
    """Zero-LLM-cost answer, built from the EvidencePack where one is given.

    Normal case: the matched topic(s)' already-vetted intuition/key_concepts/
    key_equations verbatim from library.py, plus retrieved book chunks with
    their real [S#] tags - real, already-trusted content (the same text the
    Topics tab shows) assembled without generating new prose.

    degraded=True: nothing cleared the CURRICULUM floor and book evidence is
    only "weak" - too thin to confidently narrate. Shows the raw retrieved
    material only, with no framing that implies it was vetted as an answer.
    This is the "related material" case, distinct from a real offline answer.

    When a pack is supplied, its prerequisite_concepts/conflicting_evidence/
    knowledge_gaps are appended - real, already-computed structure that a
    deterministic response can show even with no LLM available, rather than
    silently dropping it the way the raw parameters alone would.
    """
    parts = []
    if insufficient:
        parts.append("No curriculum topic or library passage cleared the relevance "
                     "threshold for this question, so no answer is offered - "
                     "guessing from surface keyword overlap is exactly the failure "
                     "mode this is meant to avoid.")
    elif degraded:
        parts.append("The reasoning service is unavailable, and nothing retrieved is "
                     "strong enough to confidently turn into an explanation - showing "
                     "the genuinely relevant material found, without synthesis.")

    def _degraded_block(label, kept):
        lines = [f"## {label}" if label else "## This question"]
        side_kept = [c for c in kept if c.get("side") == label] if label else kept
        if not side_kept:
            lines.append("_Nothing retrieved cleared even a weak relevance bar for this part._")
        for c in side_kept:
            lines.append(f"[{c['tag']}] From *{c['source']}*:\n\n{c['text'][:600]}")
        return "\n\n".join(lines)

    def _side_block(label, side_topics, kept):
        lines = [f"## {label}" if label else "## This question"]
        side_kept = [c for c in kept if c.get("side") == label] if label else kept
        if side_topics:
            for t in side_topics:
                lines.append(f"**{t.title}** ({t.level})\n\n{t.intuition}\n\n"
                            f"Key concepts: {', '.join(t.key_concepts)}\n\n"
                            f"Key equations: {' ; '.join(t.key_equations)}")
        elif side_kept:
            # The 41-topic curriculum is a teaching/navigation layer, not the
            # boundary of what this app knows - real book evidence answers a
            # question the curriculum happens not to name, so say so plainly
            # rather than opening with a line that reads like a refusal right
            # above the evidence that contradicts it.
            concept_label = label or question
            lines.append(f"**{concept_label}** isn't one of the named curriculum topics, but "
                        f"the library has relevant material on it:")
        else:
            lines.append("_No curriculum topic covers this, and no library passage cleared "
                         "the relevance floor either._")
        for c in side_kept:
            lines.append(f"[{c['tag']}] From *{c['source']}*:\n\n{c['text'][:600]}")
        if side_topics and not side_kept and not insufficient:
            lines.append("_No library passage cleared the relevance floor for this "
                         "part either - the curriculum summary above is all there is._")
        return "\n\n".join(lines)

    kept = book_ev.get("kept", [])
    block_fn = _degraded_block if degraded else _side_block
    if sides:
        for side in sides:
            parts.append(block_fn(side["label"], kept) if degraded
                        else block_fn(side["label"], side["topics"], kept))
    elif not insufficient:
        parts.append(block_fn(None, kept) if degraded else block_fn(None, topics, kept))

    if computed and computed.get("ran"):
        parts.append(f"**Computed value** — formula {computed['result'].get('formula','')}, "
                     f"result: { {k: v for k, v in computed['result'].items() if k != 'formula'} }")

    if pack is not None and not insufficient:
        if pack.conflicting_evidence:
            parts.append("**Conflicting evidence**\n" +
                        "\n".join(f"- {c}" for c in pack.conflicting_evidence))
        if pack.knowledge_gaps:
            parts.append("**What the evidence doesn't cover**\n" +
                        "\n".join(f"- {g}" for g in pack.knowledge_gaps))
        if pack.prerequisite_concepts:
            parts.append(f"**Related topics**: {', '.join(pack.prerequisite_concepts[:5])}")

    return "\n\n".join(parts)


def evidence_quality(topics, book_ev, sides) -> str:
    """"usable" | "weak" | "none" - distinguishes a confident OFFLINE answer
    (curriculum coverage, or book evidence strong enough to narrate) from a
    DEGRADED one (something was retrieved, but nowhere near strong enough
    to present as an explanation - only as related material) from having
    nothing at all. For a comparison, the best side sets the overall level:
    a per-side breakdown that's weak on one side and solid on the other still
    gets a confident answer for the solid side, exactly like offline_synthesis
    already renders each side independently.
    """
    if sides:
        best = 0
        for s in sides:
            if s["covered_by_curriculum"] or s["evidence_strength"] == "usable":
                best = max(best, 2)
            elif s["evidence_strength"] == "weak":
                best = max(best, 1)
        return "usable" if best == 2 else "weak" if best == 1 else "none"
    if topics or book_ev.get("evidence_strength") == "usable":
        return "usable"
    if book_ev.get("evidence_strength") == "weak":
        return "weak"
    return "none"


def quality_gate(question, u, assessment, topics, sides, book_ev, symbolic, prose, checks) -> dict:
    """Ten deterministic, regex/field-based checks - no LLM call, no
    auto-retry. Flags problems for the trace panel; never regenerates the
    answer (that would be exactly the "spend more calls to compensate for
    poor retrieval" the whole redesign is meant to avoid).
    """
    prose_l = (prose or "").lower()
    gate: dict = {}

    def check(name, ok, note=""):
        gate[name] = {"pass": bool(ok), "note": note}

    check("answered_actual_question",
         bool(prose) and len(prose.strip()) > 20
         and not prose_l.strip().startswith(("sorry", "i don't know", "i cannot", "i can't")))

    word_count = len(prose.split()) if prose else 0
    requested_depth = u.get("requested_depth", "standard")
    if requested_depth == "shallow":
        check("satisfied_requested_depth", word_count <= 180,
             "" if word_count <= 180 else "a brief answer was requested but the reply runs long")
    elif requested_depth == "deep":
        check("satisfied_requested_depth", word_count >= 120,
             "" if word_count >= 120 else "an in-depth answer was requested but the reply is short")
    else:
        check("satisfied_requested_depth", True, "no specific depth requested")

    wanted = list(u.get("explicit_constraints") or []) + list(u.get("comparison_targets") or [])
    missing = [w for w in wanted if w.lower() not in prose_l]
    check("addressed_explicit_components", not missing,
         f"not echoed in the answer: {missing}" if missing else "")

    check("retrieved_correct_domain", domain_for(topics) == u.get("domain", domain_for(topics)))

    off_topic_all_kept = bool(book_ev.get("kept")) and _off_topic_tags(assessment) >= {
        c["tag"] for c in book_ev.get("kept", [])}
    check("sources_relevant",
         book_ev.get("evidence_strength", "none") != "none" or bool(topics),
         "every retrieved source was judged off-topic" if off_topic_all_kept else "")

    if sides:
        both_mentioned = all(s["label"].lower() in prose_l for s in sides)
        check("comparison_both_sides_covered", both_mentioned,
             "" if both_mentioned else "a comparison side is never named in the answer")
        ids_by_side = [{t.id for t in s["topics"]} for s in sides if s["topics"]]
        confused = len(ids_by_side) >= 2 and bool(set.intersection(*ids_by_side))
        check("no_adjacent_discipline_confusion", not confused,
             "the same curriculum topic matched two different comparison sides" if confused else "")
    else:
        check("comparison_both_sides_covered", True, "not a comparison question")
        check("no_adjacent_discipline_confusion", True)

    if u.get("equations_required"):
        has_math = bool(re.search(r"\$|\\\(|\\\[|\[T1\]|\[X1\]", prose))
        check("math_provided_if_requested", has_math,
             "" if has_math else "equations were requested but none appear in the answer")
    else:
        check("math_provided_if_requested", True, "not requested")

    if symbolic and symbolic.get("ok"):
        check("equations_consistent", symbolic.get("verdict") != "does not hold")
    else:
        check("equations_consistent", True, "no symbolic check ran")

    check("grounded", (checks or {}).get("verdict", "pass") != "fail")

    uncovered = ((sides and any(not s["covered_by_curriculum"]
                                and s["evidence_strength"] == "none" for s in sides))
                or (not sides and not topics
                    and book_ev.get("evidence_strength", "none") == "none"))
    admits_it = any(p in prose_l for p in ("no curriculum topic", "insufficient",
                                           "does not cover", "not covered"))
    check("refused_to_guess_when_insufficient", (not uncovered) or admits_it,
         "" if (not uncovered) or admits_it
         else "evidence was missing for part of the question but the answer doesn't say so")

    return gate


def run(question: str, mode: str = "explain",
        depth: str = "intermediate", prior: dict | None = None) -> Iterator[tuple[str, dict]]:
    question = (question or "").strip()
    if not question:
        yield "error", {"message": "empty question"}
        return

    budget = Budget()
    t_start = time.monotonic()

    # ── deterministic baseline: understanding + retrieval, zero LLM cost ───
    yield "understand", {"msg": "Reading the question…"}
    u = deterministic_understand(question, prior)

    sides = None
    if u["comparison_targets"]:
        book_ev, topics, sides = gather_comparison_sides(u["comparison_targets"],
                                                          equations_required=u["equations_required"])
        u["domain"] = domain_for(topics)
        u["subdomain"] = subdomain_for(topics)
        u["concepts"] = concepts_for(topics)
        yield "understand", {"msg": f"{u['intent']} · {u['domain']} · {u['difficulty']} · "
                                    f"comparing {' vs '.join(u['comparison_targets'])}",
                             "understanding": u}
        probe = compute_for(question, topics[0].id if topics else None)
        computed = probe if (probe and probe.get("ran")) else None
        r = route(u, topics, probe, depth)
        yield "route", {"msg": " + ".join(k for k in
                                          ("curriculum", "books", "solver", "arxiv", "sympy")
                                          if r[k]),
                        "routing": r,
                        "topics": [{"id": t.id, "title": t.title, "level": t.level}
                                   for t in topics]}
        t0 = time.monotonic()
        query = question + " " + " ".join(u["topics"][:4])
        papers, symbolic = [], None
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_arx = pool.submit(R.search_arxiv, query, 5) if r["arxiv"] else None
            if r["sympy"] and "=" in u["identity"]:
                lhs, _, rhs = u["identity"].partition("=")
                symbolic = R.check_identity(lhs.strip(), rhs.strip())
            if f_arx:
                papers = (f_arx.result() or {}).get("papers", [])
        yield "gather", {"msg": (f"{len(topics)} topic(s), {len(book_ev.get('kept', []))} passage(s)"
                                 + (f", {len(papers)} paper(s)" if papers else "")
                                 + (", 1 computed" if computed else "")
                                 + (", 1 symbolic check" if symbolic and symbolic.get("ok") else "")
                                 + f" · {int((time.monotonic()-t0)*1000)}ms"),
                         "evidence": book_ev, "papers": papers,
                         "computed": computed, "symbolic": symbolic, "sides": sides}
    else:
        topics = match_topics(question, k=4)
        u["domain"] = domain_for(topics)
        u["subdomain"] = subdomain_for(topics)
        u["concepts"] = concepts_for(topics)
        yield "understand", {"msg": f"{u['intent']} · {u['domain']} · {u['difficulty']} · "
                                    f"{', '.join(u['topics'][:4])}",
                             "understanding": u}
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
            f_books = pool.submit(_retrieve_with_topic_context, query, topics,
                                  u["equations_required"])
            f_arx = pool.submit(R.search_arxiv, query, 5) if r["arxiv"] else None
            if r["sympy"] and "=" in u["identity"]:
                lhs, _, rhs = u["identity"].partition("=")
                symbolic = R.check_identity(lhs.strip(), rhs.strip())
            book_ev, linked_topics = f_books.result()
            for item in book_ev.get("kept", []):
                item["linked_topics"] = linked_topics
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

    mastery = record_visit(topics[0].id if topics else None, mode, depth)

    # ── relevance gate: is there anything here worth reasoning about? ──────
    # A pure relevance decision, independent of whether the LLM is even
    # reachable - a healthy DeepSeek account gets this too, which is both
    # the correct behaviour for an out-of-curriculum question and a direct
    # cost saving (nothing downstream gets called for nothing).
    if sides is not None:
        nothing_found = (not computed and
                         all(not s["covered_by_curriculum"] and s["evidence_strength"] == "none"
                             for s in sides))
    else:
        nothing_found = (not topics and not computed and not papers
                         and book_ev.get("evidence_strength", "none") == "none")

    empty_assessment = {"skipped": True, "usable": [], "off_topic": [], "agreements": [],
                        "conflicts": [], "gaps": [], "covered_by_curriculum": bool(topics),
                        "confidence": "unassessed", "excluded": []}
    empty_reasoning = {"skipped": True, "text": ""}
    empty_validation = {"verdict": "pass", "cited": [], "offered": [], "fabricated_tags": [],
                        "sentences": 0, "uncited_sentences": 0}

    def _done(prose, answer_mode, assessment, reasoning, checks, extra_honesty_note,
             provider_error=None, pack=None, execution=None, verification=None):
        payload = {
            "question": question, "mode": mode, "depth": depth, "prose": prose,
            "understanding": u, "routing": r,
            "topics": [{"id": t.id, "title": t.title, "level": t.level} for t in topics],
            "evidence": book_ev, "papers": papers, "computed": computed,
            "symbolic": symbolic, "sides": sides, "assessment": assessment,
            "reasoning": reasoning, "validation": checks, "mastery": mastery,
            "familiarity": _familiarity(mastery), "budget": budget.summary(),
            "elapsed_s": int(time.monotonic() - t_start), "answer_mode": answer_mode,
            "quality_gate": quality_gate(question, u, assessment, topics, sides, book_ev,
                                        symbolic, prose, checks),
            "honesty": {
                "numbers_are_computed_not_generated": bool(computed),
                "identity_checked_symbolically": bool(symbolic and symbolic.get("ok")),
                "covered_by_curriculum": assessment.get("covered_by_curriculum", bool(topics)),
                "note": extra_honesty_note,
            },
        }
        if answer_mode != "online":
            payload["suggestions"] = [t.id for t in suggest_related(question)]
        if provider_error is not None:
            payload["provider_error"] = {"kind": provider_error.kind,
                                         "http_status": provider_error.http_status,
                                         "message": provider_error.user_message}
        if pack is not None:
            payload["evidence_pack"] = {
                "concepts": pack.concepts,
                "sources": pack.sources,
                "prerequisite_concepts": pack.prerequisite_concepts,
                "supporting_evidence": pack.supporting_evidence,
                "conflicting_evidence": pack.conflicting_evidence,
                "knowledge_gaps": pack.knowledge_gaps,
                "curriculum_backed": pack.curriculum_backed,
                "passage_count": len(pack.passages),
                "mathematical_objects": pack.mathematical_objects,
                "givens": pack.givens,
                "unknowns": pack.unknowns,
                "assumptions": pack.assumptions,
                "strategy": pack.strategy,
            }
        if execution is not None:
            payload["execution"] = execution
        if verification is not None:
            payload["verification"] = verification.to_dict()
        return payload

    if nothing_found:
        prose = offline_synthesis(question, u, topics, sides, book_ev, computed,
                                  insufficient=True)
        yield "done", _done(prose, "insufficient_evidence", empty_assessment, empty_reasoning,
                            empty_validation,
                            "No curriculum topic or library passage cleared the relevance "
                            "threshold for this question - nothing below is a guess.")
        return

    # ── attempt the LLM path; any classified provider failure falls back to
    # offline synthesis, reusing the retrieval work already done above ─────
    gateway = get_gateway()
    client = None
    provider_error = gateway.blocked()
    if provider_error is None:
        key = _api_key()
        if key:
            client = gateway.client()
        else:
            provider_error = ProviderError("missing_key", None, "No DEEPSEEK_API_KEY found",
                                           RuntimeError("missing DEEPSEEK_API_KEY"))

    assessment, reasoning, checks, prose = empty_assessment, empty_reasoning, empty_validation, None

    if provider_error is None:
        try:
            llm_u = understand(question, depth, client, budget, prior)
            u.update(llm_u)
            u["domain"] = domain_for(topics)
            u["subdomain"] = subdomain_for(topics)
            u["concepts"] = concepts_for(topics)
        except ProviderError as pe:
            provider_error = pe
            gateway.record_failure(pe)

    pack = None
    if provider_error is None:
        yield "evidence", {"msg": "Verifying and comparing…"}
        try:
            assessment = evidence_engine(question, topics, book_ev, papers, computed,
                                         symbolic, depth, client, budget, sides=sides)
        except ProviderError as pe:
            provider_error = pe
            gateway.record_failure(pe)
        else:
            assessment["excluded"] = sorted(_off_topic_tags(assessment))
            pack = build_evidence_pack(question, u, topics, sides, book_ev, papers,
                                       assessment, computed, symbolic)
            yield "evidence", {"msg": ("skipped (intro)" if assessment.get("skipped") else
                                       f"{len(assessment['usable'])} usable · "
                                       f"curriculum covers it: {assessment.get('covered_by_curriculum')} · "
                                       f"confidence {assessment.get('confidence')}"),
                               "assessment": assessment}

    if provider_error is None:
        yield "reasoning", {"msg": "Building the argument…"}
        try:
            reasoning = reasoning_engine(question, u, topics, book_ev, papers, computed,
                                         symbolic, assessment, depth, client, budget, sides=sides,
                                         pack=pack)
        except ProviderError as pe:
            provider_error = pe
            gateway.record_failure(pe)
        else:
            yield "reasoning", {"msg": ("folded into teaching (intro)" if reasoning.get("skipped")
                                        else f"derivation plan + physical interpretation, "
                                            f"{len(reasoning['text'].split())} words"),
                                "reasoning": reasoning}

    # ── DETERMINISTIC EXECUTION + VERIFICATION: both zero LLM cost, both run
    # on whatever the Derivation stage produced (even at intro depth, where
    # reasoning is skipped but computed/symbolic results still exist).
    # Execution makes the real SymPy/Python computation (numerical, symbolic,
    # matrix, unit conversion) an explicit record BEFORE verification judges
    # the derivation against it - the same distinction as "here is what was
    # actually computed" vs. "here is whether the claim matches it". Reported
    # as a second "reasoning" event rather than a new SSE stage name - the
    # frontend already re-renders a stage's row each time it fires (exactly
    # how every other multi-yield stage in this function already works), and
    # inventing a new stage name here would silently drift out of sync with
    # the frontend's listener list the same way it once already did.
    execution = None
    verification = None
    if provider_error is None:
        execution = execute_deterministically(question, u, pack, reasoning, computed, symbolic)
        verification = verify_derivation(question, u, pack, reasoning, computed, symbolic)
        yield "reasoning", {"msg": f"verification: {verification.status} "
                                   f"({len(verification.passed)} passed, {len(verification.failed)} failed)",
                            "reasoning": reasoning, "execution": execution,
                            "verification": verification.to_dict()}

    if provider_error is None:
        box: dict = {}

        def _teach():
            try:
                box["prose"] = professor_engine(question, u, topics, book_ev, papers,
                                                computed, symbolic, assessment, reasoning,
                                                mode, depth, mastery, client, budget, sides=sides,
                                                pack=pack, execution=execution,
                                                verification=verification)
            except Exception as exc:
                box["exc"] = exc

        worker = threading.Thread(target=_teach, daemon=True)
        t_p = time.monotonic()
        yield "professor", {"msg": "Teaching…", "elapsed_s": 0}
        worker.start()
        while worker.is_alive():
            worker.join(timeout=5.0)
            if worker.is_alive():
                yield "professor", {"msg": f"Teaching… {int(time.monotonic()-t_p)}s",
                                    "elapsed_s": int(time.monotonic() - t_p)}
        exc = box.get("exc")
        if exc is not None:
            if is_llm_sdk_error(exc):
                kind, status, msg = classify_llm_error(exc)
                provider_error = ProviderError(kind, status, msg, exc)
                gateway.record_failure(provider_error)
            else:
                yield "error", {"message": f"{type(exc).__name__}: {exc}"}
                return
        else:
            prose = (box.get("prose") or "").strip()
            if not prose:
                yield "error", {"message": "the teaching stage returned nothing"}
                return
            yield "professor", {"msg": f"Taught in {int(time.monotonic()-t_p)}s",
                                "elapsed_s": int(time.monotonic() - t_p)}

    if provider_error is None:
        yield "validation", {"msg": "Checking the answer against its sources…"}
        try:
            checks = validation(question, prose, book_ev, papers, computed, topics,
                                depth, client, budget, sides=sides)
        except ProviderError as pe:
            provider_error = pe
            gateway.record_failure(pe)
        else:
            # Show what was cited, not how many sentences went untagged. "43/55
            # uncited" sat next to the verdict and read like an accusation, when a
            # 55-sentence explanation carrying four sources is exactly what good
            # teaching looks like. The count stays in the payload for the evidence
            # panel; it just stops being the headline.
            yield "validation", {"msg": (f"{checks['verdict']} · {len(checks['cited'])} "
                                         f"source{'' if len(checks['cited']) == 1 else 's'} cited"
                                         + (f" · {len(checks['fabricated_tags'])} fabricated"
                                            if checks["fabricated_tags"] else "")),
                                 "validation": checks}

    if provider_error is not None:
        if pack is None:
            pack = build_evidence_pack(question, u, topics, sides, book_ev, papers,
                                       assessment, computed, symbolic)
        # Execution and verification are both deterministic, so they run
        # offline too - neither ever needed the LLM that just failed.
        # Whatever the Derivation stage managed to produce before the
        # failure (or nothing, if it failed on the very first call) is what
        # gets executed and checked.
        execution = execute_deterministically(question, u, pack, reasoning, computed, symbolic)
        verification = verify_derivation(question, u, pack, reasoning, computed, symbolic)
        quality = evidence_quality(topics, book_ev, sides)
        if quality == "usable":
            prose = offline_synthesis(question, u, topics, sides, book_ev, computed, pack=pack)
            yield "done", _done(prose, "offline", assessment, reasoning, checks,
                                "DeepSeek is unavailable - this is assembled directly from "
                                "the curriculum and your library, with no AI reasoning applied.",
                                provider_error=provider_error, pack=pack, execution=execution,
                                verification=verification)
        else:
            prose = offline_synthesis(question, u, topics, sides, book_ev, computed,
                                      degraded=(quality == "weak"),
                                      insufficient=(quality == "none"), pack=pack)
            yield "done", _done(prose, "degraded" if quality == "weak" else "insufficient_evidence",
                                assessment, reasoning, checks,
                                "DeepSeek is unavailable and nothing retrieved is strong "
                                "enough to confidently explain - showing related material only.",
                                provider_error=provider_error, pack=pack, execution=execution,
                                verification=verification)
        return

    trailer = knowledge_trailer(topics, book_ev, _off_topic_tags(assessment))
    if trailer:
        prose = f"{prose}\n\n---\n\n{trailer}"

    gateway.record_success()
    yield "done", _done(prose, "online", assessment, reasoning, checks,
                        "[C:] curriculum · [S#] your books · [A#] arXiv · [T1] solver · "
                        "[X1] symbolic check. Untagged sentences are the model's synthesis.",
                        pack=pack, execution=execution, verification=verification)


def answer(question: str, mode: str = "explain", depth: str = "intermediate") -> dict:
    last: dict = {}
    for stage, payload in run(question, mode, depth):
        if stage in ("done", "error"):
            last = {"stage": stage, **payload}
    return last
