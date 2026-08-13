"""Grounded tutor: retrieval + LLM + deterministic solver, with a visible trace.

The point of this module is not "ask an LLM about physics" — it is to make the
provenance of every sentence and every digit inspectable. Three sources, kept
strictly apart so the UI can label them:

  computed  — physics.py solvers, CODATA constants. The model is never asked
              for a number and any figure it invents is not displayed as one.
  retrieved — chunks from the user's own library via the second_brain gateway.
              Prose must cite these by tag; uncited sentences are reported.
  curated   — the 32-topic curriculum in library.py (intuition, equations).

Each stage yields an event so the caller can stream the pipeline to the screen;
the trace is the product, not debug output.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator

from .library import TOPICS
from .physics import M_E, SOLVERS, solve

# ── configuration ─────────────────────────────────────────────────────────
# The gateway lives in the sibling Agentic-AI workspace. Kept as a path bridge
# rather than a copy so there is exactly one index and one retrieval policy.
BRAIN_ROOT = Path(
    os.getenv("SECOND_BRAIN_ROOT",
              "/Users/vaibhavgupta/Desktop/Project Agentic AI/Agentic-AI")
).expanduser()
PHYSICS_CORPORA = ["desk-physics", "desk-quantum-computing"]

# Filter on RAW score, never on the gateway's "confidence".
#
# The gateway min-max normalises within each result set, so the worst hit is
# always 0.000 and the best always 1.000 no matter how close they really are.
# Observed on a live query: raw 0.6274 / 0.5963 / 0.5898 — a 0.04 spread —
# came back as confidence 1.000 / 0.173 / 0.000, with the *most* relevant chunk
# scoring 0.000. Thresholding on that number would discard the best evidence.
MIN_RAW_SCORE = 0.30
MIN_CHUNK_CHARS = 200      # below this it's a heading fragment, not evidence
WEAK_EVIDENCE_RAW = 0.45   # top hit under this ⇒ tell the user grounding is thin
MODEL = "deepseek-v4-pro"


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key:
        return key
    for env in (Path(__file__).parent / ".env",
                BRAIN_ROOT / "learn_agent" / ".env",
                BRAIN_ROOT.parent / "stock_agent" / ".env"):
        try:
            for line in env.read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip()
        except OSError:
            continue
    return ""


# ── retrieval ─────────────────────────────────────────────────────────────

def retrieve_evidence(question: str, top_k: int = 6) -> dict:
    """Pull grounding chunks from the user's physics library.

    Returns kept/rejected separately: showing what was searched and thrown
    away is a large part of what makes the provenance claim credible.
    """
    if str(BRAIN_ROOT) not in sys.path:
        sys.path.insert(0, str(BRAIN_ROOT))
    try:
        from second_brain.gateway import retrieve
    except Exception as exc:                     # gateway absent → degrade, don't crash
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                "kept": [], "rejected": [], "scope": PHYSICS_CORPORA}

    try:
        res = retrieve(question, corpora=PHYSICS_CORPORA, top_k=top_k)
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                "kept": [], "rejected": [], "scope": PHYSICS_CORPORA}

    scored = []
    for h in res.get("hits", []):
        scored.append({
            "text": (h.get("text") or "").strip(),
            "source": Path(str(h.get("source", "?"))).name,
            "corpus": h.get("corpus"),
            "raw_score": round(float(h.get("raw_score", 0.0)), 4),
            "normalised": round(float(h.get("confidence", 0.0)), 3),
        })
    # Rank by raw score — the gateway's ordering already does this, but making
    # it explicit means the tags we hand the model match what we display.
    scored.sort(key=lambda c: -c["raw_score"])

    kept, rejected = [], []
    for item in scored:
        if item["raw_score"] < MIN_RAW_SCORE:
            item["why"] = f"raw score {item['raw_score']} below floor {MIN_RAW_SCORE}"
            rejected.append(item)
        elif len(item["text"]) < MIN_CHUNK_CHARS:
            item["why"] = "too short to ground on (heading fragment)"
            rejected.append(item)
        else:
            item["tag"] = f"S{len(kept) + 1}"
            kept.append(item)

    top_raw = kept[0]["raw_score"] if kept else 0.0
    return {"available": True, "kept": kept, "rejected": rejected,
            "scope": PHYSICS_CORPORA, "n_returned": res.get("n", 0),
            "top_raw_score": top_raw,
            "evidence_strength": ("none" if not kept
                                  else "weak" if top_raw < WEAK_EVIDENCE_RAW
                                  else "usable"),
            "scoring_note": ("Ranked and filtered on raw_score. The gateway's "
                             "'normalised' figure is min-max scaled within this "
                             "result set, so its 0.000 and 1.000 are relative, "
                             "not absolute.")}


# ── curriculum match ──────────────────────────────────────────────────────

_STOP = set("the a an of for on in is are do does can could would should will "
            "me my you your i we us what how why with about and or but so to "
            "that this it its at be by as from into than then there".split())


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z-]{2,}", text.lower())
            if w not in _STOP}


def match_topic(question: str):
    """Best curriculum topic for a question, or None. Mirrors the client-side
    matcher so the ask panel and this endpoint agree on the subject."""
    q = _tokens(question)
    if not q:
        return None, []
    scored = []
    for t in TOPICS.values():
        hay = " ".join([t.title, t.id.replace("-", " "), t.intuition,
                        " ".join(t.key_concepts)])
        overlap = len(q & _tokens(hay))
        if re.search(re.escape(t.title.lower()), question.lower()):
            overlap += 5
        if overlap:
            scored.append((overlap, t))
    scored.sort(key=lambda p: (-p[0], p[1].title))
    if not scored:
        return None, []
    return scored[0][1], [t for _s, t in scored[1:4]]


# ── parameter extraction (ported from the client so both agree) ───────────

_ORD = {"first": 2, "second": 3, "third": 4, "fourth": 5, "fifth": 6}


def _level_n(t: str):
    t = t.lower()
    m = re.search(r"\bn\s*=\s*(\d+)\b", t) or re.search(r"\blevel\s+(\d+)\b", t)
    if m:
        return int(m.group(1))
    if "ground state" in t:
        return 1
    m = re.search(r"\b(first|second|third|fourth|fifth)\s+excited state\b", t)
    return _ORD[m.group(1)] if m else None


def _length_nm(t: str):
    for pat, mul in ((r"(\d+(?:\.\d+)?)\s*nm\b", 1.0),
                     (r"(\d+(?:\.\d+)?)\s*pm\b", 1e-3),
                     (r"(\d+(?:\.\d+)?)\s*(?:å|angstroms?)\b", 0.1)):
        m = re.search(pat, t, re.I)
        if m:
            return float(m.group(1)) * mul
    return None


def _delta_x_m(t: str):
    for pat, mul in ((r"(\d+(?:\.\d+)?)\s*nm\b", 1e-9),
                     (r"(\d+(?:\.\d+)?)\s*pm\b", 1e-12),
                     (r"(\d+(?:\.\d+)?)\s*(?:å|angstroms?)\b", 1e-10),
                     (r"(\d+(?:\.\d+)?)\s*m\b(?!\w)", 1.0)):
        m = re.search(pat, t, re.I)
        if m:
            return float(m.group(1)) * mul
    return None


def _transition(t: str):
    t = t.lower()
    m = (re.search(r"from\s+n?\s*=?\s*(\d+)\s+to\s+n?\s*=?\s*(\d+)", t)
         or re.search(r"n_?i\s*=\s*(\d+).{0,20}?n_?f\s*=\s*(\d+)", t))
    return {"n_i": int(m.group(1)), "n_f": int(m.group(2))} if m else None


def _de_broglie(t: str):
    t = t.lower()
    if "electron" not in t:
        return None
    m = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*m/s", t, re.I)
    return {"mass_kg": M_E, "speed_m_s": float(m.group(1))} if m else None


def _harmonic(t: str):
    n = _level_n(t)
    m = re.search(r"(?:omega|ω)\s*=\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", t, re.I)
    return {"n": n, "omega": float(m.group(1))} if (n is not None and m) else None


TOPIC_TO_SOLVER = {
    "particle-in-a-box": "particle-in-a-box", "harmonic-oscillator": "harmonic-oscillator",
    "hydrogen-atom": "hydrogen-atom", "bohr-model": "hydrogen-transition",
    "de-broglie": "de-broglie", "photoelectric-effect": "photon-energy",
    "blackbody-radiation": "photon-energy",
    "uncertainty-principle": "uncertainty-principle", "quantum-information": "qubit",
}

_EXTRACT = {
    "particle-in-a-box": lambda t: ({"n": _level_n(t), "L": _length_nm(t) * 1e-9}
                                    if _level_n(t) is not None and _length_nm(t) is not None else None),
    "hydrogen-atom":     lambda t: ({"n": _level_n(t)} if _level_n(t) is not None else None),
    "hydrogen-transition": _transition,
    "photon-energy":     lambda t: ({"wavelength_nm": _length_nm(t)} if _length_nm(t) is not None else None),
    "uncertainty-principle": lambda t: ({"delta_x_m": _delta_x_m(t)} if _delta_x_m(t) is not None else None),
    "de-broglie":        _de_broglie,
    "harmonic-oscillator": _harmonic,
}


def compute_for(question: str, topic_id: str | None) -> dict | None:
    """Run the deterministic solver when the question carries real parameters.

    Returns None rather than guessing: a made-up default would be a number the
    UI would then present as 'computed', which is exactly the lie to avoid.
    """
    solver_id = TOPIC_TO_SOLVER.get(topic_id or "")
    if not solver_id or solver_id not in SOLVERS:
        return None
    extractor = _EXTRACT.get(solver_id)
    params = extractor(question) if extractor else None
    if not params:
        return {"solver": solver_id, "ran": False,
                "reason": "question gives no numeric parameters to compute with"}
    try:
        out = solve(solver_id, **params)
    except Exception as exc:
        return {"solver": solver_id, "ran": False, "inputs": params,
                "reason": f"{type(exc).__name__}: {exc}"}
    return {"solver": solver_id, "ran": True, "inputs": params, "result": out}


# ── composition ───────────────────────────────────────────────────────────

_SYSTEM = """You are a physics tutor in the style of Feynman: build intuition \
first, then formalise. You are one stage in a pipeline that shows the user \
exactly where each part of the answer came from, so you must follow two rules \
absolutely.

1. NEVER state a numeric result of your own. Any arithmetic has already been \
done by a deterministic solver and is shown to the user separately. You may \
refer to a computed value in words ("the ground-state energy shown below"), \
but never write your own figure for it. Quoting a constant or an exponent that \
appears in a formula is fine.

2. Ground your explanation in the SOURCES provided and cite them inline as \
[S1], [S2]. If the sources do not cover part of what you say, say so plainly \
in that sentence rather than implying the books support it. If the sources are \
irrelevant, ignore them and answer from the curriculum material, and open with \
one short sentence saying the library had nothing directly on point.

Write for the requested depth. Be concrete. Use LaTeX for mathematics."""


def _compose(question, topic, evidence, computed, mode, depth, client):
    src_block = "\n\n".join(
        f"[{c['tag']}] (from {c['source']}, raw relevance {c['raw_score']})\n{c['text'][:1100]}"
        for c in evidence["kept"]
    ) or "(no usable sources retrieved)"
    if evidence.get("evidence_strength") == "weak":
        src_block += ("\n\nNOTE: the best of these scored only "
                      f"{evidence.get('top_raw_score')} — treat them as possibly "
                      "off-topic and say so if they are.")

    cur = ""
    if topic is not None:
        cur = (f"Curriculum topic: {topic.title} ({topic.level})\n"
               f"Intuition: {topic.intuition}\n"
               f"Key concepts: {', '.join(topic.key_concepts)}\n"
               f"Key equations (LaTeX): {' ; '.join(topic.key_equations)}")

    comp = "(no computation — the question carries no numeric parameters)"
    if computed and computed.get("ran"):
        comp = (f"A solver already computed this. Formula {computed['result'].get('formula','')}, "
                f"inputs {computed['inputs']}, outputs "
                f"{ {k: v for k, v in computed['result'].items() if k != 'formula'} }. "
                f"Refer to it in words; do NOT restate the digits.")

    user = (f"QUESTION: {question}\n\nMODE: {mode}    DEPTH: {depth}\n\n"
            f"CURRICULUM MATERIAL:\n{cur}\n\nSOURCES FROM THE USER'S LIBRARY:\n{src_block}\n\n"
            f"COMPUTED:\n{comp}\n\nWrite the explanation now.")

    resp = client.chat.completions.create(
        model=MODEL, max_tokens=8000,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


def _audit(prose: str, evidence: dict) -> dict:
    """Report how well the prose actually honoured the grounding contract."""
    cited = set(re.findall(r"\[(S\d+)\]", prose))
    offered = {c["tag"] for c in evidence["kept"]}
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", prose) if len(s.strip()) > 40]
    uncited = [s for s in sentences if not re.search(r"\[S\d+\]", s)]
    return {
        "cited": sorted(cited),
        "offered": sorted(offered),
        "hallucinated_tags": sorted(cited - offered),   # cited a source that wasn't given
        "unused_sources": sorted(offered - cited),
        "sentences": len(sentences),
        "uncited_sentences": len(uncited),
    }


# ── the pipeline, as a stream of stages ───────────────────────────────────

def answer_stream(question: str, mode: str = "explain",
                  depth: str = "intermediate") -> Iterator[tuple[str, dict]]:
    """Yield (stage, payload) so the caller can render the pipeline live."""
    question = (question or "").strip()
    if not question:
        yield "error", {"message": "empty question"}
        return

    yield "match", {"msg": "Matching against the 32-topic curriculum…"}
    topic, related = match_topic(question)
    yield "match", {"msg": (f"Topic: {topic.title}" if topic else "No curriculum topic matched"),
                    "topic": topic.id if topic else None,
                    "topic_title": topic.title if topic else None,
                    "related": [t.id for t in related]}

    yield "retrieve", {"msg": f"Searching {', '.join(PHYSICS_CORPORA)}…"}
    evidence = retrieve_evidence(question)
    if not evidence["available"]:
        yield "retrieve", {"msg": f"Library unavailable — {evidence['reason']}", "evidence": evidence}
    else:
        yield "retrieve", {"msg": (f"{len(evidence['kept'])} usable source(s), "
                                   f"{len(evidence['rejected'])} rejected"),
                           "evidence": evidence}

    yield "compute", {"msg": "Checking for a deterministic solver…"}
    computed = compute_for(question, topic.id if topic else None)
    if computed and computed.get("ran"):
        yield "compute", {"msg": f"Computed via {computed['solver']}", "computed": computed}
    else:
        yield "compute", {"msg": (computed or {}).get("reason", "no solver for this topic"),
                          "computed": computed}

    key = _api_key()
    if not key:
        yield "error", {"message": "No DEEPSEEK_API_KEY found — set it in quantum-professor-/.env"}
        return
    try:
        from openai import OpenAI
    except ImportError:
        yield "error", {"message": "pip3 install openai"}
        return

    yield "compose", {"msg": "Composing the explanation…"}
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com",
                    timeout=120.0, max_retries=1)
    try:
        prose = _compose(question, topic, evidence, computed, mode, depth, client)
    except Exception as exc:
        yield "error", {"message": f"{type(exc).__name__}: {exc}"}
        return
    if not prose.strip():
        yield "error", {"message": "model returned an empty completion"}
        return

    audit = _audit(prose, evidence)
    yield "done", {
        "question": question, "mode": mode, "depth": depth,
        "topic": topic.id if topic else None,
        "topic_title": topic.title if topic else None,
        "related": [{"id": t.id, "title": t.title} for t in related],
        "prose": prose,
        "evidence": evidence,
        "computed": computed,
        "audit": audit,
        "honesty": {
            "numbers_are_computed_not_generated": bool(computed and computed.get("ran")),
            "grounded_in_library": bool(evidence.get("kept")),
            "confidence_is_per_corpus_normalised": True,
            "note": ("Scores are max-normalised within each corpus, so 1.000 means "
                     "'best of its corpus', not 'certain'. raw_score is the "
                     "comparable figure."),
        },
    }


def answer(question: str, mode: str = "explain", depth: str = "intermediate") -> dict:
    """Non-streaming convenience wrapper — returns the final payload."""
    last: dict = {}
    for stage, payload in answer_stream(question, mode, depth):
        if stage in ("done", "error"):
            last = {"stage": stage, **payload}
    return last
