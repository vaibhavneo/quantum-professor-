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
import threading
import time
from pathlib import Path
from typing import Iterator

try:                       # package import (local) / flat script (deployment)
    from .library import TOPICS
    from .physics import M_E, SOLVERS, solve
except ImportError:
    from library import TOPICS
    from physics import M_E, SOLVERS, solve

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

_DOT_LEADER = re.compile(r"\.{4,}")
_SPACED_OUT = re.compile(r"(?:\b[A-Za-z]\s){6,}")   # "T h e P h o t o e l e c t r i c"


def looks_like_frontmatter(text: str) -> str | None:
    """Detect contents pages and indexes, which survive good chunking.

    Better chunking fixed chunk *size*, not the fact that a physics PDF opens
    with a table of contents. Such a page is dense in the query's own
    vocabulary and so scores very high — an observed Pauli query ranked a
    contents page first at raw 0.81, above a real Slater-determinant passage.
    Returns a reason string when the chunk should be dropped, else None.
    """
    if not text:
        return "empty"
    if len(_DOT_LEADER.findall(text)) >= 3:
        return "contents page (dot leaders)"
    if len(_SPACED_OUT.findall(text)) >= 2:
        return "broken PDF letter-spacing (front matter)"
    digits = sum(c.isdigit() for c in text)
    if digits / max(len(text), 1) > 0.14:
        return "index or contents page (digit-dense)"
    letters = sum(c.isalpha() for c in text)
    if letters / max(len(text), 1) < 0.55:
        return "not prose (symbol/number dense)"
    return None


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
        elif (reason := looks_like_frontmatter(item["text"])):
            item["why"] = reason
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


def _singular(w: str) -> str:
    """Crude plural fold so a question and the curriculum agree on a word.

    "fermion" in a question never matched "fermions" in key_concepts, so the
    two most specific terms in the question contributed nothing to the score.
    Both sides run through the same fold, so only consistency matters, not
    linguistic correctness."""
    if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is", "as")):
        return w[:-1]
    return w


def _tokens(text: str) -> set[str]:
    return {_singular(w) for w in re.findall(r"[a-z][a-z-]{2,}", text.lower())
            if w not in _STOP}


def _topic_fields(t):
    """(strong, weak) token sets. Title/id/concepts name the subject; the
    intuition paragraph merely mentions things."""
    strong = _tokens(" ".join([t.title, t.id.replace("-", " "),
                               " ".join(t.key_concepts)]))
    return strong, _tokens(t.intuition) - strong


# Document frequency over the curriculum: "quantum" and "particle" appear in
# most topics and so say almost nothing about which one is meant, while
# "fermion" or "tunneling" pin it down. Counting bare overlap let a generic
# word decide the match — a question about photons vs fermions was answered as
# "Infinite Square Well (Particle in a Box)" purely because "particle" is in
# that title.
_DF: dict[str, int] = {}
for _t in TOPICS.values():
    _s, _w = _topic_fields(_t)
    for _tok in (_s | _w):
        _DF[_tok] = _DF.get(_tok, 0) + 1
_NTOPICS = max(len(TOPICS), 1)


def _idf(tok: str) -> float:
    import math
    return math.log(1.0 + _NTOPICS / (1 + _DF.get(tok, 0)))


def match_topic(question: str):
    """Best curriculum topic for a question, or None. Scores by IDF-weighted
    overlap so a rare, specific term outweighs a ubiquitous one."""
    q = _tokens(question)
    if not q:
        return None, []
    scored = []
    for t in TOPICS.values():
        strong, weak = _topic_fields(t)
        score = (sum(_idf(w) for w in q & strong) * 2.0
                 + sum(_idf(w) for w in q & weak))
        if t.title.lower() in question.lower():
            score += 10.0                    # the question names the topic outright
        if score > 0:
            scored.append((score, t))
    scored.sort(key=lambda p: (-p[0], p[1].title))
    if not scored:
        return None, []
    # A near-zero best match means nothing in the curriculum really fits; say so
    # rather than dressing up an unrelated topic as the answer's subject.
    if scored[0][0] < 1.0:
        return None, [t for _s, t in scored[:3]]
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

# ── pedagogy ──────────────────────────────────────────────────────────────
# Passing the bare words "socratic" / "advanced" into the prompt barely moves
# the output. Each setting gets an explicit behavioural instruction, and depth
# also scales the token budget, so the controls are functional rather than
# decorative.
MODE_DIRECTIVE = {
    "explain": "Give a ground-up explanation. Build the intuition first, then "
               "formalise it. Lead with the physical picture, not the algebra.",
    "socratic": "Teach by guided questioning. Ask 3-5 short questions in "
                "sequence that lead the learner to the answer themselves, giving "
                "just enough after each to make the next one answerable. Do not "
                "state the conclusion up front.",
    "exercise": "Practice-first. Pose one concrete worked problem on this topic, "
                "walk through the solution step by step showing the reasoning at "
                "each line, then state the general principle it illustrates.",
    "compare": "Structure the whole answer as a comparison. Identify the two (or "
               "more) things being contrasted, treat them side by side across the "
               "same dimensions, and finish with when each one applies.",
}
DEPTH_DIRECTIVE = {
    "intro": "Assume no physics background beyond school algebra. Avoid "
             "operators and Dirac notation; use analogies and plain language.",
    "intermediate": "Assume undergraduate mechanics and calculus, and comfort "
                    "with basic wavefunctions. Standard notation is fine.",
    "advanced": "Assume graduate-level fluency. Use Dirac notation and operator "
                "algebra freely, and give the full derivation rather than "
                "sketching it.",
}
# deepseek-v4-pro reasons before answering and those tokens count against
# max_tokens. A measured socratic/advanced call spent 7,047 of 8,573 completion
# tokens on reasoning alone — so a 9,000 cap left ~450 tokens of headroom and a
# slightly longer chain came back with empty content. Budgets are sized for the
# reasoning, not the prose.
DEPTH_TOKENS = {"intro": 8000, "intermediate": 12000, "advanced": 18000}
# That same call took 115.1s; 120s was cutting it far too fine.
LLM_TIMEOUT_S = 300.0


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


# ── mastery: what the learner has already worked through ──────────────────
MASTERY_PATH = Path(__file__).parent / "memory" / "mastery.json"


def load_mastery() -> dict:
    try:
        return json.loads(MASTERY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"topics": {}}


def record_visit(topic_id: str | None, mode: str, depth: str) -> dict:
    """Bump the counter for a topic and return its state.

    Deliberately a plain JSON file rather than a database: it is inspectable,
    and this is study history, not something worth a schema.
    """
    if not topic_id:
        return {}
    store = load_mastery()
    t = store.setdefault("topics", {}).setdefault(
        topic_id, {"visits": 0, "modes": [], "last_depth": None})
    t["visits"] += 1
    t["last_depth"] = depth
    if mode not in t["modes"]:
        t["modes"].append(mode)
    try:
        MASTERY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MASTERY_PATH.write_text(json.dumps(store, indent=1))
    except OSError:
        pass                      # history is a nicety; never fail the answer
    return dict(t)


def _familiarity(state: dict) -> str:
    v = (state or {}).get("visits", 0)
    return "new" if v <= 1 else "revisited" if v <= 3 else "familiar"


def _compose(question, topic, evidence, computed, mode, depth, client, mastery=None,
             token_override=None):
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

    fam = _familiarity(mastery)
    hist = {
        "new": "This is the learner's first time on this topic.",
        "revisited": "The learner has been here once or twice before — don't "
                     "re-lay the basics at length; go a level deeper than an "
                     "introduction would.",
        "familiar": "The learner has worked this topic several times. Skip the "
                    "introductory framing entirely and go straight to the "
                    "subtleties, edge cases and connections.",
    }[fam]

    user = (f"QUESTION: {question}\n\n"
            f"HOW TO ANSWER: {MODE_DIRECTIVE.get(mode, MODE_DIRECTIVE['explain'])}\n"
            f"WHO YOU ARE ANSWERING: {DEPTH_DIRECTIVE.get(depth, DEPTH_DIRECTIVE['intermediate'])}\n"
            f"LEARNER HISTORY: {hist}\n\n"
            f"CURRICULUM MATERIAL:\n{cur}\n\nSOURCES FROM THE USER'S LIBRARY:\n{src_block}\n\n"
            f"COMPUTED:\n{comp}\n\nWrite the explanation now.")

    resp = client.chat.completions.create(
        model=MODEL, max_tokens=token_override or DEPTH_TOKENS.get(depth, 12000),
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

    mastery = record_visit(topic.id if topic else None, mode, depth)
    fam = _familiarity(mastery)
    yield "adapt", {"msg": (f"{mode} · {depth} · this topic is {fam}"
                            + (f" (visit {mastery.get('visits')})" if mastery else "")),
                    "mode": mode, "depth": depth,
                    "familiarity": fam, "mastery": mastery}

    client = OpenAI(api_key=key, base_url="https://api.deepseek.com",
                    timeout=LLM_TIMEOUT_S, max_retries=1)

    # A reasoning model takes 90-200s on this prompt, and until now the stream
    # went completely silent for that whole stretch: the UI showed a frozen
    # "Composing…" with no way to tell it apart from a hang, and an idle
    # connection that long is exactly what an edge proxy drops. Run the call on
    # a worker thread and keep emitting ticks while it works.
    box: dict = {}

    def _run():
        try:
            p = _compose(question, topic, evidence, computed, mode, depth, client, mastery)
            if not p.strip():
                # Empty body ⇒ the reasoning chain ate the whole budget.
                box["retrying"] = True
                p = _compose(question, topic, evidence, computed, mode, depth, client,
                             mastery, token_override=DEPTH_TOKENS.get(depth, 12000) * 2)
            box["prose"] = p
        except Exception as exc:                     # surfaced on the main thread
            box["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=_run, daemon=True)
    t0 = time.monotonic()
    yield "compose", {"msg": "Composing the explanation… (this usually takes 1-3 minutes)",
                      "elapsed_s": 0}
    worker.start()
    announced_retry = False
    while worker.is_alive():
        worker.join(timeout=5.0)
        if not worker.is_alive():
            break
        secs = int(time.monotonic() - t0)
        if box.get("retrying") and not announced_retry:
            announced_retry = True
            yield "compose", {"msg": "Empty completion — retrying with a larger budget…",
                              "elapsed_s": secs}
        else:
            yield "compose", {"msg": f"Composing the explanation… {secs}s", "elapsed_s": secs}

    if box.get("error"):
        yield "error", {"message": box["error"]}
        return
    prose = box.get("prose") or ""
    if not prose.strip():
        yield "error", {"message": "model returned an empty completion twice"}
        return
    yield "compose", {"msg": f"Composed in {int(time.monotonic() - t0)}s",
                      "elapsed_s": int(time.monotonic() - t0)}

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
        "mastery": mastery,
        "familiarity": fam,
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
