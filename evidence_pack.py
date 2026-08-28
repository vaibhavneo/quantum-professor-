"""EvidencePack - the structured handoff between retrieval and answer
generation.

Before this, evidence_engine()/reasoning_engine()/professor_engine()/
validation() each independently re-derived their own citation text from the
same raw book_ev/papers/topics/sides/assessment - four near-identical but
subtly different re-readings of the same underlying data. This module is
now the one place that consolidation happens: retrieval produces raw
results, build_evidence_pack() assembles them into one structured object,
and the professor reasons from THAT rather than re-interpreting five loose
parameters itself.

Nothing here calls an LLM or does new retrieval - it is a pure, deterministic
reshaping of data qp_pipeline.py already computed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .library import TOPICS
except ImportError:
    from library import TOPICS


@dataclass
class EvidencePack:
    question: str
    intent: str = "explain"
    concepts: list = field(default_factory=list)
    topics: list = field(default_factory=list)          # curriculum Topic objects, union across sides
    requested_depth: str = "standard"
    math_required: bool = False
    passages: list = field(default_factory=list)        # book/arXiv items, each a dict with a "tag"
    sources: list = field(default_factory=list)         # unique source titles across all passages
    prerequisite_concepts: list = field(default_factory=list)  # titles one step back from matched topics
    supporting_evidence: list = field(default_factory=list)    # assessment["agreements"]
    conflicting_evidence: list = field(default_factory=list)   # assessment["conflicts"]
    knowledge_gaps: list = field(default_factory=list)         # assessment["gaps"]
    sides: list | None = None                            # comparison sides_meta, when applicable
    computed: dict | None = None
    symbolic: dict | None = None
    curriculum_backed: bool = False                      # ANY topic matched anywhere (not a gate)
    mathematical_objects: list = field(default_factory=list)  # see extract_mathematical_objects()
    givens: list = field(default_factory=list)            # see decompose_problem()
    unknowns: list = field(default_factory=list)          # see decompose_problem()
    assumptions: list = field(default_factory=list)       # see decompose_problem()
    strategy: str = ""                                    # see solution_strategy()

    @property
    def has_book_evidence(self) -> bool:
        return bool(self.passages)

    @property
    def is_comparison(self) -> bool:
        return bool(self.sides)

    def passages_for_side(self, label: str | None):
        if label is None:
            return self.passages
        return [p for p in self.passages if p.get("side") == label]

    def topics_for_side(self, label: str | None):
        if not self.sides or label is None:
            return self.topics
        for s in self.sides:
            if s["label"] == label:
                return s.get("topics") or []
        return []


def extract_mathematical_objects(topics, computed, symbolic) -> list:
    """The "Mathematical Objects" stage - deterministic by design (per the
    user's own call: this needs no LLM, everything it names was already
    computed or curated elsewhere). Pulls out the actual named equations in
    play - from curriculum topics' own key_equations, the solver's result,
    and any symbolic identity check - as one flat, structured list the
    Derivation Plan stage can reason over, instead of that stage having to
    re-scan prose for equations itself.
    """
    objects = []
    for t in topics or []:
        for eq in t.key_equations:
            objects.append({"name": t.title, "expression": eq, "kind": "curriculum_equation",
                           "topic_id": t.id, "tag": f"C:{t.id}"})
    if computed and computed.get("ran"):
        formula = computed["result"].get("formula", "")
        if formula:
            objects.append({"name": "computed relation", "expression": formula,
                           "kind": "computed_result", "topic_id": None, "tag": "T1"})
    if symbolic and symbolic.get("ok"):
        objects.append({"name": "symbolic identity check", "expression": symbolic.get("verdict", ""),
                       "kind": "symbolic_check", "topic_id": None, "tag": "X1"})
    return objects


# Standing physical-model assumptions per solver-backed topic - curated once,
# the same way verification.py curates per-topic reference data, rather than
# invented per-question. Only topics with a real closed-form solver get an
# entry: these are exactly the questions where "assumptions" means something
# concrete (which idealization the numbers rest on), not vague hedging.
_MODEL_ASSUMPTIONS = {
    "particle-in-a-box": [
        "the potential is infinite outside the well (the particle is never found there)",
        "the particle is non-relativistic",
        "the confinement is treated as strictly one-dimensional",
    ],
    "harmonic-oscillator": [
        "the restoring potential is exactly quadratic (an ideal spring, no anharmonic terms)",
        "the particle is non-relativistic",
    ],
    "hydrogen-atom": [
        "the proton is treated as fixed (infinite nuclear mass, no recoil)",
        "only the Coulomb interaction is included - no spin, no relativistic corrections",
    ],
    "hydrogen-transition": [
        "the transition is treated as an idealized two-level jump, ignoring selection rules "
        "and finite linewidth",
    ],
    "de-broglie": [
        "the particle is treated non-relativistically",
    ],
    "blackbody-radiation": [
        "the emitter is treated as an ideal blackbody in thermal equilibrium",
    ],
    "photon-energy": [
        "light is treated as a stream of discrete photons of definite energy",
    ],
    "uncertainty-principle": [
        "the bound used is the theoretical minimum (Delta_x * Delta_p = hbar/2 exactly); a "
        "real state's actual uncertainty product is generally larger",
    ],
}


def decompose_problem(question: str, u: dict, topics, computed) -> dict:
    """Problem Decomposition (Givens / Unknowns / Assumptions) - deterministic,
    like extract_mathematical_objects() above: the solver's own already-
    extracted inputs ARE the givens, the question's own phrasing and intent
    name the unknowns, and the standing physical-model assumptions come from
    the curated table above. Nothing here is invented or asks an LLM - it
    only restates what the pipeline already knows or extracted.

    Returns empty lists when there's no real solver result to decompose (a
    pure conceptual or symbolic-only question) - this framework is for
    genuine multi-step numeric problems, not forced onto every question.
    """
    if not (computed and computed.get("ran")):
        return {"givens": [], "unknowns": [], "assumptions": []}
    result = computed["result"]
    topic_id = result.get("topic", "")
    givens = ([f"physical system: {topics[0].title}"] if topics else [])
    givens += [f"{k} = {v}" for k, v in (computed.get("inputs") or {}).items()]

    ql = question.lower()
    unknowns = []
    if u.get("intent") == "derive" or "derive" in ql:
        unknowns.append("the general symbolic expression")
    if any(kw in ql for kw in ("calculate", "compute", "what is", "find", "how much", "determine")):
        formula = result.get("formula", "")
        lhs = formula.split("=", 1)[0].strip() if "=" in formula else "the requested quantity"
        unknowns.append(f"the numeric value of {lhs}")
    if not unknowns:
        unknowns.append("the requested result")

    return {"givens": givens, "unknowns": unknowns,
           "assumptions": list(_MODEL_ASSUMPTIONS.get(topic_id, []))}


def solution_strategy(u: dict, computed, symbolic) -> str:
    """Solution Strategy - one deterministic sentence naming the general
    approach, chosen from what evidence is actually available. The right
    strategy for "a real solver ran" is always the same shape (derive, then
    substitute), so this is a lookup over evidence kind, not something that
    needs an LLM's judgement to state."""
    if computed and computed.get("ran"):
        return ("Derive the general symbolic result from the governing equation for this "
               "system, then substitute the given numeric values to compute the requested "
               "quantity.")
    if symbolic and symbolic.get("ok"):
        return ("Verify the stated identity algebraically, then use it to justify the "
               "derivation step it supports.")
    if (u or {}).get("intent") == "derive":
        return "Derive the result from the relevant governing equation and definitions."
    return "Explain the underlying physics directly from the established curriculum concepts."


def build_evidence_pack(question, u, topics, sides, book_ev, papers, assessment, computed,
                        symbolic) -> EvidencePack:
    """Pure consolidation - no new retrieval, no LLM call. Every field here
    already existed somewhere in qp_pipeline.py's scattered parameters; this
    is just the one place they get read into a single structured shape.
    """
    if sides:
        all_topics, seen_t = [], set()
        for s in sides:
            for t in s.get("topics") or []:
                if t.id not in seen_t:
                    seen_t.add(t.id)
                    all_topics.append(t)
    else:
        all_topics = list(topics or [])

    prereq_titles, seen_p = [], set()
    for t in all_topics:
        for pid in t.prerequisites:
            if pid not in seen_p and pid in TOPICS:
                seen_p.add(pid)
                prereq_titles.append(TOPICS[pid].title)

    passages = list((book_ev or {}).get("kept", []))
    passages += [{"tag": f"A{i}", "source": p.get("title", ""), "side": p.get("side"),
                 "text": p.get("summary", ""), "published": p.get("published", "")}
                for i, p in enumerate(papers or [], 1)]
    sources = sorted({p.get("source", "") for p in passages if p.get("source")})

    curriculum_backed = bool(topics) or bool(sides and any(s.get("covered_by_curriculum")
                                                           for s in sides))
    assessment = assessment or {}
    mathematical_objects = extract_mathematical_objects(all_topics, computed, symbolic)
    decomposition = decompose_problem(question, u, all_topics, computed)
    return EvidencePack(
        question=question,
        intent=u.get("intent", "explain"),
        concepts=list(u.get("concepts") or []),
        topics=all_topics,
        requested_depth=u.get("requested_depth", "standard"),
        math_required=bool(u.get("equations_required")),
        passages=passages,
        sources=sources,
        prerequisite_concepts=prereq_titles,
        supporting_evidence=list(assessment.get("agreements") or []),
        conflicting_evidence=list(assessment.get("conflicts") or []),
        knowledge_gaps=list(assessment.get("gaps") or []),
        sides=sides,
        computed=computed,
        symbolic=symbolic,
        curriculum_backed=curriculum_backed,
        mathematical_objects=mathematical_objects,
        givens=decomposition["givens"],
        unknowns=decomposition["unknowns"],
        assumptions=decomposition["assumptions"],
        strategy=solution_strategy(u, computed, symbolic),
    )
