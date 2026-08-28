"""Dedicated end-to-end regression suite for the ten named scenarios (A-J)
requested for the question-driven retrieval upgrade. Each test drives the
real qp_pipeline.run() generator with a fake LLM client (or none at all) -
no network, no DeepSeek balance required anywhere in this file.

For every comparison scenario (A, B, F), the point being verified is not
just "did retrieval find something" but "is EACH side judged independently
and honestly" - the exact failure this whole upgrade targets is a quantum-
flavoured curriculum topic standing in as the answer to a question that is,
in whole or in part, about classical physics.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qp_pipeline import FakeClient, make_status_error  # noqa: E402

from quantum_prof import qp_pipeline as qp  # noqa: E402


def _run(question, monkeypatch, retrieve_map=None, provider_status=None, depth="intermediate",
        mode="explain", prior=None, client=None):
    """Drives run() end-to-end. retrieve_map, if given, is {substring-of-query:
    kept-list}; anything not matching returns no evidence. provider_status, if
    given, makes every LLM call fail with that HTTP status (simulating
    DeepSeek being down); otherwise a benign FakeClient answers every call.
    Pass an explicit `client` (e.g. one built with a dispatch table) when a
    scenario needs different canned replies for different pipeline stages.
    """
    def fake_retrieve(q, top_k=6):
        if retrieve_map:
            for key, items in retrieve_map.items():
                if key in q:
                    return {"available": True, "kept": list(items), "rejected": [],
                           "evidence_strength": "usable"}
        return {"available": True, "kept": [], "rejected": [], "evidence_strength": "none"}

    monkeypatch.setattr(qp, "retrieve_evidence", fake_retrieve)
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key")
    if client is None:
        if provider_status is not None:
            boom = make_status_error(openai.APIStatusError, provider_status, "simulated failure")
            client = FakeClient(raise_on_call=boom)
        else:
            client = FakeClient(text="A benign generated explanation of the material above.")
    with patch("openai.OpenAI", return_value=client):
        events = list(qp.run(question, mode=mode, depth=depth, prior=prior))
    stages = [s for s, _ in events]
    assert stages[-1] in ("done", "error"), f"unexpected final stage: {stages}"
    return events[-1][1], events


# ── A: classical vs quantum mechanics, mathematical, two sentences ─────────

def test_A_classical_vs_quantum_mechanics_mathematical(monkeypatch):
    q = ("What is the difference between quantum physics and classical physics? "
        "Give me the mathematical explanation.")
    payload, _ = _run(q, monkeypatch, retrieve_map={
        "classical mechanics": [{"tag": "S1", "source": "Newtonian Mechanics",
                                 "text": "n" * 250, "raw_score": 0.5}],
        "quantum mechanics": [{"tag": "S1", "source": "Griffiths",
                               "text": "g" * 250, "raw_score": 0.6}],
    }, provider_status=402)

    u = payload["understanding"]
    assert u["comparison_targets"] == ["quantum mechanics", "classical mechanics"]
    assert u["equations_required"] is True
    assert payload["answer_mode"] == "offline"

    by_label = {s["label"]: s for s in payload["sides"]}
    assert by_label["classical mechanics"]["covered_by_curriculum"] is False
    assert by_label["classical mechanics"]["topics"] == []
    assert by_label["quantum mechanics"]["covered_by_curriculum"] is True

    # Sections follow the question's own word order ("quantum ... classical
    # ..."), so locate each by name rather than assuming which comes first.
    prose_l = payload["prose"].lower()
    c0 = prose_l.index("## classical mechanics")
    q0 = prose_l.index("## quantum mechanics")
    classical_block = prose_l[c0:] if c0 > q0 else prose_l[c0:q0]
    # the exact reported bug: a quantum-flavoured topic standing in for the
    # classical side it has nothing to do with.
    assert "quantum information" not in classical_block
    assert "quantum statistical mechanics" not in classical_block
    assert "quantum computing" not in classical_block
    assert "newtonian" in classical_block  # real evidence, not silence either


# ── B: three-way comparison ─────────────────────────────────────────────────

def test_B_three_way_mechanics_comparison(monkeypatch):
    q = "Compare Newtonian, Lagrangian and Hamiltonian mechanics."
    payload, _ = _run(q, monkeypatch, provider_status=402)
    u = payload["understanding"]
    assert u["comparison_targets"] == ["Newtonian mechanics", "Lagrangian mechanics",
                                       "Hamiltonian mechanics"]
    assert len(payload["sides"]) == 3
    # Updated for the retrieval phase: the curriculum now has a real
    # Lagrangian/Hamiltonian mechanics topic, so those two sides are
    # genuinely covered - Newtonian mechanics specifically still has no
    # dedicated topic and must keep reporting that honestly, rather than
    # borrowing the Lagrangian/Hamiltonian topic to look answered.
    by_label = {s["label"]: s for s in payload["sides"]}
    assert by_label["Newtonian mechanics"]["covered_by_curriculum"] is False
    assert by_label["Lagrangian mechanics"]["covered_by_curriculum"] is True
    assert by_label["Hamiltonian mechanics"]["covered_by_curriculum"] is True
    assert payload["answer_mode"] in ("offline", "degraded", "insufficient_evidence")
    prose_l = payload["prose"].lower()
    for label in ("newtonian mechanics", "lagrangian mechanics", "hamiltonian mechanics"):
        assert label in prose_l


# ── C: Schrödinger equation, mathematically - real curriculum match ────────

def test_C_schrodinger_equation_mathematically(monkeypatch):
    payload, _ = _run("Explain the Schrödinger equation mathematically.", monkeypatch)
    u = payload["understanding"]
    assert u["equations_required"] is True
    matched_ids = {t["id"] for t in payload["topics"]}
    assert "schrodinger-equation" in matched_ids
    assert payload["answer_mode"] == "online"


# ── D: Heisenberg uncertainty principle, mathematically ────────────────────

def test_D_heisenberg_uncertainty_mathematically(monkeypatch):
    payload, _ = _run("Explain the Heisenberg uncertainty principle mathematically.", monkeypatch)
    matched_ids = {t["id"] for t in payload["topics"]}
    assert "uncertainty-principle" in matched_ids
    assert payload["understanding"]["equations_required"] is True
    assert payload["answer_mode"] == "online"


# ── E: quantum entanglement - a legitimate, non-comparison match ───────────

def test_E_quantum_entanglement(monkeypatch):
    payload, _ = _run("Explain quantum entanglement.", monkeypatch)
    matched_ids = {t["id"] for t in payload["topics"]}
    # genuinely relevant curriculum topics, not a comparison question at all
    assert matched_ids & {"decoherence", "quantum-information"}
    assert payload["understanding"]["comparison_targets"] == []
    assert payload["answer_mode"] == "online"


# ── F: classical vs quantum probability ─────────────────────────────────────

def test_F_classical_vs_quantum_probability(monkeypatch):
    payload, _ = _run("Classical probability vs quantum probability.", monkeypatch,
                      provider_status=402)
    u = payload["understanding"]
    assert u["comparison_targets"] == ["Classical probability", "quantum probability"]
    by_label = {s["label"]: s for s in payload["sides"]}
    assert by_label["Classical probability"]["covered_by_curriculum"] is False
    prose_l = payload["prose"].lower()
    c0 = prose_l.index("## classical probability")
    q0 = prose_l.index("## quantum probability")
    classical_block = prose_l[c0:] if c0 > q0 else prose_l[c0:q0]
    assert "quantum information" not in classical_block
    assert "quantum statistical mechanics" not in classical_block
    assert "quantum statistical mechanics" not in prose_l[c0:q0]


# ── G: outside the knowledge base entirely ──────────────────────────────────

def test_G_outside_the_curriculum(monkeypatch):
    payload, _ = _run("How do transistors work in silicon wafers?", monkeypatch)
    assert payload["answer_mode"] == "insufficient_evidence"
    assert payload["topics"] == []


# ── H: correction of a previous question ────────────────────────────────────

def test_H_correction_of_previous_question(monkeypatch):
    prior = {"restate": "what is Newtonian mechanics?", "topics": ["newtonian", "mechanic"],
            "comparison_targets": None, "domain": "quantum-mechanics", "difficulty": "intermediate"}
    payload, _ = _run("I actually meant classical mechanics and quantum mechanics.",
                      monkeypatch, prior=prior)
    u = payload["understanding"]
    assert u["is_correction"] is True
    assert u["comparison_targets"] == ["classical mechanics", "quantum mechanics"]


# ── I: DeepSeek 402 Insufficient Balance ─────────────────────────────────────

def test_I_deepseek_402_never_surfaces_as_a_raw_error(monkeypatch):
    payload, events = _run("what is the uncertainty principle", monkeypatch, provider_status=402)
    assert [s for s, _ in events][-1] == "done"
    assert payload["answer_mode"] in ("offline", "degraded")
    assert payload["provider_error"]["kind"] == "insufficient_balance"
    assert "Insufficient Balance" not in payload["prose"]  # no raw SDK text leaked into the answer


# ── J: weak lexical match that must be rejected ─────────────────────────────

def test_J_weak_match_rejected_not_guessed(monkeypatch):
    payload, _ = _run("classical mechanics", monkeypatch)
    assert payload["answer_mode"] == "insufficient_evidence"
    assert payload["topics"] == []
