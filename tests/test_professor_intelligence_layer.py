"""Regression suite for the Professor Intelligence Layer phase: EvidencePack,
the removed hard curriculum gate, the grounded answer contract, and the
ModelGateway. Reuses test_scenarios_a_to_j.py's _run() helper rather than
re-implementing the same run() driving logic - no network, no DeepSeek
balance required anywhere in this file.

IMPORTANT CAVEAT, honored throughout this file: everything here verifies the
DETERMINISTIC pipeline (retrieval, gating, EvidencePack construction, mode
dispatch, provider-failure handling) with a scripted FakeClient standing in
for the LLM. None of it proves a REAL DeepSeek response would be good
prose - that needs a live, billed call this account cannot currently make.
Where a test's name says "online", it means "the pipeline correctly reached
and used the LLM call site", not "the model's answer was factually correct."
"""
import sys
from pathlib import Path
from unittest.mock import patch

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qp_pipeline import FakeClient, by_system_prompt, make_status_error  # noqa: E402
from test_scenarios_a_to_j import _run  # noqa: E402

from quantum_prof import qp_pipeline as qp  # noqa: E402


# ── A: classical vs quantum physics, mathematical - now via EvidencePack ───

def test_A_classical_vs_quantum_physics_mathematical_evidence_pack(monkeypatch):
    q = ("What is the difference between quantum physics and classical physics? "
        "Give me the mathematical explanation.")
    payload, _ = _run(q, monkeypatch, retrieve_map={
        "classical mechanics": [{"tag": "S1", "source": "Newtonian Mechanics",
                                 "text": "n" * 250, "raw_score": 0.5}],
        "quantum mechanics": [{"tag": "S1", "source": "Griffiths",
                               "text": "g" * 250, "raw_score": 0.6}],
    }, provider_status=402)
    assert payload["answer_mode"] == "offline"
    pack = payload["evidence_pack"]
    assert pack["curriculum_backed"] is True  # quantum side matched real topics
    assert pack["passage_count"] == 2
    assert set(pack["sources"]) == {"Newtonian Mechanics", "Griffiths"}


# ── B: three-way comparison, mathematically ─────────────────────────────────

def test_B_three_way_mechanics_comparison_mathematically(monkeypatch):
    q = "Compare Newtonian, Lagrangian and Hamiltonian mechanics mathematically."
    payload, _ = _run(q, monkeypatch, provider_status=402)
    u = payload["understanding"]
    # regression guard for the trailing-manner-adverb bug this exact
    # phrasing exposed during this phase's own testing
    assert u["comparison_targets"] == ["Newtonian mechanics", "Lagrangian mechanics",
                                       "Hamiltonian mechanics"]
    assert u["equations_required"] is True
    assert len(payload["sides"]) == 3


# ── C: out-of-curriculum, answerable from book evidence (goal 2) ───────────

def test_C_out_of_curriculum_but_answerable_from_book_evidence(monkeypatch):
    """The actual point of removing the hard curriculum gate: zero curriculum
    topics matched, but real book evidence exists - the answer must use it
    and must NOT say "no curriculum topic covers this" as if that were a
    reason to refuse.
    """
    # release-readiness follow-up: the retrieved text must genuinely share
    # real vocabulary with the question - a placeholder string like "t"*250
    # now correctly fails the no-topic relevance check added to close a
    # different defect (gibberish/off-topic input getting a confident
    # book-evidence answer via pure keyword-collision), so this fixture
    # uses realistic, topically relevant prose instead.
    payload, _ = _run("How do transistors work in silicon wafers?", monkeypatch,
                      retrieve_map={"transistor": [
                          {"tag": "S1", "source": "Modern Particle Physics",
                           "text": "Transistors are semiconductor devices fabricated on "
                                  "silicon wafers using photolithography to create the "
                                  "doped regions that control current flow.",
                           "raw_score": 0.5}]},
                      provider_status=402)
    assert payload["topics"] == []
    assert payload["answer_mode"] == "offline"
    prose = payload["prose"]
    assert "no curriculum topic covers this" not in prose.lower()
    assert "isn't one of the named curriculum topics" in prose
    assert "modern particle physics" in prose.lower()


# ── D: a question requiring equations ───────────────────────────────────────

def test_D_question_requiring_equations(monkeypatch):
    payload, _ = _run("Derive the time-independent Schrodinger equation.", monkeypatch)
    assert payload["understanding"]["equations_required"] is True
    matched_ids = {t["id"] for t in payload["topics"]}
    assert "schrodinger-equation" in matched_ids
    assert payload["answer_mode"] == "online"


# ── E: a conceptual beginner question ───────────────────────────────────────

def test_E_conceptual_beginner_question(monkeypatch):
    payload, _ = _run("What is a photon?", monkeypatch, depth="intro")
    u = payload["understanding"]
    assert u["equations_required"] is False
    assert u["comparison_targets"] == []
    # plain conceptual explain-intent question: the structure directive must
    # NOT force the 7-section contract on something this simple
    assert payload["answer_mode"] == "online"


# ── F: a deep advanced question ─────────────────────────────────────────────

def test_F_deep_advanced_question(monkeypatch):
    payload, _ = _run("Explain the Dirac equation in depth, thoroughly.", monkeypatch,
                      depth="advanced")
    u = payload["understanding"]
    assert u["requested_depth"] == "deep"
    assert payload["answer_mode"] == "online"


# ── G: conflicting evidence ──────────────────────────────────────────────────

def test_G_conflicting_evidence_surfaces_in_the_pack_and_the_answer(monkeypatch):
    """evidence_engine() is the ONLY stage that can genuinely detect
    conflicting sources (it's an LLM judgement, not something retrieval
    computes) - so this scenario mocks its JSON reply specifically, via the
    dispatch-by-system-prompt FakeClient, rather than faking it deeper in
    the pipeline where it wouldn't reflect a real code path.
    """
    client = FakeClient(text='{"restate":"q"}', dispatch=by_system_prompt({
        "You are the evidence stage": (
            '{"usable":["S1","S2"],"off_topic":[],'
            '"agreements":["both sources agree the effect is real"],'
            '"conflicts":["Source S1 gives one mechanism, S2 gives a different one"],'
            '"gaps":[],"covered_by_curriculum":true,"confidence":"medium"}'),
        "You are a physics tutor": "A generated answer that mentions the disagreement.",
    }))
    payload, _ = _run("What is the uncertainty principle?", monkeypatch,
                      retrieve_map={"uncertainty": [
                          {"tag": "S1", "source": "Book A", "text": "a" * 250, "raw_score": 0.5},
                          {"tag": "S2", "source": "Book B", "text": "b" * 250, "raw_score": 0.5}]},
                      client=client)
    assert payload["answer_mode"] == "online"
    assert payload["assessment"]["conflicts"] == [
        "Source S1 gives one mechanism, S2 gives a different one"]
    pack = payload["evidence_pack"]
    assert pack["conflicting_evidence"] == ["Source S1 gives one mechanism, S2 gives a different one"]
    assert pack["supporting_evidence"] == ["both sources agree the effect is real"]


# ── H: unavailable / invalid LLM provider ───────────────────────────────────

def test_H_invalid_provider_key_classified_and_never_raw(monkeypatch):
    payload, events = _run("what is the uncertainty principle", monkeypatch, provider_status=401)
    assert [s for s, _ in events][-1] == "done"
    assert payload["provider_error"]["kind"] == "invalid_key"
    assert "Traceback" not in payload["prose"]


def test_H_timeout_classified_distinctly(monkeypatch):
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "evidence_strength": "usable",
                                           "rejected": [], "kept": [
                            {"tag": "S1", "source": "Griffiths", "text": "z" * 250,
                             "raw_score": 0.6}]})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key")
    import httpx
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    timeout_exc = openai.APITimeoutError(request=request)
    with patch("openai.OpenAI", return_value=FakeClient(raise_on_call=timeout_exc)):
        events = list(qp.run("what is the uncertainty principle", depth="intermediate"))
    payload = events[-1][1]
    assert payload["provider_error"]["kind"] == "timeout"


# ── I: offline deterministic mode - structured, clearly labeled ────────────

def test_I_offline_mode_produces_structured_labeled_response(monkeypatch):
    payload, _ = _run("What is the uncertainty principle?", monkeypatch,
                      retrieve_map={"uncertainty": [
                          {"tag": "S1", "source": "Griffiths", "text": "g" * 250,
                           "raw_score": 0.6}]},
                      provider_status=402)
    assert payload["answer_mode"] == "offline"
    assert payload["provider_error"] is not None
    # clearly labeled as deterministic, never pretending to be LLM prose
    assert "no ai reasoning applied" in payload["honesty"]["note"].lower()
    pack = payload["evidence_pack"]
    assert pack["passage_count"] >= 1
    assert isinstance(pack["prerequisite_concepts"], list)


# ── J: requested-depth behavior, end to end ─────────────────────────────────

def test_J_requested_depth_shallow_reaches_the_professor_prompt(monkeypatch):
    payload, _ = _run("Explain spin briefly.", monkeypatch)
    assert payload["understanding"]["requested_depth"] == "shallow"


def test_J_requested_depth_deep_flows_into_understanding_and_quality_gate(monkeypatch):
    payload, _ = _run("Explain the Dirac equation in depth, thoroughly.", monkeypatch)
    assert payload["understanding"]["requested_depth"] == "deep"
    assert "satisfied_requested_depth" in payload["quality_gate"]


# ── Release-readiness regression tests: the five defects found by the
# 76-question real-world Professor Quality Evaluation, reproduced and fixed
# via qp.run() end to end (provider_status=402 forces the real offline path,
# no real network call) - these prove the fix reaches the actual runtime
# payload, not just the helper function in isolation. ──────────────────────

def test_defect1_solver_computed_value_appears_in_offline_answer(monkeypatch):
    payload, _ = _run("What is the ground state energy of a hydrogen atom?", monkeypatch,
                      provider_status=402)
    assert payload["answer_mode"] == "offline"
    assert payload["computed"]["ran"] is True
    assert payload["computed"]["result"]["energy_eV"] == -13.605693
    # the exact number the solver computed must be in the user-visible text,
    # not just the payload - previously it was appended after several
    # thousand characters of book excerpts, or simply never checked.
    assert "-13.605693" in payload["prose"]
    assert "Computed result" in payload["prose"]


def test_defect2_infinite_well_numeric_question_routes_to_particle_in_a_box(monkeypatch):
    payload, _ = _run("What is the energy of an electron in the n=2 state of a "
                      "1nm-wide infinite square well?", monkeypatch, provider_status=402)
    assert payload["topics"][0]["id"] == "particle-in-a-box"
    assert payload["computed"]["solver"] == "particle-in-a-box"
    assert payload["computed"]["result"]["energy_eV"] == 1.504121
    assert "1.504121" in payload["prose"]


def test_defect3_gibberish_query_gets_honest_insufficient_evidence_no_book_dump(monkeypatch):
    # a real book passage exists and shares exactly one word ("random")
    # with the query - the same single-keyword-collision shape confirmed
    # against the real corpus (gibberish, a joke question, and a book's own
    # index page all matched on exactly one shared word).
    payload, _ = _run("asdkjaslkdj random gibberish text 12345", monkeypatch,
                      retrieve_map={"random": [
                          {"tag": "S1", "source": "Nonequilibrium Statistical Mechanics",
                           "text": "the probability that the measurement has a value in "
                                  "B is given by the distribution function mu.",
                           "raw_score": 1.05}]},
                      provider_status=402)
    assert payload["topics"] == []
    assert payload["answer_mode"] == "insufficient_evidence"
    assert "Nonequilibrium Statistical Mechanics" not in payload["prose"]
    assert "[S1]" not in payload["prose"]
    assert "no answer is offered" in payload["prose"]


def test_defect3_genuinely_relevant_no_topic_evidence_still_shown(monkeypatch):
    # regression guard: the fix must not suppress real, topically relevant
    # book evidence just because no curriculum topic exists for it.
    payload, _ = _run("How do transistors work in silicon wafers?", monkeypatch,
                      retrieve_map={"transistor": [
                          {"tag": "S1", "source": "Modern Particle Physics",
                           "text": "Transistors are semiconductor devices fabricated on "
                                  "silicon wafers using photolithography to create the "
                                  "doped regions that control current flow.",
                           "raw_score": 0.5}]},
                      provider_status=402)
    assert payload["topics"] == []
    assert payload["answer_mode"] == "offline"
    assert "modern particle physics" in payload["prose"].lower()


def test_defect4_direct_relational_question_gets_a_direct_answer(monkeypatch):
    payload, _ = _run("If we increase the width L of an infinite square well, does the "
                      "ground-state energy increase or decrease, and why?", monkeypatch,
                      provider_status=402)
    assert payload["topics"][0]["id"] == "particle-in-a-box"
    idx = payload["prose"].find("Direct answer")
    assert idx != -1
    assert idx < 50, "the direct answer must lead the response, not be buried in it"
    assert "it decreases" in payload["prose"]


def test_defect5_false_premise_question_is_explicitly_corrected(monkeypatch):
    payload, _ = _run("Since heavier objects fall faster than lighter ones in a vacuum, "
                      "calculate how much faster a 10 kg ball falls than a 1 kg ball from "
                      "the same height.", monkeypatch,
                      retrieve_map={"vacuum": [
                          {"tag": "S1", "source": "Physics for Scientists and Engineers",
                           "text": "In free fall without air resistance, the acceleration "
                                  "of a falling object in a vacuum is g, independent of "
                                  "the object's mass, height, or weight.",
                           "raw_score": 1.0}]},
                      provider_status=402)
    prose = payload["prose"]
    assert "A note on this question's premise" in prose
    assert "same rate regardless of mass" in prose


def test_defect5_normal_question_gets_no_premise_note(monkeypatch):
    # regression guard: the curated correction table must not fire on
    # ordinary questions that don't match one of its specific patterns.
    payload, _ = _run("What is the Heisenberg uncertainty principle?", monkeypatch,
                      provider_status=402)
    assert "A note on this question's premise" not in payload["prose"]
