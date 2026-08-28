"""Offline tests for the Physics + Mathematical Verification Layer
(verification.py). Every check here is real sympy/deterministic logic - no
LLM, no network, no mocking of the math itself. Where a test needs a
"derivation", it's a plain string standing in for what reasoning_engine()
would have produced - the point is to prove the CHECK is correct, not to
re-test the LLM call (already covered elsewhere with mocks).
"""
import physics
from quantum_prof import verification as v
from quantum_prof.evidence_pack import EvidencePack
from quantum_prof.library import TOPICS


def _topic(tid):
    return TOPICS[tid]


# ── 1. Symbol/equation consistency ──────────────────────────────────────────

def test_symbol_consistency_passes_for_real_offered_tag():
    pack = EvidencePack(question="q", topics=[_topic("schrodinger-equation")])
    r = v.check_symbol_consistency("using [C:schrodinger-equation] we get...", pack)
    assert r.status == "pass"


def test_symbol_consistency_fails_for_fabricated_tag():
    pack = EvidencePack(question="q", topics=[])
    r = v.check_symbol_consistency("using [C:not-a-real-topic] we get...", pack)
    assert r.status == "fail"
    assert "C:not-a-real-topic" in r.detail


def test_symbol_consistency_not_applicable_with_no_tags():
    r = v.check_symbol_consistency("plain prose with no citations at all", EvidencePack(question="q"))
    assert r.status == "not_applicable"


# ── 2. Dimensional/unit consistency ─────────────────────────────────────────

def test_dimensional_consistency_passes_with_expected_unit():
    computed = {"ran": True, "result": physics.solve("harmonic-oscillator", n=0, omega=1e14)}
    r = v.check_dimensional_consistency("the zero-point energy is about 0.033 eV", computed)
    assert r.status == "pass"


def test_dimensional_consistency_fails_with_wrong_unit_kind():
    computed = {"ran": True, "result": physics.solve("harmonic-oscillator", n=0, omega=1e14)}
    r = v.check_dimensional_consistency("the result is 0.033 meters", computed)
    assert r.status == "fail"
    assert r.correction is not None


def test_dimensional_consistency_not_applicable_without_computed():
    assert v.check_dimensional_consistency("some text", None).status == "not_applicable"


# ── 3. Algebraic consistency (reuses research.check_identity - real sympy) ──

def test_algebraic_consistency_passes_for_real_identity():
    r = v.check_algebraic_consistency({"identity": "sin(x)**2 + cos(x)**2 = 1"}, None)
    assert r.status == "pass"


def test_algebraic_consistency_fails_for_false_identity():
    r = v.check_algebraic_consistency({"identity": "sin(x)**2 + cos(x)**2 = 2"}, None)
    assert r.status == "fail"
    assert r.correction is not None


def test_algebraic_consistency_reuses_already_computed_symbolic_result():
    # symbolic already ran upstream (route()'s sympy stage) - must not re-run it
    symbolic = {"ok": True, "equal": True, "lhs": "1", "rhs": "1", "verdict": "identity holds"}
    r = v.check_algebraic_consistency({"identity": "irrelevant"}, symbolic)
    assert r.status == "pass"


def test_algebraic_consistency_not_applicable_with_no_identity():
    assert v.check_algebraic_consistency({}, None).status == "not_applicable"


# ── 4. Operator consistency (real sympy.physics.quantum-style algebra) ─────

def test_operator_consistency_pauli_matrices_pass():
    pack = EvidencePack(question="explain qubits and Pauli matrices")
    r = v.check_operator_consistency(pack)
    assert r.status == "pass"
    assert "Pauli" in r.detail


def test_operator_consistency_commutator_pass():
    pack = EvidencePack(question="what is the canonical commutation relation", concepts=["commutator"])
    r = v.check_operator_consistency(pack)
    assert r.status == "pass"


def test_operator_consistency_not_applicable_otherwise():
    r = v.check_operator_consistency(EvidencePack(question="what is a photon"))
    assert r.status == "not_applicable"


# ── 5. Boundary/initial-condition consistency ───────────────────────────────

def test_boundary_conditions_pass_for_particle_in_a_box():
    computed = {"ran": True, "result": {"topic": "particle-in-a-box"}}
    r = v.check_boundary_conditions(None, computed)
    assert r.status == "pass"


def test_boundary_conditions_not_applicable_otherwise():
    computed = {"ran": True, "result": {"topic": "harmonic-oscillator"}}
    assert v.check_boundary_conditions(None, computed).status == "not_applicable"


# ── 6/7. Limiting-case + classical-limit checks (real sympy limits) ────────

def test_limiting_case_harmonic_oscillator():
    computed = {"ran": True, "result": {"topic": "harmonic-oscillator"}}
    assert v.check_limiting_case(computed).status == "pass"


def test_limiting_case_particle_in_a_box():
    computed = {"ran": True, "result": {"topic": "particle-in-a-box"}}
    assert v.check_limiting_case(computed).status == "pass"


def test_classical_limit_harmonic_oscillator():
    computed = {"ran": True, "result": {"topic": "harmonic-oscillator"}}
    r = v.check_classical_limit(computed)
    assert r.status == "pass"
    assert "hbar -> 0" in r.detail


def test_classical_limit_not_applicable_for_uncurated_topic():
    computed = {"ran": True, "result": {"topic": "hydrogen-atom"}}
    assert v.check_classical_limit(computed).status == "not_applicable"


# ── 8. Conservation-law check (real sympy differentiation) ─────────────────

def test_conservation_law_passes_for_shm_via_topic_id():
    pack = EvidencePack(question="q", topics=[_topic("harmonic-oscillator")]) \
        if "harmonic-oscillator" in TOPICS else EvidencePack(
            question="classical harmonic oscillator Hamiltonian mechanics")
    r = v.check_conservation_law(pack)
    assert r.status == "pass"
    assert "dE/dt = 0" in r.detail


def test_conservation_law_passes_for_hamiltonian_mechanics_question_text():
    # classical/Hamiltonian mechanics has NO curriculum topic at all - the
    # check must still fire from the raw question text, not just topic ids.
    pack = EvidencePack(question="Explain Hamiltonian mechanics for a simple harmonic oscillator")
    r = v.check_conservation_law(pack)
    assert r.status == "pass"


def test_conservation_law_not_applicable_otherwise():
    pack = EvidencePack(question="what is a photon")
    assert v.check_conservation_law(pack).status == "not_applicable"


# ── 9. Known-result / reference check (cross-checks physics.py's solvers) ──

def test_known_result_passes_when_derivation_states_the_right_number():
    computed = {"ran": True, "result": physics.solve("harmonic-oscillator", n=0, omega=1e14)}
    r = v.check_known_result("the zero-point energy is 0.032911 eV", computed)
    assert r.status == "pass"


def test_known_result_fails_on_deliberately_incorrect_derivation():
    # scenario: a deliberately incorrect derivation the verifier must detect
    computed = {"ran": True, "result": physics.solve("harmonic-oscillator", n=0, omega=1e14)}
    wrong = "The zero-point energy of the ground state is approximately 0.5 eV."
    r = v.check_known_result(wrong, computed)
    assert r.status == "fail"
    assert r.correction == "the correct value is 0.032911 eV"


def test_known_result_not_applicable_without_a_solver_run():
    assert v.check_known_result("some text", None).status == "not_applicable"


# ── 10. Assumption/approximation inventory ──────────────────────────────────

def test_assumptions_inventory_finds_flagged_sentences():
    text = "Assuming the potential is symmetric. We neglect relativistic corrections."
    found = v.check_assumptions(text)
    assert len(found) == 2


def test_assumptions_inventory_empty_for_plain_text():
    assert v.check_assumptions("The energy levels are quantized.") == []


# ── Orchestrator: verify_derivation() end to end ────────────────────────────

def test_verify_derivation_correct_derivation_is_verified_mathematically():
    computed = {"ran": True, "result": physics.solve("harmonic-oscillator", n=0, omega=1e14)}
    reasoning = {"derivation_plan": "The zero-point energy is E_0 = 0.032911 eV.", "text": "x"}
    result = v.verify_derivation("ground state energy", {}, None, reasoning, computed, None)
    assert result.status == "verified_mathematically"
    assert not result.failed


def test_verify_derivation_incorrect_derivation_is_not_silently_passed():
    # scenario 7: a deliberately incorrect derivation the verifier must detect
    computed = {"ran": True, "result": physics.solve("harmonic-oscillator", n=0, omega=1e14)}
    reasoning = {"derivation_plan": "The zero-point energy is approximately 0.5 eV.", "text": "x"}
    result = v.verify_derivation("ground state energy", {}, None, reasoning, computed, None)
    assert result.status != "verified_mathematically"
    assert result.failed
    assert result.corrections
    assert result.corrections[0]["correction"] == "the correct value is 0.032911 eV"


def test_verify_derivation_nothing_checkable_is_honest_not_a_false_pass():
    reasoning = {"derivation_plan": "some prose with nothing checkable", "text": "x"}
    result = v.verify_derivation("what is quantum entanglement", {}, None, reasoning, None, None)
    assert result.status == "not_independently_verified"
    assert not result.passed and not result.failed


def test_verify_derivation_never_raises_on_malformed_input():
    result = v.verify_derivation("q", {}, None, {}, None, None)
    assert result.status in ("verified_mathematically", "partially_verified",
                             "failed", "not_independently_verified")


def test_verify_derivation_includes_full_check_ledger():
    result = v.verify_derivation("what is a photon", {}, None, {"derivation_plan": "", "text": ""},
                                 None, None)
    assert len(result.checks) == 9  # all ten minus the assumptions inventory, which isn't pass/fail
    assert all("status" in c for c in result.checks)


def test_verification_result_serializes_to_a_plain_dict():
    result = v.verify_derivation("q", {}, None, {}, None, None)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert set(d.keys()) == {"status", "confidence", "passed", "failed", "warnings",
                             "assumptions", "corrections", "checks"}


# ── regression: dimensional_consistency word-boundary matching ─────────────

def test_dimensional_consistency_does_not_false_positive_on_a_substring_match():
    # "level" contains the letters "ev" - a naive `"ev" in text` scan would
    # wrongly read this as the unit "eV" being stated, turning a text that
    # names NO unit at all into a false "pass". Found by direct testing
    # while building the FAILED-status scenario below.
    computed = {"ran": True, "result": physics.solve("harmonic-oscillator", n=0, omega=1e14)}
    r = v.check_dimensional_consistency("the n=2 energy level comes out to about -1.2", computed)
    assert r.status == "not_applicable"
    assert "no unit stated" in r.detail


# ── the fourth status: FAILED (checked and actively wrong, nothing else
#    corroborates it - distinct from "nothing was applicable to check") ────

def test_verify_derivation_pure_failure_with_nothing_corroborating_is_status_failed():
    # hydrogen-atom has no curated boundary/limiting-case/classical-limit
    # check, so a wrong numeric claim with no stated unit leaves known_result
    # as the ONLY applicable check - and it fails. That must read as FAILED,
    # not the same "not_independently_verified" a genuinely uncheckable
    # question would get.
    computed = {"ran": True, "result": physics.solve("hydrogen-atom", n=2)}
    wrong = {"derivation_plan": "the n=2 energy level comes out to about -1.2", "text": "x"}
    result = v.verify_derivation("energy of hydrogen n=2", {}, None, wrong, computed, None)
    assert result.status == "failed"
    assert not result.passed
    assert result.failed
    assert result.corrections[0]["correction"] == "the correct value is -3.401423 eV"


# ── requirement: unsupported equations must not produce a false "verified" ─

def test_verify_derivation_fabricated_citation_never_yields_false_verified():
    pack = EvidencePack(question="q", topics=[_topic("schrodinger-equation")])
    # cites a tag that was never offered - an "unsupported equation" in the
    # sense that matters here: the derivation leans on something not
    # actually in evidence.
    reasoning = {"derivation_plan": "from [C:not-a-real-topic] we derive...", "text": "x"}
    result = v.verify_derivation("q", {}, pack, reasoning, None, None)
    assert result.status != "verified_mathematically"
    assert any(f["check"] == "symbol_consistency" for f in result.failed)


# ── requirement: incomplete evidence must never be reported as full
#    verification - it must land on PARTIALLY or NOT INDEPENDENTLY verified ─

def test_verify_derivation_incomplete_evidence_never_reports_full_verification():
    # No pack, no computed result, no symbolic check - genuinely thin
    # evidence. The only honest outcomes are "nothing could be checked" or,
    # if something WAS checkable and wrong, a failure - never a claim that
    # the math was independently confirmed.
    reasoning = {"derivation_plan": "the energy increases with n, roughly speaking", "text": "x"}
    result = v.verify_derivation("some under-evidenced question", {}, None, reasoning, None, None)
    assert result.status != "verified_mathematically"
    assert result.status in ("not_independently_verified", "partially_verified", "failed")


# ── requirement: malformed derivations are handled safely (never raises) ───

def test_verify_derivation_handles_none_reasoning_without_raising():
    result = v.verify_derivation("q", {}, None, None, None, None)
    assert result.status == "not_independently_verified"


def test_verify_derivation_handles_reasoning_with_wrong_value_types_without_raising():
    # derivation_plan is a list, not a string - a malformed upstream reply
    # (e.g. a provider returning structured content unexpectedly) must not
    # crash the check, only degrade to "nothing to check".
    malformed = {"derivation_plan": ["not", "a", "string"], "text": 12345}
    result = v.verify_derivation("q", {"identity": None}, None, malformed, None, None)
    assert result.status in ("not_independently_verified", "partially_verified",
                             "failed", "verified_mathematically")


def test_verify_derivation_handles_pack_with_empty_mathematical_objects_without_raising():
    # "missing mathematical objects" at the verifier's own level: a pack
    # that exists but carries none - must degrade honestly, not crash.
    pack = EvidencePack(question="q", topics=[], mathematical_objects=[])
    result = v.verify_derivation("q", {}, pack, {"derivation_plan": "some claim", "text": "x"},
                                 None, None)
    assert result.status in ("not_independently_verified", "partially_verified", "failed")


# ── requirement: verification never silently turns uncertainty into
#    correctness - "verified_mathematically" must always be EARNED by a real
#    passing check, never handed out by default when nothing was checkable ─

def test_verify_derivation_never_reports_verified_when_nothing_actually_passed():
    uncertain_inputs = [
        ("q", {}, None, {}, None, None),
        ("what is quantum entanglement", {}, None,
         {"derivation_plan": "some prose with nothing checkable", "text": "x"}, None, None),
        ("q", {"identity": ""}, None, {"derivation_plan": "", "text": ""}, None, None),
        ("q", {}, EvidencePack(question="q"), None, None, None),
    ]
    for args in uncertain_inputs:
        result = v.verify_derivation(*args)
        assert not result.passed, f"no check actually passed for {args!r}, yet result.passed is non-empty"
        assert result.status != "verified_mathematically", \
            f"uncertainty was silently reported as verified for input {args!r}"
