"""Offline tests for the Physics + Mathematical Verification Layer
(verification.py). Every check here is real sympy/deterministic logic - no
LLM, no network, no mocking of the math itself. Where a test needs a
"derivation", it's a plain string standing in for what reasoning_engine()
would have produced - the point is to prove the CHECK is correct, not to
re-test the LLM call (already covered elsewhere with mocks).
"""
import physics
from quantum_prof import verification as v
from quantum_prof.evidence_pack import EvidencePack, extract_mathematical_objects
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


# ── 4. Operator consistency: verifies the CLAIM the derivation makes, not
#    just that a keyword ("pauli"/"qubit"/"commutator") is present ─────────

def test_operator_consistency_correct_pauli_multiplication_passes():
    text = "multiplying the matrices directly, sigma_x sigma_y = i*sigma_z."
    r = v.check_operator_consistency(text, None)
    assert r.status == "pass"
    assert "sigma_x*sigma_y" in r.detail


def test_operator_consistency_incorrect_pauli_multiplication_missing_i_fails():
    # scenario 2's required case: the true relation is i*sigma_z, not sigma_z -
    # dropping the i is a deliberately incorrect derivation this must catch.
    text = "multiplying the matrices directly, sigma_x sigma_y = sigma_z."
    r = v.check_operator_consistency(text, None)
    assert r.status == "fail"
    assert r.correction is not None


def test_operator_consistency_unrelated_prose_mentioning_pauli_is_not_applicable():
    # a mention, not a claim - must not be enough to trigger a pass on its own.
    text = "Wolfgang Pauli made many foundational contributions to quantum theory."
    r = v.check_operator_consistency(text, None)
    assert r.status == "not_applicable"


def test_operator_consistency_prose_mentioning_qubits_without_a_claim_is_not_applicable():
    text = "A qubit can exist in a superposition of the 0 and 1 states."
    r = v.check_operator_consistency(text, None)
    assert r.status == "not_applicable"


def test_operator_consistency_correct_canonical_commutator_passes():
    r = v.check_operator_consistency("direct computation gives [x,p] = i*hbar.", None)
    assert r.status == "pass"


def test_operator_consistency_finds_the_real_claim_past_an_earlier_restated_definition():
    # a derivation restating "[x,p] = xp - px" (the DEFINITION, not a value)
    # before the actual claimed value later in the text must not give up on
    # the first, unparseable match - it should keep looking.
    text = "- [x,p] = xp - px\n- evaluating on a test function gives [x,p] = i*hbar"
    r = v.check_operator_consistency(text, None)
    assert r.status == "pass"


def test_operator_consistency_commutator_missing_i_fails():
    r = v.check_operator_consistency("direct computation gives [x,p] = hbar.", None)
    assert r.status == "fail"
    assert r.correction is not None


def test_operator_consistency_commutator_wrong_coefficient_fails():
    r = v.check_operator_consistency("direct computation gives [x,p] = 2*i*hbar.", None)
    assert r.status == "fail"


def test_operator_consistency_unparseable_pauli_claim_is_honestly_not_applicable():
    # a real claim shape is present, but the right-hand side isn't one of
    # the bounded forms this can safely evaluate - refuse, don't guess.
    text = "sigma_x sigma_y = some complicated tensor expression we won't write out"
    r = v.check_operator_consistency(text, None)
    assert r.status == "not_applicable"
    assert "could not safely parse" in r.detail


def test_operator_consistency_not_applicable_when_nothing_relevant_at_all():
    r = v.check_operator_consistency("", EvidencePack(question="what is a photon"))
    assert r.status == "not_applicable"


def test_verify_derivation_incorrect_pauli_claim_does_not_reach_verified_mathematically():
    # the exact benchmark-confirmed false positive this fix closes.
    reasoning = {"derivation_plan": "multiplying directly, sigma_x sigma_y = sigma_z.", "text": "x"}
    result = v.verify_derivation("Pauli algebra", {}, None, reasoning, None, None)
    assert result.status != "verified_mathematically"
    assert any(f["check"] == "operator_consistency" for f in result.failed)


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


def test_conservation_law_passes_for_hamiltonian_mechanics_conservation_claim_text():
    # classical/Hamiltonian mechanics has NO curriculum topic at all - the
    # check must still fire from the raw question text, not just topic ids -
    # but ONLY when the text actually names the SHM system AND claims
    # conservation, not from "Hamiltonian mechanics" alone (see the
    # regression test below for that distinction).
    pack = EvidencePack(question="Use Hamiltonian mechanics to show energy is conserved for a "
                                 "simple harmonic oscillator")
    r = v.check_conservation_law(pack)
    assert r.status == "pass"


def test_conservation_law_does_not_fire_on_hamiltonian_mechanics_wording_alone():
    # REGRESSION (benchmark-confirmed false positive): "Hamiltonian
    # mechanics" appearing in a question is not itself a conservation-law
    # claim - naming the formalism is not the same as claiming something is
    # conserved. Previously this fired and unconditionally verified a fixed,
    # unrelated SHM fact regardless of what was actually asked.
    pack = EvidencePack(question="Derive Hamilton's equations from the Lagrangian")
    r = v.check_conservation_law(pack)
    assert r.status == "not_applicable"


def test_conservation_law_does_not_fire_on_lagrangian_mechanics_wording_alone():
    pack = EvidencePack(question="Derive the Euler-Lagrange equation from the principle of "
                                 "least action")
    r = v.check_conservation_law(pack)
    assert r.status == "not_applicable"


def test_conservation_law_does_not_fire_for_unrelated_quantum_mechanics_conservation_claim():
    # "conserved" IS present, but the system is not the SHM this check
    # verifies - a real conservation claim about a DIFFERENT system must not
    # be answered by the fixed SHM computation.
    pack = EvidencePack(question="Show that momentum is conserved in an isolated quantum system")
    r = v.check_conservation_law(pack)
    assert r.status == "not_applicable"


def test_conservation_law_does_not_fire_for_unrelated_problem_merely_mentioning_hamiltonian():
    pack = EvidencePack(question="What is the physical meaning of the Hamiltonian operator in "
                                 "quantum mechanics?")
    r = v.check_conservation_law(pack)
    assert r.status == "not_applicable"


def test_conservation_law_primary_topic_match_is_sufficient_even_without_conservation_wording():
    # the OTHER legitimate trigger: a genuine primary curriculum match for
    # harmonic-oscillator is enough on its own, matching
    # test_conservation_law_passes_for_shm_via_topic_id above.
    pack = EvidencePack(question="q", topics=[_topic("harmonic-oscillator")])
    r = v.check_conservation_law(pack)
    assert r.status == "pass"


# ── REGRESSION: a SECONDARY (non-primary) topic match must not be able to
#    inflate verification confidence - the benchmark-confirmed case where a
#    hydrogen-atom question pulled in harmonic-oscillator as evidence ──────

def test_conservation_law_ignores_a_secondary_non_primary_harmonic_oscillator_match():
    pack = EvidencePack(question="What is the ground state energy of a hydrogen atom?",
                        topics=[_topic("hydrogen-atom"), _topic("harmonic-oscillator")])
    r = v.check_conservation_law(pack)
    assert r.status == "not_applicable"


def test_boundary_conditions_ignores_a_secondary_non_primary_particle_in_a_box_match():
    pack = EvidencePack(question="What is the ground state energy of a hydrogen atom?",
                        topics=[_topic("hydrogen-atom"), _topic("particle-in-a-box")])
    computed = {"ran": True, "result": {"topic": "hydrogen-atom"}}
    r = v.check_boundary_conditions(pack, computed)
    assert r.status == "not_applicable"


def test_verify_derivation_wrong_hydrogen_answer_now_lands_on_failed_not_masked_by_secondary_match():
    # end-to-end: with the spurious secondary match correctly ignored, a
    # wrong hydrogen-atom numeric claim with nothing else to corroborate it
    # lands on the honest, decisive FAILED - rather than partially_verified
    # propped up by an unrelated topic's fixed conservation check.
    pack = EvidencePack(question="What is the ground state energy of a hydrogen atom?",
                        topics=[_topic("hydrogen-atom"), _topic("harmonic-oscillator")])
    computed = {"ran": True, "result": physics.solve("hydrogen-atom", n=1)}
    reasoning = {"derivation_plan": "the ground state energy comes out to about -1.2", "text": "x"}
    result = v.verify_derivation("What is the ground state energy of a hydrogen atom?", {}, pack,
                                 reasoning, computed, None)
    assert result.status == "failed"


def test_verify_derivation_hamiltons_equations_no_longer_falsely_verified():
    # end-to-end: the exact benchmark scenario, through the orchestrator.
    reasoning = {"derivation_plan": "Legendre transform gives H, then Hamilton's equations follow.",
                "text": "x"}
    result = v.verify_derivation("Derive Hamilton's equations from the Lagrangian", {}, None,
                                 reasoning, None, None)
    assert result.status == "not_independently_verified"
    assert not any(p["check"] == "conservation_law" for p in result.passed)


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


# ── REGRESSION: citation tags must never be read as numeric claims ─────────

def test_known_result_ignores_the_digit_in_a_T1_citation_tag():
    # the exact benchmark-confirmed bug: citing [T1] - exactly what the
    # system's own prompt instructs - must not make known_result read the
    # tag's own digit as a stated (and therefore "wrong") result.
    computed = {"ran": True, "result": physics.solve("particle-in-a-box", n=2, L=1e-9)}
    text = "the result is given by the solver [T1], referred to in words, not restated"
    r = v.check_known_result(text, computed)
    assert r.status != "fail"
    assert r.status == "warning"  # honestly: no real number was stated to check


def test_known_result_still_catches_a_real_wrong_number_alongside_a_tag():
    # stripping the tag must not disable real detection of an actually wrong
    # number stated elsewhere in the same sentence.
    computed = {"ran": True, "result": physics.solve("particle-in-a-box", n=2, L=1e-9)}
    text = "citing [T1], the energy works out to about 5.0 eV"
    r = v.check_known_result(text, computed)
    assert r.status == "fail"


def test_known_result_strips_all_citation_tag_shapes_not_just_T1():
    computed = {"ran": True, "result": physics.solve("harmonic-oscillator", n=0, omega=1e14)}
    text = "per [C:harmonic-oscillator] and [S12] and [A3] and [X1], the zero-point energy is 0.032911 eV"
    r = v.check_known_result(text, computed)
    assert r.status == "pass"


def test_verify_derivation_correctly_cited_answer_with_T1_is_not_downgraded():
    # end-to-end: a textbook-correct reply that cites [T1] and defers to it
    # in words must reach verified_mathematically, not be penalized for
    # doing exactly what it was asked to do. Uses a real pack with the T1
    # mathematical object build_evidence_pack() actually adds whenever a
    # solver ran, matching the real pipeline rather than an artificial
    # pack=None that would separately flag [T1] as an unoffered citation.
    computed = {"ran": True, "result": physics.solve("particle-in-a-box", n=2, L=1e-9)}
    pack = EvidencePack(question="q", mathematical_objects=[
        {"tag": "T1", "name": "solver result", "expression": "", "kind": "computed_result",
        "topic_id": None}])
    reasoning = {"derivation_plan": "apply the particle-in-a-box energy formula for this "
                                    "configuration [T1] - the solver's computed result is the "
                                    "answer, referred to in words, not restated", "text": "x"}
    result = v.verify_derivation("energy of n=2 electron in a 1nm box", {}, pack, reasoning,
                                 computed, None)
    assert result.status == "verified_mathematically"
    assert not result.failed


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


# ── hydrogen-transition data contract: physics.py's solver result must
# expose a "formula" field like every other solver does, so
# extract_mathematical_objects() actually offers [T1] and a compliant
# citation of it is recognized as legitimate rather than flagged as
# fabricated. The fix lives entirely in physics.py's solver output - none
# of these tests special-case hydrogen-transition text in verification.py
# itself, which still only ever checks "was this tag actually offered".

def _hydrogen_transition_pack(n_i=3, n_f=2):
    computed = {"ran": True, "inputs": {"n_i": n_i, "n_f": n_f},
               "result": physics.solve("hydrogen-transition", n_i=n_i, n_f=n_f)}
    topics = [_topic("bohr-model")]
    objects = extract_mathematical_objects(topics, computed, None)
    pack = EvidencePack(question="q", topics=topics, mathematical_objects=objects,
                        computed=computed)
    return pack, computed


def test_hydrogen_transition_result_now_offers_a_legitimate_t1_object():
    _, computed = _hydrogen_transition_pack()
    assert computed["result"]["formula"], "hydrogen-transition must expose a formula like every other solver"
    objects = extract_mathematical_objects([], computed, None)
    assert any(o["tag"] == "T1" for o in objects)


def test_hydrogen_transition_legitimate_t1_citation_is_not_falsely_failed():
    # the exact benchmark-confirmed false negative this fix closes:
    # solve-hydrogen-transition-derive-and-calculate.
    pack, computed = _hydrogen_transition_pack()
    reasoning = {"derivation_plan": (
        "the Rydberg formula gives the transition energy [C:bohr-model]\n"
        "the solver's computed result [T1] gives the wavelength, referred to in words")}
    result = v.verify_derivation("hydrogen transition", {}, pack, reasoning, computed, None)
    assert result.status == "verified_mathematically"
    assert not result.failed
    assert any(c["check"] == "symbol_consistency" for c in result.passed)


def test_hydrogen_transition_genuinely_fabricated_citation_still_detected():
    # legitimizing [T1] must not make symbol_consistency toothless - a tag
    # that names nothing real is still caught.
    pack, computed = _hydrogen_transition_pack()
    reasoning = {"derivation_plan": (
        "the Rydberg formula gives the transition energy [C:bohr-model]\n"
        "the solver's computed result [T2] gives the wavelength, referred to in words")}
    result = v.verify_derivation("hydrogen transition", {}, pack, reasoning, computed, None)
    assert result.status == "failed"
    assert any(f["check"] == "symbol_consistency" for f in result.failed)
    assert any("T2" in f["detail"] for f in result.failed)


# ── release-readiness Phase 5 (adversarial correctness): documented, real
# limitations found by testing claim types the original 6 planted-error
# benchmark problems never targeted. These are NOT bugs fixed here - fixing
# them would mean teaching boundary_conditions/limiting_case/classical_limit/
# conservation_law to parse and compare against the derivation's OWN stated
# claim, which none of them do today; they each independently RE-DERIVE one
# fixed, already-true fact (e.g. "psi vanishes at the well's walls") and
# report pass/fail on THAT, regardless of what the surrounding prose
# actually asserts. That is real, load-bearing verification (the fact
# itself is genuinely checked, not asserted) - it is just not the same as
# checking the DERIVATION'S claim. These tests exist so this gap is a known,
# tracked property of the system rather than a silent assumption.

def test_known_limitation_boundary_conditions_check_ignores_a_contradicting_claim():
    computed = {"ran": True, "result": physics.solve("particle-in-a-box", n=2, L=1e-9)}
    pack = EvidencePack(question="q", topics=[_topic("particle-in-a-box")])
    wrong_claim = ("the wavefunction does NOT need to vanish at the walls of the infinite "
                  "well - psi(0) and psi(L) can be nonzero since the box is only "
                  "approximately infinite")
    reasoning = {"derivation_plan": wrong_claim, "text": wrong_claim}
    result = v.verify_derivation("q", {}, pack, reasoning, computed, None)
    # Documents the actual (limited) behavior: boundary_conditions independently
    # reconfirms psi(0)=psi(L)=0 - a true fact - without ever reading that the
    # text asserted the opposite, so the false prose claim is not caught.
    assert result.status == "verified_mathematically"
    bc = next(c for c in result.checks if c["name"] == "boundary_conditions")
    assert bc["status"] == "pass"


def test_known_limitation_conservation_law_check_ignores_a_contradicting_claim():
    pack = EvidencePack(question="q", topics=[_topic("harmonic-oscillator")])
    wrong_claim = ("energy is NOT conserved for the classical harmonic oscillator - it decays "
                  "over each cycle due to the restoring force")
    reasoning = {"derivation_plan": wrong_claim, "text": wrong_claim}
    result = v.verify_derivation("q", {}, pack, reasoning, None, None)
    assert result.status == "verified_mathematically"
    cl = next(c for c in result.checks if c["name"] == "conservation_law")
    assert cl["status"] == "pass"


def test_known_limitation_physical_interpretation_correctness_is_not_checked():
    # No check evaluates the PHYSICAL INTERPRETATION prose's semantic
    # correctness (doing so would need an LLM judgement call, out of scope
    # for this deterministic layer) - a correct citation and correct number
    # paired with a backwards physical claim ("tighter confinement lowers
    # the energy", the opposite of the truth) still verifies.
    computed = {"ran": True, "result": physics.solve("particle-in-a-box", n=2, L=1e-9)}
    pack = EvidencePack(question="q", topics=[_topic("particle-in-a-box")],
                        mathematical_objects=[{"tag": "T1", "name": "computed relation",
                                              "expression": "E_n = n^2 pi^2 hbar^2/(2mL^2)",
                                              "kind": "computed_result", "topic_id": None}])
    backwards = ("the solver's computed result [T1] gives the energy; a SMALLER box would "
                "give a LOWER confinement energy, so tighter confinement always reduces the "
                "energy level")
    reasoning = {"derivation_plan": backwards, "text": backwards}
    result = v.verify_derivation("q", {}, pack, reasoning, computed, None)
    assert result.status == "verified_mathematically"


def test_hydrogen_transition_incorrect_mathematics_still_detected():
    # a legitimate [T1] citation does not exempt the derivation from having
    # to state the RIGHT number - the real answer for n=3->n=2 is
    # 656.1123 nm, not 500.
    pack, computed = _hydrogen_transition_pack()
    reasoning = {"derivation_plan": (
        "the Rydberg formula gives the transition energy [C:bohr-model]\n"
        "the solver's computed result [T1] gives a wavelength of 500 nm")}
    result = v.verify_derivation("hydrogen transition", {}, pack, reasoning, computed, None)
    assert result.status != "verified_mathematically"
    assert any(f["check"] == "known_result" for f in result.failed)
