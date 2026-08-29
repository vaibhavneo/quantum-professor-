"""Offline tests for deterministic_execution.py - the Deterministic
Execution layer between Derivation and Verification. Every check here is
real sympy/Python - no LLM, no network. Reuses verification.py's own
Pauli/commutator parsing directly (imported, not reimplemented), so these
tests focus on what THIS module adds: consolidation, and unit conversion.
"""
import physics
from quantum_prof import deterministic_execution as de
from quantum_prof import research as R


# ── unit conversion (the one genuinely new capability) ──────────────────────

def test_convert_units_ev_to_joules():
    r = de.convert_units(1.504121, "eV")
    assert r["to_unit"] == "J"
    assert abs(r["to_value"] - 2.409867520908714e-19) < 1e-25


def test_convert_units_nm_to_metres():
    r = de.convert_units(1.0, "nm")
    assert r == {"from_value": 1.0, "from_unit": "nm", "to_value": 1e-9, "to_unit": "m"}


def test_convert_units_unrecognized_unit_returns_none_not_a_guess():
    assert de.convert_units(5.0, "furlongs") is None


# ── real solver outputs: unit conversions present or honestly absent ───────

def test_result_unit_conversions_particle_in_a_box_shows_both_pairs():
    computed = {"ran": True, "inputs": {"n": 2, "L": 1e-9},
               "result": physics.solve("particle-in-a-box", n=2, L=1e-9)}
    convs = de._result_unit_conversions(computed)
    units_seen = {(c["from_unit"], c["to_unit"]) for c in convs}
    assert ("J", "eV") in units_seen
    assert ("m", "nm") in units_seen


def test_result_unit_conversions_de_broglie_shows_metre_to_picometre():
    computed = {"ran": True, "inputs": {"mass_kg": 9.109e-31, "speed_m_s": 1e6},
               "result": physics.solve("de-broglie", mass_kg=9.109e-31, speed_m_s=1e6)}
    convs = de._result_unit_conversions(computed)
    assert any(c["from_unit"] == "m" and c["to_unit"] == "pm" for c in convs)


def test_result_unit_conversions_honestly_empty_when_no_pair_exists():
    # uncertainty-principle's result has no same-quantity unit pair anywhere -
    # must return [] rather than fabricate one.
    computed = {"ran": True, "inputs": {"delta_x_m": 1e-10},
               "result": physics.solve("uncertainty-principle", delta_x_m=1e-10)}
    assert de._result_unit_conversions(computed) == []


# ── matrix operations (reuses verification.py's own parsing + sympy) ───────

def test_execute_matrix_claim_correct_pauli_product():
    r = de.execute_matrix_claim("multiplying directly, sigma_x sigma_y = i*sigma_z.")
    assert r["kind"] == "pauli_product"
    assert r["claim"] == "sigma_x*sigma_y = i*sigma_z"
    assert r["computed"] is not None


def test_execute_matrix_claim_reports_the_real_computation_even_for_a_wrong_claim():
    # execution is not judgement - it reports what was actually computed
    # regardless of whether the derivation's claim matches it; VERIFICATION
    # is the stage that judges pass/fail, separately.
    r = de.execute_matrix_claim("multiplying directly, sigma_x sigma_y = sigma_z.")
    assert r["kind"] == "pauli_product"
    assert r["computed"] is not None


def test_execute_matrix_claim_canonical_commutator():
    r = de.execute_matrix_claim("direct computation gives [x,p] = i*hbar.")
    assert r["kind"] == "canonical_commutator"


def test_execute_matrix_claim_none_for_unrelated_prose():
    assert de.execute_matrix_claim("Wolfgang Pauli made many contributions to physics.") is None


def test_execute_matrix_claim_none_for_empty_text():
    assert de.execute_matrix_claim("") is None
    assert de.execute_matrix_claim(None) is None


# ── the orchestrator: execute_deterministically() ───────────────────────────

def test_execute_deterministically_numerical_calculation_present():
    computed = {"ran": True, "inputs": {"n": 2, "L": 1e-9},
               "result": physics.solve("particle-in-a-box", n=2, L=1e-9)}
    reasoning = {"derivation_plan": "some prose", "text": "some prose"}
    rec = de.execute_deterministically("q", {}, None, reasoning, computed, None)
    assert rec["numerical_calculation"]["result"]["energy_eV"] == 1.504121
    assert rec["numerical_calculation"]["inputs"] == {"n": 2, "L": 1e-9}
    assert len(rec["unit_conversions"]) >= 1


def test_execute_deterministically_symbolic_algebra_present():
    u = {"identity": "sin(x)**2 + cos(x)**2 = 1"}
    reasoning = {"derivation_plan": "", "text": ""}
    rec = de.execute_deterministically("q", u, None, reasoning, None, None)
    assert rec["symbolic_algebra"]["equal"] is True


def test_execute_deterministically_reuses_already_computed_symbolic_result():
    # must not re-derive when route()'s own sympy stage already ran -
    # the same "compute once, reuse" discipline check_algebraic_consistency
    # already follows.
    symbolic = {"ok": True, "equal": True, "lhs": "1", "rhs": "1", "verdict": "identity holds"}
    rec = de.execute_deterministically("q", {"identity": "irrelevant"}, None,
                                       {"derivation_plan": "", "text": ""}, None, symbolic)
    assert rec["symbolic_algebra"]["verdict"] == "identity holds"


def test_execute_deterministically_matrix_operations_present():
    reasoning = {"derivation_plan": "sigma_x sigma_y = i*sigma_z", "text": "x"}
    rec = de.execute_deterministically("q", {}, None, reasoning, None, None)
    assert rec["matrix_operations"]["kind"] == "pauli_product"


def test_execute_deterministically_all_none_for_pure_concept_question():
    reasoning = {"derivation_plan": "just prose, no equations or numbers", "text": "x"}
    rec = de.execute_deterministically("what is a photon", {}, None, reasoning, None, None)
    assert rec == {"numerical_calculation": None, "symbolic_algebra": None,
                   "matrix_operations": None, "unit_conversions": [],
                   "differential_equation": None}


def test_execute_deterministically_never_raises_on_malformed_input():
    rec = de.execute_deterministically("q", {}, None, {}, None, None)
    assert rec["numerical_calculation"] is None
    rec2 = de.execute_deterministically("q", {}, None, None, None, None)
    assert rec2["numerical_calculation"] is None


# ── general matrix operations: product and eigenvalues (new "kind" values) ──

def test_execute_matrix_literal_claim_computes_matrix_product():
    r = de.execute_matrix_literal_claim(
        "Compute the product of the matrices with rows [1,2],[3,4] and [0,1],[1,0].")
    assert r["kind"] == "matrix_product"
    assert r["computed"] == [[2, 1], [4, 3]]


def test_execute_matrix_literal_claim_computes_eigenvalues():
    r = de.execute_matrix_literal_claim(
        "Find the eigenvalues of the matrix with rows [2,1] and [1,2].")
    assert r["kind"] == "eigenvalues"
    assert sorted(r["computed"]) == ["1", "3"]


def test_execute_matrix_literal_claim_none_for_unrelated_prose():
    assert de.execute_matrix_literal_claim("Wolfgang Pauli made many contributions.") is None


def test_execute_matrix_literal_claim_ignores_citation_tags():
    # [T1]/[C:topic] must never be mistaken for a numeric matrix row - the
    # row regex requires 2+ comma-separated numbers, which a citation tag
    # (letters, no comma) can never satisfy.
    text = "the eigenvalue result [T1] confirms the claim [C:some-topic]"
    assert de.execute_matrix_literal_claim(text) is None


# ── single-qubit gate application and Born-rule probability ────────────────

def test_parse_ket_state_basis_state():
    sp = R._sympy()
    state = de.parse_ket_state(sp, "|0>")
    assert state.tolist() == [[1], [0]]


def test_parse_ket_state_equal_superposition():
    sp = R._sympy()
    state = de.parse_ket_state(sp, "(|0>+|1>)/sqrt(2)")
    probs = de.born_rule_probabilities(sp, state)
    assert probs[0] == sp.Rational(1, 2)
    assert probs[1] == sp.Rational(1, 2)


def test_parse_ket_state_none_for_unrecognized_text():
    sp = R._sympy()
    assert de.parse_ket_state(sp, "some unrelated text") is None


def test_apply_gate_hadamard_on_zero_state():
    sp = R._sympy()
    result = de.apply_gate(sp, "hadamard", de.parse_ket_state(sp, "|0>"))
    probs = de.born_rule_probabilities(sp, result)
    assert probs[0] == sp.Rational(1, 2)
    assert probs[1] == sp.Rational(1, 2)


def test_apply_gate_none_for_unsupported_gate():
    sp = R._sympy()
    assert de.apply_gate(sp, "cnot", de.parse_ket_state(sp, "|0>")) is None


def test_execute_quantum_state_claim_gate_application():
    r = de.execute_quantum_state_claim(
        "applying the Hadamard gate to |0> gives an equal superposition")
    assert r["kind"] == "gate_application"
    assert r["computed"]["probabilities"] == {"0": "1/2", "1": "1/2"}


def test_execute_quantum_state_claim_born_rule_without_named_gate():
    r = de.execute_quantum_state_claim(
        "the qubit state is (|0>+|1>)/sqrt(2); apply the Born rule")
    assert r["kind"] == "born_rule_probability"
    assert r["computed"] == {"0": "1/2", "1": "1/2"}


def test_execute_quantum_state_claim_none_for_unrelated_prose():
    assert de.execute_quantum_state_claim("Wolfgang Pauli made many contributions.") is None


# ── differential equation solving (new top-level field) ────────────────────

def test_solve_ode_exponential_decay():
    r = R.solve_ode("dy/dx = -k*y")
    assert r["ok"] is True
    assert r["solution"] == "Eq(y(x), C1*exp(-k*x))"


def test_solve_ode_prime_notation_matches_leibniz_notation():
    assert R.solve_ode("y' = -k*y")["solution"] == R.solve_ode("dy/dx = -k*y")["solution"]


def test_solve_ode_unparseable_input_reports_error_not_a_guess():
    r = R.solve_ode("not an equation at all")
    assert r["ok"] is False


def test_execute_differential_equation_claim_dy_dx_notation():
    r = de.execute_differential_equation_claim(
        "Solve the differential equation dy/dx = -k*y and verify the solution.", "")
    assert r["equation"] == "dy/dx = -k*y"
    assert r["solution"] == "Eq(y(x), C1*exp(-k*x))"


def test_execute_differential_equation_claim_prime_notation():
    r = de.execute_differential_equation_claim("Solve y' = -k*y and find the decay law.", "")
    assert r is not None
    assert "exp(-k*x)" in r["solution"]


def test_execute_differential_equation_claim_none_for_unrelated_prose():
    assert de.execute_differential_equation_claim("what is a photon", "just prose") is None


# ── orchestrator: the three new capabilities reach execute_deterministically ─

def test_execute_deterministically_matrix_literal_reaches_matrix_operations():
    q = "Compute the product of the matrices with rows [1,2],[3,4] and [0,1],[1,0]."
    reasoning = {"derivation_plan": "multiply the matrices directly", "text": "x"}
    rec = de.execute_deterministically(q, {}, None, reasoning, None, None)
    assert rec["matrix_operations"]["kind"] == "matrix_product"


def test_execute_deterministically_quantum_state_reaches_matrix_operations():
    q = "Derive the action of the Hadamard gate on the |0> state."
    reasoning = {"derivation_plan": "applying the Hadamard gate to |0>", "text": "x"}
    rec = de.execute_deterministically(q, {}, None, reasoning, None, None)
    assert rec["matrix_operations"]["kind"] == "gate_application"


def test_execute_deterministically_differential_equation_present():
    q = "Solve the differential equation dy/dx = -k*y."
    reasoning = {"derivation_plan": "separate variables and integrate", "text": "x"}
    rec = de.execute_deterministically(q, {}, None, reasoning, None, None)
    assert rec["differential_equation"]["equation"] == "dy/dx = -k*y"


def test_execute_deterministically_existing_pauli_claim_still_takes_priority():
    # execute_matrix_claim() (Pauli/commutator) must still win when its own
    # pattern matches, even though the new literal/quantum-state parsers now
    # also run on the same orchestrator call - no regression in precedence.
    reasoning = {"derivation_plan": "sigma_x sigma_y = i*sigma_z", "text": "x"}
    rec = de.execute_deterministically("q", {}, None, reasoning, None, None)
    assert rec["matrix_operations"]["kind"] == "pauli_product"
