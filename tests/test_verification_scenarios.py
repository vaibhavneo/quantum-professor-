"""Dedicated scenario suite for the seven named examples requested for the
Physics + Mathematical Verification Layer:

1. Quantum harmonic oscillator     5. Hamiltonian mechanics
2. Schrodinger equation            6. Qubit / Pauli matrices
3. Particle in a box               7. A deliberately incorrect derivation
4. Classical harmonic oscillator

Each test below calls verify_derivation() - the same orchestrator run()
calls in the real pipeline - directly, so it's obvious at a glance which
named example each test proves. The proof that this orchestrator's result
actually reaches the Professor's LLM prompt inside the real run() pipeline
(not just that it's callable) lives separately in test_qp_pipeline.py's
test_derivation_verification_reaches_the_professor_prompt_in_the_real_pipeline.

No network, no LLM, no DeepSeek balance required anywhere in this file -
every check is real sympy/deterministic Python running against real
physics.py solver output and real library.py curriculum data.
"""
import physics
from quantum_prof import verification as v
from quantum_prof.evidence_pack import EvidencePack
from quantum_prof.library import TOPICS


def _topic(tid):
    return TOPICS[tid]


# ── 1. Quantum harmonic oscillator ──────────────────────────────────────────

def test_scenario_1_quantum_harmonic_oscillator_correct_derivation_verifies():
    computed = {"ran": True, "result": physics.solve("harmonic-oscillator", n=0, omega=1e14)}
    pack = EvidencePack(question="what is the zero-point energy of a quantum harmonic oscillator",
                        topics=[_topic("harmonic-oscillator")])
    reasoning = {"derivation_plan": "E_0 = hbar*omega/2, which comes out to 0.032911 eV.", "text": "x"}
    result = v.verify_derivation("zero-point energy of a QHO", {}, pack, reasoning, computed, None)
    assert result.status == "verified_mathematically"
    assert any(p["check"] == "known_result" for p in result.passed)
    # a real, curated large-n limiting-case check also fires for this topic
    assert any(p["check"] == "limiting_case" for p in result.passed)


# ── 2. Schrodinger equation ─────────────────────────────────────────────────

def test_scenario_2_schrodinger_equation_citation_and_identity_are_checked():
    pack = EvidencePack(question="derive the time-independent Schrodinger equation",
                        topics=[_topic("schrodinger-equation")])
    derivation = "starting from [C:schrodinger-equation] we separate variables..."
    symbol_check = v.check_symbol_consistency(derivation, pack)
    assert symbol_check.status == "pass"
    # a real sympy identity check for an algebraic step in the same derivation
    algebra_check = v.check_algebraic_consistency({"identity": "sin(x)**2 + cos(x)**2 = 1"}, None)
    assert algebra_check.status == "pass"


# ── 3. Particle in a box ────────────────────────────────────────────────────

def test_scenario_3_particle_in_a_box_boundary_conditions_and_known_result():
    computed = {"ran": True, "result": physics.solve("particle-in-a-box", n=2, L=1e-9)}
    assert computed["result"]["energy_eV"] == 1.504121
    boundary = v.check_boundary_conditions(None, computed)
    assert boundary.status == "pass"
    correct = v.check_known_result("the n=2 energy level is 1.504121 eV", computed)
    assert correct.status == "pass"


# ── 4. Classical harmonic oscillator ────────────────────────────────────────

def test_scenario_4_classical_harmonic_oscillator_conservation_and_classical_limit():
    pack = EvidencePack(question="analyze the total energy of a classical harmonic oscillator "
                                 "using simple harmonic motion",
                        topics=[_topic("harmonic-oscillator")])
    conservation = v.check_conservation_law(pack)
    assert conservation.status == "pass"
    assert "dE/dt = 0" in conservation.detail
    # the quantum-side curriculum topic's own hbar -> 0 classical limit
    computed = {"ran": True, "result": {"topic": "harmonic-oscillator"}}
    classical_limit = v.check_classical_limit(computed)
    assert classical_limit.status == "pass"
    assert "hbar -> 0" in classical_limit.detail


# ── 5. Hamiltonian mechanics ─────────────────────────────────────────────────

def test_scenario_5_hamiltonian_mechanics_has_no_curriculum_topic_but_still_verifies():
    # Hamiltonian/classical mechanics has NO curriculum topic in library.py at
    # all (confirmed during the question-driven-retrieval work) - the
    # conservation-law check must still fire from the question's own text,
    # not merely from a topic id that doesn't exist for this domain.
    assert "hamiltonian-mechanics" not in TOPICS
    pack = EvidencePack(question="use Hamiltonian mechanics to show energy is conserved "
                                 "for a simple harmonic oscillator")
    result = v.check_conservation_law(pack)
    assert result.status == "pass"


# ── 6. Qubit / Pauli matrices ────────────────────────────────────────────────

def test_scenario_6_qubit_pauli_matrix_algebra_is_verified_by_real_sympy():
    pack = EvidencePack(question="explain qubits using the Pauli matrices")
    result = v.check_operator_consistency(pack)
    assert result.status == "pass"
    assert "Pauli" in result.detail  # sigma_i^2 = I and [sigma_x, sigma_y] = 2i*sigma_z


# ── 7. A deliberately incorrect derivation ──────────────────────────────────

def test_scenario_7_deliberately_incorrect_derivation_is_caught_not_silently_passed():
    computed = {"ran": True, "result": physics.solve("particle-in-a-box", n=2, L=1e-9)}
    wrong_reasoning = {"derivation_plan": "the n=2 energy level works out to approximately 5.0 eV.",
                       "text": "x"}
    result = v.verify_derivation("energy of n=2 electron in a 1nm box", {}, None,
                                 wrong_reasoning, computed, None)
    # particle-in-a-box has curated boundary/limiting-case checks that pass
    # independently of the wrong number, so this specific example lands on
    # PARTIALLY VERIFIED (something else genuinely corroborates the topic,
    # but the stated value is still wrong) rather than FAILED (nothing
    # corroborates it at all) - see
    # test_verify_derivation_pure_failure_with_nothing_corroborating_is_status_failed
    # in test_verification.py for that variant. Both are "detected, not
    # silently passed" - the point this scenario proves either way.
    assert result.status == "partially_verified"
    assert any(f["check"] == "known_result" for f in result.failed)
    assert result.corrections and "1.504121" in result.corrections[0]["correction"]
    # the full ledger is reported even though most checks are not_applicable
    # here - the verifier never hides a check behind a false pass.
    assert len(result.checks) == 9
