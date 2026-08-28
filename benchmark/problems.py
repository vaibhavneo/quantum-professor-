"""The 42-problem Physics Reasoning Benchmark data set.

Every question here was checked against the REAL, unmocked match_topics()
and compute_for() before being written down (see the session's diagnostic
pass) - "expect_*" fields state what the current system actually does
today, not what an idealized system should do. Several of them deliberately
capture a gap (empty topic match, a solver that never runs, a check that's
narrower than it looks) precisely because the point of this benchmark is to
surface those gaps, not to hand-pick only the questions that already work.

Only the LLM boundary is scripted (understand / reasoning / professor
replies) - retrieval, topic matching, the solver, and verification all run
for real against this repo's actual data and code.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Problem:
    id: str
    category: str          # conceptual | derivation | operator_eigenvalue | multi_step |
                           # classical_limit | conservation_law | quantum_computing |
                           # incorrect_derivation
    domain: str            # classical-mechanics | lagrangian-hamiltonian | quantum-mechanics |
                           # quantum-statistical-mechanics | qft | quantum-computing | math-physics
    question: str
    derivation_reply: str  # scripted "DERIVATION PLAN / PHYSICAL INTERPRETATION" LLM reply
    understand_json: str = "{}"          # scripted understand-stage JSON reply
    is_wrong: bool = False                # a deliberately incorrect derivation
    expected_status: tuple = ()           # any status in this tuple counts as "as expected"
    targeted_check: str | None = None     # for is_wrong problems: the check name that should
                                          # appear in verification["failed"] if the error is caught
    expect_topics_matched: bool | None = None   # None = not asserted
    expect_solver_ran: bool | None = None       # None = not asserted
    notes: str = ""


_GENERIC_CONCEPTUAL_REPLY = (
    "DERIVATION PLAN\n"
    "- this is a conceptual question with no closed-form derivation required\n\n"
    "PHYSICAL INTERPRETATION\n"
    "- {gist}")


def _concept(gist: str) -> str:
    return _GENERIC_CONCEPTUAL_REPLY.format(gist=gist)


PROBLEMS: list[Problem] = [

    # ── 1. Conceptual questions ─────────────────────────────────────────────
    Problem(
        id="concept-uncertainty-principle", category="conceptual", domain="quantum-mechanics",
        question="What is the Heisenberg uncertainty principle?",
        derivation_reply=_concept("position and momentum cannot both be known to arbitrary "
                                  "precision simultaneously"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="FIXED (was a confirmed weakness): matches the real curriculum "
             "(uncertainty-principle, commutators), but the reply makes zero mathematical claim. "
             "operator_consistency previously fired anyway from the matched topic TITLE's "
             "'commutator' keyword and passed a fixed reference fact. It now requires an actual "
             "parseable claim (e.g. '[x,p] = ...') in the text - a pure-prose concept answer "
             "correctly reports not_independently_verified instead of overclaiming rigor.",
    ),
    Problem(
        id="concept-wavefunction-collapse", category="conceptual", domain="quantum-mechanics",
        question="What is the physical meaning of wavefunction collapse?",
        derivation_reply=_concept("measurement selects one eigenstate from the superposition"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="GAP: matches zero curriculum topics despite being a core QM concept the "
             "curriculum's measurement-postulates topic clearly covers.",
    ),
    Problem(
        id="concept-qubit-vs-bit", category="conceptual", domain="quantum-computing",
        question="What is a qubit and how does it differ from a classical bit?",
        derivation_reply=_concept("a qubit can be in a superposition of 0 and 1, a classical "
                                  "bit cannot"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="GAP (still present, unrelated to the fixes): the most basic possible "
             "quantum-computing question matches zero curriculum topics despite "
             "quantum-information existing - a retrieval gap, not a verification one. "
             "FIXED separately: operator_consistency previously fired purely because 'qubit' "
             "appeared in the question text and passed a fixed Pauli-algebra reference check "
             "regardless of content; it now requires an actual parseable operator claim, so a "
             "reply with no equation correctly reports not_independently_verified.",
    ),
    Problem(
        id="concept-schrodinger-vs-heisenberg-picture", category="conceptual", domain="math-physics",
        question="What is the difference between the Schrodinger picture and the Heisenberg picture?",
        derivation_reply=_concept("in the Schrodinger picture states evolve and operators are "
                                  "fixed; in the Heisenberg picture it's the reverse"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="GAP: no curriculum topic covers this standard graduate-level QM distinction.",
    ),
    Problem(
        id="concept-symmetry-breaking", category="conceptual", domain="qft",
        question="What is spontaneous symmetry breaking in the Standard Model?",
        derivation_reply=_concept("the vacuum state does not share the symmetry of the "
                                  "underlying Lagrangian"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="Matches standard-model topic.",
    ),
    Problem(
        id="concept-renormalization", category="conceptual", domain="qft",
        question="What is the physical meaning of renormalization in quantum field theory?",
        derivation_reply=_concept("infinities from high-energy virtual processes are absorbed "
                                  "into redefined physical parameters"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="Matches qft-fundamentals topic.",
    ),

    # ── 2. Mathematical derivations ──────────────────────────────────────────
    Problem(
        id="derive-qho-energy-levels", category="derivation", domain="quantum-mechanics",
        question="Derive the energy levels of the quantum harmonic oscillator and explain their "
                "physical meaning.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- start from the quantum harmonic oscillator Hamiltonian [C:harmonic-oscillator]\n"
            "- solving the Schrodinger equation for this potential quantizes the energy\n"
            "- result: E_n = hbar*omega*(n + 1/2) for n = 0, 1, 2, ...\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the ground state (n=0) still has nonzero zero-point energy\n"
            "- levels are evenly spaced by hbar*omega"),
        expected_status=("verified_mathematically",),
        targeted_check="symbol_consistency",
        expect_topics_matched=True,
        notes="Real citation of an offered tag + real SHM conservation-law check both pass "
             "(confirmed in the prior turn's full-chain integration test).",
    ),
    Problem(
        id="derive-tise-from-tdse", category="derivation", domain="quantum-mechanics",
        question="Derive the time-independent Schrodinger equation from the time-dependent "
                "Schrodinger equation.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- start from the time-dependent Schrodinger equation [C:schrodinger-equation]\n"
            "- separate variables: Psi(x,t) = psi(x) f(t)\n"
            "- this yields the time-independent equation for stationary states\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- stationary states have time-independent probability density"),
        expected_status=("verified_mathematically", "not_independently_verified"),
        expect_topics_matched=True,
        notes="symbol_consistency should pass on the real citation; nothing numeric to check "
             "beyond that, so verified_mathematically or an honest abstention are both fine.",
    ),
    Problem(
        id="derive-particle-in-box-symbolic", category="derivation", domain="quantum-mechanics",
        question="Derive the energy levels of a particle in an infinite square well.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- impose psi(0) = psi(L) = 0 on the infinite square well [C:particle-in-a-box]\n"
            "- the allowed wavefunctions are psi_n(x) = sin(n*pi*x/L)\n"
            "- result: E_n = n^2 pi^2 hbar^2 / (2 m L^2)\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- confinement forces discrete, quantized energy levels"),
        expected_status=("verified_mathematically",),
        targeted_check="boundary_conditions",
        expect_topics_matched=True,
        notes="Real boundary-condition check (psi(0)=psi(L)=0) fires and passes for this topic.",
    ),
    Problem(
        id="derive-hamiltons-equations", category="derivation", domain="lagrangian-hamiltonian",
        question="Derive Hamilton's equations from the Lagrangian.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- define H = sum(p_i * qdot_i) - L via the Legendre transform\n"
            "- taking partial derivatives gives qdot_i = dH/dp_i and pdot_i = -dH/dq_i\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- Hamilton's equations describe the same dynamics as the Euler-Lagrange equations, "
            "in phase space instead of configuration space"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="FIXED (the headline weakness this pass targeted): GAP (still present) - the "
             "curriculum has zero classical/Lagrangian-Hamiltonian mechanics topics at all. "
             "conservation_law previously fired on the word 'Lagrangian' alone and "
             "unconditionally verified a FIXED, unrelated claim (classical SHM energy "
             "conservation) that has nothing to do with Hamilton's equations. It now requires "
             "the primary topic to genuinely be harmonic-oscillator, or the text to both name "
             "SHM specifically AND claim conservation - neither holds here, so this correctly "
             "reports not_independently_verified instead of a false 'verified'.",
    ),
    Problem(
        id="derive-canonical-commutator", category="derivation", domain="math-physics",
        question="Derive the commutator of the position and momentum operators.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- acting on a test wavefunction, [x,p]psi = x(-i hbar psi') - (-i hbar (x psi))'\n"
            "- this simplifies to i hbar psi\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the nonzero commutator [x,p] = i*hbar is the algebraic root of the uncertainty "
            "principle"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="Under the fixed, claim-parsing operator_consistency this now honestly reports "
             "not_independently_verified rather than the old keyword-triggered false pass: the "
             "actual '[x,p] = i*hbar' restatement lands in the PHYSICAL INTERPRETATION half of "
             "the reply, which verify_derivation() never sees (only derivation_plan is checked, "
             "by design - verification sits BEFORE the interpretation stage). The DERIVATION "
             "PLAN half phrases the claim as '[x,p]psi = ...(expansion)...simplifies to i hbar "
             "psi', which doesn't match the bounded '[x,p] = <rhs>' grammar this check safely "
             "parses. A legitimate claim, honestly not verified rather than guessed at.",
    ),
    Problem(
        id="derive-two-level-partition-function", category="derivation",
        domain="quantum-statistical-mechanics",
        question="Derive the partition function for a two level quantum system.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- for two states of energy 0 and epsilon, Z = sum over states of e^(-beta E) "
            "[C:quantum-statistical-mechanics]\n"
            "- result: Z = 1 + e^(-beta*epsilon)\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- at low temperature Z -> 1 (ground state only); at high temperature both states "
            "become equally populated"),
        expected_status=("verified_mathematically", "not_independently_verified"),
        expect_topics_matched=True,
    ),

    # ── 3. Operator / eigenvalue problems ────────────────────────────────────
    Problem(
        id="operator-pauli-z-eigenvalues", category="operator_eigenvalue", domain="quantum-computing",
        question="Find the eigenvalues of the Pauli-Z operator.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- Pauli-Z is diagonal: Z = diag(1, -1) [C:operators-observables]\n"
            "- the eigenvalues of a diagonal matrix are its diagonal entries\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the eigenvalues +1 and -1 correspond to the computational basis states |0> and |1>"),
        expected_status=("verified_mathematically", "not_independently_verified"),
        expect_topics_matched=True,
    ),
    Problem(
        id="operator-hamiltonian-hermitian", category="operator_eigenvalue", domain="quantum-mechanics",
        question="Show that the Hamiltonian operator is Hermitian and its eigenvalues are real.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- a Hermitian operator satisfies H = H-dagger [C:operators-observables]\n"
            "- for any Hermitian operator, <psi|H|psi> is real, so its eigenvalues are real\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- physical observables must be Hermitian so that measured values are real numbers"),
        expected_status=("verified_mathematically", "not_independently_verified"),
        expect_topics_matched=True,
    ),
    Problem(
        id="operator-number-operator-eigenstates", category="operator_eigenvalue",
        domain="quantum-mechanics",
        question="What are the eigenstates of the number operator for the quantum harmonic "
                "oscillator?",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the number operator is N = a-dagger*a [C:harmonic-oscillator]\n"
            "- its eigenstates are the Fock states |n> with eigenvalue n\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- n counts the number of quanta of excitation in the oscillator"),
        expected_status=("verified_mathematically", "not_independently_verified"),
        expect_topics_matched=True,
    ),
    Problem(
        id="operator-compute-xp-commutator", category="operator_eigenvalue", domain="math-physics",
        question="Compute the commutator of the x and p operators.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- [x,p] = xp - px\n"
            "- evaluating on a test function gives [x,p] = i*hbar\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- x and p are incompatible observables and cannot be simultaneously diagonalized"),
        expected_status=("verified_mathematically",),
        targeted_check="operator_consistency",
        expect_topics_matched=False,
        notes="Same retrieval gap as derive-canonical-commutator, same text-triggered "
             "verification success.",
    ),
    Problem(
        id="operator-pauli-algebra-correct", category="operator_eigenvalue", domain="quantum-computing",
        question="Show that the Pauli matrices satisfy the algebra sigma_x sigma_y = i sigma_z.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- multiply the explicit 2x2 Pauli matrices sigma_x and sigma_y directly "
            "[C:spin-pauli]\n"
            "- the product works out to i*sigma_z\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this algebra underlies how single-qubit gates compose"),
        expected_status=("verified_mathematically",),
        targeted_check="operator_consistency",
        expect_topics_matched=True,
    ),

    # ── 4. Multi-step calculations ───────────────────────────────────────────
    Problem(
        id="calc-particle-in-box-n2-1nm", category="multi_step", domain="quantum-mechanics",
        question="What is the energy of an electron in the n=2 state of a 1nm box?",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- apply the particle-in-a-box energy formula for this configuration [T1]\n"
            "- the solver's computed result is the answer, referred to in words, not restated\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- doubling the quantum number quadruples the energy for this potential"),
        expected_status=("verified_mathematically",),
        expect_solver_ran=True,
        notes="FIXED (headline bug from the benchmark): this reply does exactly what the "
             "system's own prompt asks - cites [T1] and refers to the computed value in words, "
             "never restating the digit. known_result previously misread the citation tag's own "
             "'1' as a stated (and 'plausibly wrong') value; citation tags are now stripped "
             "before number extraction, so this correctly reaches verified_mathematically.",
    ),
    Problem(
        id="calc-hydrogen-ground-state", category="multi_step", domain="quantum-mechanics",
        question="What is the ground state energy of a hydrogen atom?",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- apply the Bohr/Schrodinger hydrogen ground-state energy formula [T1]\n"
            "- the solver's computed result is the answer, referred to in words, not restated\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the negative sign indicates a bound state"),
        expected_status=("verified_mathematically",),
        expect_solver_ran=True,
        notes="FIXED, same as calc-particle-in-box-n2-1nm: citing [T1] no longer makes "
             "known_result misread the tag's own '1' as the stated energy. Separately, "
             "match_topics() still returns 'harmonic-oscillator' as a secondary match here "
             "(weak coincidental overlap on 'energy'/'state'), but conservation_law now only "
             "fires from the PRIMARY topic, so that secondary match no longer contributes a "
             "pass either way - this reaches verified_mathematically on symbol_consistency alone.",
    ),
    Problem(
        id="calc-de-broglie-wavelength-generic", category="multi_step", domain="quantum-mechanics",
        question="What is the de Broglie wavelength of an electron with a given momentum?",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- lambda = h / p [C:de-broglie]\n"
            "- without a numeric momentum this stays symbolic\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- larger momentum means a shorter de Broglie wavelength"),
        expected_status=("verified_mathematically", "not_independently_verified"),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="GAP: topic matches, but the phrase 'a given momentum' carries no actual number, "
             "so the solver never runs - a realistic way a student might phrase this.",
    ),
    Problem(
        id="calc-hydrogen-transition-wavelength", category="multi_step", domain="quantum-mechanics",
        question="What is the wavelength of light emitted in the hydrogen n=3 to n=2 transition?",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- use the Rydberg/Bohr transition formula for n=3 to n=2\n"
            "- this gives the Balmer-alpha line\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this is the red line at the heart of the visible hydrogen spectrum"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="GAP: does not match bohr-model despite naming the exact transition it teaches - "
             "the hydrogen-transition solver never gets a chance to run.",
    ),
    Problem(
        id="calc-uncertainty-min-momentum", category="multi_step", domain="quantum-mechanics",
        question="What is the minimum uncertainty in momentum for an electron confined to a "
                "region of 1e-10 m?",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- apply the Heisenberg uncertainty relation delta_x * delta_p >= hbar/2\n"
            "- solve for the minimum delta_p given delta_x\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- tighter confinement in position forces a larger momentum uncertainty"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="GAP: scientific notation ('1e-10 m') is not recognized by retrieval/extraction "
             "even though the uncertainty-principle solver exists and could have run.",
    ),

    # ── 5. Classical-limit problems ──────────────────────────────────────────
    Problem(
        id="limit-qho-classical", category="classical_limit", domain="quantum-mechanics",
        question="Show that the quantum harmonic oscillator energy reduces to the classical "
                "result as hbar goes to zero.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- start from E_n = hbar*omega*(n+1/2) [C:harmonic-oscillator]\n"
            "- take the limit hbar -> 0\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the discreteness of energy levels vanishes, recovering the classical continuum"),
        expected_status=("not_independently_verified", "verified_mathematically"),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="GAP: check_classical_limit keys off computed's topic, not pack.topics - since no "
             "specific numbers were given, compute_for() never ran, so the curated hbar->0 check "
             "for this EXACT topic never actually fires despite the topic being matched.",
    ),
    Problem(
        id="limit-particle-in-box-large-n", category="classical_limit", domain="quantum-mechanics",
        question="Show that quantum particle in a box energy levels become continuous for large n.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- start from E_n = n^2 pi^2 hbar^2 / (2 m L^2) [C:particle-in-a-box]\n"
            "- consider the limit n -> infinity\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the relative spacing between adjacent levels shrinks, approaching a continuum"),
        expected_status=("not_independently_verified", "verified_mathematically"),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="Same gap as limit-qho-classical: check_limiting_case needs computed, not just a "
             "topic match.",
    ),
    Problem(
        id="limit-correspondence-principle", category="classical_limit", domain="quantum-mechanics",
        question="How does the correspondence principle relate quantum and classical mechanics?",
        derivation_reply=_concept("quantum predictions must reduce to classical ones in the "
                                  "limit of large quantum numbers"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="FIXED (same pattern as derive-hamiltons-equations): the phrase 'classical "
             "mechanics' previously triggered conservation_law to unconditionally verify a "
             "fixed, unrelated SHM fact. It now requires the primary topic to genuinely be "
             "harmonic-oscillator, or the text to name SHM specifically alongside a conservation "
             "claim - neither holds for a question about the correspondence principle, so this "
             "correctly reports not_independently_verified.",
    ),
    Problem(
        id="limit-fermi-dirac-to-boltzmann", category="classical_limit",
        domain="quantum-statistical-mechanics",
        question="Show that the Fermi-Dirac distribution reduces to the Maxwell-Boltzmann "
                "distribution in the classical limit.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- start from n_FD = 1/(e^(beta(eps-mu)) + 1) [C:quantum-statistics]\n"
            "- in the dilute/high-temperature limit the +1 becomes negligible\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this recovers the classical Maxwell-Boltzmann distribution"),
        expected_status=("not_independently_verified", "verified_mathematically"),
        expect_topics_matched=True,
        notes="GAP: classical_limit is curated ONLY for the harmonic-oscillator topic - a "
             "genuine, well-posed classical-limit claim for quantum-statistics has no check to "
             "run against it at all.",
    ),
    Problem(
        id="limit-ehrenfest-newton", category="classical_limit", domain="classical-mechanics",
        question="Show that the Ehrenfest theorem recovers Newton's second law in the classical "
                "limit.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- Ehrenfest's theorem gives d<p>/dt = -<dV/dx>\n"
            "- in the classical limit this becomes m d^2<x>/dt^2 = F(<x>)\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- expectation values obey classical equations of motion"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="match_topics() returns quantum-statistics for this classical-mechanics question "
             "(likely shared vocabulary like 'classical'/'quantum'/'theorem') - a loosely "
             "related match, not a real one; there is still no genuine classical-mechanics "
             "curriculum coverage, so verification correctly finds nothing to check regardless.",
    ),

    # ── 6. Conservation-law problems ─────────────────────────────────────────
    Problem(
        id="conservation-shm-hamiltonian", category="conservation_law", domain="classical-mechanics",
        question="Show that total energy is conserved for a classical harmonic oscillator using "
                "Hamiltonian mechanics.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- write E = p^2/2m + (1/2) k x^2 for the classical oscillator\n"
            "- differentiate along the equations of motion\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- dE/dt = 0 confirms energy conservation for this conservative system"),
        expected_status=("verified_mathematically",),
        targeted_check="conservation_law",
        expect_topics_matched=True,
    ),
    Problem(
        id="conservation-hamiltonian-mechanics-generic", category="conservation_law",
        domain="lagrangian-hamiltonian",
        question="Use Hamiltonian mechanics to show energy conservation for a conservative "
                "classical system.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- for a time-independent Hamiltonian, dH/dt = 0 along trajectories\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- H is conserved because it has no explicit time dependence"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="INTENTIONAL CONSEQUENCE of the fix, not a regression: this question asks about a "
             "GENERIC conservative system, not simple harmonic motion specifically. The only "
             "implemented conservation check is SHM-specific (verifies dE/dt=0 for "
             "E=1/2mv^2+1/2kx^2 with x=A*cos(wt+phi)) - it must not silently stand in for a "
             "general-conservative-system check that was never actually run. 'Hamiltonian "
             "mechanics' wording alone (without naming SHM) is no longer sufficient, so this "
             "honestly reports not_independently_verified rather than a claim the pipeline can't "
             "actually back.",
    ),
    Problem(
        id="conservation-isolated-system-momentum", category="conservation_law",
        domain="quantum-mechanics",
        question="Show that momentum is conserved in an isolated quantum system.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- for a translationally-invariant Hamiltonian, [H,p] = 0\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- by Noether's theorem this symmetry implies momentum conservation"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="GAP: conservation_law only implements classical SHM energy conservation - general "
             "quantum momentum conservation (a real, well-posed physics claim) has no check.",
    ),
    Problem(
        id="conservation-probability-unitarity", category="conservation_law",
        domain="quantum-mechanics",
        question="Show that probability is conserved under time evolution governed by the "
                "Schrodinger equation.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- define the probability current J and show d(rho)/dt + dJ/dx = 0 [C:schrodinger-equation]\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this continuity equation is the QM statement of unitarity"),
        expected_status=("verified_mathematically", "not_independently_verified"),
        expect_topics_matched=True,
        notes="GAP: symbol_consistency may pass on the real citation, but conservation_law "
             "itself has no check for probability/unitarity conservation - only classical SHM "
             "energy conservation is covered.",
    ),

    # ── 7. Quantum-computing problems ────────────────────────────────────────
    Problem(
        id="qc-superposition-parallelism", category="quantum_computing", domain="quantum-computing",
        question="Explain superposition and how it enables quantum parallelism.",
        derivation_reply=_concept("a register of n qubits in superposition represents 2^n "
                                  "basis states at once"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="GAP: another basic quantum-computing concept with zero curriculum topic match.",
    ),
    Problem(
        id="qc-bloch-sphere", category="quantum_computing", domain="quantum-computing",
        question="What is the Bloch sphere representation of a qubit?",
        derivation_reply=_concept("any single-qubit pure state maps to a point on the unit "
                                  "sphere"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="FIXED, same as concept-qubit-vs-bit: 'qubit' in the question text alone no longer "
             "triggers a pass with no actual claim to verify. The zero-curriculum-match gap is "
             "unrelated and remains.",
    ),
    Problem(
        id="qc-hadamard-involution", category="quantum_computing", domain="quantum-computing",
        question="Show that the Hadamard gate applied twice returns the original state.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- H = (1/sqrt(2)) * [[1,1],[1,-1]]\n"
            "- computing H*H directly gives the identity matrix\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- H is its own inverse (an involution), so applying it twice is a no-op"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        notes="GAP (compound): no topic match AND operator_consistency only covers the "
             "canonical commutator and Pauli algebra specifically - Hadamard-gate algebra has no "
             "check even if it were matched.",
    ),
    Problem(
        id="qc-entanglement-no-signaling", category="quantum_computing", domain="quantum-computing",
        question="What is quantum entanglement and why can't it be used for faster-than-light "
                "communication?",
        derivation_reply=_concept("measurement correlations are real, but no controllable "
                                  "signal can be sent this way (the no-communication theorem)"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="FIXED: matches quantum-information, whose curriculum key_concepts list includes "
             "'qubits' - that curriculum-metadata word alone previously triggered a fixed "
             "Pauli-algebra pass for a question that never mentions a qubit or a matrix. Now "
             "requires an actual parseable claim, which this reply (correctly) never makes.",
    ),
    Problem(
        id="qc-cnot-truth-table", category="quantum_computing", domain="quantum-computing",
        question="Derive the truth table of the CNOT gate and explain its role in creating "
                "entanglement.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- CNOT flips the target qubit iff the control qubit is |1> [C:quantum-information]\n"
            "- applying it to a superposed control produces an entangled Bell-like state\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- CNOT plus single-qubit gates form a universal gate set"),
        expected_status=("verified_mathematically", "not_independently_verified"),
        expect_topics_matched=True,
        notes="Topic matches and the citation is real, but no check verifies gate-level claims "
             "like a truth table - a real citation is the most this can earn credit for.",
    ),

    # ── 8. Deliberately incorrect derivations ────────────────────────────────
    Problem(
        id="wrong-particle-in-box-numeric", category="incorrect_derivation",
        domain="quantum-mechanics",
        question="What is the energy of an electron in the n=2 state of a 1nm box?",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- apply the particle-in-a-box energy formula for this configuration\n"
            "- the result comes out to approximately 5.0 eV\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- confinement quantizes the allowed energies"),
        is_wrong=True,
        expected_status=("partially_verified",),
        targeted_check="known_result",
        notes="DETECTED: known_result fails (real value is 1.504121 eV) even though "
             "boundary_conditions/dimensional_consistency/limiting_case still genuinely pass for "
             "this topic, landing on partially_verified rather than a clean 'failed'.",
    ),
    Problem(
        id="wrong-hydrogen-numeric-no-unit", category="incorrect_derivation",
        domain="quantum-mechanics",
        question="What is the ground state energy of a hydrogen atom?",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- apply the hydrogen energy formula for the ground state\n"
            "- the result comes out to about -1.2\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the negative sign indicates a bound state"),
        is_wrong=True,
        expected_status=("failed",),
        targeted_check="known_result",
        notes="FIXED to a cleaner, more decisive outcome: known_result correctly fails "
             "(unchanged), and conservation_law no longer fires from harmonic-oscillator's "
             "spurious secondary topic match (it now requires that to be the PRIMARY match), so "
             "nothing masks the error anymore - this now lands on the honest, decisive FAILED "
             "instead of a partially_verified propped up by unrelated evidence.",
    ),
    Problem(
        id="wrong-trig-identity", category="incorrect_derivation", domain="math-physics",
        question="Verify the identity sin(theta)^2 + cos(theta)^2 = 2 used in this derivation.",
        understand_json='{"needs_symbolic": true, "identity": "sin(theta)**2 + cos(theta)**2 = 2"}',
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the Pythagorean identity is asserted as sin^2(theta) + cos^2(theta) = 2\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this identity is used to simplify a wave-mechanics normalization integral"),
        is_wrong=True,
        expected_status=("partially_verified", "failed"),
        targeted_check="algebraic_consistency",
        notes="DETECTED: real sympy simplification of sin(theta)**2+cos(theta)**2-2 is nonzero, "
             "so algebraic_consistency fails - a genuine false mathematical identity is caught.",
    ),
    Problem(
        id="wrong-fabricated-citation", category="incorrect_derivation", domain="quantum-mechanics",
        question="Derive the energy levels of the quantum harmonic oscillator and explain their "
                "physical meaning.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- start from the result established in [C:not-a-real-topic]\n"
            "- result: E_n = hbar*omega*(n + 1/2)\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the ground state still has nonzero zero-point energy"),
        is_wrong=True,
        expected_status=("partially_verified", "failed"),
        targeted_check="symbol_consistency",
        notes="DETECTED: the cited tag was never offered, so symbol_consistency fails even "
             "though the physics claimed happens to be correct - citation honesty is checked "
             "independently of numeric correctness.",
    ),
    Problem(
        id="wrong-dimensional-unit", category="incorrect_derivation", domain="quantum-mechanics",
        question="What is the energy of an electron in the n=2 state of a 1nm box?",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- apply the particle-in-a-box energy formula for this configuration\n"
            "- the result comes out to approximately 5.0 meters\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- confinement quantizes the allowed energies"),
        is_wrong=True,
        expected_status=("partially_verified", "failed"),
        targeted_check="dimensional_consistency",
        notes="DETECTED (regression guard for last turn's word-boundary fix): stating an energy "
             "in meters is a different quantity kind entirely, caught by dimensional_consistency.",
    ),
    Problem(
        id="wrong-pauli-algebra-claim", category="incorrect_derivation", domain="quantum-computing",
        question="Show that the Pauli matrices satisfy the algebra sigma_x sigma_y = i sigma_z.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- multiply the explicit 2x2 Pauli matrices sigma_x and sigma_y directly "
            "[C:spin-pauli]\n"
            "- the product works out to sigma_z\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this algebra underlies how single-qubit gates compose"),
        is_wrong=True,
        expected_status=("partially_verified", "failed"),
        targeted_check="operator_consistency",
        notes="FIXED (was the confirmed false positive this hardening pass targeted): the "
             "derivation drops the i factor (sigma_x*sigma_y = sigma_z is false; the true "
             "relation is i*sigma_z). operator_consistency now parses the specific claim (across "
             "the 'multiply X and Y ... the product works out to Z' prose pattern) and verifies "
             "it by direct matrix multiplication, correctly failing it. symbol_consistency still "
             "passes (the [C:spin-pauli] citation is genuinely real), so the honest overall "
             "outcome is partially_verified - detected, not silently passed.",
    ),
]


assert len({p.id for p in PROBLEMS}) == len(PROBLEMS), "duplicate problem id"
