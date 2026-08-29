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
    expect_decomposition: bool | None = None    # None = not asserted; True = givens/unknowns/
                                                # assumptions all expected non-empty
    expect_unknowns_contains: tuple = ()        # substrings each expected somewhere in unknowns
    expect_strategy_contains: str | None = None # substring expected in the strategy sentence
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
        expect_topics_matched=True,
        notes="FIXED (retrieval phase): measurement-postulates now has 'wavefunction collapse "
             "upon measurement' as an explicit key_concept, so this correctly matches the "
             "topic that already covered this idea in prose.",
    ),
    Problem(
        id="concept-qubit-vs-bit", category="conceptual", domain="quantum-computing",
        question="What is a qubit and how does it differ from a classical bit?",
        derivation_reply=_concept("a qubit can be in a superposition of 0 and 1, a classical "
                                  "bit cannot"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="FIXED, both layers: (1) retrieval - quantum-information now has 'qubit vs "
             "classical bit' as an explicit key_concept, closing what was previously a zero-match "
             "retrieval gap. (2) verification - operator_consistency previously fired purely "
             "because 'qubit' appeared in the question text and passed a fixed Pauli-algebra "
             "reference check regardless of content; it now requires an actual parseable "
             "operator claim, so a reply with no equation correctly reports "
             "not_independently_verified.",
    ),
    Problem(
        id="concept-no-cloning-theorem", category="conceptual", domain="quantum-computing",
        question="What is the no-cloning theorem and why is it important in quantum computing?",
        derivation_reply=_concept("an unknown quantum state cannot be copied exactly, which is "
                                  "why quantum information cannot simply be duplicated like "
                                  "classical bits"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="NEW (release-readiness Phase 4): quantum-information's key_concepts already "
             "named 'no-cloning theorem' explicitly, but nothing in the benchmark exercised "
             "it - added to close that coverage gap. Matches quantum-information; "
             "not_independently_verified is correct since the scripted reply cites no tag.",
    ),
    Problem(
        id="concept-schrodinger-vs-heisenberg-picture", category="conceptual", domain="math-physics",
        question="What is the difference between the Schrodinger picture and the Heisenberg picture?",
        derivation_reply=_concept("in the Schrodinger picture states evolve and operators are "
                                  "fixed; in the Heisenberg picture it's the reverse"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="FIXED (retrieval phase, content gap): added a new curriculum topic, "
             "schrodinger-heisenberg-pictures, covering exactly this graduate-level QM "
             "distinction. Status stays not_independently_verified since this scripted reply "
             "makes no citable/checkable claim - the point here is retrieval grounding, not "
             "verification.",
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
    Problem(
        id="concept-klein-gordon-equation", category="conceptual", domain="qft",
        question="What is the Klein-Gordon equation and what does it describe?",
        derivation_reply=_concept("it is the relativistic wave equation for spin-0 particles, "
                                  "obtained by quantizing the relativistic energy-momentum "
                                  "relation"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="NEW (release-readiness Phase 4): relativistic-qm's key_concepts already named "
             "'Klein-Gordon equation' explicitly, but no benchmark problem exercised it - QFT "
             "coverage before this addition was only symmetry-breaking and renormalization.",
    ),
    Problem(
        id="concept-creation-annihilation-operators", category="conceptual", domain="qft",
        question="What are creation and annihilation operators in quantum field theory?",
        derivation_reply=_concept("they raise or lower the number of field quanta in a given "
                                  "mode, the ladder operators of the quantized field"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="NEW (release-readiness Phase 4): matches second-quantization primarily, with "
             "qft-fundamentals and harmonic-oscillator as legitimate secondary matches (the "
             "same ladder-operator formalism the QHO already uses) - closes an untested but "
             "real curriculum concept.",
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
        expect_topics_matched=True,
        notes="FIXED, both phases: (1) verification hardening - conservation_law previously "
             "fired on the word 'Lagrangian' alone and unconditionally verified a FIXED, "
             "unrelated claim (classical SHM energy conservation); it now requires the primary "
             "topic to genuinely be harmonic-oscillator or explicit SHM+conservation wording, "
             "neither of which holds here. (2) retrieval - a new curriculum topic "
             "(lagrangian-hamiltonian-mechanics, added for this content gap) now genuinely "
             "matches. Status stays not_independently_verified since this scripted reply doesn't "
             "cite that topic or state anything checkable - real grounding without a checkable "
             "claim is still honestly reported, not oversold.",
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
        expect_topics_matched=True,
        notes="Retrieval FIXED: commutators now has 'commutator of position and momentum "
             "operators' as an explicit key_concept, closing what was previously a zero-match "
             "gap (it primary-matches commutators, with uncertainty-principle as a related "
             "secondary). Verification status is unaffected by that - under the fixed, "
             "claim-parsing operator_consistency this now honestly reports "
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
        expect_topics_matched=True,
        notes="Retrieval FIXED (same as derive-canonical-commutator, via the same new "
             "commutators key_concept). Verification already succeeded here via text-parsing "
             "operator_consistency (this reply restates '[x,p] = i*hbar' cleanly within the "
             "Derivation Plan itself, unlike derive-canonical-commutator's awkward phrasing).",
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
        expect_decomposition=True,
        expect_unknowns_contains=("numeric value",),
        expect_strategy_contains="substitute",
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
        expect_decomposition=True,
        expect_unknowns_contains=("numeric value",),
        expect_strategy_contains="substitute",
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
        expect_topics_matched=True,
        notes="Retrieval PARTIALLY FIXED: bohr-model now has 'Balmer series transition "
             "wavelengths' as an explicit key_concept, so this correctly matches the topic. "
             "RESIDUAL GAP: the hydrogen-transition solver's extractor requires the literal "
             "word 'from' ('from n=3 to n=2') - 'n=3 to n=2' without 'from' still doesn't "
             "trigger it, so the solver still doesn't run for this exact phrasing. Retrieval "
             "grounding and solver triggering are two separate mechanisms.",
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
        expect_topics_matched=True,
        expect_solver_ran=True,
        expect_decomposition=True,
        expect_unknowns_contains=("numeric value",),
        expect_strategy_contains="substitute",
        notes="FULLY FIXED: the numeric extractor's regex now accepts scientific notation "
             "('1e-10 m'), and uncertainty-principle's key_concepts now include plain "
             "'position'/'momentum'/'minimum' tokens (previously only a hyphenated "
             "'position-momentum uncertainty' phrase, which doesn't tokenize the same as the "
             "separate words) so it wins the match over commutators, which shares 'uncertainty'. "
             "The solver now genuinely runs. Status stays not_independently_verified because "
             "this reply states no number to check known_result against - real computed "
             "evidence exists, but nothing in the text makes a checkable numeric claim.",
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
        notes="FIXED, both phases: (1) verification - the phrase 'classical mechanics' "
             "previously triggered conservation_law to unconditionally verify a fixed, unrelated "
             "SHM fact; it now requires the primary topic to genuinely be harmonic-oscillator or "
             "explicit SHM+conservation wording, neither of which holds here. (2) retrieval - "
             "this now primary-matches the new ehrenfest-correspondence-principle topic (added "
             "for a related content gap) - a genuine, precise match, replacing the previous "
             "loosely-related density-matrix/quantum-statistics matches.",
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
        notes="FIXED (content gap, then genuinely improved): previously matched "
             "quantum-statistics only loosely (shared vocabulary, not a real relationship). Now "
             "matches the new ehrenfest-correspondence-principle topic (added for this content "
             "gap) precisely - a real, dedicated match rather than a coincidental one. Status "
             "stays not_independently_verified since this scripted reply cites no tag and states "
             "nothing checkable.",
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
        expect_topics_matched=True,
        notes="FIXED (retrieval phase): quantum-information now has 'superposition and quantum "
             "parallelism' as an explicit key_concept.",
    ),
    Problem(
        id="qc-bloch-sphere", category="quantum_computing", domain="quantum-computing",
        question="What is the Bloch sphere representation of a qubit?",
        derivation_reply=_concept("any single-qubit pure state maps to a point on the unit "
                                  "sphere"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        notes="FIXED, both layers, same pattern as concept-qubit-vs-bit: retrieval now matches "
             "via quantum-information's new 'Bloch sphere representation' key_concept, and "
             "operator_consistency no longer passes from the bare word 'qubit' with no actual "
             "claim to verify.",
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
        expect_topics_matched=True,
        notes="Retrieval FIXED: quantum-information now has 'Hadamard gate' as an explicit "
             "key_concept. RESIDUAL GAP: operator_consistency only parses Pauli-product and "
             "canonical-commutator claims specifically - Hadamard-gate algebra has no check "
             "even now that the topic matches, so this correctly stays honest rather than "
             "guessing at a claim it can't verify.",
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
        expect_topics_matched=False,
        notes="DETECTED: real sympy simplification of sin(theta)**2+cos(theta)**2-2 is nonzero, "
             "so algebraic_consistency fails - a genuine false mathematical identity is caught. "
             "The only remaining unmatched question in the benchmark by design, not a gap: a "
             "generic trigonometric identity check has no dedicated curriculum topic to match, "
             "and doesn't need one - check_algebraic_consistency works directly from the "
             "identity string, independent of topic retrieval.",
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

    # ── Multi-step problem solving: a genuine "derive the general formula AND
    #    calculate a specific numeric value" combined ask - qualitatively
    #    different from the existing multi_step category's pure-numeric
    #    questions, which never populate the "general symbolic expression"
    #    half of decompose_problem()'s unknowns. Added because no existing
    #    problem exercised this combined pattern. ───────────────────────────
    Problem(
        id="solve-particle-in-box-derive-and-calculate", category="problem_solving",
        domain="quantum-mechanics",
        question="A particle is in a one-dimensional infinite potential well of width L. "
                "Derive the energy eigenvalues and calculate the ground-state energy for an "
                "electron when L = 1 nm.",
        derivation_reply=(
            "GIVEN / FIND / ASSUMPTIONS\n"
            "- given: the ground-state configuration of an electron in a narrow infinite well "
            "[T1]\n"
            "- find: the general energy eigenvalue formula and the numeric ground-state energy\n"
            "- assumptions: infinite potential walls, non-relativistic particle, strictly "
            "one-dimensional confinement\n\n"
            "DERIVATION PLAN\n"
            "- start from the infinite-square-well governing equation [C:particle-in-a-box]\n"
            "- solving the Schrodinger equation with these boundary conditions gives the "
            "quantized energy formula\n"
            "- substituting the given configuration, the solver's computed result [T1] is the "
            "ground-state answer, referred to in words rather than restated\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- confinement in a smaller well raises the energy scale\n"
            "- the ground state has nonzero energy, unlike a classical particle at rest"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=True,
        expect_decomposition=True,
        expect_unknowns_contains=("general symbolic expression", "numeric value"),
        expect_strategy_contains="substitute",
        notes="THE flagship example for this capability - this exact question was confirmed "
             "BROKEN before the multi-step phase: _level_n() only recognized 'ground state' "
             "(space), not the 'ground-state' (hyphen) used here, so compute_for() never ran "
             "and no numeric answer was ever actually computed. Now fully solved: real n=1, "
             "L=1nm, energy_eV=0.37603, with genuine Given/Find/Assumptions/Strategy reaching "
             "the reasoning prompt and a fully verified numeric result.",
    ),
    Problem(
        id="solve-hydrogen-ground-state-derive-and-calculate", category="problem_solving",
        domain="quantum-mechanics",
        question="Derive the general energy formula for the hydrogen atom and calculate the "
                "ground-state energy.",
        derivation_reply=(
            "GIVEN / FIND / ASSUMPTIONS\n"
            "- given: the ground-state configuration of the hydrogen atom [T1]\n"
            "- find: the general energy formula and the numeric ground-state energy\n"
            "- assumptions: the proton is treated as fixed, only the Coulomb interaction is "
            "included\n\n"
            "DERIVATION PLAN\n"
            "- solve the radial Schrodinger equation for the Coulomb potential "
            "[C:hydrogen-atom]\n"
            "- this gives the quantized Bohr-like energy formula\n"
            "- for the ground state, the solver's computed result [T1] is the answer, referred "
            "to in words rather than restated\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the negative sign indicates a bound state\n"
            "- higher-n states approach the ionization threshold"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=True,
        expect_decomposition=True,
        expect_unknowns_contains=("general symbolic expression", "numeric value"),
        expect_strategy_contains="substitute",
        notes="Second solver-backed topic exercising the same combined derive+calculate "
             "pattern, confirming decompose_problem()/solution_strategy() generalize beyond "
             "particle-in-a-box. Real n=1, energy_eV=-13.605693.",
    ),

    # ── Expanded problem-solving coverage (step 4): 19 more problems across
    #    classical mechanics, quantum mechanics, quantum computing, and
    #    mathematical physics, specifically to expose whether the
    #    Deterministic Execution layer generalizes beyond the handful of
    #    pre-registered solver/matrix-claim patterns it currently covers.
    #    Every expected_* value below was checked against the ACTUAL current
    #    behavior (not an idealized target) before being written down - many
    #    of these are deliberately "not_independently_verified" or
    #    "verified_mathematically-via-citation-only", because that is
    #    honestly what happens today, not because it's the desired end state.

    # -- classical mechanics --------------------------------------------------
    Problem(
        id="solve-newtons-second-law", category="problem_solving", domain="classical-mechanics",
        question="A 2 kg block experiences a net force of 10 N. Derive Newton's second law and "
                "calculate the resulting acceleration.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- Newton's second law states F = ma\n"
            "- solving for acceleration gives a = F/m\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the block accelerates in the direction of the net force"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="RETRIEVAL FIXED: added a newtons-laws-of-motion curriculum topic (release-"
             "readiness Phase 3), so this now matches on real vocabulary ('Newton's second "
             "law', 'net force', 'acceleration'). status stays not_independently_verified - "
             "there is still no classical-mechanics numeric solver in physics.py, and this "
             "scripted reply cites no [C:]/[T1] tag for symbol_consistency to check - "
             "F=ma with concrete numbers is still never actually computed by this pipeline.",
    ),
    Problem(
        id="solve-shm-angular-frequency", category="problem_solving", domain="classical-mechanics",
        question="Derive the equation of motion for a mass-spring simple harmonic oscillator "
                "and calculate the angular frequency for m=1 kg, k=100 N/m.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- Newton's second law gives m*x'' = -k*x\n"
            "- this yields angular frequency omega = sqrt(k/m)\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the mass oscillates sinusoidally about equilibrium"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="FINDING (domain-crossing retrieval risk): a CLASSICAL mass-spring question "
             "primary-matches the QUANTUM harmonic-oscillator curriculum topic (shared "
             "vocabulary: 'harmonic oscillator', 'angular frequency') - there is no separate "
             "classical-SHM topic to match instead. No solver exists for classical m/k inputs "
             "either. It still reaches verified_mathematically via conservation_law, which "
             "happens to be physically correct here (it independently checks classical SHM "
             "energy conservation, regardless of which topic triggered it) - a coincidence of "
             "the check's own content, not a designed guarantee that a quantum-topic match "
             "will always be classically valid.",
    ),
    Problem(
        id="solve-energy-conservation-falling-ball", category="problem_solving",
        domain="classical-mechanics",
        question="Using conservation of energy, derive and calculate the speed of a 1 kg ball "
                "after falling 5 m from rest.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- conservation of energy: mgh = (1/2)mv^2\n"
            "- solving gives v = sqrt(2gh)\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- gravitational PE converts entirely to kinetic energy"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="RETRIEVAL FIXED: now matches the same newtons-laws-of-motion topic added for "
             "Newton's second law above ('speed after falling from rest', 'gravitational free "
             "fall'). status stays not_independently_verified - no numeric solver, no cited "
             "tag. Also still reveals a real scope limit of check_conservation_law: it only "
             "verifies SIMPLE HARMONIC MOTION energy conservation (a fixed E=1/2mv^2+1/2kx^2 "
             "computation) - gravitational PE-to-KE conversion is a different conservation-of-"
             "energy claim entirely, which no check covers.",
    ),
    Problem(
        id="solve-lagrangian-euler-lagrange", category="problem_solving",
        domain="lagrangian-hamiltonian",
        question="Derive the Euler-Lagrange equation from the principle of least action for a "
                "simple pendulum.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the action S = integral of L dt is stationary for the true path "
            "[C:lagrangian-hamiltonian-mechanics]\n"
            "- this yields d/dt(dL/dtheta_dot) - dL/dtheta = 0\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this is equivalent to Newton's second law in generalized coordinates"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="CITATION-ONLY: matches the real lagrangian-hamiltonian-mechanics topic (a "
             "genuine retrieval win), but nothing about the Euler-Lagrange derivation ITSELF is "
             "executed or checked - verified_mathematically here is earned entirely by "
             "symbol_consistency (the citation is real), not by any computation confirming the "
             "specific derivation steps. No symbolic/numeric/matrix execution fires at all.",
    ),
    Problem(
        id="solve-hamiltonian-mechanics-conserved-quantity", category="problem_solving",
        domain="lagrangian-hamiltonian",
        question="Derive Hamilton's equations for a simple harmonic oscillator and identify "
                "the conserved quantity.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the Hamiltonian is H = p^2/2m + (1/2)kx^2 "
            "[C:lagrangian-hamiltonian-mechanics]\n"
            "- Hamilton's equations follow from partial derivatives of H\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- H itself is conserved since it has no explicit time dependence"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="CITATION-ONLY, PLUS a real but coincidentally-relevant check: primary-matches "
             "lagrangian-hamiltonian-mechanics (real citation), and ALSO happens to pick up "
             "harmonic-oscillator as a secondary match, which lets conservation_law fire - "
             "genuinely checking SHM energy conservation, which is at least topically related "
             "to this question. Still, Hamilton's EQUATIONS themselves (the actual dp/dt, dq/dt "
             "relations asked for) are never symbolically derived or checked by anything.",
    ),

    # -- quantum mechanics ------------------------------------------------------
    Problem(
        id="solve-qho-derive-and-calculate-v2", category="problem_solving",
        domain="quantum-mechanics",
        question="Derive the quantum harmonic oscillator energy levels and calculate the "
                "ground-state energy for omega = 2e14 rad/s.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- start from the QHO Hamiltonian [C:harmonic-oscillator]\n"
            "- solving gives quantized levels; the solver's computed result [T1] is the "
            "ground-state answer, referred to in words\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- nonzero zero-point energy persists even in the ground state"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=True,
        expect_decomposition=True,
        notes="FULL EXECUTION, confirming the ground-state fix (n=0) generalizes to a second "
             "omega value: real numerical_calculation AND a real unit conversion (J to eV) "
             "both reach the execution record, plus limiting_case/classical_limit/"
             "conservation_law all genuinely fire for this topic.",
    ),
    Problem(
        id="solve-uncertainty-derive-and-calculate", category="problem_solving",
        domain="quantum-mechanics",
        question="Derive the Heisenberg uncertainty relation and calculate the minimum "
                "momentum uncertainty for an electron confined to 0.5 nm.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the uncertainty relation follows from the canonical commutator "
            "[C:uncertainty-principle]\n"
            "- the solver's computed result [T1] gives the minimum momentum uncertainty, "
            "referred to in words\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- tighter confinement forces larger momentum uncertainty"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=True,
        expect_decomposition=True,
        notes="Real numerical_calculation reaches the execution record (delta_x=0.5nm "
             "correctly parsed and computed) - confirms the plain-decimal (non-scientific-"
             "notation) numeric path also works, complementing the earlier scientific-notation "
             "fix. No unit-conversion pair exists in this solver's result, so unit_conversions "
             "is honestly empty.",
    ),
    Problem(
        id="solve-canonical-commutator-derive-and-verify", category="problem_solving",
        domain="math-physics",
        question="Derive the canonical commutator of position and momentum and verify it "
                "equals i hbar.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- acting on a test function, [x,p] = i*hbar [C:commutators]\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this nonzero commutator is the algebraic root of the uncertainty principle"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="Real matrix_operations execution: the canonical-commutator claim is parsed from "
             "the derivation text and compared against the reference i*hbar relation, "
             "confirming the execution layer's commutator path generalizes to a differently-"
             "phrased question than the one it was originally built against.",
    ),
    Problem(
        id="solve-hydrogen-transition-derive-and-calculate", category="problem_solving",
        domain="quantum-mechanics",
        question="Derive the transition energy formula and calculate the wavelength emitted "
                "when a hydrogen electron transitions from n=3 to n=2.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the Rydberg formula gives the transition energy [C:bohr-model]\n"
            "- the solver's computed result [T1] gives the wavelength, referred to in words\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this transition produces the Balmer-alpha line"),
        is_wrong=False,
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=True,
        notes="FIXED: physics.py's hydrogen-transition solver result now includes 'topic' and "
             "'formula' fields like every other solver, so extract_mathematical_objects() "
             "legitimately offers [T1] and the derivation's compliant citation of it is no "
             "longer flagged as fabricated (symbol_consistency now passes). This was a false "
             "negative before the fix (a correct derivation reported 'failed'), confirmed by "
             "dedicated regression tests in tests/test_verification.py that a genuinely "
             "fabricated citation, and a genuinely wrong stated wavelength, both still get "
             "caught - the fix closed a real gap without weakening either check. Reaches "
             "verified_mathematically via symbol_consistency alone (citation-only) - no check "
             "here re-derives the transition energy from the Rydberg formula itself, only that "
             "the cited tags are real and that any number the text states matches the solver.",
    ),
    Problem(
        id="solve-expectation-value-position", category="problem_solving",
        domain="quantum-mechanics",
        question="Calculate the expectation value of position for a particle in the ground "
                "state of an infinite square well.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the expectation value is <x> = integral of psi* x psi dx "
            "[C:particle-in-a-box]\n"
            "- by symmetry of the ground state wavefunction about the well's center, <x> = L/2\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the particle is, on average, found at the center of the well"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="GAP in the execution layer specifically: there is no expectation-value "
             "integrator anywhere (research.py's derive() supports integrate, but nothing "
             "wires a wavefunction integral like this one through it). Reaches "
             "verified_mathematically via symbol_consistency plus boundary_conditions (the "
             "latter genuinely fires because particle-in-a-box is the primary topic, even "
             "though it checks psi(0)=psi(L)=0, not the <x>=L/2 claim actually made here).",
    ),
    Problem(
        id="solve-tunneling-probability", category="problem_solving", domain="quantum-mechanics",
        question="Derive the tunneling probability for a particle incident on a potential "
                "barrier and explain how it depends on barrier width.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- inside the barrier the wavefunction decays exponentially "
            "[C:step-barrier-tunneling]\n"
            "- the transmission probability falls off exponentially with barrier width\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- tunneling allows classically forbidden penetration through the barrier"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="CITATION-ONLY: matches the real step-barrier-tunneling topic, but there is no "
             "tunneling-probability solver in physics.py and no check verifies the exponential "
             "decay claim - verified_mathematically is earned by symbol_consistency alone.",
    ),

    # -- quantum computing --------------------------------------------------
    Problem(
        id="solve-pauli-multiplication", category="problem_solving", domain="quantum-computing",
        question="Derive the product sigma_x sigma_y using the explicit Pauli matrices and "
                "verify the result.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- multiply the explicit 2x2 Pauli matrices directly [C:spin-pauli]\n"
            "- sigma_x sigma_y = i*sigma_z\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this algebra underlies how single-qubit gates compose"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="FULL EXECUTION: real matrix_operations record (direct sympy matrix "
             "multiplication) AND operator_consistency both fire correctly - the "
             "'problem_solving' framing of the same Pauli-product pattern already proven in "
             "the operator_eigenvalue category, confirming it isn't category-specific.",
    ),
    Problem(
        id="solve-hadamard-on-zero-state", category="problem_solving", domain="quantum-computing",
        question="Derive the action of the Hadamard gate on the |0> state and calculate the "
                "resulting probabilities.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the Hadamard gate is H = (1/sqrt(2))*[[1,1],[1,-1]] [C:quantum-information]\n"
            "- applying H to |0> gives (|0>+|1>)/sqrt(2)\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- measurement now gives 0 or 1 with equal 50% probability"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="Now execution-backed: execute_quantum_state_claim() parses the Hadamard-on-|0> "
             "claim and computes the real state and 50/50 probabilities via sympy, instead of "
             "no parser existing at all. status/passed stay the same (verified_mathematically "
             "via symbol_consistency alone) since verify_derivation() never consumed execution "
             "output and still doesn't - execution and verification remain parallel, "
             "independent computations; this problem is now genuinely executed AND cited, not "
             "just cited.",
    ),
    Problem(
        id="solve-measurement-probability-born-rule", category="problem_solving",
        domain="quantum-computing",
        question="Using the Born rule and the measurement postulate, calculate the "
                "probability of a specific measurement outcome for a qubit in the state "
                "(|0>+|1>)/sqrt(2).",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the Born rule gives P(outcome) = |amplitude|^2 [C:measurement-postulates]\n"
            "- applying this to the given amplitude yields the probability\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- probabilities must sum to 1 across all outcomes"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="Now execution-backed: the question was reworded to state a concrete qubit "
             "state, (|0>+|1>)/sqrt(2), so execute_quantum_state_claim() has something to "
             "parse and computes the real Born-rule probabilities (0.5/0.5) via sympy. "
             "status/passed stay the same (verified_mathematically via symbol_consistency "
             "alone) since verify_derivation() never consumed execution output and still "
             "doesn't; this problem is now genuinely executed AND cited, not just cited.",
    ),
    Problem(
        id="solve-bell-state-entanglement", category="problem_solving", domain="quantum-computing",
        question="Derive the entangled Bell state formed by applying a Hadamard gate and a "
                "CNOT gate to two qubits.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- applying H to the first qubit then CNOT gives (|00>+|11>)/sqrt(2) "
            "[C:quantum-information]\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this state cannot be written as a product of two single-qubit states"),
        expected_status=("verified_mathematically",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="CITATION-ONLY: real quantum-information match, but nothing executes the two-gate "
             "sequence or checks that the resulting state is genuinely entangled (non-"
             "factorizable) - a real gap, since entanglement verification is exactly the kind "
             "of claim a matrix/tensor-aware execution layer could eventually check.",
    ),
    Problem(
        id="solve-tensor-product-two-qubit-state", category="problem_solving",
        domain="quantum-computing",
        question="Derive the combined two-qubit basis state using the tensor product of "
                "individual qubit states.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- the combined state is the tensor product |0> tensor |1> = |01>\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- tensor products are how independent quantum systems combine"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=True,
        expect_solver_ran=False,
        notes="RETRIEVAL FIXED: quantum-information's key_concepts/key_equations now include "
             "tensor-product vocabulary (release-readiness Phase 3), so this matches. status "
             "stays not_independently_verified, and execution_backed stays False - there is "
             "still no tensor/Kronecker-product operation anywhere in the execution layer, "
             "only a curriculum citation is now available where none existed before.",
    ),

    # -- mathematical physics --------------------------------------------------
    Problem(
        id="solve-eigenvalue-problem-2x2", category="problem_solving", domain="math-physics",
        question="Find the eigenvalues of the matrix with rows [2,1] and [1,2] and verify "
                "them by direct computation.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- solve det(A - lambda*I) = 0\n"
            "- this gives lambda = 1 and lambda = 3\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- the eigenvalues are the matrix's characteristic values"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        expect_solver_ran=False,
        notes="Now execution-backed: execute_matrix_literal_claim() parses the explicit "
             "[2,1]/[1,2] matrix and computes real eigenvalues via Matrix.eigenvals() (1 and "
             "3, confirmed). status stays not_independently_verified - no curriculum topic "
             "covers generic eigenvalue problems and verify_derivation() doesn't consume "
             "execution output - so this is now honestly executed-but-still-topically-"
             "ungrounded, not unexecuted.",
    ),
    Problem(
        id="solve-matrix-multiplication-generic", category="problem_solving",
        domain="math-physics",
        question="Compute the product of the matrices with rows [1,2],[3,4] and [0,1],[1,0] "
                "and verify the result.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- multiply the matrices directly\n"
            "- the product has rows [2,1] and [4,3]\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- matrix multiplication is not commutative in general"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        expect_solver_ran=False,
        notes="Now execution-backed: execute_matrix_literal_claim() parses the explicit "
             "[1,2],[3,4] and [0,1],[1,0] matrices and computes the real product via sympy "
             "(confirmed: [[2,1],[4,3]], matching the stated claim). status stays "
             "not_independently_verified - no curriculum topic covers generic matrix "
             "operations and verify_derivation() doesn't consume execution output.",
    ),
    Problem(
        id="solve-differential-equation-exponential-decay", category="problem_solving",
        domain="math-physics",
        question="Solve the differential equation dy/dx = -k*y and verify the solution "
                "satisfies the original equation.",
        derivation_reply=(
            "DERIVATION PLAN\n"
            "- separate variables and integrate\n"
            "- this gives y = C*exp(-k*x)\n\n"
            "PHYSICAL INTERPRETATION\n"
            "- this describes exponential decay"),
        expected_status=("not_independently_verified",),
        expect_topics_matched=False,
        expect_solver_ran=False,
        notes="Now execution-backed: research.solve_ode() (a new sibling of derive(), using "
             "sympy's dsolve() - ODEs need sp.Function/Derivative, not derive()'s all-Symbol "
             "namespace) solves dy/dx = -k*y and confirms the stated solution "
             "y=C*exp(-k*x). status stays not_independently_verified - no curriculum topic "
             "covers generic ODEs and verify_derivation() doesn't consume execution output.",
    ),
]


assert len({p.id for p in PROBLEMS}) == len(PROBLEMS), "duplicate problem id"
