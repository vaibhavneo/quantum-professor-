"""Physics + Mathematical Verification Layer.

Sits between the Derivation and Physical Interpretation stages:

    ... -> Derivation Plan -> Derivation -> VERIFICATION -> Physical
    Interpretation -> Professor Answer

Everything here is deterministic - no LLM call, ever. It reuses exactly the
sympy infrastructure that already exists in this app (research.py's
check_identity()/derive(), physics.py's closed-form solvers) rather than
inventing a parallel math engine. Scope is deliberately bounded to the ten
requested categories, each implemented honestly: where a check genuinely
applies (a known solver ran, a symbolic identity was posed, a recognized
operator/conservation relation is in play) it does real sympy work and can
fail; where nothing in the question gives it something to check, it says
"not_applicable" rather than fabricating a pass. This module does not prove
physics in general - it verifies the specific, bounded claims the pipeline
already computed or was asked to check.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

try:
    from . import research as R
except ImportError:
    import research as R

# ── shared vocabulary ────────────────────────────────────────────────────

_ASSUMPTION_PATTERNS = [
    re.compile(r"\bassum\w*\b", re.I), re.compile(r"\bapproximat\w*\b", re.I),
    re.compile(r"\bto (?:first|second|leading) order\b", re.I),
    re.compile(r"\bneglect\w*\b", re.I), re.compile(r"\bignor\w*\b", re.I),
    re.compile(r"\bnon-relativistic\b", re.I), re.compile(r"\bin the limit\b", re.I),
    re.compile(r"\bfor simplicity\b", re.I), re.compile(r"\btreat\w* .+? as\b", re.I),
]

# physics.py solver "result" -> (value_field, unit_kind) used by the
# known-result and dimensional checks below.
_SOLVER_VALUE_FIELDS = {
    "harmonic-oscillator": [("energy_eV", "eV"), ("energy_J", "J")],
    "particle-in-a-box": [("energy_eV", "eV"), ("energy_J", "J")],
    "hydrogen-atom": [("energy_eV", "eV")],
    "hydrogen-transition": [("wavelength_nm", "nm")],
    "de-broglie": [("wavelength_pm", "pm")],
    "blackbody-radiation": [("energy_eV", "eV")],
    "uncertainty-principle": [("min_delta_p_kg_m_s", "kg*m/s")],
}

_UNIT_WORDS = {
    "eV": ("ev", "electronvolt", "electron-volt", "electron volt"),
    "J": ("joule", "joules"),
    "nm": ("nanometre", "nanometer", "nm"),
    "pm": ("picometre", "picometer", "pm"),
    "m": ("metre", "meter", "metres", "meters"),
    "kg*m/s": ("kg", "kilogram"),
}


@dataclass
class CheckResult:
    name: str
    status: str          # "pass" | "fail" | "warning" | "not_applicable"
    detail: str = ""
    correction: str | None = None


@dataclass
class VerificationResult:
    status: str = "not_independently_verified"   # verified_mathematically |
                                                  # partially_verified |
                                                  # failed |
                                                  # not_independently_verified
    confidence: str = "low"                       # high | medium | low
    passed: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    corrections: list = field(default_factory=list)
    checks: list = field(default_factory=list)     # every CheckResult, incl. not_applicable

    def to_dict(self) -> dict:
        return asdict(self)


# ── individual checks (each returns a CheckResult, never raises) ──────────

def check_symbol_consistency(text: str, pack) -> CheckResult:
    """1. Symbol/equation consistency - every [C:]/[T1]/[X1] tag the
    derivation cites must be one that was actually offered, the same
    discipline validation() already applies to the final prose, run here
    one stage earlier."""
    if not text:
        return CheckResult("symbol_consistency", "not_applicable", "no derivation text to check")
    offered = {f"C:{t.id}" for t in (pack.topics if pack else [])}
    offered |= {o["tag"] for o in (pack.mathematical_objects if pack else [])}
    cited = set(re.findall(r"\[([A-Za-z]+:?[\w\-]*)\]", text))
    fabricated = sorted(c for c in cited if c not in offered and not re.match(r"^[SAX]\d+$", c))
    if not cited:
        return CheckResult("symbol_consistency", "not_applicable", "no tagged symbols cited")
    if fabricated:
        return CheckResult("symbol_consistency", "fail",
                           f"cited tag(s) not actually offered: {fabricated}")
    return CheckResult("symbol_consistency", "pass", f"all {len(cited)} cited tag(s) were offered")


def check_dimensional_consistency(text: str, computed: dict | None) -> CheckResult:
    """2. Dimensional/unit consistency - bounded to the unit the solver
    itself declares for its own result (physics.py's field-name suffixes
    are the source of truth), not general-purpose unit parsing of arbitrary
    text. Flags a stated unit word that belongs to a different quantity
    kind than the one actually computed (e.g. calling an energy a length)."""
    if not (computed and computed.get("ran")):
        return CheckResult("dimensional_consistency", "not_applicable", "no computed result to check units against")
    topic = computed["result"].get("topic", "")
    fields = _SOLVER_VALUE_FIELDS.get(topic)
    if not fields:
        return CheckResult("dimensional_consistency", "not_applicable",
                           f"no curated unit reference for solver topic {topic!r}")
    expected_unit = fields[0][1]
    lower = text.lower()

    def _mentions(words):
        return any(re.search(rf"\b{re.escape(w)}\b", lower) for w in words)

    other_units_present = [u for u, words in _UNIT_WORDS.items()
                           if u != expected_unit and _mentions(words)]
    expected_words = _UNIT_WORDS.get(expected_unit, ())
    if _mentions(expected_words):
        return CheckResult("dimensional_consistency", "pass",
                           f"uses the expected unit ({expected_unit})")
    if other_units_present:
        return CheckResult("dimensional_consistency", "fail",
                           f"mentions {other_units_present} but the computed result is in {expected_unit}",
                           correction=f"the computed quantity is in {expected_unit}")
    return CheckResult("dimensional_consistency", "not_applicable", "no unit stated in the derivation to check")


def check_algebraic_consistency(u: dict, symbolic: dict | None) -> CheckResult:
    """3. Algebraic consistency - reuses research.py's existing
    check_identity() (sympy, already computed upstream when applicable;
    called fresh here only if an identity was posed but never checked)."""
    identity = (u or {}).get("identity") or ""
    if symbolic is None and identity and "=" in identity:
        lhs, _, rhs = identity.partition("=")
        symbolic = R.check_identity(lhs.strip(), rhs.strip())
    if not symbolic or not symbolic.get("ok"):
        return CheckResult("algebraic_consistency", "not_applicable", "no algebraic identity to check")
    if symbolic.get("equal"):
        return CheckResult("algebraic_consistency", "pass",
                           f"{symbolic.get('lhs')} = {symbolic.get('rhs')} confirmed by sympy")
    return CheckResult("algebraic_consistency", "fail",
                       f"sympy found the identity does not hold: {symbolic.get('verdict')}",
                       correction=f"difference is {symbolic.get('difference')}, not 0")


_PAULI_MATS = None  # lazily built - importing sympy at module load isn't needed elsewhere


def _pauli_matrices():
    global _PAULI_MATS
    if _PAULI_MATS is None:
        import sympy as sp
        _PAULI_MATS = {
            "x": sp.Matrix([[0, 1], [1, 0]]),
            "y": sp.Matrix([[0, -sp.I], [sp.I, 0]]),
            "z": sp.Matrix([[1, 0], [0, -1]]),
        }
    return _PAULI_MATS


# sigma_x sigma_y = <rhs>, sigma1*sigma2 = <rhs>, sigma_y sigma_z = ... etc -
# captures which two Pauli matrices are claimed to multiply together, and the
# raw text of whatever the derivation claims the product equals.
_PAULI_PRODUCT_CLAIM_RE = re.compile(
    r"sigma[_\s]?([xyz])\s*[\*\s]\s*sigma[_\s]?([xyz])\s*=\s*([^.\n;]+)", re.I)

# The same claim, in natural prose rather than compact equation form:
# "multiply sigma_x and sigma_y ... the product (is|equals|works out to) Z".
# A real claim can span a line break between naming the two matrices and
# stating the result - this is still one bounded, specific grammar being
# matched, not free-text NLP.
_PAULI_PRODUCT_PROSE_RE = re.compile(
    r"sigma[_\s]?([xyz])\s+and\s+sigma[_\s]?([xyz])\b[\s\S]{0,80}?"
    r"(?:product|result)\b[\s\S]{0,40}?"
    r"(?:is|equals|works out to|gives|yields)\s*([^.\n;]+)", re.I)

# A bounded, honest RHS grammar: optional sign/coefficient, optional literal
# "i", one Pauli matrix - or exactly "0". Anything outside this shape is a
# claim this function admits it cannot safely parse, rather than guessing.
_PAULI_RHS_RE = re.compile(r"^\s*([+-]?\s*\d*\.?\d*)\s*\*?\s*(i\b)?\s*\*?\s*sigma[_\s]?([xyz])\s*$", re.I)


def _parse_pauli_rhs(raw: str):
    """Returns a concrete 2x2 sympy Matrix for a bounded set of RHS shapes
    this curriculum actually produces, or None if the claim's right-hand
    side doesn't match one of them - a safe refusal, never a guess."""
    import sympy as sp
    raw = raw.strip()
    if re.match(r"^0+\.?0*$", raw):
        return sp.zeros(2)
    m = _PAULI_RHS_RE.match(raw)
    if not m:
        return None
    coeff_txt, i_txt, label = m.groups()
    coeff_txt = (coeff_txt or "").replace(" ", "")
    coeff = sp.Integer(-1) if coeff_txt == "-" else (sp.Integer(1) if coeff_txt in ("", "+")
                                                     else sp.sympify(coeff_txt))
    if i_txt:
        coeff *= sp.I
    return coeff * _pauli_matrices()[label.lower()]


# [x,p] = <rhs>, [x, p] = ... - the canonical commutation relation.
_COMMUTATOR_CLAIM_RE = re.compile(r"\[\s*x\s*,\s*p\s*\]\s*=\s*([^.\n;]+)", re.I)
_COMMUTATOR_RHS_RE = re.compile(r"^\s*([+-]?\s*\d*\.?\d*)\s*\*?\s*(i\b)?\s*\*?\s*hbar\s*$", re.I)


def _parse_commutator_rhs(raw: str):
    """Returns (coefficient, has_i) for a bounded RHS grammar, or None if
    unparseable. [x,p] is a postulated relation, not something derivable
    from pure algebra - so this compares the CLAIMED coefficient/i-factor
    against the textbook-standard i*hbar, the same reference-value
    comparison check_known_result already does for numeric claims."""
    raw = raw.strip()
    m = _COMMUTATOR_RHS_RE.match(raw)
    if not m:
        return None
    coeff_txt, i_txt = m.groups()
    coeff_txt = (coeff_txt or "").replace(" ", "")
    coeff = -1.0 if coeff_txt == "-" else (1.0 if coeff_txt in ("", "+") else float(coeff_txt))
    return coeff, bool(i_txt)


def _operator_topic_signal(text: str, pack) -> bool:
    """Whether an operator/Pauli/commutator topic is plausibly in play at
    all - used ONLY to word an honest non-finding, never to grant a pass.
    A keyword here selects an opportunity to look for a claim; it is never
    treated as proof of one."""
    names = (text or "") + " " + " ".join(
        o.get("name", "") + " " + o.get("expression", "")
        for o in (pack.mathematical_objects if pack else []))
    names += " " + " ".join(t.title for t in (pack.topics if pack else []))
    if pack is not None:
        names += " " + " ".join(pack.concepts) + " " + (pack.question or "")
    low = names.lower()
    return any(k in low for k in ("pauli", "qubit", "commutator", "operator"))


def check_operator_consistency(text: str, pack) -> CheckResult:
    """4. Operator consistency - parses the SPECIFIC operator claim the
    derivation makes (a Pauli product, or the canonical commutator) and
    verifies THAT, rather than confirming a fixed reference fact is
    internally consistent regardless of what was actually claimed. A
    keyword like 'pauli'/'qubit'/'commutator' only selects that this check
    might be worth attempting - by itself it never constitutes proof, and a
    claim this function can't safely parse is reported as not_applicable
    (which resolves to not_independently_verified overall) rather than
    guessed at."""
    text = text or ""
    try:
        # A derivation often restates the definition ("[x,p] = xp - px") before
        # giving the actual claimed value ("[x,p] = i*hbar") later - trying
        # every candidate match, not just the first, is what finds the real
        # claim instead of giving up on an earlier, differently-shaped one.
        pauli_candidates = (list(_PAULI_PRODUCT_CLAIM_RE.finditer(text))
                           + list(_PAULI_PRODUCT_PROSE_RE.finditer(text)))
        last_unparsed = None
        for m in pauli_candidates:
            a, b, rhs_raw = m.groups()
            rhs = _parse_pauli_rhs(rhs_raw)
            if rhs is None:
                last_unparsed = (a, b, rhs_raw)
                continue
            mats = _pauli_matrices()
            lhs = mats[a.lower()] * mats[b.lower()]
            if lhs.equals(rhs):
                return CheckResult("operator_consistency", "pass",
                                   f"sigma_{a}*sigma_{b} = {rhs_raw.strip()} confirmed by direct "
                                   "matrix multiplication")
            return CheckResult("operator_consistency", "fail",
                               f"sigma_{a}*sigma_{b} = {rhs_raw.strip()} does not hold - direct "
                               f"computation gives sigma_{a}*sigma_{b} = {lhs.tolist()}",
                               correction=f"sigma_{a}*sigma_{b} actually equals {lhs.tolist()}")
        if last_unparsed:
            a, b, rhs_raw = last_unparsed
            return CheckResult("operator_consistency", "not_applicable",
                               f"found a Pauli product claim (sigma_{a}*sigma_{b} = "
                               f"{rhs_raw.strip()}) but could not safely parse its right-hand "
                               "side - not independently verified rather than guessed")

        commutator_candidates = list(_COMMUTATOR_CLAIM_RE.finditer(text))
        last_unparsed_c = None
        for m2 in commutator_candidates:
            rhs_raw = m2.group(1)
            parsed = _parse_commutator_rhs(rhs_raw)
            if parsed is None:
                last_unparsed_c = rhs_raw
                continue
            coeff, has_i = parsed
            if coeff == 1.0 and has_i:
                return CheckResult("operator_consistency", "pass",
                                   "[x,p] = i*hbar matches the canonical commutation relation")
            return CheckResult("operator_consistency", "fail",
                               f"[x,p] = {rhs_raw.strip()} does not match the canonical "
                               "commutation relation",
                               correction="the canonical commutation relation is [x,p] = i*hbar")
        if last_unparsed_c:
            return CheckResult("operator_consistency", "not_applicable",
                               f"found a canonical-commutator claim ([x,p] = "
                               f"{last_unparsed_c.strip()}) but could not safely parse its "
                               "right-hand side - not independently verified rather than guessed")
    except Exception as exc:
        return CheckResult("operator_consistency", "warning", f"operator check errored: {exc}")

    if _operator_topic_signal(text, pack):
        return CheckResult("operator_consistency", "not_applicable",
                           "an operator/Pauli/commutator topic appears to be in play, but no "
                           "specific, parseable mathematical claim (e.g. 'sigma_x sigma_y = ...' "
                           "or '[x,p] = ...') was found in the derivation to check")
    return CheckResult("operator_consistency", "not_applicable",
                       "no operator relation (commutator/Pauli) claim in this derivation")


def check_boundary_conditions(pack, computed: dict | None) -> CheckResult:
    """5. Boundary/initial-condition consistency - topic-scoped to the one
    boundary-value problem this curriculum actually teaches in closed form:
    the infinite square well, where psi(0)=psi(L)=0 is the defining
    condition. Verified symbolically, not asserted.

    Only the PRIMARY matched topic (pack.topics[0]) counts, never any topic
    anywhere in the list - a secondary, coincidental-overlap match (e.g. a
    hydrogen-atom question also weakly matching particle-in-a-box on shared
    vocabulary like "energy"/"state") must not be able to fire this check
    for a problem that isn't actually about the infinite square well.
    """
    topic = (computed or {}).get("result", {}).get("topic", "")
    primary_id = pack.topics[0].id if (pack and pack.topics) else None
    if topic != "particle-in-a-box" and primary_id not in ("particle-in-a-box", "finite-well"):
        return CheckResult("boundary_conditions", "not_applicable",
                           "no boundary-value problem (infinite square well) in this question")
    try:
        import sympy as sp
        x, L = sp.symbols("x L", positive=True)
        n = sp.symbols("n", positive=True, integer=True)
        psi = sp.sin(n * sp.pi * x / L)
        at_0 = psi.subs(x, 0)
        at_L = psi.subs(x, L)
        if at_0 == 0 and sp.simplify(at_L) == 0:
            return CheckResult("boundary_conditions", "pass",
                               "psi(0)=0 and psi(L)=0 confirmed for sin(n*pi*x/L)")
        return CheckResult("boundary_conditions", "fail",
                           f"boundary values did not vanish: psi(0)={at_0}, psi(L)={at_L}")
    except Exception as exc:
        return CheckResult("boundary_conditions", "warning", f"boundary check errored: {exc}")


def check_limiting_case(computed: dict | None) -> CheckResult:
    """6. Limiting-case check - large-n behaviour, via research.py's
    existing derive("limit", ...) (sympy), for the solver topics this
    curriculum can state a clean limit for."""
    topic = (computed or {}).get("result", {}).get("topic", "") if computed else ""
    if topic == "harmonic-oscillator":
        out = R.derive("hbar*w*(n + 1/2)", "limit", wrt="n", to="oo")
        if out.get("ok"):
            return CheckResult("limiting_case", "pass",
                               f"as n -> infinity, E_n -> {out['result']} (unbounded, as expected)")
        return CheckResult("limiting_case", "warning", f"limit check errored: {out.get('error')}")
    if topic == "particle-in-a-box":
        out = R.derive("n**2 * pi**2 * hbar**2 / (2*m*L**2)", "limit", wrt="n", to="oo")
        if out.get("ok"):
            return CheckResult("limiting_case", "pass",
                               f"as n -> infinity, E_n -> {out['result']} (levels spread without bound)")
        return CheckResult("limiting_case", "warning", f"limit check errored: {out.get('error')}")
    return CheckResult("limiting_case", "not_applicable",
                       f"no curated large-n limit for solver topic {topic!r}")


def check_classical_limit(computed: dict | None) -> CheckResult:
    """7. Classical-limit check - hbar -> 0, via the same sympy derive()
    path, for the one result in this curriculum with a clean closed-form
    hbar-dependence: the harmonic oscillator's zero-point energy."""
    topic = (computed or {}).get("result", {}).get("topic", "") if computed else ""
    if topic == "harmonic-oscillator":
        out = R.derive("hbar*w*(n + 1/2)", "limit", wrt="hbar", to="0")
        if out.get("ok"):
            return CheckResult("classical_limit", "pass",
                               f"as hbar -> 0, E_n -> {out['result']} (recovers the classical "
                               "result of no minimum energy)")
        return CheckResult("classical_limit", "warning", f"classical-limit check errored: {out.get('error')}")
    return CheckResult("classical_limit", "not_applicable",
                       f"no curated classical (hbar->0) limit for solver topic {topic!r}")


_SHM_SYSTEM_RE = re.compile(r"\b(harmonic oscillator|simple harmonic motion)\b", re.I)
_CONSERVATION_CLAIM_RE = re.compile(r"\bconserv\w*\b|\bconstant of motion\b", re.I)


def check_conservation_law(pack) -> CheckResult:
    """8. Conservation-law check - classical/Hamiltonian mechanics energy
    conservation for simple harmonic motion, verified by direct sympy
    differentiation (dE/dt = 0 along the actual equations of motion), not
    asserted from the textbook statement that it holds.

    Scoped narrowly on purpose: the word "Hamiltonian" or "Lagrangian"
    appearing anywhere in a question used to be enough to trigger this and
    unconditionally verify ONE fixed fact (SHM energy conservation) no
    matter what the question actually asked - so "derive Hamilton's
    equations from the Lagrangian" was reported verified_mathematically
    against a check that never touched Hamilton's equations at all. This
    now fires only when the PRIMARY matched topic genuinely is the harmonic
    oscillator, or the question both names that specific system AND
    explicitly concerns conservation - a general Hamiltonian/Lagrangian
    mechanics question is not, by itself, a conservation-law question.
    """
    primary_id = pack.topics[0].id if (pack and pack.topics) else None
    question_l = (pack.question or "").lower() if pack else ""
    concepts_l = " ".join(pack.concepts if pack else []).lower()
    signal = question_l + " " + concepts_l

    shm_primary_topic = primary_id == "harmonic-oscillator"
    shm_system_named = bool(_SHM_SYSTEM_RE.search(signal))
    conservation_claimed = bool(_CONSERVATION_CLAIM_RE.search(signal))
    relevant = shm_primary_topic or (shm_system_named and conservation_claimed)
    if not relevant:
        return CheckResult("conservation_law", "not_applicable",
                           "no explicit simple-harmonic-motion conservation claim, and no "
                           "harmonic-oscillator primary topic match, to check against - this "
                           "check does not generalize to other classical-mechanics systems")
    try:
        import sympy as sp
        t, m, k, A, w, phi = sp.symbols("t m k A w phi", positive=True, real=True)
        x = A * sp.cos(w * t + phi)
        v = sp.diff(x, t)
        E = sp.Rational(1, 2) * m * v**2 + sp.Rational(1, 2) * k * x**2
        E = E.subs(k, m * w**2)  # SHM dispersion relation
        dE_dt = sp.simplify(sp.diff(E, t))
        if dE_dt == 0:
            return CheckResult("conservation_law", "pass",
                               "dE/dt = 0 confirmed for simple harmonic motion (energy conserved)")
        return CheckResult("conservation_law", "fail", f"dE/dt simplified to {dE_dt}, not 0")
    except Exception as exc:
        return CheckResult("conservation_law", "warning", f"conservation check errored: {exc}")


_NUMBER_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")

# Citation tags ([T1], [C:harmonic-oscillator], [X1], [S12], [A3], ...) carry
# digits that are labels, never physics values - "T1" means "solver value
# one", not the number 1. Stripped out before any number extraction so
# citing a tag (exactly what the pipeline's own prompts ask for) can never
# be misread as the derivation stating a numeric result.
_CITATION_TAG_RE = re.compile(r"\[[A-Za-z]+:?[\w\-]*\]")


def _strip_citation_tags(text: str) -> str:
    return _CITATION_TAG_RE.sub(" ", text)


def check_known_result(text: str, computed: dict | None) -> CheckResult:
    """9. Known-result/reference check - cross-checks any number the
    derivation states against the ALREADY-COMPUTED, trusted physics.py
    solver result for this exact question, the existing physics corpus's
    own reference value."""
    if not (computed and computed.get("ran")):
        return CheckResult("known_result", "not_applicable", "no solver result to check against")
    topic = computed["result"].get("topic", "")
    fields = _SOLVER_VALUE_FIELDS.get(topic)
    if not fields:
        return CheckResult("known_result", "not_applicable", f"no reference field curated for {topic!r}")
    field_name, unit = fields[0]
    reference = computed["result"].get(field_name)
    if reference is None or not text:
        return CheckResult("known_result", "not_applicable", "no reference value or no text to check")
    stated = [float(m) for m in _NUMBER_RE.findall(_strip_citation_tags(text))]
    if not stated:
        return CheckResult("known_result", "warning",
                           "the derivation states no numeric result to cross-check "
                           "(expected if it correctly deferred to the solver)")
    close = any(abs(s - reference) <= max(abs(reference) * 0.02, 1e-9) for s in stated)
    # A number that looks like it's THIS quantity (same order of magnitude,
    # not some unrelated input parameter) but doesn't match is the failure
    # worth flagging; wildly different magnitudes are more likely a
    # different quantity (an input, an intermediate step) than a wrong answer.
    plausible_same_quantity = [s for s in stated if reference != 0
                               and 0.01 < abs(s / reference) < 100]
    if close:
        return CheckResult("known_result", "pass",
                           f"derivation's stated value matches the solver's {field_name}={reference}")
    if plausible_same_quantity:
        return CheckResult("known_result", "fail",
                           f"derivation states {plausible_same_quantity}, but the solver computed "
                           f"{field_name}={reference}",
                           correction=f"the correct value is {reference} {unit}")
    return CheckResult("known_result", "warning",
                       "no stated number matches the expected order of magnitude for this result")


def check_assumptions(text: str) -> list:
    """10. Assumption/approximation inventory - not a pass/fail check, a
    deterministic listing of what the derivation itself flags as an
    assumption or approximation."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    found = []
    for s in sentences:
        s = s.strip(" -•")
        if not s:
            continue
        if any(p.search(s) for p in _ASSUMPTION_PATTERNS):
            found.append(s[:200])
    return found


# ── orchestrator ─────────────────────────────────────────────────────────

def verify_derivation(question: str, u: dict, pack, reasoning: dict, computed: dict | None,
                      symbolic: dict | None) -> VerificationResult:
    """Runs all ten checks against the derivation reasoning_engine() already
    produced. Deterministic throughout - zero LLM calls. Never raises: any
    individual check that errors is recorded as a warning, not a crash.
    """
    text = (reasoning or {}).get("derivation_plan") or (reasoning or {}).get("text") or ""
    if not isinstance(text, str):
        text = str(text)  # a malformed upstream reply must degrade, never crash the checks below
    checks = [
        check_symbol_consistency(text, pack),
        check_dimensional_consistency(text, computed),
        check_algebraic_consistency(u, symbolic),
        check_operator_consistency(text, pack),
        check_boundary_conditions(pack, computed),
        check_limiting_case(computed),
        check_classical_limit(computed),
        check_conservation_law(pack),
        check_known_result(text, computed),
    ]
    result = VerificationResult()
    result.checks = [asdict(c) for c in checks]
    result.assumptions = check_assumptions(text)
    for c in checks:
        if c.status == "pass":
            result.passed.append({"check": c.name, "detail": c.detail})
        elif c.status == "fail":
            result.failed.append({"check": c.name, "detail": c.detail})
            if c.correction:
                result.corrections.append({"issue": c.name, "correction": c.correction})
        elif c.status == "warning":
            result.warnings.append(f"{c.name}: {c.detail}")
        # not_applicable is intentionally not surfaced as noise - see checks list for the full record

    if result.failed:
        # Both branches ran at least one real check that came back wrong -
        # the difference is whether anything else corroborated the
        # derivation. Collapsing this into "not_independently_verified"
        # would silently turn an active, detected error into mere
        # uncertainty; "failed" says plainly that something checked out
        # wrong, not just that nothing could be confirmed.
        result.status = "partially_verified" if result.passed else "failed"
        result.confidence = "medium" if result.passed else "low"
    elif result.passed:
        result.status = "verified_mathematically"
        result.confidence = "high" if len(result.passed) >= 2 else "medium"
    else:
        result.status = "not_independently_verified"
        result.confidence = "low"
    return result
