"""Deterministic Execution layer.

Sits between Derivation and Verification:

    ... -> Derivation Plan -> Derivation -> DETERMINISTIC EXECUTION ->
    Verification -> Physical Interpretation -> Professor Answer

Makes the real SymPy/Python computation the pipeline already performs an
explicit, visible, structured record - not a re-derivation, and not a
second opinion. Numerical calculation reuses physics.py's already-computed
solver result; symbolic algebra reuses research.py's check_identity();
matrix operations reuse verification.py's own Pauli/commutator parsing and
sympy computation (called here, not reimplemented, so there is exactly one
place that arithmetic runs), extended in this module with general numeric
matrix products/eigenvalues and single-qubit gate/Born-rule execution
(folded into the same matrix_operations shape, new "kind" values); ODE
solving is a new top-level field via research.solve_ode(); unit conversion
is the other genuinely new capability - a small curated table covering the
units this app's physics actually produces, not a general-purpose units
library.

Zero LLM calls anywhere in this file. The professor gets to cite this
record as ground truth, exactly like [T1]/[X1] already work today, just
now gathered under one explicitly labeled stage instead of being scattered
and implicit.
"""
from __future__ import annotations

import re

try:
    from . import research as R
    from .verification import (_COMMUTATOR_CLAIM_RE, _PAULI_PRODUCT_CLAIM_RE,
                               _PAULI_PRODUCT_PROSE_RE, _pauli_matrices)
except ImportError:
    import research as R
    from verification import (_COMMUTATOR_CLAIM_RE, _PAULI_PRODUCT_CLAIM_RE,
                              _PAULI_PRODUCT_PROSE_RE, _pauli_matrices)

# Curated, deterministic - the exact non-SI units this app's physics.py
# solvers actually produce (confirmed by reading their result dicts), not a
# general-purpose unit system. Value: (SI unit, multiply-by-this-for-SI).
_TO_SI = {
    "eV": ("J", 1.602176634e-19),
    "nm": ("m", 1e-9),
    "pm": ("m", 1e-12),
    "angstrom": ("m", 1e-10),
}


def convert_units(value: float, from_unit: str) -> dict | None:
    """Deterministic unit conversion to SI, for the bounded set of units
    this app's physics actually deals in. Returns None for an unrecognized
    unit rather than guessing a conversion factor - the same "refuse rather
    than guess" discipline verification.py's checks already follow."""
    if from_unit not in _TO_SI:
        return None
    si_unit, factor = _TO_SI[from_unit]
    return {"from_value": value, "from_unit": from_unit,
           "to_value": value * factor, "to_unit": si_unit}


def _result_unit_conversions(computed: dict) -> list:
    """Real unit-conversion pairs the solver's own result already contains
    or implies - never invented. Two patterns, both confirmed against the
    actual physics.py solver outputs:
    (1) the SAME result dict states a quantity in two units directly (e.g.
        de-broglie's wavelength_m AND wavelength_pm together);
    (2) the SI value only lives in the solver's recorded INPUT (e.g.
        particle-in-a-box's L, in metres) while the result shows a display
        unit (L_nm) - the two are cross-referenced, not each computed fresh.
    Returns [] when neither pattern is present, which most solver topics
    genuinely don't have - an honest empty list, not a forced entry.
    """
    result = computed.get("result", {}) or {}
    inputs = computed.get("inputs", {}) or {}
    found = []

    # Pattern 1: same-dict SI/non-SI pair sharing a base name.
    for unit, (si_unit, _factor) in _TO_SI.items():
        for key, value in result.items():
            if not key.endswith(f"_{unit}"):
                continue
            base = key[: -len(unit) - 1]
            si_key = f"{base}_{si_unit}"
            if si_key in result:
                found.append({"from_value": result[si_key], "from_unit": si_unit,
                             "to_value": value, "to_unit": unit})

    # Pattern 2: SI input cross-referenced against a display-unit result
    # field with the same base name (particle-in-a-box's L / L_nm).
    for in_key, in_value in inputs.items():
        for unit in _TO_SI:
            display_key = f"{in_key}_{unit}"
            if display_key in result:
                found.append({"from_value": in_value, "from_unit": "m" if in_key == "L" else "SI",
                             "to_value": result[display_key], "to_unit": unit})

    return found


def execute_matrix_claim(text: str) -> dict | None:
    """Matrix operations - reuses verification.py's own Pauli-product and
    canonical-commutator parsing and sympy computation directly (the same
    regex, the same matrices), so there is exactly one place this
    arithmetic runs; this only ALSO exposes the raw computed result as its
    own labeled fact, distinct from the pass/fail verdict verification
    reports separately."""
    text = text or ""
    for m in list(_PAULI_PRODUCT_CLAIM_RE.finditer(text)) + list(_PAULI_PRODUCT_PROSE_RE.finditer(text)):
        a, b, rhs_raw = m.groups()
        mats = _pauli_matrices()
        lhs = mats[a.lower()] * mats[b.lower()]
        return {"kind": "pauli_product", "claim": f"sigma_{a}*sigma_{b} = {rhs_raw.strip()}",
               "computed": lhs.tolist()}
    for m2 in _COMMUTATOR_CLAIM_RE.finditer(text):
        return {"kind": "canonical_commutator", "claim": f"[x,p] = {m2.group(1).strip()}",
               "computed": "i*hbar (the postulated canonical commutation relation)"}
    return None


# Requires 2+ comma-separated numbers inside the brackets, so this can never
# match a citation tag like [T1] or [C:topic] (letters, no comma) - only an
# actual numeric matrix row such as "[1,2]".
_ROW_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)\s*\]")


def _rows_to_matrix(sp, rows_text_list):
    return sp.Matrix([[sp.sympify(v.strip()) for v in row.split(",")] for row in rows_text_list])


def execute_matrix_literal_claim(text: str) -> dict | None:
    """General numeric matrix operations - product and eigenvalues - for
    claims that state matrices as explicit bracketed rows (e.g. "matrix with
    rows [1,2],[3,4]"), distinct from execute_matrix_claim()'s Pauli/
    commutator-specific parsing above. Disambiguates a single matrix
    (eigenvalues) from two (product) by bracket count plus a keyword, since
    the claim text lists rows flatly rather than nesting them."""
    text = text or ""
    rows = _ROW_RE.findall(text)
    low = text.lower()
    try:
        sp = R._sympy()
        if "eigenvalue" in low and len(rows) >= 2:
            m = _rows_to_matrix(sp, rows[:2])
            eigenvals = sorted((str(v) for v in m.eigenvals().keys()))
            return {"kind": "eigenvalues", "claim": f"eigenvalues of {m.tolist()}",
                   "computed": eigenvals}
        if any(k in low for k in ("product", "multiply", "multiplying")) and len(rows) >= 4:
            a = _rows_to_matrix(sp, rows[:2])
            b = _rows_to_matrix(sp, rows[2:4])
            return {"kind": "matrix_product", "claim": f"{a.tolist()} * {b.tolist()}",
                   "computed": (a * b).tolist()}
    except Exception:
        return None
    return None


_KET_RE = re.compile(r"\|([01])>")
_SUPERPOSITION_RE = re.compile(
    r"\(\s*\|([01])>\s*([+-])\s*\|([01])>\s*\)\s*/\s*sqrt\(\s*(\d+)\s*\)", re.IGNORECASE)


def parse_ket_state(sp, text: str):
    """Parses a single basis ket "|0>"/"|1>" or an explicit equal-weight
    superposition "(|0>+|1>)/sqrt(2)" into a 2-component sympy column
    vector. Returns None for anything else - deliberately narrow (single
    qubit, real coefficients only), matching the states this app's
    quantum-computing questions actually state explicitly, not a general
    quantum-state parser."""
    basis = {"0": sp.Matrix([1, 0]), "1": sp.Matrix([0, 1])}
    compact = text.replace(" ", "")
    m = re.fullmatch(r"\|([01])>", compact)
    if m:
        return basis[m.group(1)]
    m = re.fullmatch(r"\(\|([01])>([+-])\|([01])>\)/sqrt\((\d+)\)", compact, re.IGNORECASE)
    if m:
        a, sign, b, n = m.groups()
        coeff = 1 / sp.sqrt(int(n))
        sign_val = 1 if sign == "+" else -1
        return coeff * basis[a] + sign_val * coeff * basis[b]
    return None


def apply_gate(sp, gate_name: str, state):
    """Applies a named single-qubit gate to a state vector. Only Hadamard is
    supported - the only gate this app's benchmark questions name
    explicitly; multi-gate/multi-qubit circuits (CNOT, Bell states) are
    out of scope for this deterministic layer, not silently approximated."""
    if gate_name.lower() != "hadamard":
        return None
    h = (sp.Integer(1) / sp.sqrt(2)) * sp.Matrix([[1, 1], [1, -1]])
    return h * state


def born_rule_probabilities(sp, state) -> dict:
    """|amplitude|^2 per basis state - real values, not complex amplitudes,
    since every state parse_ket_state() can produce is real-coefficient."""
    return {i: sp.nsimplify(sp.simplify(sp.Abs(state[i]) ** 2)) for i in range(state.rows)}


def execute_quantum_state_claim(text: str) -> dict | None:
    """Gate-on-state execution and Born-rule probability - folded into the
    same {"kind","claim","computed"} shape as execute_matrix_claim() above,
    single-qubit and real-coefficient only. Tries, in order: a named gate
    applied to a stated basis ket, then a directly-stated superposition
    with no named gate."""
    text = text or ""
    try:
        sp = R._sympy()
    except Exception:
        return None

    kets = _KET_RE.findall(text)
    if "hadamard" in text.lower() and kets:
        base = parse_ket_state(sp, f"|{kets[0]}>")
        result = apply_gate(sp, "hadamard", base) if base is not None else None
        if result is not None:
            probs = born_rule_probabilities(sp, result)
            return {"kind": "gate_application",
                   "claim": f"Hadamard applied to |{kets[0]}>",
                   "computed": {"state": [str(c) for c in result],
                                "probabilities": {str(k): str(v) for k, v in probs.items()}}}

    sup = _SUPERPOSITION_RE.search(text)
    if sup:
        a, sign, b, n = sup.groups()
        state = parse_ket_state(sp, f"(|{a}>{sign}|{b}>)/sqrt({n})")
        if state is not None:
            probs = born_rule_probabilities(sp, state)
            return {"kind": "born_rule_probability",
                   "claim": f"|amplitude|^2 for (|{a}>{sign}|{b}>)/sqrt({n})",
                   "computed": {str(k): str(v) for k, v in probs.items()}}
    return None


# The literal equation ("dy/dx = -k*y") usually appears only in the ORIGINAL
# QUESTION - the derivation plan talks about solving it, not restating it -
# so callers pass question+text combined, unlike execute_matrix_claim()
# above which only ever needs the derivation text.
_ODE_DY_DX_RE = re.compile(r"\bd([A-Za-z])/d([A-Za-z])\s*=\s*(.+?)(?:\s+and\b|\s+which\b|[.,]|$)")
_ODE_PRIME_RE = re.compile(r"\b([A-Za-z])'\s*=\s*(.+?)(?:\s+and\b|\s+which\b|[.,]|$)")


def execute_differential_equation_claim(question: str, text: str) -> dict | None:
    """Solves a stated first-order ODE via research.solve_ode() when the
    combined question/derivation text contains recognizable ODE notation
    ("dy/dx = ..." or "y' = ..."). Returns None (never raises) when nothing
    matches or research.solve_ode() itself can't solve it - refusing rather
    than guessing, the same discipline verification.py's checks follow."""
    combined = f"{question or ''}\n{text or ''}"
    m = _ODE_DY_DX_RE.search(combined)
    if m:
        func_name, wrt, rhs = m.group(1), m.group(2), m.group(3).strip()
    else:
        m = _ODE_PRIME_RE.search(combined)
        if not m:
            return None
        func_name, wrt, rhs = m.group(1), "x", m.group(2).strip()
    equation = f"d{func_name}/d{wrt} = {rhs}"
    result = R.solve_ode(equation, func_name=func_name, wrt=wrt)
    if not result.get("ok"):
        return None
    return {"equation": result["equation"], "solution": result["solution"]}


def execute_deterministically(question, u, pack, reasoning, computed, symbolic) -> dict:
    """The orchestrator: one structured record naming everything actually
    executed deterministically for this answer - numerical calculation,
    symbolic algebra, matrix operations, unit conversion. Never invented,
    never asked of the LLM; every field is None/empty when nothing of that
    kind applies to this particular question, which is the normal case for
    most questions, not a failure of this function."""
    text = (reasoning or {}).get("derivation_plan") or (reasoning or {}).get("text") or ""
    if not isinstance(text, str):
        text = str(text)

    numerical_calculation = None
    unit_conversions = []
    if computed and computed.get("ran"):
        numerical_calculation = {
            "formula": computed["result"].get("formula", ""),
            "inputs": computed.get("inputs", {}),
            "result": {k: v for k, v in computed["result"].items()
                      if k not in ("topic", "formula")},
        }
        unit_conversions = _result_unit_conversions(computed)

    symbolic_algebra = None
    sym = symbolic
    identity = (u or {}).get("identity") or ""
    if sym is None and identity and "=" in identity:
        lhs, _, rhs = identity.partition("=")
        sym = R.check_identity(lhs.strip(), rhs.strip())
    if sym and sym.get("ok"):
        symbolic_algebra = {"lhs": sym.get("lhs"), "rhs": sym.get("rhs"),
                            "equal": sym.get("equal"), "verdict": sym.get("verdict")}

    search_text = f"{question or ''}\n{text}"
    matrix_operations = (execute_matrix_claim(text)
                        or execute_matrix_literal_claim(search_text)
                        or execute_quantum_state_claim(search_text))
    differential_equation = execute_differential_equation_claim(question, text)

    return {
        "numerical_calculation": numerical_calculation,
        "symbolic_algebra": symbolic_algebra,
        "matrix_operations": matrix_operations,
        "unit_conversions": unit_conversions,
        "differential_equation": differential_equation,
    }
