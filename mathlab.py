"""Math Lab — type a problem in mathematical terms and have it solved.

The Solver tab only ever offered eight canned problems with fixed parameter
boxes, so there was no way to hand the app an arbitrary expression, an ODE, or
a potential of your own. This module takes free-form mathematical input.

Everything here is computed by sympy or numpy, never by the model. The
professor may talk *about* a result; the engine decides what it is.

Three surfaces:

  evaluate()      one symbolic operation — simplify, differentiate, integrate,
                  solve, limit, series, eigenvalues of a matrix.
  solve_ode()     an ordinary differential equation, including the
                  time-dependent Schrodinger equation written directly.
  schrodinger()   bound states of the 1-D time-independent Schrodinger
                  equation for an arbitrary V(x), by finite difference. This is
                  the one a student actually wants: give it a potential, get
                  energies and wavefunctions back.
"""
from __future__ import annotations

import re

import numpy as np
import sympy as sp

# CODATA, matching physics.py so the two engines never disagree
HBAR = 1.054571817e-34
M_E = 9.1093837015e-31
E_CHARGE = 1.602176634e-19
NM = 1e-9

# Only these names resolve. sympify() with an open namespace will happily
# evaluate attribute access and calls, so the namespace is the sandbox.
_ALLOWED = {n: getattr(sp, n) for n in (
    "sin cos tan asin acos atan sinh cosh tanh exp log sqrt Abs "
    "pi I oo Symbol Function Derivative Integral Sum factorial "
    "erf gamma besselj legendre hermite laguerre conjugate re im "
    # Piecewise potentials are the bread and butter of 1-D QM — square wells,
    # barriers, steps. Without these a user can only type smooth functions.
    "Piecewise Heaviside Min Max sign floor ceiling".split())}
_ALLOWED.update({"ln": sp.log, "Infinity": sp.oo,
                 # needed because the indicator rewrite below emits `True`,
                 # and _ns() would otherwise turn it into a plain symbol
                 "True": sp.true, "False": sp.false})

# E is deliberately NOT bound to Euler's number. sympy maps E -> 2.718..., so
# the Schrodinger equation "-hbar**2/(2*m)*psi''(x) = E*psi(x)" silently
# solved for energy = e and returned exp(1/2) inside the exponent. In a physics
# app E means energy far more often than it means Euler's constant, so it is
# left to become a plain symbol; write exp(1) if you want the number.

_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ns(text: str, extra: dict | None = None) -> dict:
    """Namespace for sympify: known functions plus real symbols for the rest."""
    ns = dict(_ALLOWED)
    ns.update(extra or {})
    for name in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)):
        if name not in ns and _SAFE_NAME.match(name):
            ns[name] = sp.Symbol(name, real=True)
    return ns


_CMP = ("<", ">", "=")


def _indicator_rewrite(text: str) -> str:
    """`-V0*(Abs(x) < a)` -> `-V0*Piecewise((1, Abs(x) < a), (0, True))`.

    Every textbook writes a square well as a constant times an inequality, but
    sympify evaluates the multiplication while parsing and dies with
    "unsupported operand type(s) for *: 'Float' and 'StrictLessThan'" before
    any post-processing gets a chance. So the rewrite has to happen on the
    text. A parenthesised comparison means its indicator function — that is
    what the notation has always meant — so that is what it becomes.
    """
    def match(s, i):                     # index of ')' closing s[i] == '('
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
                if depth == 0:
                    return j
        return -1

    def has_cmp(s):                      # a comparison at this paren level
        depth = 0
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0 and ch in _CMP:
                return True
        return False

    out, i = [], 0
    while i < len(text):
        if text[i] == "(":
            j = match(text, i)
            if j < 0:                    # unbalanced: leave it for sympify to report
                out.append(text[i]); i += 1; continue
            inner = text[i + 1:j]
            out.append(f"Piecewise((1, {_indicator_rewrite(inner)}), (0, True))"
                       if has_cmp(inner) else f"({_indicator_rewrite(inner)})")
            i = j + 1
        else:
            out.append(text[i]); i += 1
    return "".join(out)


def _parse(text: str, extra: dict | None = None, indicators: bool = False):
    if len(text) > 2000:
        raise ValueError("expression too long")
    if "__" in text or "import" in text:
        raise ValueError("disallowed token")
    if indicators:
        text = _indicator_rewrite(text)
    return sp.sympify(text, locals=_ns(text, extra))


def _bools_to_piecewise(expr):
    """A bare condition as the whole potential, e.g. `V = Abs(x) < a`.

    _indicator_rewrite handles the parenthesised form; this catches the case
    where the comparison *is* the expression and so was never wrapped. Only
    the top level is examined on purpose — walking the tree also rewrites the
    ExprCondPairs inside a Piecewise the rewrite just built, which fails with
    "Expecting Boolean or bool but got Zero".
    """
    from sympy.logic.boolalg import Boolean
    from sympy.core.relational import Relational

    if isinstance(expr, (Relational, Boolean)):
        return sp.Piecewise((1, expr), (0, True))
    return expr


def _count_nodes(psi, rel_floor: float = 0.02) -> int:
    """Interior zero crossings, ignoring the numerically-zero tails.

    A node is a genuine sign change of the wavefunction. Where |psi| has
    decayed to rounding error the sign is meaningless, so only samples above a
    small fraction of the peak amplitude are considered.
    """
    a = np.abs(psi)
    peak = a.max()
    if peak <= 0:
        return 0
    sig = psi[a > rel_floor * peak]
    if sig.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(np.signbit(sig))))


def _out(expr):
    return {"result": str(expr), "latex": sp.latex(expr)}


# ── 1. general symbolic operations ────────────────────────────────────────

OPERATIONS = ["simplify", "expand", "factor", "differentiate", "integrate",
              "solve", "limit", "series", "eigenvalues", "evaluate"]


def evaluate(expression: str, operation: str = "simplify", variable: str = "x",
             at=None, order: int = 6, subs: dict | None = None) -> dict:
    """Run one symbolic operation on a typed expression."""
    try:
        sub_syms = {}
        if subs:
            for k, v in subs.items():
                if _SAFE_NAME.match(k):
                    sub_syms[sp.Symbol(k, real=True)] = sp.sympify(v)
        e = _parse(expression)
        x = sp.Symbol(variable, real=True)

        if operation == "eigenvalues":
            M = sp.Matrix(e) if not isinstance(e, sp.MatrixBase) else e
            evs = M.eigenvals()
            return {"ok": True, "operation": operation, "input": str(e),
                    "eigenvalues": [{"value": str(k), "latex": sp.latex(k),
                                     "multiplicity": int(v)} for k, v in evs.items()],
                    **_out(sp.Matrix(list(evs.keys())))}

        ops = {
            "simplify":      lambda: sp.simplify(e),
            "expand":        lambda: sp.expand(e),
            "factor":        lambda: sp.factor(e),
            "differentiate": lambda: sp.diff(e, x),
            "integrate":     lambda: sp.integrate(e, x),
            "solve":         lambda: sp.solve(sp.Eq(e, 0) if not isinstance(e, sp.Eq) else e, x),
            "limit":         lambda: sp.limit(e, x, sp.sympify(at if at is not None else 0)),
            "series":        lambda: sp.series(e, x, 0, int(order)),
            "evaluate":      lambda: (e.subs(sub_syms) if sub_syms else e).evalf(),
        }
        if operation not in ops:
            return {"ok": False, "error": f"unknown operation {operation!r}",
                    "available": OPERATIONS}
        res = ops[operation]()
        if sub_syms and operation != "evaluate":
            res = sp.simplify(res.subs(sub_syms)) if hasattr(res, "subs") else res
        payload = {"ok": True, "operation": operation, "input": str(e),
                   "input_latex": sp.latex(e), **_out(res)}
        # solve() returns a list; str(list) is unreadable next to the roots
        # themselves, so hand them over individually as well.
        if isinstance(res, (list, tuple)):
            payload["solutions"] = [sp.latex(r) for r in res]
        # a numeric value alongside the closed form, when one exists
        try:
            if hasattr(res, "free_symbols") and not res.free_symbols:
                payload["numeric"] = float(res)
        except (TypeError, ValueError):
            pass
        return payload
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ── 2. differential equations ─────────────────────────────────────────────

def solve_ode(equation: str, func: str = "psi", variable: str = "x",
              ics: dict | None = None) -> dict:
    """Solve an ODE written as an equation in `func(variable)`.

    Accepts either 'Eq(lhs, rhs)' or plain 'lhs = rhs'. Derivatives may be
    written Derivative(psi(x), x, 2) or psi''(x).
    """
    try:
        x = sp.Symbol(variable, real=True)
        f = sp.Function(func)
        text = equation.strip()
        # psi''(x) → Derivative(psi(x), x, 2)
        for n in (4, 3, 2, 1):
            text = re.sub(rf"{func}{chr(39)*n}\(\s*{variable}\s*\)",
                          f"Derivative({func}({variable}), {variable}, {n})", text)
        if "=" in text and not text.startswith("Eq("):
            lhs, _, rhs = text.partition("=")
            expr = sp.Eq(_parse(lhs, {func: f}), _parse(rhs, {func: f}))
        else:
            expr = _parse(text, {func: f})
            if not isinstance(expr, sp.Eq):
                expr = sp.Eq(expr, 0)
        sol = sp.dsolve(expr, f(x))
        out = {"ok": True, "equation": str(expr), "equation_latex": sp.latex(expr),
               "solution": str(sol), "solution_latex": sp.latex(sol),
               # kept under the old names too: existing callers read these
               "general_solution": str(sol), "latex": sp.latex(sol)}
        try:
            # What kind of equation this is, in sympy's own words — useful when
            # you are trying to recognise the standard form you were taught.
            kinds = sp.classify_ode(expr, f(x))
            if kinds:
                out["classification"] = str(kinds[0]).replace("_", " ")
        except Exception:
            pass
        if ics:
            try:
                parsed = {_parse(k, {func: f}): sp.sympify(v) for k, v in ics.items()}
                part = sp.dsolve(expr, f(x), ics=parsed)
                out["particular_solution"] = str(part)
                out["particular_latex"] = sp.latex(part)
            except Exception as exc:
                out["ics_error"] = f"{type(exc).__name__}: {exc}"
        return out
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ── 3. the time-independent Schrodinger equation, any V(x) ────────────────

# Potentials are written the way a physicist says them out loud: x in
# nanometres, V in electron-volts. Writing the harmonic oscillator as
# 0.5*m*omega**2*x**2 forces the reader into kilograms, metres and joules to
# get a sane number out, which is not how anyone thinks at the desk. Every
# preset below therefore carries its own parameters in eV/nm, and `units="SI"`
# stays available for the rare case where you really do want joules.
PRESET_POTENTIALS = {
    "infinite-well": {
        "V": "0", "x_min": 0.0, "x_max": 1.0, "params": {},
        "label": "Infinite square well (hard walls at the domain edges)",
        "note": "E_n = n²h²/8mL². With L = 1 nm, E_1 = 0.376 eV."},
    "harmonic": {
        "V": "0.5*k*x**2", "x_min": -5.0, "x_max": 5.0, "params": {"k": 1.0},
        "label": "Harmonic oscillator",
        "note": "k in eV/nm². Levels are equally spaced by ħω, ω = √(k/m)."},
    "finite-well": {
        "V": "-V0*(Abs(x) < a)", "x_min": -3.0, "x_max": 3.0,
        "params": {"V0": 5.0, "a": 0.5},
        "label": "Finite square well",
        "note": "Depth V0 in eV, half-width a in nm. Bound states sit below 0."},
    "barrier": {
        "V": "V0*(Abs(x) < a)", "x_min": -3.0, "x_max": 3.0,
        "params": {"V0": 5.0, "a": 0.2},
        "label": "Rectangular barrier",
        "note": "Height V0 in eV, half-width a in nm."},
    "linear": {
        "V": "F*x", "x_min": 0.0, "x_max": 5.0, "params": {"F": 1.0},
        "label": "Linear (triangular) potential",
        "note": "Field F in eV/nm — the quantum bouncing ball."},
    "morse": {
        "V": "D*(1 - exp(-alpha*x))**2", "x_min": -0.5, "x_max": 5.0,
        "params": {"D": 5.0, "alpha": 1.0},
        "label": "Morse potential (a real diatomic bond)",
        "note": "Well depth D in eV, width alpha in 1/nm. Levels crowd near the top."},
    "coulomb-1d": {
        "V": "-1.43996/sqrt(x**2 + soft**2)", "x_min": -2.0, "x_max": 2.0,
        "params": {"soft": 0.05},
        "label": "Softened Coulomb (1-D hydrogen)",
        "note": "e²/4πε₀ = 1.43996 eV·nm. `soft` removes the singularity at the origin."},
}

UNIT_SYSTEMS = {
    "eV_nm": "x in nanometres, V(x) in electron-volts (what you'd write by hand)",
    "SI": "x in metres, V(x) in joules; `m` and `hbar` are in scope",
}


def schrodinger(potential: str = "0.5*k*x**2",
                x_min_nm: float = -5.0, x_max_nm: float = 5.0,
                n_points: int = 800, n_states: int = 5,
                mass_me: float = 1.0, params: dict | None = None,
                units: str = "eV_nm") -> dict:
    """Bound states of -ħ²/2m ψ'' + V(x)ψ = Eψ on [x_min, x_max], in nm and eV.

    Finite difference with Dirichlet walls: the box edges are hard walls, so
    choose a domain wide enough that the states you care about have decayed,
    or the walls will shift the energies. That caveat is returned with the
    result rather than left for the reader to discover.

    `units` controls only how V(x) is *read*. Energies always come back in eV
    and positions in nm, because that is what the answer gets quoted in.
    """
    try:
        if not 50 <= n_points <= 5000:
            return {"ok": False, "error": "n_points must be between 50 and 5000"}
        if x_max_nm <= x_min_nm:
            return {"ok": False, "error": "x_max must exceed x_min"}
        if units not in UNIT_SYSTEMS:
            return {"ok": False, "error": f"units must be one of {sorted(UNIT_SYSTEMS)}"}

        x_nm = np.linspace(x_min_nm, x_max_nm, int(n_points))
        x_m = x_nm * NM
        dx = x_m[1] - x_m[0]
        m = mass_me * M_E

        # V(x) is evaluated symbolically then sampled, so any typed expression
        # works — presets are a convenience, not a restriction.
        xs = sp.Symbol("x", real=True)
        extra = {"m": sp.Float(m), "hbar": sp.Float(HBAR)}
        for k, v in (params or {}).items():
            if _SAFE_NAME.match(k):
                extra[k] = sp.Float(float(v))
        Vexpr = _bools_to_piecewise(
            _parse(potential, {**extra, "x": xs}, indicators=True))
        free = {s.name for s in getattr(Vexpr, "free_symbols", set())} - {"x"}
        if free:
            return {"ok": False,
                    "error": f"undefined symbol(s) in V(x): {sorted(free)}",
                    "hint": "give each one a value, e.g. V0=5, a=0.5"}
        Vf = sp.lambdify(xs, Vexpr, "numpy")
        # Sample in whichever coordinate the expression was written in, then
        # convert the result to joules for the eigensolve.
        x_sample = x_nm if units == "eV_nm" else x_m
        V_raw = np.asarray(Vf(x_sample), dtype=float) * np.ones_like(x_m)
        V_J = V_raw * E_CHARGE if units == "eV_nm" else V_raw

        # interior points only — Dirichlet walls at both ends
        N = len(x_m) - 2
        coeff = HBAR ** 2 / (2 * m * dx ** 2)
        main = 2 * coeff + V_J[1:-1]
        off = -coeff * np.ones(N - 1)
        E_J, vecs = np.linalg.eigh(np.diag(main) + np.diag(off, 1) + np.diag(off, -1))

        k = min(int(n_states), N)
        states = []
        for i in range(k):
            psi = np.concatenate(([0.0], vecs[:, i], [0.0]))
            norm = np.sqrt(np.trapezoid(psi ** 2, x_m)) if hasattr(np, "trapezoid") \
                else np.sqrt(np.trapz(psi ** 2, x_m))
            psi = psi / norm
            step = max(1, len(x_nm) // 160)          # thin for transport
            states.append({
                "n": i + 1,
                "energy_eV": float(E_J[i] / E_CHARGE),
                "energy_J": float(E_J[i]),
                # Count sign changes only where the wavefunction is actually
                # non-zero. In the exponentially-decaying tails psi is at
                # rounding-error level and flips sign at random, which counted
                # the harmonic oscillator's first five states as 7, 37, 5, 17, 5
                # nodes instead of 0, 1, 2, 3, 4.
                "nodes": _count_nodes(psi),
                "x_nm": [round(float(v), 5) for v in x_nm[::step]],
                "psi": [round(float(v), 6) for v in psi[::step]],
            })
        return {
            "ok": True,
            "method": "finite-difference eigensolve of the 1-D TISE",
            "potential": str(Vexpr), "potential_latex": sp.latex(Vexpr),
            "units": units,
            "domain_nm": [x_min_nm, x_max_nm], "n_points": int(n_points),
            "mass_me": mass_me,
            "V_min_eV": float(V_J.min() / E_CHARGE),
            "V_max_eV": float(V_J.max() / E_CHARGE),
            "V_curve": {"x_nm": [round(float(v), 5) for v in x_nm[::max(1, len(x_nm)//160)]],
                        "V_eV": [round(float(v), 6) for v in
                                 (V_J / E_CHARGE)[::max(1, len(x_nm)//160)]]},
            "states": states,
            "caveat": ("Hard walls sit at the domain edges. Widen the domain until the "
                       "energies stop moving, or the box itself is setting them."),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "schrodinger":
        print(json.dumps({k: v for k, v in schrodinger().items() if k != "states"}, indent=1))
    else:
        print(json.dumps(evaluate("exp(-x**2)", "integrate"), indent=1))
