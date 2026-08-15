"""End-to-end smoke test against a deployed Quantum Professor.

    python3 smoke.py https://your-app.up.railway.app

Checks the keyless surfaces, the whole Math Lab, and one full streaming
answer. Every numeric assertion has a closed form behind it, so a pass means
the deployed solver is right, not merely reachable.
"""
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "").rstrip("/")
if not BASE:
    sys.exit("usage: smoke.py <base-url>")

fails = []


def get(path, timeout=60):
    url = BASE + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:                       # DNS, TLS, refused, timeout
        return None, str(e).encode()


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  — ' + detail) if detail else ''}")
    if not ok:
        fails.append(label)
    return ok


print(f"\n=== {BASE} ===\n[static]")
st, body = get("/")
check("GET /", st == 200 and b"Quantum Professor" in body, f"HTTP {st}, {len(body)} bytes")
check("Math Lab tab present", b'data-tab="math"' in body)
check("curriculum blob refreshed", b'"quantum-statistical-mechanics"' in body)
st, body = get("/vendor/katex/katex.min.js")
check("vendored KaTeX served", st == 200 and len(body) > 100_000, f"HTTP {st}, {len(body)} bytes")

print("[keyless api]")
for p in ("topics", "books", "problems", "mathlab"):
    st, body = get(f"/api/{p}")
    check(f"/api/{p}", st == 200, f"HTTP {st}")
st, body = get("/api/topics")
if st == 200:
    n = len(json.loads(body)["topics"])
    check("41 topics live", n == 41, f"got {n}")

print("[solver]")
st, body = get("/api/solve?topic=particle-in-a-box&n=2&L=1e-9")
if check("/api/solve", st == 200, f"HTTP {st}"):
    e = json.loads(body)["energy_eV"]
    check("n=2 in a 1nm box = 1.504121 eV", abs(e - 1.504121) < 1e-5, f"got {e}")
st, _ = get("/api/solve?topic=particle-in-a-box&L_nm=1")
check("bad parameter returns 400 not a dropped connection", st == 400, f"HTTP {st}")

print("[math lab]")
st, body = get("/api/math?expr=x**2-4&op=factor")
check("symbolic factor", st == 200 and b"(x - 2)*(x + 2)" in body, f"HTTP {st}")
st, body = get("/api/ode?" + urllib.parse.urlencode(
    {"eq": "Derivative(psi(x), x, 2) + k**2*psi(x) = 0", "func": "psi", "var": "x"}))
check("ODE solved", st == 200 and b"C1" in body, f"HTTP {st}")

st, body = get("/api/schrodinger?V=0&xmin=0&xmax=1&states=4&points=2000")
if check("TISE endpoint", st == 200, f"HTTP {st}"):
    d = json.loads(body)
    HBAR, ME, NM, EC = 1.054571817e-34, 9.1093837015e-31, 1e-9, 1.602176634e-19
    worst = max(abs(s["energy_eV"] - s["n"] ** 2 * (math.pi * HBAR) ** 2
                    / (2 * ME * NM ** 2) / EC) for s in d["states"])
    check("infinite-well energies match n^2*0.376 eV", worst < 2e-3, f"max err {worst:.2e}")
    check("node counts 0,1,2,3", [s["nodes"] for s in d["states"]] == [0, 1, 2, 3])
    check("wavefunctions returned for plotting", len(d["states"][0]["psi"]) > 50)

st, body = get("/api/schrodinger?" + urllib.parse.urlencode(
    {"V": "-V0*(Abs(x) < a)", "params": "V0=5,a=0.5", "xmin": -3, "xmax": 3,
     "states": 8, "points": 3000}))
if st == 200:
    d = json.loads(body)
    bound = [s for s in d["states"] if s["energy_eV"] < 0]
    z0 = math.sqrt(2 * 9.1093837015e-31 * 5 * 1.602176634e-19 * (0.5e-9) ** 2) / 1.054571817e-34
    check("piecewise potential parses (the bug that killed every well)",
          d.get("ok") is True)
    check("finite well bound-state count matches z0",
          len(bound) == math.ceil(z0 * 2 / math.pi), f"{len(bound)} bound")

print("[streaming answer]")
q = urllib.parse.urlencode({"q": "why is the ground state energy of a box not zero",
                            "mode": "explain", "depth": "intro"})
st, body = get("/api/ask?" + q, timeout=300)
if check("/api/ask streams", st == 200, f"HTTP {st}"):
    text = body.decode("utf-8", "replace")
    stages = [ln.split(": ", 1)[1] for ln in text.splitlines() if ln.startswith("event: ")]
    check("pipeline stages streamed", len(set(stages)) >= 4, ", ".join(sorted(set(stages))))
    check("no API-key error", "No DEEPSEEK_API_KEY" not in text,
          "set DEEPSEEK_API_KEY on the service" if "No DEEPSEEK_API_KEY" in text else "")
    check("an answer came back", '"prose"' in text)
    low = text.lower()
    for phrase in ("no source loaded", "nothing in the curriculum",
                   "the curriculum does not cover"):
        check(f"opener free of {phrase!r}", low.count(phrase) == 0)

print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILED: ' + str(fails)}")
sys.exit(1 if fails else 0)
