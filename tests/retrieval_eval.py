"""Retrieval evaluation — does the right book come back for the question?

WHY THIS EXISTS BEFORE ANY TUNING

    The observation that started this: asked about cavity decoherence, the
    top hits were a general survey ("Quantum Mechanics II: Advanced Topics")
    and Haroche's "Exploring the Quantum" — the book that IS the cavity-QED
    reference, whose authors won the Nobel for the experiments — sat at rank
    four.

    That is a plausible-sounding complaint, and plausible-sounding complaints
    are exactly what should not be acted on directly. BM25 rewards term
    frequency, so a broad survey that mentions a word often can outrank a
    specialist text that develops it properly. Whether that actually costs
    answer quality is a measurement, not an intuition.

    So: this file was written BEFORE looking at any ranking output, from what
    each book demonstrably is. Every expectation below names books that are
    genuinely the authority for that question — several where several are
    legitimate. If a case is wrong, fix the CASE and say so; do not delete a
    failing one, and do not add a case chosen because it already passes.

WHAT IS MEASURED

    RECORDED BASELINE, BM25 alone, before any re-ranking existed:
        top-1 36%   hit@3 68%   hit@5 75%   spread 3.29
        specialist@3 76%   core@3 55%   popular-level books in a top-3: 2

    top1    the first hit is one of the expected books
    hit@3   an expected book appears in the first three
    hit@5   an expected book appears in the first five
    spread  distinct books among the top five — a ranking that returns one
            book five times has given the reader one opinion, not evidence

    top1 is the weakest signal and is reported, not optimised: for "solve the
    particle in a box" it is genuinely arbitrary whether Griffiths or Zettili
    comes first. hit@3 is the number that matters, because the tutor is
    handed several chunks and cites from them.

Run:  python3 tests/retrieval_eval.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (question, {substrings identifying acceptable authoritative books})
CASES = [
    # ── quantum optics and cavity QED ───────────────────────────────────────
    ("How does a cavity QED experiment watch a superposition decohere?",
     {"Exploring the quantum"}),
    ("What is a squeezed state of light?",
     {"Introduction To Quantum Optics", "An Introduction to Quantum Optics",
      "Quantum Photonics"}),
    ("Explain the Jaynes-Cummings model of an atom in a cavity.",
     {"Exploring the quantum", "Introduction To Quantum Optics",
      "An Introduction to Quantum Optics", "Atom - Photon Interactions"}),
    ("How does a laser field drive spontaneous emission from a two-level atom?",
     {"Atom - Photon Interactions", "Introduction To Quantum Optics",
      "An Introduction to Quantum Optics"}),
    ("How are photonic qubits realised on an integrated chip?",
     {"Integrated Quantum Photonics", "Quantum Photonics"}),

    # ── quantum field theory ────────────────────────────────────────────────
    ("What is renormalization in quantum field theory?",
     {"Quantum Field Theory for the Gifted Amateur",
      "Relativistic Quantum Field Theory",
      "A First Graduate Course in Quantum Field Theory"}),
    ("Explain second quantization and creation operators for a many-body system.",
     {"Quantum Field Theory for the Gifted Amateur",
      "Quantum Many-Body Physics", "Quantum Mechanics in a Nutshell",
      "A First Graduate Course in Quantum Field Theory"}),
    ("Derive the Dirac equation for a relativistic electron.",
     {"Relativistic Quantum Field Theory",
      "A First Graduate Course in Quantum Field Theory",
      "Quantum Field Theory for the Gifted Amateur"}),
    ("What is a path integral formulation of quantum mechanics?",
     {"Quantum Many-Body Physics", "Quantum Mechanics A Concise Textbook",
      "Quantum Mechanics II Advanced Topics"}),

    # ── foundations and interpretation ──────────────────────────────────────
    ("What did the EPR paper actually argue, and what was Bohr's reply?",
     {"The Einstein Paradox"}),
    ("Explain the pilot-wave interpretation and quantum equilibrium.",
     {"Beyond the Quantum"}),
    ("How did von Neumann formalise the measurement process?",
     {"Mathematical foundations of quantum mechanics"}),
    ("What is a Bell inequality and how is it violated experimentally?",
     {"Modern Quantum Theory", "The Einstein Paradox"}),
    ("Why must an observable be self-adjoint rather than merely Hermitian?",
     {"A Mathematical Primer on Quantum Mechanics",
      "Mathematical foundations of quantum mechanics",
      "Introduction to Quantum Mechanics (Horst R. Beyer)"}),

    # ── quantum information ─────────────────────────────────────────────────
    ("How does quantum teleportation transfer an unknown state?",
     {"Design of Quantum Teleportation Schemes", "Modern Quantum Theory",
      "Quantum Mechanics for Beginners"}),
    ("What is a quantum error correcting code and how does it protect a qubit?",
     {"From Classical to Quantum Coding", "Modern Quantum Theory"}),
    ("How do you numerically compute an entanglement measure for a spin chain?",
     {"Numerical Recipes in Quantum Information Theory"}),

    # ── core course material ────────────────────────────────────────────────
    ("Solve the infinite square well and find its energy eigenvalues.",
     {"Introduction to Quantum Mechanics, Third Edition",
      "Quantum Mechanics. Concepts and Applications", "Basic Quantum Mechanics",
      "Fundamentals of Quantum Mechanics"}),
    ("Use ladder operators to find the harmonic oscillator spectrum.",
     {"Mastering Quantum Mechanics Essentials", "Principles of Quantum Mechanics",
      "Introduction to Quantum Mechanics, Third Edition",
      "Quantum Mechanics. Concepts and Applications"}),
    ("Explain the radial equation for the hydrogen atom.",
     {"Introduction to Quantum Mechanics, Third Edition",
      "Quantum Mechanics. Concepts and Applications",
      "Quantum Mechanics, Volume 1", "Principles of Quantum Mechanics"}),
    ("What does the Stern-Gerlach experiment demonstrate about spin?",
     {"Quantum Mechanics  - A Paradigms Approach",
      "Quantum Mechanics The Theoretical Minimum", "Modern Quantum Mechanics"}),
    ("Explain bra-ket notation and the inner product of states.",
     {"Quantum Mechanics The Theoretical Minimum",
      "Mastering Quantum Mechanics Essentials", "Principles of Quantum Mechanics",
      "Quantum Mechanics - A Mathematical Introduction"}),
    ("How does time-independent perturbation theory correct an energy level?",
     {"Introduction to Quantum Mechanics, Third Edition",
      "Principles of Quantum Mechanics", "Modern Quantum Mechanics",
      "Quantum Mechanics, Volume 2"}),
    ("Explain the addition of two angular momenta and Clebsch-Gordan coefficients.",
     {"Quantum Mechanics, Volume 2", "Modern Quantum Mechanics",
      "Principles of Quantum Mechanics"}),
    ("What is the density matrix for a mixed state?",
     {"Quantum Mechanics, Volume 3", "Modern Quantum Theory",
      "A Condensed Course of Quantum Mechanics", "Quantum Mechanics (Mark Julian Everitt)"}),
    ("Describe identical particles, bosons and fermions, and exchange symmetry.",
     {"Quantum Mechanics, Volume 3", "Quantum Mechanics in a Nutshell",
      "Introduction to Quantum Mechanics, Third Edition"}),
    ("Explain the WKB approximation and tunnelling through a barrier.",
     {"Introduction to Quantum Mechanics, Third Edition",
      "Quantum Mechanics. Concepts and Applications",
      "Problem Solving in Quantum Mechanics", "Quantum Mechanics I The Fundamentals"}),
    ("What is scattering theory and the Born approximation?",
     {"Modern Quantum Mechanics", "Quantum Mechanics in a Nutshell",
      "Mastering Quantum Mechanics (Zoltán Papp)",
      "Quantum Mechanics - A Second Course"}),
]


def _book(source: str) -> str:
    return os.path.basename(str(source))


def evaluate(search_fn, k: int = 5) -> dict:
    rows, top1, h3, h5, spreads = [], 0, 0, 0, []
    for question, expected in CASES:
        hits = search_fn(question, top_k=k)
        books = [_book(h["source"]) for h in hits]
        def ok(name):
            return any(e.lower() in name.lower() for e in expected)
        got1 = bool(books) and ok(books[0])
        got3 = any(ok(b) for b in books[:3])
        got5 = any(ok(b) for b in books[:5])
        top1 += got1
        h3 += got3
        h5 += got5
        spreads.append(len(set(books[:5])))
        rows.append({"question": question, "expected": sorted(expected),
                     "books": books, "top1": got1, "hit3": got3, "hit5": got5})
    n = len(CASES)
    return {"n": n, "top1": top1, "hit3": h3, "hit5": h5,
            "top1_pct": top1 / n, "hit3_pct": h3 / n, "hit5_pct": h5 / n,
            "mean_spread": sum(spreads) / len(spreads), "rows": rows}


def _report(r: dict, label: str) -> None:
    print(f"══ {label} ══")
    print(f"  cases {r['n']}")
    print(f"  top-1  {r['top1']:2}/{r['n']}  ({r['top1_pct']:.0%})")
    print(f"  hit@3  {r['hit3']:2}/{r['n']}  ({r['hit3_pct']:.0%})   <- the one that matters")
    print(f"  hit@5  {r['hit5']:2}/{r['n']}  ({r['hit5_pct']:.0%})")
    print(f"  mean distinct books in top 5: {r['mean_spread']:.2f}")
    miss = [x for x in r["rows"] if not x["hit3"]]
    if miss:
        print(f"\n  MISSED at 3 ({len(miss)}):")
        for m in miss:
            print(f"    {m['question'][:64]}")
            print(f"       wanted: {', '.join(e[:34] for e in m['expected'][:3])}")
            print(f"       got   : {', '.join(b.split(' (')[0][:34] for b in m['books'][:3])}")


def main() -> int:
    import local_corpus
    r = evaluate(local_corpus.search)
    _report(r, "current ranking (BM25 + title affinity + level prior)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
