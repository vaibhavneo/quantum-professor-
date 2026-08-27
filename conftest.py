"""Registers this checkout as an importable ``quantum_prof`` package.

The directory is named ``quantum-professor-`` (hyphens, trailing dash), which
Python can't import as-is, and there's no setup.py/pyproject.toml to bridge
the gap. Tests import from ``quantum_prof.<module>`` regardless, so point a
synthetic package entry at this directory before collection starts.
"""
import pathlib
import sys
import types

_root = pathlib.Path(__file__).resolve().parent
if "quantum_prof" not in sys.modules:
    pkg = types.ModuleType("quantum_prof")
    pkg.__path__ = [str(_root)]
    sys.modules["quantum_prof"] = pkg
