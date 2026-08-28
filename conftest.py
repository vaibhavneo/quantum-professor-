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


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_model_gateway_singleton():
    """The ModelGateway is a process-wide singleton by design (the cooldown
    only means anything if every real request shares it) - but that same
    persistence would leak a cached mock client or a blocked-provider state
    from one test into the next. Reset it before and after every test so
    each test's own patch("openai.OpenAI", ...) is what actually gets used.
    """
    from quantum_prof.model_gateway import get_gateway
    get_gateway().reset()
    yield
    get_gateway().reset()
