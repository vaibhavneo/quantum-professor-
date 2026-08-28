"""A minimal, self-contained OpenAI-SDK-shaped mock client for the benchmark
runner. Deliberately NOT imported from tests/test_qp_pipeline.py's FakeClient
- the benchmark is meant to be runnable on its own, independent of the
pytest suite's internal test helpers, which are free to change shape without
this permanent asset following along.

Every benchmark problem only needs two things scripted (the understand-stage
JSON reply and the Derivation Plan & Physical Interpretation reply) - the
evidence, professor, and validation stages get a harmless generic fallback,
since this benchmark measures the DETERMINISTIC pipeline (retrieval, topic
matching, the solver, verify_derivation()), not the professor's prose, which
cannot be scored without a real LLM.
"""
from __future__ import annotations

import types


def by_system_prompt(mapping: dict) -> callable:
    def _dispatch(kwargs):
        sys_prompt = kwargs["messages"][0]["content"]
        for key, reply in mapping.items():
            if key in sys_prompt:
                return reply
        return None
    return _dispatch


class _FakeCompletions:
    def __init__(self, fallback_text: str, dispatch):
        self.fallback_text = fallback_text
        self.dispatch = dispatch
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.dispatch(kwargs)
        if text is None:
            text = self.fallback_text
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                      completion_tokens_details=None)
        msg = types.SimpleNamespace(content=text)
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice], usage=usage)


class ScriptedClient:
    """OpenAI-client-shaped. `mapping` routes a canned reply by matching a
    substring of the call's system prompt; anything unmatched gets a
    harmless generic fallback (empty JSON for the JSON-parsing stages,
    "stub answer" for the professor's free-text stage)."""

    def __init__(self, mapping: dict, fallback_text: str = "{}"):
        self.completions = _FakeCompletions(fallback_text, by_system_prompt(mapping))
        self.chat = types.SimpleNamespace(completions=self.completions)

    @property
    def calls(self):
        return self.completions.calls
