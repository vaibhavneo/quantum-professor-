"""ModelGateway - the one place that knows how to obtain an LLM client and
remembers when the provider is broken, so the rest of the pipeline never has
to think about DeepSeek specifically, and a known-bad key or empty balance
doesn't get re-attempted on every single question.

A permanent-looking failure (invalid key, zero balance, account forbidden)
blocks new attempts for a cooldown window - repeating a call that failed
this way wastes nothing but time, since none of them will ever succeed
until a human fixes the account. A transient failure (rate limit, timeout,
a network blip, an unclassified 5xx) does NOT block - it may well have
cleared by the very next request, and the whole point of the cooldown is
to stop *doomed* retries, not to give up on a provider that might recover.
"""
from __future__ import annotations

import time

try:
    from .llm_errors import ProviderError
    from .tutor import _api_key
except ImportError:
    from llm_errors import ProviderError
    from tutor import _api_key

PERMANENT_KINDS = {"invalid_key", "insufficient_balance", "missing_key", "forbidden"}
DEFAULT_COOLDOWN_SECONDS = 60.0


class ModelGateway:
    """Independent of any one provider in principle - today it only knows
    how to build the DeepSeek-backed OpenAI-SDK client, but nothing above
    this line (qp_pipeline.py's _call()) reaches into that detail directly.
    """

    def __init__(self, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS):
        self._client = None
        self._cooldown_seconds = cooldown_seconds
        self._blocked_until = 0.0
        self._blocked_error: ProviderError | None = None

    def blocked(self) -> ProviderError | None:
        """A remembered permanent failure, if we're still inside its
        cooldown window - callers should skip the network call entirely
        and go straight to the offline path when this returns non-None."""
        if self._blocked_error is not None and time.monotonic() < self._blocked_until:
            return self._blocked_error
        return None

    def client(self):
        """The shared LLM client, built once per process and reused.
        Raises RuntimeError if no key is configured at all - callers should
        check that (or blocked()) before calling this."""
        if self._client is None:
            key = _api_key()
            if not key:
                raise RuntimeError("no API key configured")
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url="https://api.deepseek.com",
                                  timeout=300.0, max_retries=1)
        return self._client

    def record_failure(self, exc: ProviderError) -> None:
        if exc.kind in PERMANENT_KINDS:
            self._blocked_until = time.monotonic() + self._cooldown_seconds
            self._blocked_error = exc

    def record_success(self) -> None:
        self._blocked_until = 0.0
        self._blocked_error = None

    def reset(self) -> None:
        """Test/ops hook - clears the cooldown and the cached client so a
        fixed key/balance takes effect immediately rather than waiting out
        the cooldown, and so tests never leak state into each other."""
        self._client = None
        self._blocked_until = 0.0
        self._blocked_error = None


_GATEWAY = ModelGateway()


def get_gateway() -> ModelGateway:
    """Process-wide singleton - the cooldown only means anything if every
    request goes through the same instance."""
    return _GATEWAY
