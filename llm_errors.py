"""Classifies a DeepSeek/OpenAI-SDK-shaped exception into a stable kind, so
the pipeline can decide "fall back to offline mode" versus "this is a real
bug, let it raise" without scattering isinstance() checks everywhere.

DeepSeek's 402 Insufficient Balance has no dedicated exception class in the
openai SDK (only 401/403/404/409/422/429 do - everything else 4xx/5xx falls
through to the bare APIStatusError), so classification keys off the numeric
status_code every HTTP-shaped SDK error carries, not the exception's class.
"""
from __future__ import annotations

import openai

_STATUS_KIND = {
    401: "invalid_key",
    402: "insufficient_balance",
    403: "forbidden",
    429: "rate_limited",
}


def is_llm_sdk_error(exc: Exception) -> bool:
    """True only for an error the LLM SDK itself raised - never for a bug in
    our own code (KeyError, AttributeError, ...), which must always
    propagate as a real error rather than being read as a provider outage."""
    return isinstance(exc, openai.OpenAIError)


def classify_llm_error(exc: Exception) -> tuple[str, int | None, str]:
    """(kind, http_status, user_message) for any openai SDK error.

    Call only after is_llm_sdk_error(exc) confirms this is actually an SDK
    error - this function doesn't re-check, it just classifies.
    """
    status = getattr(exc, "status_code", None)
    if status in _STATUS_KIND:
        kind = _STATUS_KIND[status]
    elif isinstance(status, int) and status >= 500:
        kind = "server_error"
    elif isinstance(status, int):
        kind = "request_error"
    elif isinstance(exc, openai.APITimeoutError):
        kind = "timeout"
    elif isinstance(exc, openai.APIConnectionError):
        kind = "network_error"
    else:
        kind = "unknown_provider_error"

    messages = {
        "invalid_key": "The configured DeepSeek API key is invalid.",
        "insufficient_balance": "The DeepSeek account has run out of balance.",
        "forbidden": "The DeepSeek account isn't permitted to use this model.",
        "rate_limited": "DeepSeek is rate-limiting requests right now.",
        "server_error": "DeepSeek's servers returned an error.",
        "request_error": "DeepSeek rejected the request.",
        "timeout": "DeepSeek didn't respond in time.",
        "network_error": "Couldn't reach DeepSeek's servers.",
        "unknown_provider_error": "DeepSeek returned an unexpected error.",
    }
    return kind, status, messages[kind]


class ProviderError(RuntimeError):
    """Raised in place of a caught SDK error so the pipeline can catch one
    specific type and fall back to offline synthesis, while any other
    exception (a real bug) still propagates untouched."""

    def __init__(self, kind: str, http_status: int | None, user_message: str,
                original: Exception):
        super().__init__(user_message)
        self.kind = kind
        self.http_status = http_status
        self.user_message = user_message
        self.original = original
