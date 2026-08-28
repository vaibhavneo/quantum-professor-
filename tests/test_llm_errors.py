"""Offline tests for llm_errors.py - real openai SDK exceptions built from a
genuine minimal httpx.Request/Response, no network involved."""
import httpx
import openai

from quantum_prof.llm_errors import classify_llm_error, is_llm_sdk_error


def _status_error(cls, status_code, message="boom"):
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": message}})
    return cls(message, response=response, body={"message": message})


def test_classify_402_as_insufficient_balance():
    exc = _status_error(openai.APIStatusError, 402, "Insufficient Balance")
    kind, status, msg = classify_llm_error(exc)
    assert kind == "insufficient_balance"
    assert status == 402


def test_classify_401_as_invalid_key():
    exc = _status_error(openai.AuthenticationError, 401, "invalid api key")
    kind, status, msg = classify_llm_error(exc)
    assert kind == "invalid_key"
    assert status == 401


def test_classify_429_as_rate_limited():
    exc = _status_error(openai.RateLimitError, 429, "slow down")
    kind, status, msg = classify_llm_error(exc)
    assert kind == "rate_limited"
    assert status == 429


def test_classify_5xx_as_server_error():
    exc = _status_error(openai.APIStatusError, 503, "upstream down")
    kind, status, msg = classify_llm_error(exc)
    assert kind == "server_error"
    assert status == 503


def test_classify_connection_error_as_network_error():
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    exc = openai.APIConnectionError(request=request)
    kind, status, msg = classify_llm_error(exc)
    assert kind == "network_error"


def test_is_llm_sdk_error_true_for_sdk_exceptions():
    exc = _status_error(openai.APIStatusError, 402)
    assert is_llm_sdk_error(exc) is True


def test_non_sdk_exception_is_not_classified():
    assert is_llm_sdk_error(ValueError("not an SDK error")) is False
    assert is_llm_sdk_error(KeyError("boom")) is False
