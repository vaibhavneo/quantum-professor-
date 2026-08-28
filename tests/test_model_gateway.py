"""Offline tests for model_gateway.py - no network, no real client built
(client() is only exercised far enough to prove it raises without a key;
constructing a real openai.OpenAI object needs no network call either, but
we don't need to prove that here - qp_pipeline.py's own tests already patch
openai.OpenAI directly for the full run() path)."""
import pytest

from quantum_prof.llm_errors import ProviderError
from quantum_prof.model_gateway import PERMANENT_KINDS, ModelGateway


def _err(kind, status=None):
    return ProviderError(kind, status, f"simulated {kind}", RuntimeError("x"))


def test_not_blocked_initially():
    gw = ModelGateway()
    assert gw.blocked() is None


def test_permanent_failure_blocks_within_cooldown():
    gw = ModelGateway(cooldown_seconds=60)
    gw.record_failure(_err("insufficient_balance", 402))
    blocked = gw.blocked()
    assert blocked is not None
    assert blocked.kind == "insufficient_balance"


def test_transient_failure_does_not_block():
    gw = ModelGateway(cooldown_seconds=60)
    gw.record_failure(_err("rate_limited", 429))
    assert gw.blocked() is None


def test_timeout_does_not_block():
    gw = ModelGateway(cooldown_seconds=60)
    gw.record_failure(_err("timeout"))
    assert gw.blocked() is None


def test_network_error_does_not_block():
    gw = ModelGateway(cooldown_seconds=60)
    gw.record_failure(_err("network_error"))
    assert gw.blocked() is None


def test_cooldown_expires(monkeypatch):
    from quantum_prof import model_gateway
    gw = ModelGateway(cooldown_seconds=10)
    t = [1000.0]
    monkeypatch.setattr(model_gateway.time, "monotonic", lambda: t[0])
    gw.record_failure(_err("invalid_key", 401))
    assert gw.blocked() is not None
    t[0] += 11
    assert gw.blocked() is None


def test_record_success_clears_a_block():
    gw = ModelGateway(cooldown_seconds=60)
    gw.record_failure(_err("invalid_key", 401))
    assert gw.blocked() is not None
    gw.record_success()
    assert gw.blocked() is None


def test_reset_clears_client_and_block():
    gw = ModelGateway()
    gw.record_failure(_err("invalid_key", 401))
    gw.reset()
    assert gw.blocked() is None
    assert gw._client is None


def test_client_raises_without_a_key(monkeypatch):
    from quantum_prof import model_gateway
    monkeypatch.setattr(model_gateway, "_api_key", lambda: "")
    gw = ModelGateway()
    with pytest.raises(RuntimeError):
        gw.client()


def test_client_is_cached_across_calls(monkeypatch):
    from quantum_prof import model_gateway
    monkeypatch.setattr(model_gateway, "_api_key", lambda: "fake-key")
    gw = ModelGateway()
    c1 = gw.client()
    c2 = gw.client()
    assert c1 is c2


def test_all_permanent_kinds_actually_block():
    for kind in PERMANENT_KINDS:
        gw = ModelGateway(cooldown_seconds=60)
        gw.record_failure(_err(kind))
        assert gw.blocked() is not None, f"{kind} should block"


def test_get_gateway_is_a_process_wide_singleton():
    from quantum_prof.model_gateway import get_gateway
    assert get_gateway() is get_gateway()
