"""Offline tests for the qp_pipeline LLM-facing stages.

No network call anywhere - a fake OpenAI-shaped client stands in for the real
DeepSeek client, matching how this app's LLM boundary is verified elsewhere
when the account has no balance to spend on live calls.

These specifically guard the bug where evidence_engine()'s "off_topic"
judgement was computed but never enforced: reasoning_engine() and
professor_engine() used to build their source context from every retrieved
book/paper regardless of what the relevance gate said about it.
"""
import pathlib
import re
import types
from unittest.mock import patch

import httpx
import openai

from quantum_prof import qp_pipeline as qp


def by_system_prompt(mapping):
    """Dispatcher for _FakeCompletions(dispatch=...): routes a canned reply by
    matching the call's system prompt against a substring key, so a full
    run() traversal can hand each pipeline stage its own correct content
    instead of one blanket string. Falls back to None (caller's default)
    when nothing matches."""
    def _dispatch(kwargs):
        sys_prompt = kwargs["messages"][0]["content"]
        for key, reply in mapping.items():
            if key in sys_prompt:
                return reply
        return None
    return _dispatch


def make_status_error(cls, status_code, message="boom"):
    """A real openai SDK status error, not a mock - built from a genuine
    minimal httpx.Request/Response so provider-failure tests exercise the
    same exception shape DeepSeek actually raises."""
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": message}})
    return cls(message, response=response, body={"message": message})


class _FakeCompletions:
    def __init__(self, text="stub answer", responses=None, raise_on_call=None, dispatch=None):
        self.text = text
        self.responses = responses
        self.raise_on_call = raise_on_call
        self.dispatch = dispatch
        self.calls = []
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        n = len(self.calls)
        self.last_kwargs = kwargs
        if self.raise_on_call is not None:
            exc = (self.raise_on_call(kwargs, n) if callable(self.raise_on_call)
                   else self.raise_on_call)
            if exc is not None:
                raise exc
        text = self.text
        if self.dispatch is not None:
            picked = self.dispatch(kwargs)
            if picked is not None:
                text = picked
        elif self.responses is not None:
            text = self.responses[min(n - 1, len(self.responses) - 1)]
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                       completion_tokens_details=None)
        msg = types.SimpleNamespace(content=text)
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    @property
    def last_prompt(self):
        return self.last_kwargs["messages"][1]["content"]


class FakeClient:
    def __init__(self, text="stub answer", responses=None, raise_on_call=None, dispatch=None):
        self.completions = _FakeCompletions(text, responses=responses,
                                            raise_on_call=raise_on_call, dispatch=dispatch)
        self.chat = types.SimpleNamespace(completions=self.completions)


# ── Phase 1: test-harness extensions ────────────────────────────────────────

def test_fakeclient_responses_queue_returns_per_call():
    client = FakeClient(responses=["first", "second", "third"])
    assert client.completions.create(messages=[{"role": "system", "content": "s"},
                                                {"role": "user", "content": "u"}]
                                     ).choices[0].message.content == "first"
    assert client.completions.create(messages=[{"role": "system", "content": "s"},
                                                {"role": "user", "content": "u"}]
                                     ).choices[0].message.content == "second"
    # exhausted queue holds its last entry rather than raising an IndexError
    assert client.completions.create(messages=[{"role": "system", "content": "s"},
                                                {"role": "user", "content": "u"}]
                                     ).choices[0].message.content == "third"
    assert client.completions.create(messages=[{"role": "system", "content": "s"},
                                                {"role": "user", "content": "u"}]
                                     ).choices[0].message.content == "third"
    assert len(client.completions.calls) == 4


def test_fakeclient_raise_on_call_raises_on_matching_call_number():
    boom = make_status_error(openai.APIStatusError, 429, "slow down")
    client = FakeClient(raise_on_call=lambda kwargs, n: boom if n == 2 else None)
    msg = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    client.completions.create(messages=msg)          # call 1: fine
    try:
        client.completions.create(messages=msg)      # call 2: raises
        assert False, "expected the 429 to raise"
    except openai.APIStatusError as exc:
        assert exc.status_code == 429
    client.completions.create(messages=msg)          # call 3: fine again


def test_fakeclient_dispatch_by_system_prompt_routes_correctly():
    dispatch = by_system_prompt({"You classify a physics question": "understand-reply",
                                 "You are the evidence stage": "evidence-reply"})
    client = FakeClient(text="fallback", dispatch=dispatch)
    r1 = client.completions.create(messages=[{"role": "system", "content": "You classify a physics question..."},
                                              {"role": "user", "content": "u"}])
    r2 = client.completions.create(messages=[{"role": "system", "content": "You are the evidence stage..."},
                                              {"role": "user", "content": "u"}])
    r3 = client.completions.create(messages=[{"role": "system", "content": "unrelated system prompt"},
                                              {"role": "user", "content": "u"}])
    assert r1.choices[0].message.content == "understand-reply"
    assert r2.choices[0].message.content == "evidence-reply"
    assert r3.choices[0].message.content == "fallback"


BOOK_EV_TWO_TAGS = {"kept": [
    {"tag": "S1", "source": "Griffiths", "text": "the ON-topic passage about tunneling"},
    {"tag": "S2", "source": "Random", "text": "the OFF-topic passage about baking"},
]}


def _assessment(usable, off_topic, skipped=False):
    return {"skipped": skipped, "usable": usable, "off_topic": off_topic,
            "agreements": [], "conflicts": [], "gaps": [],
            "covered_by_curriculum": True, "confidence": "high"}


def test_reasoning_engine_excludes_off_topic_book_tag():
    assessment = _assessment(["S1"], ["S2"])
    client = FakeClient()
    qp.reasoning_engine("q", {"restate": "q"}, [], BOOK_EV_TWO_TAGS, [], None, None,
                        assessment, "intermediate", client, qp.Budget())
    prompt = client.completions.last_prompt
    assert "tunneling" in prompt and "[S1]" in prompt
    assert "baking" not in prompt and "[S2]" not in prompt


def test_professor_engine_excludes_off_topic_book_tag():
    assessment = _assessment(["S1"], ["S2"])
    client = FakeClient()
    qp.professor_engine("q", {"restate": "q"}, [], BOOK_EV_TWO_TAGS, [], None, None,
                        assessment, {"skipped": True, "text": ""}, "explain",
                        "intermediate", {}, client, qp.Budget())
    prompt = client.completions.last_prompt
    assert "tunneling" in prompt and "[S1]" in prompt
    assert "baking" not in prompt and "[S2]" not in prompt


def test_professor_engine_paper_tags_not_renumbered_when_one_excluded():
    papers = [{"published": "2020", "title": "Off-topic paper", "summary": "irrelevant chatter"},
              {"published": "2021", "title": "On-topic paper", "summary": "relevant discussion"}]
    assessment = _assessment(["A2"], ["A1"])
    client = FakeClient()
    qp.professor_engine("q", {"restate": "q"}, [], {"kept": []}, papers, None, None,
                        assessment, {"skipped": True, "text": ""}, "explain",
                        "intermediate", {}, client, qp.Budget())
    prompt = client.completions.last_prompt
    assert "irrelevant chatter" not in prompt
    assert "[A2]" in prompt and "relevant discussion" in prompt
    assert "[A1]" not in prompt


def test_intro_depth_skip_leaves_nothing_excluded():
    # evidence_engine() itself stubs off_topic=[] when the stage is skipped
    # (intro depth) - the filter must be a no-op there, not a special case.
    assessment = qp.evidence_engine("q", [], BOOK_EV_TWO_TAGS, [], None, None,
                                    "intro", FakeClient(), qp.Budget())
    assert assessment["skipped"] is True
    assert assessment["off_topic"] == []
    client = FakeClient()
    qp.professor_engine("q", {"restate": "q"}, [], BOOK_EV_TWO_TAGS, [], None, None,
                        assessment, {"skipped": True, "text": ""}, "explain",
                        "intermediate", {}, client, qp.Budget())
    prompt = client.completions.last_prompt
    assert "tunneling" in prompt and "baking" in prompt


def test_frontend_stage_listeners_match_pipeline_stage_names(monkeypatch):
    """Guards the frontend/backend contract drift bug: standalone.html's SSE
    listener list must name exactly the stages run() actually yields. Deriving
    the expected set by running the real pipeline - rather than hardcoding a
    second copy of the stage names in this test - means this test can't
    itself drift out of sync the way the frontend once did.
    """
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q: {"available": False, "kept": [], "rejected": [], "scope": []})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key-for-test")

    with patch("openai.OpenAI", return_value=FakeClient()):
        stages = {stage for stage, _ in qp.run("what is a photon", depth="intermediate")}
    stages -= {"done", "error"}

    html_path = pathlib.Path(__file__).resolve().parent.parent / "web" / "standalone.html"
    html = html_path.read_text()
    m = re.search(r'\[("[a-z]+"(?:,"[a-z]+")*)\]\.forEach\(st=>es\.addEventListener', html)
    assert m, "could not find the stage-listener array in standalone.html"
    listened = set(re.findall(r'"([a-z]+)"', m.group(1)))

    assert listened == stages


# ── Phase 4: Question Engine domain + difficulty ───────────────────────────

class _FakeTopic:
    def __init__(self, id):
        self.id = id


def test_domain_for_override_topic():
    assert qp.domain_for([_FakeTopic("general-relativity")]) == "relativity"
    assert qp.domain_for([_FakeTopic("quantum-optics")]) == "quantum-computing"


def test_domain_for_core_qm_default():
    assert qp.domain_for([_FakeTopic("harmonic-oscillator")]) == "quantum-mechanics"


def test_domain_for_no_topics():
    assert qp.domain_for([]) == "general-physics"


def _u(**over):
    base = {"intent": "explain", "topics": [], "needs_literature": True,
            "needs_symbolic": False, "identity": "", "difficulty": "intermediate"}
    base.update(over)
    return base


def test_route_lets_advanced_difficulty_override_intro_ui_depth():
    r = qp.route(_u(difficulty="advanced"), [], None, "intro")
    assert r["arxiv"] is True


def test_route_intro_ui_depth_and_intro_difficulty_still_suppresses_arxiv():
    r = qp.route(_u(difficulty="intro"), [], None, "intro")
    assert r["arxiv"] is False


def test_route_advanced_ui_depth_unaffected_by_intro_difficulty():
    r = qp.route(_u(difficulty="intro"), [], None, "advanced")
    assert r["arxiv"] is True


def test_understand_carries_through_valid_difficulty():
    client = FakeClient('{"intent":"explain","topics":["photon"],'
                        '"needs_literature":false,"needs_symbolic":false,'
                        '"identity":"","difficulty":"advanced","restate":"q"}')
    u = qp.understand("q", "intermediate", client, qp.Budget())
    assert u["difficulty"] == "advanced"


def test_understand_falls_back_to_intermediate_when_difficulty_missing():
    client = FakeClient('{"intent":"explain","topics":["photon"],'
                        '"needs_literature":false,"needs_symbolic":false,'
                        '"identity":"","restate":"q"}')
    u = qp.understand("q", "intermediate", client, qp.Budget())
    assert u["difficulty"] == "intermediate"


# ── Phase 2: hardened curriculum matching ───────────────────────────────────

def test_match_topics_rejects_single_coincidental_token_overlap():
    # density-matrix's own key_concepts name "classical vs quantum
    # uncertainty" - a topic that CONTRASTS itself against classical
    # mechanics used to score as a match FOR a question about it, on the
    # strength of the single word "classical" alone.
    assert qp.match_topics("classical mechanics") == []


def test_match_topics_still_finds_legitimate_single_token_match():
    topics = qp.match_topics("what is spin")
    assert any(t.id == "spin-pauli" for t in topics)


def test_match_topics_related_list_is_floored():
    # Every entry match_topics() returns must individually clear the floor,
    # not just occupy a top-4 rank behind a real primary match.
    from tutor import MIN_TOPIC_SCORE, _score_topics
    scored = _score_topics("what is the uncertainty principle")
    topics = qp.match_topics("what is the uncertainty principle", k=len(scored) + 5)
    returned_ids = {t.id for t in topics}
    for score, t in scored:
        if score >= MIN_TOPIC_SCORE:
            assert t.id in returned_ids
        else:
            assert t.id not in returned_ids


def test_suggest_related_is_unfloored_but_matching_finds_nothing():
    from tutor import suggest_related
    assert qp.match_topics("classical mechanics") == []
    # suggest_related() is explicitly allowed to surface a below-floor guess
    # for "you might instead ask about..." - it just must never be treated
    # as evidence, which match_topics() (asserted above) already guarantees.
    suggestions = suggest_related("classical mechanics")
    assert isinstance(suggestions, list)


# ── Phase 3: provider-error classification wired into _call() ──────────────

def test_call_wraps_402_as_provider_error():
    boom = make_status_error(openai.APIStatusError, 402, "Insufficient Balance")
    client = FakeClient(raise_on_call=boom)
    try:
        qp._call(client, "understand", "intermediate", "sys", "user", qp.Budget())
        assert False, "expected a ProviderError"
    except qp.ProviderError as exc:
        assert exc.kind == "insufficient_balance"
        assert exc.http_status == 402


def test_call_does_not_retry_a_provider_error():
    boom = make_status_error(openai.APIStatusError, 429, "slow down")
    client = FakeClient(raise_on_call=boom)
    try:
        qp._call(client, "understand", "intermediate", "sys", "user", qp.Budget())
    except qp.ProviderError:
        pass
    assert len(client.completions.calls) == 1


def test_call_reraises_non_sdk_exceptions_untouched():
    client = FakeClient(raise_on_call=ValueError("a real bug"))
    try:
        qp._call(client, "understand", "intermediate", "sys", "user", qp.Budget())
        assert False, "expected the ValueError to propagate"
    except ValueError:
        pass
    except qp.ProviderError:
        assert False, "a real bug must never be misclassified as a provider error"


# ── ModelGateway integration: the actual point of the cooldown ─────────────

def test_run_second_request_skips_the_network_entirely_after_a_permanent_failure(monkeypatch):
    """The real payoff of goal 4: once one request has seen a permanent
    provider failure (invalid key / zero balance), a second request must
    not even attempt the network call - it should go straight to the
    offline path using the remembered failure.
    """
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "evidence_strength": "usable",
                                           "rejected": [], "kept": [
                            {"tag": "S1", "source": "Griffiths", "text": "z" * 250,
                             "raw_score": 0.6}]})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key")
    boom = make_status_error(openai.APIStatusError, 402, "Insufficient Balance")

    with patch("openai.OpenAI", return_value=FakeClient(raise_on_call=boom)) as first_ctor:
        events1 = list(qp.run("what is the uncertainty principle", depth="intermediate"))
    assert first_ctor.called is True
    payload1 = events1[-1][1]
    assert payload1["answer_mode"] == "offline"
    assert payload1["provider_error"]["kind"] == "insufficient_balance"

    # Second request: even a client that would happily succeed must never
    # be reached, because the gateway remembers the first failure.
    with patch("openai.OpenAI", return_value=FakeClient(text="should never be seen")) as second_ctor:
        events2 = list(qp.run("what is a photon", depth="intermediate"))
    assert second_ctor.called is False, "the gateway should have short-circuited before construction"
    payload2 = events2[-1][1]
    assert payload2["answer_mode"] == "offline"
    assert payload2["provider_error"]["kind"] == "insufficient_balance"


def test_run_transient_failure_does_not_block_the_next_request(monkeypatch):
    # A single shared client, matching production reality: the gateway caches
    # one client object across requests, so what must be proven here is that
    # the SECOND request's LLM call is actually attempted (not skipped by a
    # remembered cooldown), not that a new client object gets constructed.
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "evidence_strength": "usable",
                                           "rejected": [], "kept": [
                            {"tag": "S1", "source": "Griffiths", "text": "z" * 250,
                             "raw_score": 0.6}]})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key")
    boom = make_status_error(openai.APIStatusError, 429, "slow down")
    shared_client = FakeClient(raise_on_call=lambda kwargs, n: boom if n == 1 else None,
                               text="a real answer")

    with patch("openai.OpenAI", return_value=shared_client):
        events1 = list(qp.run("what is the uncertainty principle", depth="intermediate"))
        assert events1[-1][1]["answer_mode"] == "offline"

        events2 = list(qp.run("what is a photon", depth="intermediate"))
    assert len(shared_client.completions.calls) >= 2, \
        "a transient failure must not block the next request's LLM call from being attempted"
    assert events2[-1][1]["answer_mode"] == "online"


# ── Phase 4: deterministic Question Understanding layer ────────────────────

def test_extract_comparison_targets_classical_vs_quantum():
    # scenario A
    q = "what is the mathematical difference between classical mechanics and quantum mechanics?"
    assert qp.extract_comparison_targets(q) == ["classical mechanics", "quantum mechanics"]


def test_extract_comparison_targets_newtonian_vs_lagrangian():
    # scenario B - the harder case: only one side names "mechanics" outright
    q = "Newtonian vs Lagrangian mechanics"
    assert qp.extract_comparison_targets(q) == ["Newtonian mechanics", "Lagrangian mechanics"]


def test_extract_comparison_targets_borrows_shared_head_noun():
    # scenario F
    q = "compare classical and quantum probability"
    assert qp.extract_comparison_targets(q) == ["classical probability", "quantum probability"]


def test_extract_comparison_targets_empty_for_non_comparison_question():
    assert qp.extract_comparison_targets("what is quantum entanglement") == []


def test_detect_math_requirements_heisenberg_mathematically():
    # scenario D
    required, depth = qp.detect_math_requirements(
        "explain the Heisenberg uncertainty principle mathematically")
    assert required is True and depth == "standard"


def test_detect_math_requirements_adjective_form_give_me_the_mathematical_explanation():
    # scenario A's exact phrasing - "mathematical explanation" (adjective),
    # not just "mathematically" (adverb). Both must trigger equally.
    required, depth = qp.detect_math_requirements("Give me the mathematical explanation.")
    assert required is True and depth == "standard"


def test_detect_math_requirements_does_not_trigger_on_bare_equation_mention():
    # scenario C - "equation" names the topic, it isn't a request for math
    required, depth = qp.detect_math_requirements("explain the Schrodinger equation")
    assert required is False and depth == "none"


def test_detect_correction_flags_i_actually_meant():
    # scenario H
    assert qp.detect_correction("I actually meant classical mechanics and quantum mechanics") is True
    assert qp.detect_correction("what is quantum entanglement") is False


def test_deterministic_understand_merges_prior_topics_on_correction():
    prior = {"restate": "what is Newtonian mechanics?", "topics": ["newtonian", "mechanic"],
              "comparison_targets": None}
    u = qp.deterministic_understand("sorry, I meant something else entirely", prior)
    assert u["is_correction"] is True
    assert u["corrected_from"] == prior["restate"]


def test_backfill_never_overrides_llm_supplied_comparison_targets():
    u = {"comparison_targets": ["already decided A", "already decided B"]}
    qp.backfill_deterministic_fields(u, "Newtonian vs Lagrangian mechanics")
    assert u["comparison_targets"] == ["already decided A", "already decided B"]


def test_backfill_fills_gap_when_llm_gave_nothing():
    u = {"comparison_targets": None}
    qp.backfill_deterministic_fields(u, "Newtonian vs Lagrangian mechanics")
    assert u["comparison_targets"] == ["Newtonian mechanics", "Lagrangian mechanics"]


def test_understand_extends_result_with_new_fields_via_llm_json():
    client = FakeClient('{"intent":"compare","topics":["classical","quantum"],'
                        '"needs_literature":false,"needs_symbolic":false,"identity":"",'
                        '"difficulty":"intermediate","comparison_targets":'
                        '["classical mechanics","quantum mechanics"],'
                        '"equations_required":true,"math_depth":"standard",'
                        '"explicit_constraints":[],"restate":"q"}')
    u = qp.understand("difference between classical and quantum mechanics",
                      "intermediate", client, qp.Budget())
    assert u["comparison_targets"] == ["classical mechanics", "quantum mechanics"]
    assert u["equations_required"] is True
    assert u["math_depth"] == "standard"


def test_understand_backfills_new_fields_when_llm_json_omits_them():
    client = FakeClient('{"intent":"explain","topics":["photon"],'
                        '"needs_literature":false,"needs_symbolic":false,'
                        '"identity":"","restate":"q"}')
    u = qp.understand("Newtonian vs Lagrangian mechanics", "intermediate", client, qp.Budget())
    assert u["comparison_targets"] == ["Newtonian mechanics", "Lagrangian mechanics"]


# ── Phase 5: comparison-aware, per-side retrieval ───────────────────────────

def _fake_retrieve_evidence(evidence_by_query):
    """A stand-in for tutor.retrieve_evidence() keyed by a substring of the
    query, so per-side retrieval can be tested without depending on the real
    second_brain gateway being present. Substring, not exact match:
    _retrieve_with_topic_context() widens the query with the matched topic's
    own title/concepts/domain, so a fixture keyed on the bare label must
    still match the enriched query built on top of it."""
    def _fn(query, top_k=6):
        for key, items in evidence_by_query.items():
            if key in query:
                return {"available": True, "kept": list(items), "rejected": [],
                       "evidence_strength": "usable"}
        return {"available": True, "kept": [], "rejected": [], "evidence_strength": "none"}
    return _fn


def test_gather_comparison_sides_classical_vs_quantum_mechanics(monkeypatch):
    # scenario A
    monkeypatch.setattr(qp, "retrieve_evidence", _fake_retrieve_evidence({
        "classical mechanics": [{"tag": "S1", "source": "Newtonian Mechanics",
                                 "text": "x" * 250, "raw_score": 0.5}],
        "quantum mechanics": [{"tag": "S1", "source": "Griffiths",
                               "text": "y" * 250, "raw_score": 0.6}],
    }))
    book_ev, topics, sides = qp.gather_comparison_sides(["classical mechanics", "quantum mechanics"])
    by_label = {s["label"]: s for s in sides}
    assert by_label["classical mechanics"]["covered_by_curriculum"] is False
    assert by_label["classical mechanics"]["topics"] == []
    assert by_label["quantum mechanics"]["covered_by_curriculum"] is True
    assert by_label["quantum mechanics"]["topics"] != []
    # book evidence must still reach both sides despite the query being
    # widened with the matched topic's own title/concepts/domain
    by_side_kept = {}
    for c in book_ev["kept"]:
        by_side_kept.setdefault(c["side"], []).append(c)
    assert any("Newtonian" in c["source"] for c in by_side_kept.get("classical mechanics", []))
    assert any("Griffiths" in c["source"] for c in by_side_kept.get("quantum mechanics", []))
    # and the widened query is recorded, closing the "domain/prerequisite
    # matching used only as a label, never as a real retrieval signal" gap
    assert by_side_kept["quantum mechanics"][0]["linked_topics"]


def test_gather_comparison_sides_newtonian_vs_lagrangian_both_uncovered(monkeypatch):
    # scenario B - neither classical-technique side has curriculum coverage
    monkeypatch.setattr(qp, "retrieve_evidence", _fake_retrieve_evidence({}))
    book_ev, topics, sides = qp.gather_comparison_sides(
        qp.extract_comparison_targets("Newtonian vs Lagrangian mechanics"))
    assert all(not s["covered_by_curriculum"] for s in sides)
    assert topics == []
    assert book_ev["kept"] == []
    assert book_ev["evidence_strength"] == "none"


def test_gather_comparison_sides_tags_renumbered_without_collision(monkeypatch):
    monkeypatch.setattr(qp, "retrieve_evidence", _fake_retrieve_evidence({
        "side one": [{"tag": "S1", "source": "Book A", "text": "a" * 250, "raw_score": 0.5},
                    {"tag": "S2", "source": "Book B", "text": "b" * 250, "raw_score": 0.4}],
        "side two": [{"tag": "S1", "source": "Book C", "text": "c" * 250, "raw_score": 0.5}],
    }))
    book_ev, topics, sides = qp.gather_comparison_sides(["side one", "side two"])
    tags = [c["tag"] for c in book_ev["kept"]]
    assert tags == ["S1", "S2", "S3"]
    assert [c["side"] for c in book_ev["kept"]] == ["side one", "side one", "side two"]


def test_curriculum_block_labels_empty_side_explicitly():
    sides = [{"label": "classical mechanics", "topics": []},
             {"label": "quantum mechanics", "topics": qp.match_topics("quantum mechanics")}]
    block = qp.curriculum_block([], sides)
    assert "COMPARISON SIDE: classical mechanics" in block
    assert "(no curriculum topic covers this side)" in block
    assert "COMPARISON SIDE: quantum mechanics" in block


def test_professor_engine_renders_both_sides_labeled_when_one_is_empty():
    sides = [{"label": "classical mechanics", "topics": []},
             {"label": "quantum mechanics", "topics": []}]
    book_ev = {"kept": [{"tag": "S1", "source": "Newtonian Mechanics",
                        "text": "on the classical side", "side": "classical mechanics"}]}
    assessment = _assessment([], [])
    client = FakeClient()
    qp.professor_engine("q", {"restate": "q"}, [], book_ev, [], None, None,
                        assessment, {"skipped": True, "text": ""}, "compare",
                        "intermediate", {}, client, qp.Budget(), sides=sides)
    prompt = client.completions.last_prompt
    assert "COMPARISON SIDE: classical mechanics" in prompt
    assert "COMPARISON SIDE: quantum mechanics" in prompt
    assert "(no curriculum topic covers this side)" in prompt
    assert "on the classical side" in prompt


# ── Phase 6: run() offline mode + insufficient-evidence short-circuit ──────

def _collect(gen):
    return list(gen)


def test_run_402_on_understand_falls_back_to_offline_mode_not_error(monkeypatch):
    # scenario I - the headline regression test.
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "evidence_strength": "usable",
                                           "rejected": [], "kept": [
                            {"tag": "S1", "source": "Griffiths", "text": "z" * 250,
                             "raw_score": 0.6}]})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key")
    boom = make_status_error(openai.APIStatusError, 402, "Insufficient Balance")
    with patch("openai.OpenAI", return_value=FakeClient(raise_on_call=boom)):
        events = _collect(qp.run("what is the uncertainty principle", depth="intermediate"))
    stages = [s for s, _ in events]
    assert "error" not in stages
    assert stages[-1] == "done"
    payload = events[-1][1]
    assert payload["answer_mode"] == "offline"
    assert payload["provider_error"]["kind"] == "insufficient_balance"
    matched_ids = {t["id"] for t in payload["topics"]}
    assert "uncertainty-principle" in matched_ids
    # never invents a topic beyond what was actually matched
    assert "quantum-information" not in payload["prose"].lower().replace(" ", "-")


def test_run_classical_vs_quantum_mechanics_offline_reports_classical_side_uncovered(monkeypatch):
    # scenario A, fully offline - the exact reported bug.
    monkeypatch.setattr(qp, "retrieve_evidence", _fake_retrieve_evidence({
        "classical mechanics": [{"tag": "S1", "source": "Newtonian Mechanics",
                                 "text": "n" * 250, "raw_score": 0.5}],
        "quantum mechanics": [{"tag": "S1", "source": "Griffiths",
                               "text": "g" * 250, "raw_score": 0.6}],
    }))
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key")
    boom = make_status_error(openai.APIStatusError, 402, "Insufficient Balance")
    with patch("openai.OpenAI", return_value=FakeClient(raise_on_call=boom)):
        events = _collect(qp.run(
            "what is the mathematical difference between classical mechanics and quantum mechanics?",
            depth="intermediate"))
    payload = events[-1][1]
    assert payload["answer_mode"] == "offline"
    by_label = {s["label"]: s for s in payload["sides"]}
    # the actual reported bug: classical mechanics has NO real curriculum
    # coverage, and the fix must say so honestly rather than letting a
    # quantum-flavoured topic stand in for it.
    assert by_label["classical mechanics"]["covered_by_curriculum"] is False
    assert by_label["classical mechanics"]["topics"] == []
    assert by_label["quantum mechanics"]["covered_by_curriculum"] is True
    prose = payload["prose"]
    # The curriculum is a teaching layer, not the boundary of knowledge:
    # real book evidence exists for classical mechanics, so the answer must
    # say it isn't a *named curriculum topic* while still answering from the
    # evidence - never the old "No curriculum topic covers this" framing,
    # which read as a refusal sitting right above the evidence that contradicted it.
    assert "isn't one of the named curriculum topics" in prose
    assert "no curriculum topic covers this" not in prose.lower()
    # the classical section specifically must never borrow a quantum topic -
    # split the prose at the section markers and check the right slice, since
    # a *real* quantum topic legitimately appearing under the quantum side
    # (verified above) is correct, not a repeat of the bug.
    classical_start = prose.lower().index("## classical mechanics")
    quantum_start = prose.lower().index("## quantum mechanics")
    classical_block = prose[classical_start:quantum_start].lower()
    assert "quantum information" not in classical_block
    assert "quantum statistical mechanics" not in classical_block
    assert "newtonian mechanics" in classical_block  # the real book evidence, not a guess


def test_run_outside_curriculum_short_circuits_without_llm_calls(monkeypatch):
    # scenario G
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "kept": [], "rejected": [],
                                           "evidence_strength": "none"})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    with patch("openai.OpenAI") as mock_openai:
        events = _collect(qp.run("how do transistors work", depth="intermediate"))
    assert mock_openai.called is False
    payload = events[-1][1]
    assert payload["answer_mode"] == "insufficient_evidence"
    assert payload["topics"] == []


def test_run_weak_match_rejected_offline_uses_real_matcher(monkeypatch):
    # scenario J - real match_topics(), not mocked, proving the hardened
    # scorer from Phase 2 is what actually protects this path.
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "kept": [], "rejected": [],
                                           "evidence_strength": "none"})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    with patch("openai.OpenAI") as mock_openai:
        events = _collect(qp.run("classical mechanics", depth="intermediate"))
    assert mock_openai.called is False
    payload = events[-1][1]
    assert payload["answer_mode"] == "insufficient_evidence"
    assert payload["topics"] == []


def test_run_online_path_unaffected_when_llm_succeeds(monkeypatch):
    # regression guard: a healthy account still gets the full online pipeline.
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": False, "kept": [], "rejected": [],
                                           "evidence_strength": "none"})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key")
    with patch("openai.OpenAI", return_value=FakeClient(text="stub answer")):
        events = _collect(qp.run("what is a photon", depth="intermediate"))
    stages = [s for s, _ in events]
    assert "error" not in stages
    payload = events[-1][1]
    assert payload["answer_mode"] == "online"
    # The knowledge_trailer() (Phase 7) may deterministically append real
    # prerequisite/source info after the model's own prose - that's a
    # feature, not a regression, so check the prefix, not exact equality.
    assert payload["prose"].startswith("stub answer")


# ── Phase 7: structured answer contract + math-mode prioritization ─────────

def test_structure_directive_triggers_on_equations_required():
    # scenario D
    u = {"intent": "explain", "equations_required": True, "comparison_targets": []}
    assert qp.structure_directive(u) != ""


def test_structure_directive_triggers_on_compare_intent():
    u = {"intent": "compare", "equations_required": False,
        "comparison_targets": ["classical mechanics", "quantum mechanics"]}
    assert qp.structure_directive(u) != ""


def test_structure_directive_does_not_trigger_on_plain_explain_intent():
    # scenario C - negative control: plain explanation stays flowing prose
    u = {"intent": "explain", "equations_required": False, "comparison_targets": []}
    assert qp.structure_directive(u) == ""


def test_professor_engine_includes_structure_directive_when_math_required():
    u = {"restate": "q", "intent": "derive", "equations_required": True,
        "comparison_targets": []}
    assessment = _assessment([], [])
    client = FakeClient()
    qp.professor_engine("q", u, [], {"kept": []}, [], None, None, assessment,
                        {"skipped": True, "text": ""}, "explain", "intermediate", {},
                        client, qp.Budget())
    assert "Direct Answer" in client.completions.last_prompt


def test_professor_engine_omits_structure_directive_for_plain_explain():
    u = {"restate": "q", "intent": "explain", "equations_required": False,
        "comparison_targets": []}
    assessment = _assessment([], [])
    client = FakeClient()
    qp.professor_engine("q", u, [], {"kept": []}, [], None, None, assessment,
                        {"skipped": True, "text": ""}, "explain", "intermediate", {},
                        client, qp.Budget())
    assert "Direct Answer" not in client.completions.last_prompt


def test_knowledge_trailer_only_cites_real_offered_tags():
    from library import TOPICS as REAL_TOPICS
    t = REAL_TOPICS["schrodinger-equation"]
    book_ev = {"kept": [{"tag": "S1", "text": "x"}, {"tag": "S2", "text": "y"}]}
    trailer = qp.knowledge_trailer([t], book_ev, excluded={"S2"})
    assert "S1" in trailer
    assert "S2" not in trailer


def test_knowledge_trailer_links_curriculum_topic_to_the_sources_it_surfaced():
    from library import TOPICS as REAL_TOPICS
    t = REAL_TOPICS["schrodinger-equation"]
    book_ev = {"kept": [{"tag": "S1", "text": "x", "linked_topics": ["schrodinger-equation"]},
                        {"tag": "S2", "text": "y", "linked_topics": []}]}
    trailer = qp.knowledge_trailer([t], book_ev, excluded=set())
    assert "Curriculum → Sources" in trailer
    assert t.title in trailer
    assert "S1" in trailer.split("Curriculum → Sources")[1]
    assert "S2" not in trailer.split("Curriculum → Sources")[1]


def test_knowledge_trailer_empty_when_nothing_to_report():
    assert qp.knowledge_trailer([], {"kept": []}, excluded=set()) == ""


# ── Phase 8: deterministic professor quality gate ───────────────────────────

def test_quality_gate_flags_missing_comparison_side_coverage():
    sides = [{"label": "classical mechanics", "topics": [], "covered_by_curriculum": False,
             "evidence_strength": "none"},
             {"label": "quantum mechanics", "topics": [_FakeTopic("measurement-postulates")],
              "covered_by_curriculum": True, "evidence_strength": "usable"}]
    u = {"comparison_targets": ["classical mechanics", "quantum mechanics"],
        "explicit_constraints": [], "domain": "quantum-mechanics", "equations_required": False}
    prose = "This answer only ever talks about quantum mechanics and nothing else."
    gate = qp.quality_gate("q", u, _assessment([], []), [], sides, {"kept": []}, None,
                          prose, {"verdict": "pass"})
    assert gate["comparison_both_sides_covered"]["pass"] is False


def test_quality_gate_passes_math_check_when_equation_present():
    u = {"comparison_targets": [], "explicit_constraints": [], "domain": "quantum-mechanics",
        "equations_required": True}
    prose = r"The uncertainty relation is $\Delta x \Delta p \geq \hbar/2$."
    gate = qp.quality_gate("q", u, _assessment([], []), [], None, {"kept": []}, None,
                          prose, {"verdict": "pass"})
    assert gate["math_provided_if_requested"]["pass"] is True


def test_quality_gate_flags_missing_math_when_required_but_absent():
    u = {"comparison_targets": [], "explicit_constraints": [], "domain": "quantum-mechanics",
        "equations_required": True}
    prose = "The uncertainty principle says you can't know position and momentum exactly."
    gate = qp.quality_gate("q", u, _assessment([], []), [], None, {"kept": []}, None,
                          prose, {"verdict": "pass"})
    assert gate["math_provided_if_requested"]["pass"] is False


def test_quality_gate_flags_ungrounded_low_confidence_answer():
    u = {"comparison_targets": [], "explicit_constraints": [], "domain": "quantum-mechanics",
        "equations_required": False}
    gate = qp.quality_gate("q", u, _assessment([], []), [], None, {"kept": []}, None,
                          "some answer text of reasonable length here", {"verdict": "fail"})
    assert gate["grounded"]["pass"] is False


def test_quality_gate_flags_unaddressed_insufficient_evidence():
    u = {"comparison_targets": [], "explicit_constraints": [], "domain": "general-physics",
        "equations_required": False}
    prose = "Here's an answer that doesn't admit anything is missing at all."
    gate = qp.quality_gate("q", u, _assessment([], []), [], None,
                          {"kept": [], "evidence_strength": "none"}, None, prose,
                          {"verdict": "pass"})
    assert gate["refused_to_guess_when_insufficient"]["pass"] is False


# ── Phase 9: correction/continuity transport ────────────────────────────────

def test_run_detects_correction_and_merges_prior_topics(monkeypatch):
    # scenario H
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "kept": [], "rejected": [],
                                           "evidence_strength": "none"})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    prior = {"restate": "what is Newtonian mechanics?", "topics": ["newtonian", "mechanic"],
            "comparison_targets": None, "domain": "quantum-mechanics", "difficulty": "intermediate"}
    with patch("openai.OpenAI") as mock_openai:
        events = _collect(qp.run("sorry, I meant something else entirely",
                                 depth="intermediate", prior=prior))
    payload = events[-1][1]
    assert payload["understanding"]["is_correction"] is True
    assert payload["understanding"]["corrected_from"] == prior["restate"]
    # no comparison_targets in either the correction text or the prior, so
    # the deterministic layer falls back to merging the prior's topic words
    assert "newtonian" in payload["understanding"]["topics"]


# ── Phase 10: frontend mode-aware UI + hardened standalone-file matcher ────

def _standalone_html():
    return (pathlib.Path(__file__).resolve().parent.parent / "web" / "standalone.html").read_text()


def test_standalone_html_has_mode_aware_rendering():
    html = _standalone_html()
    assert "function renderAnswerMode" in html
    assert "prov-offline" in html and "prov-insufficient" in html
    assert "prov-degraded" in html
    assert "answer_mode" in html


def test_standalone_html_sends_prior_turn_for_corrections():
    html = _standalone_html()
    assert "lastTurn" in html
    assert "&prior=" in html


def test_standalone_html_local_fallback_is_relabelled_client_only():
    html = _standalone_html()
    # the client-only path must never be labelled "offline" in the trace -
    # that label now belongs to the materially better server-driven mode.
    assert 'put("client-only"' in html
    assert 'put("offline"' not in html


def test_standalone_html_ask_matcher_has_a_relevance_floor():
    html = _standalone_html()
    assert "ASK_MIN_SCORE" in html
    assert "askIdf" in html


# ── Gap closure: domain/prerequisite-aware retrieval enrichment ────────────

def test_retrieve_with_topic_context_passes_through_bare_query_when_no_topics(monkeypatch):
    monkeypatch.setattr(qp, "retrieve_evidence", lambda q, top_k=6: {"q_seen": q})
    ev, linked = qp._retrieve_with_topic_context("classical mechanics", [])
    assert ev["q_seen"] == "classical mechanics"
    assert linked == []


def test_retrieve_with_topic_context_widens_query_with_topic_vocabulary(monkeypatch):
    topics = qp.match_topics("what is the uncertainty principle")
    seen = {}
    def fake(q, top_k=6):
        seen["q"] = q
        return {"available": True, "kept": [], "rejected": [], "evidence_strength": "usable"}
    monkeypatch.setattr(qp, "retrieve_evidence", fake)
    ev, linked = qp._retrieve_with_topic_context("uncertainty principle", topics)
    assert topics[0].title in seen["q"]
    assert linked == [topics[0].id]


def test_retrieve_with_topic_context_widens_further_via_prerequisite_when_weak(monkeypatch):
    from library import TOPICS as REAL_TOPICS
    schrodinger = REAL_TOPICS["schrodinger-equation"]
    prereq_title = REAL_TOPICS["wavefunction-born-rule"].title

    def fake(q, top_k=6):
        if prereq_title in q:
            return {"available": True, "kept": [{"tag": "S1"}], "rejected": [],
                   "evidence_strength": "usable"}
        return {"available": True, "kept": [], "rejected": [], "evidence_strength": "weak"}
    monkeypatch.setattr(qp, "retrieve_evidence", fake)
    ev, linked = qp._retrieve_with_topic_context("schrodinger equation", [schrodinger])
    assert ev["evidence_strength"] == "usable"
    assert linked == ["schrodinger-equation", "wavefunction-born-rule"]


def test_retrieve_with_topic_context_keeps_first_result_when_widening_does_not_help(monkeypatch):
    from library import TOPICS as REAL_TOPICS
    schrodinger = REAL_TOPICS["schrodinger-equation"]
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "kept": [], "rejected": [],
                                           "evidence_strength": "none"})
    ev, linked = qp._retrieve_with_topic_context("schrodinger equation", [schrodinger])
    assert linked == ["schrodinger-equation"]


def test_concepts_for_returns_real_curriculum_concepts():
    from library import TOPICS as REAL_TOPICS
    t = REAL_TOPICS["schrodinger-equation"]
    concepts = qp.concepts_for([t])
    assert concepts and concepts[0] in t.key_concepts


def test_concepts_for_empty_when_no_topics():
    assert qp.concepts_for([]) == []


# ── Gap closure: explicit_constraints detection ─────────────────────────────

def test_detect_explicit_constraints_no_calculus_and_beginner():
    found = qp.detect_explicit_constraints(
        "explain the uncertainty principle without calculus, for a beginner")
    assert "no calculus" in found and "for a beginner" in found


def test_detect_explicit_constraints_empty_for_plain_question():
    assert qp.detect_explicit_constraints("what is a photon") == []


def test_backfill_uses_detected_constraints_not_a_bare_empty_list():
    u = {}
    qp.backfill_deterministic_fields(u, "explain spin briefly, no jargon please")
    assert "keep it brief" in u["explicit_constraints"]
    assert "no jargon" in u["explicit_constraints"]


# ── Gap closure: structure directive includes Assumptions + trailer note ───

def test_structure_directive_mentions_assumptions_section():
    u = {"intent": "derive", "equations_required": True, "comparison_targets": []}
    assert "Assumptions" in qp.structure_directive(u)


# ── Gap closure: comparison reasoning gets a real, grounded structure hint ──

def test_comparison_structure_hint_uses_real_concepts_for_covered_side():
    sides = [{"label": "classical mechanics", "topics": []},
             {"label": "quantum mechanics", "topics": qp.match_topics("uncertainty principle")}]
    hint = qp.comparison_structure_hint(sides)
    assert "not covered by the curriculum" in hint
    t = qp.match_topics("uncertainty principle")[0]
    assert t.key_concepts[0] in hint


def test_comparison_structure_hint_empty_when_not_a_comparison():
    assert qp.comparison_structure_hint(None) == ""
    assert qp.comparison_structure_hint([{"label": "only one side", "topics": []}]) == ""


def test_professor_engine_includes_comparison_hint_when_sides_given():
    sides = [{"label": "classical mechanics", "topics": []},
             {"label": "quantum mechanics", "topics": qp.match_topics("uncertainty principle")}]
    u = {"restate": "q", "intent": "compare", "equations_required": False,
        "comparison_targets": ["classical mechanics", "quantum mechanics"]}
    assessment = _assessment([], [])
    client = FakeClient()
    qp.professor_engine("q", u, [], {"kept": []}, [], None, None, assessment,
                        {"skipped": True, "text": ""}, "compare", "intermediate", {},
                        client, qp.Budget(), sides=sides)
    assert "SUGGESTED PROGRESSION PER SIDE" in client.completions.last_prompt


# ── Gap closure: a real DEGRADED state, distinct from OFFLINE ──────────────

def test_evidence_quality_usable_when_curriculum_covers_it():
    assert qp.evidence_quality(qp.match_topics("uncertainty principle"),
                               {"evidence_strength": "none"}, None) == "usable"


def test_evidence_quality_weak_when_only_thin_book_evidence(monkeypatch):
    assert qp.evidence_quality([], {"evidence_strength": "weak"}, None) == "weak"


def test_evidence_quality_none_when_nothing_at_all():
    assert qp.evidence_quality([], {"evidence_strength": "none"}, None) == "none"


def test_evidence_quality_comparison_takes_the_best_side():
    sides = [{"covered_by_curriculum": False, "evidence_strength": "weak"},
             {"covered_by_curriculum": True, "evidence_strength": "usable"}]
    assert qp.evidence_quality([], {}, sides) == "usable"


def test_offline_synthesis_degraded_shows_only_raw_evidence_no_curriculum_framing():
    book_ev = {"kept": [{"tag": "S1", "source": "Some Book", "text": "thin passage"}]}
    prose = qp.offline_synthesis("q", {}, [], None, book_ev, None, degraded=True)
    assert "thin passage" in prose
    assert "reasoning service is unavailable" in prose
    assert "No curriculum topic covers this" not in prose


def test_run_degraded_mode_when_provider_down_and_evidence_only_weak(monkeypatch):
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "evidence_strength": "weak",
                                           "rejected": [], "kept": [
                            {"tag": "S1", "source": "Some Book", "text": "w" * 250,
                             "raw_score": 0.31}]})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key")
    boom = make_status_error(openai.APIStatusError, 402, "Insufficient Balance")
    with patch("openai.OpenAI", return_value=FakeClient(raise_on_call=boom)):
        # a question with no real curriculum match, so quality stays "weak"
        events = _collect(qp.run("how do transistors work in silicon wafers",
                                 depth="intermediate"))
    payload = events[-1][1]
    assert payload["answer_mode"] == "degraded"
    assert "Some Book" in payload["prose"]


def test_run_offline_not_degraded_when_curriculum_covers_it(monkeypatch):
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": False, "kept": [], "rejected": [],
                                           "evidence_strength": "none"})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key")
    boom = make_status_error(openai.APIStatusError, 402, "Insufficient Balance")
    with patch("openai.OpenAI", return_value=FakeClient(raise_on_call=boom)):
        events = _collect(qp.run("what is the uncertainty principle", depth="intermediate"))
    payload = events[-1][1]
    assert payload["answer_mode"] == "offline"


# ── Gap closure: N-way comparisons + domain synonyms ────────────────────────

def test_extract_comparison_targets_handles_three_way_comparison():
    # scenario B, new exact phrasing
    q = "Compare Newtonian, Lagrangian and Hamiltonian mechanics."
    assert qp.extract_comparison_targets(q) == [
        "Newtonian mechanics", "Lagrangian mechanics", "Hamiltonian mechanics"]


def test_extract_comparison_targets_strips_trailing_manner_adverb():
    # a trailing "mathematically" used to get welded onto the LAST target
    # only, via the shared-head-noun borrow reading it as part of the noun
    # phrase the group shares ("Hamiltonian mechanics mathematically").
    q = "Compare Newtonian, Lagrangian and Hamiltonian mechanics mathematically."
    assert qp.extract_comparison_targets(q) == [
        "Newtonian mechanics", "Lagrangian mechanics", "Hamiltonian mechanics"]
    assert qp.extract_comparison_targets("classical mechanics vs quantum mechanics mathematically") == [
        "classical mechanics", "quantum mechanics"]


def test_extract_comparison_targets_ignores_trailing_sentence():
    # scenario A, new exact two-sentence phrasing - the trailing imperative
    # sentence must never get folded into the second comparison target.
    q = ("What is the difference between quantum physics and classical physics? "
        "Give me the mathematical explanation.")
    assert qp.extract_comparison_targets(q) == ["quantum mechanics", "classical mechanics"]


def test_extract_comparison_targets_normalizes_physics_to_mechanics_synonym():
    assert qp._normalize_domain_term("quantum physics") == "quantum mechanics"
    assert qp._normalize_domain_term("classical physics") == "classical mechanics"
    assert qp._normalize_domain_term("spin") == "spin"


def test_gather_comparison_sides_three_way(monkeypatch):
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "kept": [], "rejected": [],
                                           "evidence_strength": "none"})
    targets = qp.extract_comparison_targets("Compare Newtonian, Lagrangian and Hamiltonian mechanics.")
    book_ev, topics, sides = qp.gather_comparison_sides(targets)
    assert len(sides) == 3
    assert {s["label"] for s in sides} == set(targets)


# ── Gap closure: evidence diversity + intent-aware ranking ──────────────────

def test_diversify_evidence_caps_chunks_per_source():
    kept = [{"source": "Book A", "tag": f"S{i}"} for i in range(5)] + \
           [{"source": "Book B", "tag": "S6"}]
    out = qp._diversify_evidence(kept, max_per_source=2)
    assert sum(1 for c in out if c["source"] == "Book A") == 2
    assert sum(1 for c in out if c["source"] == "Book B") == 1
    assert len(out) == 3


def test_rank_for_intent_prefers_equation_dense_passages_when_math_required():
    kept = [{"text": "a purely conceptual paragraph with no symbols at all", "raw_score": 0.9},
           {"text": r"$E = \hbar \omega$ and $[x, p] = i\hbar$", "raw_score": 0.5}]
    ranked = qp._rank_for_intent(kept, equations_required=True)
    assert ranked[0]["raw_score"] == 0.5  # the equation-dense one moves first


def test_rank_for_intent_leaves_order_alone_when_math_not_required():
    kept = [{"text": "prose", "raw_score": 0.9}, {"text": r"$x=1$", "raw_score": 0.5}]
    assert qp._rank_for_intent(kept, equations_required=False) == kept


def test_retrieve_with_topic_context_applies_diversity(monkeypatch):
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "evidence_strength": "usable",
                                           "rejected": [], "kept": [
                            {"source": "Same Book", "tag": f"S{i}", "text": "x", "raw_score": 0.5}
                            for i in range(5)]})
    ev, linked = qp._retrieve_with_topic_context("query", [])
    assert len(ev["kept"]) <= 3


# ── Gap closure: requested_depth (distinct from difficulty) ────────────────

def test_detect_requested_depth_shallow():
    assert qp.detect_requested_depth("explain spin briefly") == "shallow"


def test_detect_requested_depth_deep():
    assert qp.detect_requested_depth("explain spin in depth, thoroughly") == "deep"


def test_detect_requested_depth_standard_default():
    assert qp.detect_requested_depth("what is spin") == "standard"


def test_backfill_populates_requested_depth():
    u = {}
    qp.backfill_deterministic_fields(u, "give me a quick overview of spin")
    assert u["requested_depth"] == "shallow"


def test_professor_engine_asks_for_brevity_when_shallow_requested():
    u = {"restate": "q", "intent": "explain", "equations_required": False,
        "comparison_targets": [], "requested_depth": "shallow"}
    assessment = _assessment([], [])
    client = FakeClient()
    qp.professor_engine("q", u, [], {"kept": []}, [], None, None, assessment,
                        {"skipped": True, "text": ""}, "explain", "intermediate", {},
                        client, qp.Budget())
    assert "keep this brief" in client.completions.last_prompt


def test_quality_gate_flags_shallow_answer_that_ran_long():
    u = {"comparison_targets": [], "explicit_constraints": [], "domain": "quantum-mechanics",
        "equations_required": False, "requested_depth": "shallow"}
    prose = " ".join(["word"] * 200)
    gate = qp.quality_gate("q", u, _assessment([], []), [], None, {"kept": []}, None,
                          prose, {"verdict": "pass"})
    assert gate["satisfied_requested_depth"]["pass"] is False


def test_quality_gate_passes_shallow_answer_that_stayed_short():
    u = {"comparison_targets": [], "explicit_constraints": [], "domain": "quantum-mechanics",
        "equations_required": False, "requested_depth": "shallow"}
    prose = "A short, direct answer."
    gate = qp.quality_gate("q", u, _assessment([], []), [], None, {"kept": []}, None,
                          prose, {"verdict": "pass"})
    assert gate["satisfied_requested_depth"]["pass"] is True


# ── Professor Intelligence Layer: pack-aware math-support honesty ──────────

def test_has_math_support_true_when_topic_carries_real_equations():
    from quantum_prof.evidence_pack import EvidencePack
    t = _FakeTopic("schrodinger-equation")
    t.key_equations = [r"i\hbar \partial_t \psi = \hat H \psi"]
    pack = EvidencePack(question="q", topics=[t])
    assert qp._has_math_support(pack) is True


def test_has_math_support_false_when_nothing_backs_it():
    from quantum_prof.evidence_pack import EvidencePack
    t = _FakeTopic("x")
    t.key_equations = []
    pack = EvidencePack(question="q", topics=[t],
                        passages=[{"text": "a purely conceptual passage, no symbols"}])
    assert qp._has_math_support(pack) is False


def test_has_math_support_true_via_computed_result():
    from quantum_prof.evidence_pack import EvidencePack
    pack = EvidencePack(question="q", computed={"ran": True, "result": {}})
    assert qp._has_math_support(pack) is True


def test_structure_directive_admits_the_gap_when_math_required_but_unsupported():
    from quantum_prof.evidence_pack import EvidencePack
    pack = EvidencePack(question="q", passages=[{"text": "no equations here"}])
    u = {"intent": "derive", "equations_required": True, "comparison_targets": []}
    directive = qp.structure_directive(u, pack)
    assert "isn't supported by what was retrieved" in directive


def test_structure_directive_plain_when_math_required_and_supported():
    from quantum_prof.evidence_pack import EvidencePack
    pack = EvidencePack(question="q", computed={"ran": True, "result": {}})
    u = {"intent": "derive", "equations_required": True, "comparison_targets": []}
    directive = qp.structure_directive(u, pack)
    assert "isn't supported by what was retrieved" not in directive
    assert "Direct Answer" in directive


def test_structure_directive_no_pack_behaves_as_before():
    u = {"intent": "derive", "equations_required": True, "comparison_targets": []}
    assert "Direct Answer" in qp.structure_directive(u)


# ── Question -> Physics Intent -> Prerequisites -> Evidence ->
#    Mathematical Objects -> Derivation Plan -> Physical Interpretation ->
#    Professor Answer: the reasoning stage is now this combined call ───────

def test_reasoning_engine_splits_derivation_plan_and_physical_interpretation():
    client = FakeClient(text="DERIVATION PLAN\n- step one\n- step two\n\n"
                        "PHYSICAL INTERPRETATION\n- it means X")
    result = qp.reasoning_engine("q", {"restate": "q"}, [], {"kept": []}, [], None, None,
                                 _assessment([], []), "intermediate", client, qp.Budget())
    assert "step one" in result["derivation_plan"]
    assert "step two" in result["derivation_plan"]
    assert "PHYSICAL INTERPRETATION" not in result["derivation_plan"]
    assert result["physical_interpretation"] == "- it means X"
    # text still carries the full combined content, for backward compatibility
    assert "step one" in result["text"] and "it means X" in result["text"]


def test_reasoning_engine_feeds_mathematical_objects_into_the_prompt():
    from quantum_prof.evidence_pack import EvidencePack
    pack = EvidencePack(question="q", mathematical_objects=[
        {"name": "Schrodinger equation", "expression": "iH psi = E psi",
        "kind": "curriculum_equation", "topic_id": "x", "tag": "C:x"}])
    client = FakeClient(text="DERIVATION PLAN\n- x\n\nPHYSICAL INTERPRETATION\n- y")
    qp.reasoning_engine("q", {"restate": "q"}, [], {"kept": []}, [], None, None,
                        _assessment([], []), "intermediate", client, qp.Budget(), pack=pack)
    prompt = client.completions.last_prompt
    assert "MATHEMATICAL OBJECTS" in prompt
    assert "iH psi = E psi" in prompt


def test_reasoning_engine_intro_depth_skip_has_the_new_empty_fields():
    result = qp.reasoning_engine("q", {"restate": "q"}, [], {"kept": []}, [], None, None,
                                 _assessment([], []), "intro", FakeClient(), qp.Budget())
    assert result["skipped"] is True
    assert result["derivation_plan"] == ""
    assert result["physical_interpretation"] == ""


# ── Physics + Mathematical Verification Layer: prove Derivation ->
#    Verification -> Professor Answer actually executes in the REAL run()
#    pipeline, not just that verify_derivation() is callable in isolation ──

def test_derivation_verification_reaches_the_professor_prompt_in_the_real_pipeline(monkeypatch):
    """The definitive wiring proof. "energy of n=2 electron in a 1nm box" is a
    real question that genuinely drives match_topics() to particle-in-a-box
    and compute_for() to the real solver (energy_eV=1.504121) - nothing about
    retrieval or the solver is mocked. Only the LLM boundary is faked, and
    only the Derivation Plan stage is given a scripted reply: a deliberately
    WRONG energy value the verifier must catch.

    Proving the payload contains a "verification" key would only show
    verify_derivation() was called somewhere. The real claim - that a failed
    derivation genuinely reaches the Professor stage rather than being
    computed and then dropped on the floor - requires inspecting the ACTUAL
    text sent to the professor's LLM call, which is what this test does.
    """
    question = "energy of n=2 electron in a 1nm box"
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": False, "kept": [],
                                           "rejected": [], "scope": []})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key-for-test")

    wrong_derivation = ("DERIVATION PLAN\n"
                       "- apply the particle-in-a-box energy formula for this configuration\n"
                       "- the result comes out to approximately 5.0 eV\n\n"
                       "PHYSICAL INTERPRETATION\n"
                       "- confinement quantizes the allowed energies")
    dispatch = by_system_prompt({
        "You are the Derivation Plan & Physical Interpretation stage": wrong_derivation,
    })
    client = FakeClient(text="stub answer", dispatch=dispatch)

    with patch("openai.OpenAI", return_value=client):
        events = list(qp.run(question, depth="intermediate"))

    stage, payload = events[-1]
    assert stage == "done"
    assert payload["answer_mode"] == "online"

    # 1. The premise: compute_for() really ran the real solver, not a stub.
    assert payload["computed"]["result"]["topic"] == "particle-in-a-box"
    assert payload["computed"]["result"]["energy_eV"] == 1.504121

    # 2. verify_derivation() ran inside run() and actually caught the wrong
    #    number the scripted Derivation stage stated.
    verification = payload["verification"]
    assert verification["status"] != "verified_mathematically"
    assert any(f["check"] == "known_result" for f in verification["failed"])
    assert any("1.504121" in c["correction"] for c in verification["corrections"])

    # 3. THE key proof: that failure and correction are genuinely present in
    #    the text sent to the professor stage's own LLM call - not merely
    #    returned in a payload dict a caller could plug in separately.
    professor_calls = [c for c in client.completions.calls
                       if "You are a physics tutor in the style of Feynman"
                       in c["messages"][0]["content"]]
    assert len(professor_calls) == 1
    professor_prompt = professor_calls[0]["messages"][1]["content"]
    assert "VERIFICATION" in professor_prompt
    assert "1.504121" in professor_prompt
    assert "never present the failed value as correct" in professor_prompt

    # 4. Zero LLM calls were added by the verification layer: still exactly
    #    the 5 pre-existing stages (understand, evidence, reasoning,
    #    professor, validation) - verification itself is pure sympy/Python.
    assert len(client.completions.calls) == 5


def test_correct_derivation_is_reported_as_verified_mathematically_to_the_professor(monkeypatch):
    """Negative control for the same wiring: a Derivation stage that states
    the RIGHT number must reach the professor prompt as verified, not as a
    failure - the previous test proves failures propagate, this proves
    success does too (the professor isn't just told "not verified" always).
    """
    question = "energy of n=2 electron in a 1nm box"
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": False, "kept": [],
                                           "rejected": [], "scope": []})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key-for-test")

    right_derivation = ("DERIVATION PLAN\n"
                       "- apply the particle-in-a-box energy formula for this configuration\n"
                       "- the result comes out to approximately 1.504121 eV\n\n"
                       "PHYSICAL INTERPRETATION\n"
                       "- confinement quantizes the allowed energies")
    dispatch = by_system_prompt({
        "You are the Derivation Plan & Physical Interpretation stage": right_derivation,
    })
    client = FakeClient(text="stub answer", dispatch=dispatch)

    with patch("openai.OpenAI", return_value=client):
        events = list(qp.run(question, depth="intermediate"))

    payload = events[-1][1]
    assert payload["verification"]["status"] == "verified_mathematically"
    assert not payload["verification"]["failed"]

    professor_calls = [c for c in client.completions.calls
                       if "You are a physics tutor in the style of Feynman"
                       in c["messages"][0]["content"]]
    professor_prompt = professor_calls[0]["messages"][1]["content"]
    assert "status=verified_mathematically" in professor_prompt


def test_professor_engine_states_all_four_verification_statuses_distinctly():
    """The Professor must be able to tell all four verification outcomes
    apart in its own prompt - not just the two "happy path" ones. Builds a
    real VerificationResult for each status directly (this is about
    professor_engine()'s own wiring/phrasing contract, independent of which
    specific checks produce which status - that mapping is verification.py's
    own responsibility, covered separately in test_verification.py).
    """
    from quantum_prof.verification import VerificationResult

    cases = {
        "verified_mathematically": VerificationResult(
            status="verified_mathematically", confidence="high",
            passed=[{"check": "conservation_law", "detail": "d"}]),
        "partially_verified": VerificationResult(
            status="partially_verified", confidence="medium",
            passed=[{"check": "dimensional_consistency", "detail": "d"}],
            failed=[{"check": "known_result", "detail": "d"}],
            corrections=[{"issue": "known_result", "correction": "the correct value is 1.5 eV"}]),
        "not_independently_verified": VerificationResult(
            status="not_independently_verified", confidence="low"),
        "failed": VerificationResult(
            status="failed", confidence="low",
            failed=[{"check": "known_result", "detail": "d"}],
            corrections=[{"issue": "known_result", "correction": "the correct value is 1.5 eV"}]),
    }
    prompts = {}
    for name, verification in cases.items():
        client = FakeClient()
        qp.professor_engine("q", {"restate": "q"}, [], {"kept": []}, [], None, None,
                            _assessment([], []), {"skipped": True, "text": ""}, "explain",
                            "intermediate", {}, client, qp.Budget(), verification=verification)
        prompts[name] = client.completions.last_prompt
        assert f"status={name}" in prompts[name]

    # each status's prompt text must be distinguishable from the others -
    # the Professor can't tell them apart if they all render identically.
    assert len({p.split("VERIFICATION")[1][:80] for p in prompts.values()}) == 4


# ── Question -> Physics Intent -> Evidence + Prerequisites -> Mathematical
#    Objects -> Derivation Plan -> Derivation -> VERIFICATION -> Physical
#    Interpretation -> Professor Answer: prove the REAL chain, not just that
#    the functions exist ──────────────────────────────────────────────────

def test_full_chain_qho_derivation_traces_mathematical_objects_through_to_professor(monkeypatch):
    """One real end-to-end pipeline run on "Derive the energy levels of the
    quantum harmonic oscillator and explain their physical meaning" - a
    conceptual/derivation question (no numeric solver run, unlike the
    particle-in-a-box tests above), so this exercises the curriculum-only
    path through Mathematical Objects. Only the LLM boundary is mocked;
    match_topics(), compute_for(), build_evidence_pack() (which extracts
    Mathematical Objects), and verify_derivation() all run for real.

    Traces actual data identity/content across each stage boundary, not
    just that the pipeline reaches "done":
      Physics Intent      -> u["domain"] == "quantum-mechanics", real topics matched
      Evidence+Prereq      -> pack.prerequisite_concepts is real curriculum data
      Mathematical Objects -> pack.mathematical_objects contains the real
                              harmonic-oscillator key_equations, tagged C:harmonic-oscillator
      Derivation Plan       -> the mocked LLM reply citing that exact tag
                              appears in reasoning["derivation_plan"]
      Verification          -> verify_derivation() actually re-parses that
                              same derivation text and the same pack, and
                              genuinely PASSES it (a real citation, a real
                              conserved-energy check) - not a stub result
      Physical Interpretation -> reasoning["physical_interpretation"] is
                              non-empty and distinct from the derivation plan
      Professor Answer      -> the verification status reaches the professor
                              prompt AND (since this mock simulates a
                              compliant reply) the final prose
    """
    question = "Derive the energy levels of the quantum harmonic oscillator and explain their physical meaning."
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": False, "kept": [],
                                           "rejected": [], "scope": []})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key-for-test")

    derivation_reply = (
        "DERIVATION PLAN\n"
        "- start from the quantum harmonic oscillator Hamiltonian [C:harmonic-oscillator]\n"
        "- solving the Schrodinger equation for this potential quantizes the energy\n"
        "- result: E_n = hbar*omega*(n + 1/2) for n = 0, 1, 2, ...\n\n"
        "PHYSICAL INTERPRETATION\n"
        "- the ground state (n=0) still has nonzero zero-point energy\n"
        "- levels are evenly spaced by hbar*omega, unlike the hydrogen atom's spectrum")
    professor_reply = (
        "The quantum harmonic oscillator's energy levels are quantized as "
        "E_n = hbar*omega(n + 1/2) [C:harmonic-oscillator]. This has been verified "
        "mathematically: the citation checks out and energy conservation holds "
        "for the underlying classical motion.")
    dispatch = by_system_prompt({
        "You are the Derivation Plan & Physical Interpretation stage": derivation_reply,
        "You are a physics tutor in the style of Feynman": professor_reply,
    })
    client = FakeClient(text="stub answer", dispatch=dispatch)

    with patch("openai.OpenAI", return_value=client):
        events = list(qp.run(question, depth="intermediate"))

    by_stage = {}
    for stage, payload in events:
        by_stage.setdefault(stage, []).append(payload)
    payload = events[-1][1]
    assert events[-1][0] == "done"
    assert payload["answer_mode"] == "online"

    # Physics Intent: real deterministic + LLM-merged understanding.
    assert payload["understanding"]["domain"] == "quantum-mechanics"
    assert any(t["id"] == "harmonic-oscillator" for t in payload["topics"])

    # Evidence + Prerequisites: real curriculum prerequisite graph data.
    assert payload["evidence_pack"]["prerequisite_concepts"]

    # Mathematical Objects: the real curriculum key_equations, correctly
    # tagged - not empty, not a placeholder.
    math_objects = payload["evidence_pack"]["mathematical_objects"]
    assert math_objects, "Mathematical Objects must be populated for a real curriculum topic"
    ho_objects = [o for o in math_objects if o["topic_id"] == "harmonic-oscillator"]
    assert ho_objects and ho_objects[0]["tag"] == "C:harmonic-oscillator"

    # Derivation Plan: the mocked reply's content genuinely made it through
    # reasoning_engine()'s own parsing into the structured fields.
    reasoning = payload["reasoning"]
    assert "harmonic oscillator Hamiltonian" in reasoning["derivation_plan"]
    assert "[C:harmonic-oscillator]" in reasoning["derivation_plan"]
    # Physical Interpretation: present, and not just a copy of the plan.
    assert "zero-point energy" in reasoning["physical_interpretation"]
    assert reasoning["physical_interpretation"] != reasoning["derivation_plan"]

    # Verification: ran for real against that exact derivation text and
    # pack - a genuine pass (real citation + real conservation check),
    # not a stub. Confirms it is NOT reachable via a false "nothing to
    # check" default: something real was checked and passed.
    verification = payload["verification"]
    assert verification["status"] == "verified_mathematically"
    assert {p["check"] for p in verification["passed"]} >= {"symbol_consistency", "conservation_law"}
    assert not verification["failed"]

    # Professor Answer: the status reached the professor's own LLM prompt...
    professor_calls = [c for c in client.completions.calls
                       if "You are a physics tutor in the style of Feynman"
                       in c["messages"][0]["content"]]
    assert len(professor_calls) == 1
    assert "status=verified_mathematically" in professor_calls[0]["messages"][1]["content"]
    # ...and (since the mocked reply simulates a compliant model) the final
    # answer text itself states the verification outcome plainly.
    assert "verified" in payload["prose"].lower()

    # Exactly the 5 pre-existing LLM calls - the whole chain above added zero.
    assert len(client.completions.calls) == 5


def test_run_with_no_curriculum_match_has_empty_mathematical_objects_and_does_not_crash(monkeypatch):
    """Requirement: missing Mathematical Objects must not crash the
    pipeline. A question that matches no curriculum topic and triggers no
    solver leaves pack.mathematical_objects == [] - but real book evidence
    keeps this past the (separate, pre-existing) relevance-gate
    short-circuit, so it's actually the empty-math-objects case being
    exercised here, not the unrelated "nothing found at all" one.
    """
    monkeypatch.setattr(qp, "match_topics", lambda q, k=4: [])
    monkeypatch.setattr(qp, "retrieve_evidence",
                        lambda q, top_k=6: {"available": True, "evidence_strength": "usable",
                                           "rejected": [], "kept": [
                            {"tag": "S1", "source": "Some Book", "text": "z" * 250,
                             "raw_score": 0.6}]})
    monkeypatch.setattr(qp, "record_visit", lambda *a, **k: {})
    monkeypatch.setattr(qp, "_api_key", lambda: "fake-key-for-test")
    monkeypatch.setattr(qp, "compute_for", lambda q, topic_id: None)

    client = FakeClient(text="a general physics answer with no curriculum backing")
    with patch("openai.OpenAI", return_value=client):
        events = list(qp.run("what is the anthropic principle in cosmology", depth="intermediate"))

    assert events[-1][0] == "done"
    payload = events[-1][1]
    assert payload["evidence_pack"]["mathematical_objects"] == []
    assert payload["verification"]["status"] == "not_independently_verified"
    assert payload["verification"]["passed"] == [] and payload["verification"]["failed"] == []
