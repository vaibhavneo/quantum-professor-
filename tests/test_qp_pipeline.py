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

from quantum_prof import qp_pipeline as qp


class _FakeCompletions:
    def __init__(self, text="stub answer"):
        self.text = text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                       completion_tokens_details=None)
        msg = types.SimpleNamespace(content=self.text)
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    @property
    def last_prompt(self):
        return self.last_kwargs["messages"][1]["content"]


class FakeClient:
    def __init__(self, text="stub answer"):
        self.completions = _FakeCompletions(text)
        self.chat = types.SimpleNamespace(completions=self.completions)


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
