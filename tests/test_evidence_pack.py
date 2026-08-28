"""Offline tests for evidence_pack.py - pure data consolidation, no LLM,
no network, no retrieval of its own."""
from quantum_prof.evidence_pack import EvidencePack, build_evidence_pack, extract_mathematical_objects
from quantum_prof.library import TOPICS


def _topic(tid):
    return TOPICS[tid]


def test_build_evidence_pack_consolidates_basic_fields():
    u = {"intent": "explain", "concepts": ["Schrödinger evolution"], "requested_depth": "deep",
        "equations_required": True}
    book_ev = {"kept": [{"tag": "S1", "source": "Griffiths", "text": "x" * 50}]}
    pack = build_evidence_pack("q", u, [_topic("schrodinger-equation")], None, book_ev, [],
                               {"agreements": ["a"], "conflicts": ["c"], "gaps": ["g"]},
                               None, None)
    assert pack.question == "q"
    assert pack.intent == "explain"
    assert pack.concepts == ["Schrödinger evolution"]
    assert pack.requested_depth == "deep"
    assert pack.math_required is True
    assert pack.passages[0]["tag"] == "S1"
    assert pack.sources == ["Griffiths"]
    assert pack.supporting_evidence == ["a"]
    assert pack.conflicting_evidence == ["c"]
    assert pack.knowledge_gaps == ["g"]
    assert pack.curriculum_backed is True


def test_build_evidence_pack_prerequisite_concepts_from_real_topic_graph():
    t = _topic("schrodinger-equation")
    pack = build_evidence_pack("q", {}, [t], None, {"kept": []}, [], {}, None, None)
    assert TOPICS["wavefunction-born-rule"].title in pack.prerequisite_concepts
    assert TOPICS["de-broglie"].title in pack.prerequisite_concepts


def test_build_evidence_pack_curriculum_backed_false_when_nothing_matched():
    pack = build_evidence_pack("q", {}, [], None, {"kept": []}, [], {}, None, None)
    assert pack.curriculum_backed is False
    assert pack.topics == []


def test_build_evidence_pack_curriculum_backed_true_via_sides_even_if_flat_topics_empty():
    sides = [{"label": "classical mechanics", "topics": [], "covered_by_curriculum": False},
             {"label": "quantum mechanics", "topics": [_topic("schrodinger-equation")],
              "covered_by_curriculum": True}]
    pack = build_evidence_pack("q", {}, [], sides, {"kept": []}, [], {}, None, None)
    assert pack.curriculum_backed is True
    assert len(pack.topics) == 1  # union across sides, deduplicated


def test_build_evidence_pack_papers_become_tagged_passages():
    papers = [{"title": "A paper", "summary": "an abstract", "published": "2020"}]
    pack = build_evidence_pack("q", {}, [], None, {"kept": []}, papers, {}, None, None)
    assert pack.passages[0]["tag"] == "A1"
    assert pack.passages[0]["source"] == "A paper"


def test_passages_for_side_filters_by_side_label():
    book_ev = {"kept": [{"tag": "S1", "source": "A", "text": "x", "side": "classical mechanics"},
                        {"tag": "S2", "source": "B", "text": "y", "side": "quantum mechanics"}]}
    pack = build_evidence_pack("q", {}, [], [{"label": "classical mechanics", "topics": []},
                                             {"label": "quantum mechanics", "topics": []}],
                               book_ev, [], {}, None, None)
    assert [p["tag"] for p in pack.passages_for_side("classical mechanics")] == ["S1"]
    assert [p["tag"] for p in pack.passages_for_side("quantum mechanics")] == ["S2"]
    assert len(pack.passages_for_side(None)) == 2


def test_topics_for_side_returns_that_sides_own_topics():
    sides = [{"label": "classical mechanics", "topics": []},
             {"label": "quantum mechanics", "topics": [_topic("schrodinger-equation")]}]
    pack = build_evidence_pack("q", {}, [], sides, {"kept": []}, [], {}, None, None)
    assert pack.topics_for_side("classical mechanics") == []
    assert pack.topics_for_side("quantum mechanics") == [_topic("schrodinger-equation")]


def test_has_book_evidence_and_is_comparison_flags():
    empty = EvidencePack(question="q")
    assert empty.has_book_evidence is False
    assert empty.is_comparison is False
    with_evidence = EvidencePack(question="q", passages=[{"tag": "S1"}], sides=[{"label": "x"}])
    assert with_evidence.has_book_evidence is True
    assert with_evidence.is_comparison is True


# ── Mathematical Objects (Question -> Physics Intent -> Prerequisites ->
#    Evidence -> Mathematical Objects -> Derivation Plan -> Physical
#    Interpretation -> Professor Answer) ─────────────────────────────────

def test_extract_mathematical_objects_from_curriculum_topic():
    t = _topic("schrodinger-equation")
    objs = extract_mathematical_objects([t], None, None)
    assert len(objs) == len(t.key_equations)
    assert all(o["kind"] == "curriculum_equation" and o["topic_id"] == "schrodinger-equation"
              for o in objs)
    assert objs[0]["expression"] in t.key_equations


def test_extract_mathematical_objects_from_computed_result():
    computed = {"ran": True, "result": {"formula": "E_n = n^2 h^2 / (8 m L^2)"}}
    objs = extract_mathematical_objects([], computed, None)
    assert len(objs) == 1
    assert objs[0]["kind"] == "computed_result"
    assert objs[0]["tag"] == "T1"
    assert objs[0]["expression"] == "E_n = n^2 h^2 / (8 m L^2)"


def test_extract_mathematical_objects_from_symbolic_check():
    symbolic = {"ok": True, "verdict": "identity holds"}
    objs = extract_mathematical_objects([], None, symbolic)
    assert len(objs) == 1
    assert objs[0]["kind"] == "symbolic_check"
    assert objs[0]["tag"] == "X1"


def test_extract_mathematical_objects_empty_when_nothing_mathematical():
    assert extract_mathematical_objects([], None, None) == []
    assert extract_mathematical_objects([], {"ran": False}, {"ok": False}) == []


def test_extract_mathematical_objects_combines_all_three_sources():
    t = _topic("schrodinger-equation")
    computed = {"ran": True, "result": {"formula": "f(x)"}}
    symbolic = {"ok": True, "verdict": "holds"}
    objs = extract_mathematical_objects([t], computed, symbolic)
    kinds = {o["kind"] for o in objs}
    assert kinds == {"curriculum_equation", "computed_result", "symbolic_check"}


def test_build_evidence_pack_populates_mathematical_objects():
    t = _topic("schrodinger-equation")
    pack = build_evidence_pack("q", {}, [t], None, {"kept": []}, [], {}, None, None)
    assert len(pack.mathematical_objects) == len(t.key_equations)


def test_build_evidence_pack_mathematical_objects_use_unioned_sides_topics():
    sides = [{"label": "classical mechanics", "topics": [], "covered_by_curriculum": False},
             {"label": "quantum mechanics", "topics": [_topic("schrodinger-equation")],
              "covered_by_curriculum": True}]
    pack = build_evidence_pack("q", {}, [], sides, {"kept": []}, [], {}, None, None)
    assert len(pack.mathematical_objects) == len(_topic("schrodinger-equation").key_equations)
