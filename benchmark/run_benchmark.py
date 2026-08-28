#!/usr/bin/env python3
"""Runs the Physics Reasoning Benchmark (benchmark/problems.py) against the
REAL qp_pipeline.run() pipeline - real match_topics(), real compute_for(),
real retrieve_evidence() against the local book index, real
verify_derivation(). Only the LLM boundary (understand / reasoning replies)
is scripted, via benchmark/mock_llm.py - no DeepSeek call, no paid API,
anywhere in this file.

Usage:
    python3 -m benchmark.run_benchmark                  # summary to stdout
    python3 -m benchmark.run_benchmark --json out.json   # also write full results
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from unittest.mock import patch

import qp_pipeline as qp
from model_gateway import get_gateway

from benchmark.mock_llm import ScriptedClient
from benchmark.problems import PROBLEMS, Problem

_UNDERSTAND_KEY = "You classify a physics question"
_REASON_KEY = "You are the Derivation Plan & Physical Interpretation stage"


def run_one(problem: Problem) -> dict:
    mapping = {_UNDERSTAND_KEY: problem.understand_json, _REASON_KEY: problem.derivation_reply}
    client = ScriptedClient(mapping)
    qp._api_key = lambda: "fake-key-for-benchmark"
    get_gateway().reset()

    result: dict = {"id": problem.id, "error": None}
    t0 = time.monotonic()
    try:
        with patch("openai.OpenAI", return_value=client):
            events = list(qp.run(problem.question, depth="intermediate"))
    except Exception as exc:  # a crash is itself the most important possible finding
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_s"] = round(time.monotonic() - t0, 3)
        return result
    result["elapsed_s"] = round(time.monotonic() - t0, 3)

    stage, payload = events[-1]
    if stage != "done":
        result["error"] = f"pipeline ended in {stage!r} instead of 'done': {payload.get('message')}"
        return result

    verification = payload.get("verification") or {}
    result.update({
        "answer_mode": payload.get("answer_mode"),
        "topics_matched": [t["id"] for t in payload.get("topics", [])],
        "math_object_count": len((payload.get("evidence_pack") or {}).get("mathematical_objects", [])),
        "solver_ran": bool((payload.get("computed") or {}).get("ran")),
        "status": verification.get("status"),
        "confidence": verification.get("confidence"),
        "passed_checks": [p["check"] for p in verification.get("passed", [])],
        "failed_checks": [f["check"] for f in verification.get("failed", [])],
        "corrections": verification.get("corrections", []),
        "num_llm_calls": len(client.calls),
    })
    return result


def score(problem: Problem, result: dict) -> dict:
    """Returns a scoring dict: {"pass": bool, "findings": [...]}. "pass" is
    driven by expected_status (the primary per-problem signal); topic/solver
    mismatches are recorded as findings without failing the problem, since
    several of them are DELIBERATE - this benchmark's job is to surface a
    retrieval gap, not to quietly redefine "expected" to hide it.
    """
    if result["error"]:
        return {"pass": False, "findings": [f"CRASHED: {result['error']}"]}

    findings = []
    passed = True

    if problem.expected_status and result["status"] not in problem.expected_status:
        passed = False
        findings.append(f"status: expected one of {problem.expected_status}, got {result['status']!r}")

    if problem.expect_topics_matched is not None:
        matched = bool(result["topics_matched"])
        if matched != problem.expect_topics_matched:
            findings.append(f"topics_matched: expected {problem.expect_topics_matched}, got {matched} "
                            f"({result['topics_matched']!r})")

    if problem.expect_solver_ran is not None:
        if result["solver_ran"] != problem.expect_solver_ran:
            findings.append(f"solver_ran: expected {problem.expect_solver_ran}, got {result['solver_ran']}")

    detected = None
    if problem.is_wrong and problem.targeted_check:
        detected = problem.targeted_check in result["failed_checks"]
        expected_detected = any(s in ("failed", "partially_verified") for s in problem.expected_status)
        if detected != expected_detected:
            passed = False
            findings.append(f"detection: expected detected={expected_detected}, got detected={detected}")

    return {"pass": passed, "findings": findings, "detected": detected}


def run_all() -> list[dict]:
    rows = []
    for problem in PROBLEMS:
        result = run_one(problem)
        scoring = score(problem, result)
        rows.append({"problem": problem, "result": result, "score": scoring})
    return rows


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    crashed = sum(1 for r in rows if r["result"]["error"])
    status_pass = sum(1 for r in rows if r["score"]["pass"])

    wrong_rows = [r for r in rows if r["problem"].is_wrong]
    correct_rows = [r for r in rows if not r["problem"].is_wrong]

    detected = [r for r in wrong_rows if r["score"].get("detected") is True]
    not_detected = [r for r in wrong_rows if r["score"].get("detected") is False]
    false_positives = [r for r in wrong_rows if r["result"].get("status") == "verified_mathematically"]
    false_negatives = [r for r in correct_rows if r["result"].get("status") == "failed"]

    topic_checked = [r for r in rows if r["problem"].expect_topics_matched is not None]
    topic_as_expected = [r for r in topic_checked
                         if bool(r["result"].get("topics_matched")) == r["problem"].expect_topics_matched]
    real_topic_match_rate = sum(1 for r in rows if r["result"].get("topics_matched")) / total

    call_counts = {r["result"].get("num_llm_calls") for r in rows if not r["result"]["error"]}

    by_category = defaultdict(lambda: {"total": 0, "pass": 0})
    by_domain = defaultdict(lambda: {"total": 0, "pass": 0})
    for r in rows:
        by_category[r["problem"].category]["total"] += 1
        by_category[r["problem"].category]["pass"] += int(r["score"]["pass"])
        by_domain[r["problem"].domain]["total"] += 1
        by_domain[r["problem"].domain]["pass"] += int(r["score"]["pass"])

    return {
        "total": total, "crashed": crashed,
        "status_pass": status_pass, "status_pass_rate": status_pass / total,
        "incorrect_derivation_count": len(wrong_rows),
        "detected_count": len(detected), "not_detected_count": len(not_detected),
        "detection_rate": len(detected) / len(wrong_rows) if wrong_rows else None,
        "false_positive_count": len(false_positives),
        "false_positive_ids": [r["problem"].id for r in false_positives],
        "false_negative_count": len(false_negatives),
        "false_negative_ids": [r["problem"].id for r in false_negatives],
        "real_topic_match_rate": real_topic_match_rate,
        "topic_expectation_accuracy": (len(topic_as_expected) / len(topic_checked)
                                       if topic_checked else None),
        "distinct_llm_call_counts_seen": sorted(c for c in call_counts if c is not None),
        "by_category": dict(by_category),
        "by_domain": dict(by_domain),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write full per-problem results to this JSON file")
    ap.add_argument("--quiet", action="store_true", help="suppress the per-problem lines")
    args = ap.parse_args()

    rows = run_all()
    agg = summarize(rows)

    if not args.quiet:
        for r in rows:
            p, res, sc = r["problem"], r["result"], r["score"]
            mark = "PASS" if sc["pass"] else "FAIL"
            print(f"[{mark}] {p.id:42s} status={res.get('status') or res.get('error'):28s} "
                 f"calls={res.get('num_llm_calls', '-')}")
            for f in sc["findings"]:
                print(f"       - {f}")

    print("\n" + "=" * 78)
    print(f"Total problems:        {agg['total']}")
    print(f"Crashed:               {agg['crashed']}")
    print(f"Status as expected:    {agg['status_pass']}/{agg['total']} ({agg['status_pass_rate']:.0%})")
    print(f"Incorrect derivations: {agg['incorrect_derivation_count']} "
         f"(detected: {agg['detected_count']}, not detected: {agg['not_detected_count']}, "
         f"detection rate: {agg['detection_rate']:.0%})")
    print(f"False positives (wrong derivation reported verified_mathematically): "
         f"{agg['false_positive_count']} {agg['false_positive_ids']}")
    print(f"False negatives (correct derivation reported failed):              "
         f"{agg['false_negative_count']} {agg['false_negative_ids']}")
    print(f"Real topic-match rate (retrieval coverage): {agg['real_topic_match_rate']:.0%}")
    print(f"LLM calls per problem (should be a single value, 5): {agg['distinct_llm_call_counts_seen']}")
    print("\nBy category:")
    for cat, s in sorted(agg["by_category"].items()):
        print(f"  {cat:24s} {s['pass']}/{s['total']}")
    print("\nBy domain:")
    for dom, s in sorted(agg["by_domain"].items()):
        print(f"  {dom:28s} {s['pass']}/{s['total']}")

    if args.json:
        serializable = {
            "summary": agg,
            "problems": [
                {
                    "id": r["problem"].id, "category": r["problem"].category,
                    "domain": r["problem"].domain, "question": r["problem"].question,
                    "is_wrong": r["problem"].is_wrong, "notes": r["problem"].notes,
                    "expected_status": list(r["problem"].expected_status),
                    "result": {k: v for k, v in r["result"].items()},
                    "score": r["score"],
                }
                for r in rows
            ],
        }
        with open(args.json, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nFull results written to {args.json}")


if __name__ == "__main__":
    main()
