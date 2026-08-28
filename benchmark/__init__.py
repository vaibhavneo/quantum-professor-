"""Permanent physics reasoning benchmark for the Quantum Professor pipeline.

Run with:  python3 -m benchmark.run_benchmark

Everything here is deterministic and offline - no DeepSeek/paid API call is
ever made. Each problem scripts the LLM boundary (understand / reasoning /
professor / validation replies) the same way the test suite does, then runs
the REAL pipeline (real match_topics(), real compute_for(), real
retrieve_evidence() against the local book index, real verify_derivation())
around that scripted core, so retrieval, mathematical-object extraction, and
verification are measured against the actual production code, not a stub.
"""
