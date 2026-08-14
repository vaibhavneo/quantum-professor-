# Deploying Quantum Professor

Pure standard library apart from `openai`, so the build is trivial.

## Required environment variable

    DEEPSEEK_API_KEY=sk-...

Without it the site still serves — topics, learning path, books, problems and
the Solver are all keyless — but "Ask the Professor" returns an error instead
of an explanation.

## Binding

`web_server.py` reads `$PORT` and binds `0.0.0.0` when `PORT` is present in the
environment; locally it keeps `127.0.0.1:5052`.

## Known limitation of the hosted build: no library grounding

The retrieval layer reads the `second_brain` gateway and its `desk-physics` /
`desk-quantum-computing` indexes. Those are ~1.2 GB of chunks derived from a
local book collection — `desk-physics/chunks.json` alone is 162 MB, past
GitHub's 100 MB limit — so they are gitignored and are NOT part of this
deployment.

`retrieve_evidence()` degrades deliberately rather than failing: it returns
`available: False` and the evidence panel says "Library unavailable". Answers
are then built from the 32-topic curriculum, the deterministic solvers, and the
model, with no book citations.

Everything else is unaffected: computed numbers still come only from
`physics.py`, the pipeline trace still streams, mode/depth still apply, and the
citation audit still runs.

To restore grounding, either run locally against the indexes, or ship a slim
subset and point `SECOND_BRAIN_ROOT` at it.
