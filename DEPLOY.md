# Deploying Quantum Professor

Standard library plus three packages, all in `requirements.txt`: `openai` for
the DeepSeek client, and `sympy` + `numpy` for the Math Lab. The Math Lab is
the reason the last two are not optional — `/api/math`, `/api/ode` and
`/api/schrodinger` raise ImportError without them.

## Required environment variable

    DEEPSEEK_API_KEY=sk-...

This has to be a real environment variable on the host. `_api_key()` also
looks in `.env` files next to sibling projects on the developer's machine,
which is how it resolves locally — none of those paths exist on a deployed
box, so the variable is the only source there.

Without it the site still serves — topics, learning path, books, problems, the
Solver and the whole Math Lab are keyless — but "Ask the Professor" returns an
error instead of an explanation.

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
are then built from the 41-topic curriculum, the deterministic solvers, and the
model, with no book citations.

This path is tested, not assumed: with `BRAIN_ROOT` pointed at a directory that
does not exist, "what determines the partition function" still returns a full
answer opening on the physics and citing `[C:quantum-statistical-mechanics]`.
Curriculum topics are first-class sources, so losing the shelf costs breadth,
not grounding.

Everything else is unaffected: computed numbers still come only from
`physics.py`, the pipeline trace still streams, mode/depth still apply, and the
citation audit still runs.

To restore grounding, either run locally against the indexes, or ship a slim
subset and point `SECOND_BRAIN_ROOT` at it.
