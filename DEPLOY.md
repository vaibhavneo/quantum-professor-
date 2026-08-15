# Deploying Quantum Professor

Standard library plus four packages, all in `requirements.txt`: `openai` for
the DeepSeek client, and `sympy`, `numpy`, `scipy` for the Math Lab. The Math
Lab is the reason the rest are not optional — `/api/math`, `/api/ode` and
`/api/schrodinger` raise ImportError without sympy and numpy, and without
scipy the Schrödinger solver falls back to a dense eigensolve that takes over
two minutes on a small container where the tridiagonal one takes 0.1 s.

## How a deploy happens

Pushing to `main` builds automatically. That is worth stating because it was
not true until 2026-08-15: the service had a source repo but no deployment
trigger, and the trigger could not be created because the Railway GitHub App
had been *authorized* on the account without being *installed* — two different
things on GitHub, and the OAuth page reports the app as connected either way.
Five commits sat unbuilt while the site served an older revision and reported
itself healthy.

To force a build without a push — or if the trigger is ever lost again:

    railway redeploy --service quantum-professor --yes --from-source

`--from-source` matters. Plain `redeploy` re-runs the commit that is already
deployed, which looks like it worked and changes nothing.

## Required environment variable

    DEEPSEEK_API_KEY=sk-...

This has to be a real environment variable on the host. `_api_key()` also
looks in `.env` files next to sibling projects on the developer's machine,
which is how it resolves locally — none of those paths exist on a deployed
box, so the variable is the only source there.

Without it the site still serves — topics, learning path, books, problems, the
Solver and the whole Math Lab are keyless — but "Ask the Professor" returns an
error instead of an explanation.

## Verifying a deployment

    python3 tests/smoke_deployment.py https://your-app.up.railway.app

25 checks: the page and its vendored KaTeX, the keyless APIs, the solver, all
three Math Lab endpoints, and one full streaming answer. The numeric ones have
closed forms behind them — the 1 nm box against n²·0.376 eV, the finite well's
bound-state count against z₀ — so a pass means the deployed solver is correct,
not merely reachable. Exit code is non-zero on any failure.

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
