# Configuration and reproduction

[Back to README](../README.md) · [Examples](../examples/README.md) · [Paper comparison](paper.md)

## What each run establishes

| Check | External services | What it establishes |
| --- | --- | --- |
| Unit tests | None | Behavior under controlled model and database fakes. |
| Scripted kitchen walkthrough | None | The environment accepts a fixed three-action sequence. |
| Agent-driven kitchen episode | OpenAI API + Neo4j | A model-backed run of this toy environment, if it completes. |
| Paper benchmark reproduction | Not provided | Requires benchmark adapters, evaluation tooling and protocol alignment beyond this repository. |

Install the locked dependencies before the first two checks; installation itself may need network access. Do not report either check as evidence of benchmark performance.

## Runtime and dependencies

Python 3.14 is selected by `.python-version`; `pyproject.toml` requires Python 3.14 or newer. Use the committed lockfile:

```bash
uv sync --locked
uv run --locked python main.py --help
```

`--locked` makes uv reject an out-of-date lockfile rather than silently resolve new versions. See [uv's locking documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

Commands in this repository assume a POSIX shell. On Windows, use an equivalent shell or replace the `$(date ...)` session suffix with a manually chosen unique ID. `Settings.from_env()` reads process variables; it does not load `.env` itself. Use `uv run --env-file .env` or export variables before invoking Python.

## Configuration reference

Copy [`.env.example`](../.env.example) to `.env`. Empty required values must be filled before a model-backed run.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required | API credential for `ChatOpenAI`. |
| `NESYFS_MODEL` | `gpt-4.1-mini` | Model used by all LLM tasks. |
| `NEO4J_URI` | `bolt://localhost:7687` | Database connection URI. |
| `NEO4J_USERNAME` | `neo4j` | Database username. |
| `NEO4J_PASSWORD` | Required | Database password. |
| `NESYFS_REFLECTION_LIMIT` | `3` | Maximum reactive proposals with reflection before planning. |
| `NESYFS_REFLECTION_SAMPLES` | `5` | Votes per proposal; acceptance uses `ceil(0.8 * samples)`. |
| `NESYFS_PARTICLE_COUNT` | `6` | Number of simulated particles. |
| `NESYFS_PLANNING_HORIZON` | `5` | Future rollout rounds in this implementation. |
| `NESYFS_RETRIEVAL_LIMIT` | `100` | Maximum retrieved facts per graph query. |

All five limits must be positive integers. Planner progress-weight temperature defaults to `1.0` in its Python constructor and has no environment-variable override. LLM generation temperature is fixed at `0` in `llm.py`.

Changing `NESYFS_MODEL` alone does not reproduce the paper. Model support for structured output and generation parameters must be checked before choosing a different provider/model combination.

## Local Neo4j

The optional [Compose file](../compose.yaml) uses the versioned `neo4j:5.26.26-community` image. The 6.x Python driver supports Neo4j 5.x servers according to the [driver compatibility documentation](https://neo4j.com/docs/python-manual/current/install/).

1. Set `NEO4J_PASSWORD` in `.env` to a private password of at least eight characters.
2. Keep `NEO4J_USERNAME=neo4j` for this local setup.
3. Run `docker compose up -d`.
4. Watch `docker compose logs -f neo4j` until startup finishes.
5. Sign in at <http://localhost:7474> with the same credentials if you want to inspect the graph.

Only the database password is passed to the container; the OpenAI key is not. HTTP and Bolt ports are bound to `127.0.0.1`. The named volume retains graph data across container restarts. This configuration is for local development, not a production deployment.

Stop the database without removing its data:

```bash
docker compose down
```

`NEO4J_AUTH` sets the initial password only. Changing `.env` after the database volume already exists does not change that password. Use the existing credentials or change the database password through Neo4j. See the [official Docker authentication guidance](https://neo4j.com/docs/operations-manual/current/docker/introduction/).

To intentionally erase **all data in this project's Compose volume**, run `docker compose down --volumes`. Do not use that command if the volume contains data you want to preserve.

## Inspecting one session

In Neo4j Browser, set the ID from your run and retrieve its facts:

```cypher
:param session_id => 'your-session-id';
MATCH (f:NeSyFact {session_id: $session_id})
RETURN f.subject, f.predicate, f.object, f.observed_at, f.evidence
ORDER BY f.observed_at DESC;
```

To display entity/fact connections:

```cypher
MATCH (e:NeSyEntity {session_id: $session_id})-[:HAS_FACT]->
      (f:NeSyFact {session_id: $session_id})
RETURN e, f;
```

Reuse a session ID only to continue the same environment. A new kitchen process always starts with its initial state; it does not restore a previous episode from Neo4j.

## Cost, privacy and repeatability

Reflection makes several model calls for each candidate action. Planning adds action proposals, observation/progress predictions and memory extraction for every particle in every rollout round. Raising the particle count, horizon or reflection limits can increase latency and cost substantially. Start with the small kitchen task and a short step limit.

Task text, observations and retrieved facts are sent to the configured model service. Neo4j persists extracted facts and truncated observation evidence. Use non-sensitive example data; do not place secrets in task descriptions or observations.

The CLI has no seed flag. `TSMCPlanner` accepts a `random.Random` object for controlled local sampling, but this does not make remote LLM responses reproducible. Record the following when sharing a model-backed run:

- Commit SHA and dependency lockfile.
- Python and Neo4j versions.
- Model identifier and non-secret configuration values.
- Task, environment version, episode/session ID and step limit.
- Actions, observations, termination reason and any errors.
- Whether a result is scripted, mocked, model-backed or benchmark-derived.

Never include API keys or passwords in the run record.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Python version does not satisfy the project | Let uv install Python 3.14, or install it explicitly with `uv python install 3.14`. |
| `Missing required environment variable` | Fill the required `.env` values and use `--env-file .env`. |
| `NeSyFS limits must all be positive integers` or integer parsing error | Use positive whole numbers for every `NESYFS_*` limit. |
| Neo4j connection refused | Wait for startup; check `docker compose ps`, container logs and `NEO4J_URI`. |
| Neo4j authentication failure | Match the database's existing password. Changing `.env` does not reset an existing volume. |
| Model authentication, quota or structured-output error | Check API access and the selected model. These are live-service failures, not addressed by unit tests. |
| Agent repeats an action or never completes | Inspect its observations and facts; use a fresh session for a restarted environment. The toy parser and LLM policy can both fail. |
| Manual mode stops after one action | This is the CLI's intended manual-mode behavior. Use a Python episode loop for a custom environment. |

## Benchmark reproduction boundary

No dataset download, benchmark adapter, reward aggregation, baseline runner or experiment-results file is included. A reproduction effort would need to add those pieces, align prompts and model settings, reconcile the implementation differences in [the paper map](paper.md#paper-to-code-map), then evaluate across documented task splits and random seeds.

The paper's reported scores should remain separate from any future results obtained with this code.
