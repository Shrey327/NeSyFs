# NeSyFS

**Neuro-symbolic fast-slow agents for partially observable environments.**

A compact Python implementation inspired by [NeSyFS](https://arxiv.org/abs/2607.28942v2) by Duo Xu and Faramarz Fekri. The agent maintains a session-scoped fact graph in Neo4j, checks reactive actions through stepwise reflection, then falls back to particle-based planning when those actions are repeatedly rejected.

[Quickstart](#quickstart) · [Architecture](docs/architecture.md) · [Examples](examples/README.md) · [Configuration and reproduction](docs/reproduction.md) · [Paper reference](docs/paper.md)

![NeSyFS implementation flow: update and retrieve graph facts, propose and reflect on an action, use particle planning after repeated rejection, then execute one action and update memory.](docs/assets/architecture.svg)

## What is included

| Component | Implementation |
| --- | --- |
| **Structured memory** | Observation-to-fact extraction, stale-fact replacement and entity-based retrieval from a Neo4j graph isolated by episode ID. |
| **Fast thinking** | A LangChain policy proposes one action from the task, current observation and retrieved facts. |
| **Stepwise reflection** | Accepts a proposal when at least 80% of reflection votes agree. Rejected proposals receive corrective feedback. |
| **Slow thinking** | A lightweight TSMC-style planner simulates candidate trajectories, weights binary progress and resamples particles. |
| **Environment boundary** | A small kitchen simulator, a manual CLI mode and an adapter protocol for custom environments. |

This is an independent implementation, not the paper authors' benchmark release. ALFWorld, WebShop and ScienceWorld integrations are not included. The kitchen demo exercises the action/observation loop; it does not reproduce the paper's benchmark results. See the [paper-to-code comparison](docs/paper.md#paper-to-code-map) for implementation differences.

## Quickstart

### 1. Install the locked environment

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python **3.14+**. The repository selects Python 3.14 through `.python-version`; uv can install it if needed.

```bash
git clone https://github.com/Shrey327/NeSyFs.git
cd NeSyFs
uv sync --locked
```

Try the deterministic environment walkthrough first. It uses scripted actions with **no API key or Neo4j server**:

```bash
uv run --locked python examples/kitchen_walkthrough.py
```

### 2. Configure the agent

The full agent requires an OpenAI API key plus a running Neo4j database. API calls may incur charges.

```bash
cp .env.example .env
```

Edit `.env`: set `OPENAI_API_KEY` and choose a `NEO4J_PASSWORD` of at least eight characters. Keep the default username `neo4j` when using the bundled local database configuration.

If you have Docker with Compose, start the local database:

```bash
docker compose up -d
docker compose logs -f neo4j
```

Wait for the database to report that it has started, then press Ctrl+C to leave the log viewer. Neo4j Browser is available at <http://localhost:7474>. Both published ports are bound to localhost. If you already run Neo4j, skip Docker and set its connection details in `.env` instead.

### 3. Run a kitchen episode

```bash
uv run --locked --env-file .env python main.py \
  --session-id "kitchen-$(date +%Y%m%d-%H%M%S)" \
  --task "Put the apple in the fridge" \
  --environment kitchen-demo \
  --max-steps 10
```

The CLI prints each chosen action, whether planning was used, the number of reflection attempts and the next observation. It stops when the kitchen reports completion or the step limit is reached. Actions are model-generated, so a successful run is not guaranteed.

Use a **fresh session ID for each independent episode**. Starting the kitchen again resets its state but does not clear old graph facts. The CLI verifies the database connection and creates uniqueness constraints on startup.

For manual interaction with an external environment, see [examples](examples/README.md#manual-environment). For connection errors, model configuration or cleanup, see [the run guide](docs/reproduction.md).

## How a step works

1. Extract relevant entities from the task and observation. Retrieve existing facts, update memory, then retrieve the current context.
2. Propose a reactive action. With the defaults, four of five reflection votes must approve it.
3. Retry with reflection feedback after rejection. After three rejected proposals, invoke the planner.
4. In the planner, simulate six particles for five rollout rounds. Reweight and resample after each round, then select the first action of a surviving trajectory.
5. Execute one action through the environment adapter. Update persistent memory from the actual outcome.

Hypothetical planner facts stay in memory and are never written to the persistent graph. Reflection is model judgment, not a safety guarantee; enforce action permissions in any real environment adapter.

## Repository guide

```text
main.py                       Episode CLI
src/nesyfs/
  config.py                   Environment-backed settings
  environments.py             Local kitchen simulator
  knowledge_graph.py          Neo4j fact storage and retrieval
  llm.py                      Policy and structured LLM tasks
  models.py                   Typed facts, predictions and step results
  planner.py                  TSMC-style particle planning
  service.py                  Step orchestration and environment protocol
examples/                     Scripted walkthrough and usage examples
tests/                        Offline unit tests
docs/                         Architecture, run guide and paper comparison
2607.28942v2.pdf               Bundled reference paper
```

## Development

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m compileall -q src main.py examples tests
```

Tests use fake model responses and a fake database driver. They check local behavior without making API calls or starting Neo4j. They are not live integration tests or benchmark evaluations. See [contribution guidance](CONTRIBUTING.md) before changing research behavior.

## Research reference

**Duo Xu and Faramarz Fekri.** *NeSyFS: A Neuro-symbolic Fast-Slow Thinking Framework for LLM Agent under Partial Observability.* arXiv:2607.28942v2, 2026.

[Versioned paper](https://arxiv.org/abs/2607.28942v2) · [Bundled PDF](2607.28942v2.pdf) · [BibTeX](docs/reference.bib) · [Paper-reported results and scope](docs/paper.md)

The documentation refers to **v2**, matching the bundled PDF. Paper results are attributed to the authors and are not measurements from this repository.

## License

No software license is currently specified. The bundled paper retains its original licensing terms; this change does not relicense it.
