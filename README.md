# NeSyFS

A modular implementation of the NeSyFS fast-slow agent described in `2607.28942v2.pdf`.
It represents partial state as a session-scoped Neo4j fact graph, checks reactive
actions with stepwise self-consistency reflection, and switches to a particle-based
planner after repeated rejected actions.

## Architecture

```text
main.py
└── NeSyFSAgent                 step orchestration and environment boundary
    ├── NeSyFSLanguageModel     LangChain create_agent policy + structured LLM tasks
    ├── Neo4jKnowledgeGraph     episode-isolated facts and stale-fact replacement
    └── TSMCPlanner             simulated particles, progress weights, resampling
```

`NeSyFSAgent.step()` accepts an `EnvironmentAdapter`; an application can implement
it around its own environment API. `kitchen-demo` is a small local simulator for
running a whole episode without manually entering observations. `manual` mode
remains available for trying the agent against an external environment by hand.
Neither mode integrates the ALFWorld, WebShop, or ScienceWorld benchmarks.

## Setup

Run Neo4j locally and set `OPENAI_API_KEY`, `NEO4J_PASSWORD`, and optionally
`NEO4J_URI`, `NEO4J_USERNAME`, and `NESYFS_MODEL`. `uv run --env-file .env` loads
the repository's environment file. Configure these variables in `.env` or in
your deployment secret manager; never commit credentials.

```bash
uv sync
uv run --env-file .env python main.py \
  --session-id demo-episode \
  --task "Put the apple in the fridge" \
  --environment kitchen-demo
```

The local simulator executes each proposed action and returns the next
observation automatically. Use `--max-steps` to set an episode limit. To use
manual mode, pass `--environment manual --observation "..."` instead.

The CLI creates the Neo4j uniqueness constraints on startup. Graph facts are
isolated by `session_id`; use a fresh ID for each independent episode. Defaults
follow the paper where specified: 3 reactive/reflection attempts, 5 reflection
samples, 6 planning particles, and a planning horizon of 5. Override these with
`NESYFS_REFLECTION_LIMIT`, `NESYFS_REFLECTION_SAMPLES`,
`NESYFS_PARTICLE_COUNT`, `NESYFS_PLANNING_HORIZON`, and
`NESYFS_RETRIEVAL_LIMIT`.
