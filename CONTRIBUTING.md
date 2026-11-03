# Contributing

Start with the [README](README.md), [architecture](docs/architecture.md) and [paper-to-code map](docs/paper.md#paper-to-code-map).

## Local checks

Use the repository's Python version and lockfile:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked python examples/kitchen_walkthrough.py
uv run --locked python -m compileall -q src main.py examples tests
git diff --check
```

Unit tests run without credentials or external services. Label live Neo4j checks, model-backed episodes and benchmark evaluations separately; passing unit tests does not establish those outcomes.

## Keep research changes reviewable

- Preserve the distinction between this implementation and the paper's experiments.
- If a change affects retrieval, reflection, prompting or particle updates, explain the algorithmic difference and add a focused test.
- Keep simulated facts out of persistent memory. Keep all graph operations scoped by `session_id`.
- Use parameterized database queries. Validate environment actions before connecting real systems.
- Do not commit API keys, `.env`, database files or private model inputs.
- Keep dependency updates separate when possible. Regenerate `uv.lock` deliberately rather than removing it.
- Cite the source, model, configuration and evaluation protocol for any new result. Do not use the scripted kitchen example as benchmark evidence.

## Pull requests

Describe the change, the reason for it and the commands you ran. State which checks passed and which checks were not run. Include known limitations and screenshots for visual documentation changes.

No repository license is currently specified. A license choice requires an explicit maintainer decision; do not infer one from dependencies or the reference paper.
