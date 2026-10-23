# Examples

Run these commands from the repository root after `uv sync --locked`.

## Scripted kitchen walkthrough

```bash
uv run --locked python examples/kitchen_walkthrough.py
```

This script tests the environment with three fixed actions. It does not instantiate an agent, call an LLM or connect to Neo4j. Its deterministic output is:

```text
Scripted environment walkthrough (no agent or model calls).
Initial observation: You are in the kitchen. An apple is on the table. The fridge is closed.
Action: take apple
Observation: You picked up the apple and are holding it.
Action: open fridge
Observation: The fridge is open.
Action: put apple in fridge
Observation: You put the apple in the fridge. The task is complete.
Complete: True
```

This is an environment example, not an agent success-rate result.

## Agent-driven kitchen episode

Complete the [agent setup](../README.md#quickstart), then run:

```bash
uv run --locked --env-file .env python main.py \
  --session-id "kitchen-$(date +%Y%m%d-%H%M%S)" \
  --task "Put the apple in the fridge" \
  --environment kitchen-demo \
  --max-steps 10
```

The toy environment recognizes simple action phrases containing `take`, `pick` or `grab`; `open` or `unlock`; and `put`, `place` or `store`. It only models one apple and one fridge. It has no benchmark parser or reward system. The completion condition is always “apple inside fridge”, regardless of the supplied task, so use the task above.

The agent selects its own actions. The order, reflection counts and planning decisions can differ between runs. No live agent transcript is bundled as a claimed result.

## Manual environment

```bash
uv run --locked --env-file .env python main.py \
  --session-id "manual-$(date +%Y%m%d-%H%M%S)" \
  --task "Put the apple in the fridge" \
  --environment manual \
  --observation "You are in a kitchen. An apple is on the table. The fridge is closed."
```

The CLI proposes one action. Execute it yourself in the external environment, then paste the resulting observation when prompted. The CLI updates graph memory and exits after that one step, even if `--max-steps` is larger than one.

Manual mode does not control a browser, launch a benchmark or carry out a command on your computer. For multi-step integration, use the Python adapter boundary described below.

## Integrating an environment

An adapter implements `execute(action: str) -> str` and must return a non-empty observation. `CallableEnvironment` wraps an existing callback. The orchestration call is:

```python
result = agent.step(
    session_id=episode_id,
    task=task,
    observation=observation,
    environment=environment,
    previous_action=previous_action,
)
observation = result.observation
previous_action = result.action
```

Here `agent` is a configured `NeSyFSAgent` and `environment` is your adapter. See [main.py](../main.py) for the complete construction and connection-cleanup sequence. Your application owns termination, reward collection, action validation and authorization. Never interpret model-generated action text as an unrestricted shell command.

Use a unique session ID for a new episode. Only reuse an ID when intentionally continuing the same environment state.
