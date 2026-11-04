"""Run every agent configuration on the partial-observability tasks with the paper's settings.

Results are saved to results/ and appear in the Comparison lab. Settings are fixed here,
before any run, so a result cannot be produced by tuning them afterwards.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import uuid

from nesyfs.config import Settings
from nesyfs.eval_env import SCENARIOS
from nesyfs.evaluation import METHODS, run_comparison

# NeSyFS paper, pages 4-5 and 10: 5 votes, 80% approval, K=3 proposals, N=6 particles, L=5.
PAPER = {"reflection_samples": 5, "reflection_limit": 3, "particle_count": 6, "planning_horizon": 5}
# Votes and rollouts must be samples to differ; the policy itself stays greedy.
SAMPLE_TEMPERATURE = 0.7
HIDDEN_TASKS = [s.id for s in SCENARIOS if s.hidden]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="mlx_4b", choices=["openai", "mlx", "mlx_4b"])
    parser.add_argument("--tasks", nargs="+", default=HIDDEN_TASKS, choices=[s.id for s in SCENARIOS])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = replace(Settings.from_env(provider=args.provider), sample_temperature=SAMPLE_TEMPERATURE, **PAPER)
    output = Path(__file__).resolve().parent / "results" / f"benchmark-{uuid.uuid4()}.json"
    total = len(args.tasks) * len(args.methods) * args.repeats
    print(f"{total} episodes -> {output.name}", flush=True)

    def on_episode(run):
        ep = run["episodes"][-1]
        label = ep['status'] if ep['status'] in ('error', 'cancelled') else ('SUCCESS' if ep['success'] else ep['status'])
        print(f"[{len(run['episodes'])}/{total}] {ep['task_id']:<20} {ep['method']:<17} "
              f"{label:<17} actions={len(ep['actions']):<3} "
              f"invalid={ep['invalid_actions']:<3} {ep['elapsed_seconds']:.0f}s", flush=True)

    run = run_comparison(settings=settings, task_ids=args.tasks, methods=args.methods,
                         action_budget=args.budget, output_path=output, seed=args.seed,
                         repeats=args.repeats, on_episode=on_episode,
                         stop_on_error=False)  # Recoverable episode failures are recorded; fatal service errors stop.
    print(f"status={run['status']} {run.get('error') or ''}", flush=True)


if __name__ == "__main__":
    main()
