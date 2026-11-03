"""Matched, locally measured episodes for the toy kitchen evaluation."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from nesyfs.eval_env import SCENARIOS, EvaluationKitchen
from nesyfs.knowledge_graph import Neo4jKnowledgeGraph
from nesyfs.llm import NeSyFSLanguageModel
from nesyfs.planner import TSMCPlanner
from nesyfs.service import NeSyFSAgent

METHODS = ("reactive", "history", "graph_reflection", "adaptive")


def _local_model_provenance(model_path):
    """Identify small config files; large model weights are not hashed."""
    directory = Path(model_path)
    result = {"model_path": str(directory), "declared_model_id": None,
              "declared_revision": None, "quantization": None, "model_type": None,
              "config_sha256": {}, "weights_sha256": None,
              "runtime_version": None,
              "note": "Weight hash and separately running MLX server version are unavailable."}
    for name in ("config.json", "tokenizer_config.json"):
        try:
            path = directory / name
            if path.stat().st_size > 1_000_000:
                continue
            content = path.read_bytes()
            result["config_sha256"][name] = sha256(content).hexdigest()
            if name == "config.json":
                config = json.loads(content)
                if isinstance(config, dict):
                    model_type = config.get("model_type")
                    if isinstance(model_type, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", model_type):
                        result["model_type"] = model_type
                    quant = config.get("quantization", config.get("quantization_config"))
                    if isinstance(quant, dict):
                        result["quantization"] = {key: quant[key] for key in ("bits", "group_size")
                                                  if type(quant.get(key)) is int}
        except (OSError, ValueError, TypeError):
            continue
    try:
        install_path = directory / ".stillroom-install.json"
        if install_path.stat().st_size <= 1_000_000:
            install = json.loads(install_path.read_text())
            repo, revision = install.get("repo"), install.get("revision")
            if isinstance(repo, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
                result["declared_model_id"] = repo
            if isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
                result["declared_revision"] = revision
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    try:
        result["client_langchain_openai_version"] = metadata.version("langchain-openai")
    except metadata.PackageNotFoundError:
        result["client_langchain_openai_version"] = None
    return result


class UsageTracker(BaseCallbackHandler):
    """Count provider usage only if every started model call reported metadata."""
    def __init__(self):
        self.started = set()
        self.reported = set()
        self.input = self.output = self.total = 0

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.started.add(run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        try:
            usage = response.generations[0][0].message.usage_metadata
            values = [usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")]
            if not all(isinstance(value, int) and value >= 0 for value in values):
                return
        except (AttributeError, IndexError, KeyError, TypeError):
            return
        self.reported.add(run_id)
        self.input += values[0]
        self.output += values[1]
        self.total += values[2]

    def measured(self):
        if not self.started or self.started != self.reported:
            return None
        return {"input": self.input, "output": self.output, "total": self.total}


_usage_context = ContextVar("nesyfs_evaluation_usage", default=None)
register_configure_hook(_usage_context, inheritable=True)


class NoPlanning(Exception):
    """A rejected decision ends without an environment action in this ablation."""


class RejectPlanner:
    def choose_action(self, task, observation, facts):
        raise NoPlanning()


class RecordedEnvironment:
    def __init__(self, environment, actions):
        self.environment = environment
        self.actions = actions
        self.planning_used = False

    def execute(self, action):
        previous_invalid = self.environment.invalid_actions
        observation = self.environment.execute(action)
        self.actions.append({"action": action, "observation": observation,
                             "invalid": self.environment.invalid_actions > previous_invalid,
                             "planning_used": self.planning_used})
        return observation


def _episode(scenario, method, settings, llm, graph, action_budget, seed):
    env = EvaluationKitchen(scenario)
    session_id = str(uuid4())
    initial = env.observe()
    observation = initial
    actions, trace, history = [], [], [f"Initial observation: {initial}"]
    rejected = decisions = planner_uses = 0
    error = None
    usage = UsageTracker()
    token = _usage_context.set(usage)
    started = time.monotonic()
    recorder = RecordedEnvironment(env, actions)
    if method in ("graph_reflection", "adaptive"):
        # Stable per-task seed: matched conditions see identical task instances.
        task_seed = int.from_bytes(sha256(f"{seed}:{scenario.id}".encode()).digest()[:8], "big")
        planner = (RejectPlanner() if method == "graph_reflection" else
                   TSMCPlanner(llm, settings.particle_count, settings.planning_horizon,
                               rng=random.Random(task_seed)))
        agent = NeSyFSAgent(llm, graph, planner, settings.reflection_limit, settings.reflection_samples)
    try:
        while decisions < action_budget and not env.is_complete:
            decisions += 1
            previous_action = actions[-1]["action"] if actions else None
            recorder.planning_used = False
            if method in ("reactive", "history"):
                context = observation
                if method == "history" and history:
                    context += "\n\nPrevious action/observation history:\n" + "\n".join(history)
                proposal = llm.propose_action(scenario.task, context, [])
                new_observation = recorder.execute(proposal.action)
                history.append(f"Action: {proposal.action}\nObservation: {new_observation}")
                observation = new_observation
            else:
                def emit(stage, message, data=None):
                    nonlocal planner_uses
                    if stage == "planning" and message.startswith("Reflection rejected"):
                        if method == "adaptive":
                            planner_uses += 1
                            recorder.planning_used = True
                        else:
                            return  # The ablation stops here; it does not run the planner.
                    trace.append({"stage": stage, "message": message})
                try:
                    result = agent.step(session_id, scenario.task, observation, recorder,
                                        previous_action=previous_action, trace=emit)
                    observation = result.observation
                except NoPlanning:
                    rejected += 1
                    trace.append({"stage": "rejection", "message": "Rejected decision; no action was executed."})
            # One rejected decision consumes a turn; this prevents unbounded retries.
        status = "completed" if env.is_complete else "budget_exhausted"
    except Exception as exc:
        # Error text may contain credentials or request details; retain type only.
        error = type(exc).__name__
        status = "error"
    finally:
        elapsed = time.monotonic() - started
        _usage_context.reset(token)
    return {"episode_id": session_id, "task_id": scenario.id, "task": scenario.task,
            "scenario": {"label": getattr(scenario, "label", scenario.id),
                         "description": getattr(scenario, "description", ""),
                         "difficulty": getattr(scenario, "difficulty", "standard"),
                         "min_actions": getattr(scenario, "min_actions", None)},
            "initial_observation": initial, "method": method,
            "success": bool(env.is_complete), "status": status, "error": error,
            "actions": actions, "decision_count": decisions, "invalid_actions": env.invalid_actions,
            "rejected_decisions": rejected, "planner_uses": planner_uses,
            "elapsed_seconds": round(elapsed, 3), "tokens": usage.measured(), "cost": None,
            "usage_coverage": {"calls_started": len(usage.started), "calls_reported": len(usage.reported),
                               "reported_tokens": ({"input": usage.input, "output": usage.output,
                                                    "total": usage.total} if usage.reported else None)},
            "trace": trace}


def _write(run, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = file.name
        try:
            json.dump(run, file, indent=2, ensure_ascii=False)
            file.write("\n")
        except Exception:
            os.unlink(temporary)
            raise
    os.replace(temporary, path)


def run_comparison(settings, task_ids, methods, action_budget, output_path, seed=42,
                   on_episode=None, repeats=1, on_start=None):
    """Run each requested method on each task, persisting real results as they finish.

    All conditions share task text, model and decision limit. A reflected rejection
    in the no-planner ablation consumes one decision without executing an action.
    """
    available = {scenario.id: scenario for scenario in SCENARIOS}
    if (not task_ids or len(set(task_ids)) != len(task_ids) or
            any(task_id not in available for task_id in task_ids)):
        raise ValueError("Select distinct known task IDs.")
    if (not methods or len(set(methods)) != len(methods) or
            any(method not in METHODS for method in methods)):
        raise ValueError("Select distinct known methods.")
    if not isinstance(action_budget, int) or isinstance(action_budget, bool) or not 1 <= action_budget <= 20:
        raise ValueError("Decision budget must be an integer from 1 to 20.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Seed must be an integer.")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 10:
        raise ValueError("Repeats must be an integer from 1 to 10.")
    source_root = Path(__file__).parent
    source_files = ("evaluation.py", "eval_env.py", "service.py", "llm.py", "planner.py", "knowledge_graph.py")
    source_digests = {name: sha256((source_root / name).read_bytes()).hexdigest() for name in source_files}
    run = {"schema_version": 1, "run_id": str(uuid4()),
           "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
           "model": settings.model, "provider": getattr(settings, "provider", "openai"),
           "action_budget": action_budget, "seed": seed, "repeats": repeats,
           "model_temperature": 0, "python_version": sys.version.split()[0],
           "source_sha256": source_digests,
           "settings": {key: getattr(settings, key) for key in
                        ("reflection_limit", "reflection_samples", "particle_count", "planning_horizon")},
           "task_ids": list(task_ids), "methods": list(methods), "episodes": [], "error": None}
    if getattr(settings, "provider", "openai") in ("mlx", "mlx_4b"):
        run["local_model_provenance"] = _local_model_provenance(settings.model)
    _write(run, output_path)
    graph = None
    try:
        provider = getattr(settings, "provider", "openai")
        api_key = settings.openai_api_key if provider == "openai" else ""
        llm = NeSyFSLanguageModel(settings.model, api_key, provider=provider)
        if any(m in ("graph_reflection", "adaptive") for m in methods):
            graph = Neo4jKnowledgeGraph(settings.neo4j_uri, settings.neo4j_username,
                                         settings.neo4j_password, settings.retrieval_limit)
            graph.verify_connection()
            graph.setup()
        order = list(task_ids)
        random.Random(seed).shuffle(order)
        for repeat in range(1, repeats + 1):
            for index, task_id in enumerate(order):
                rotation = (index + repeat - 1) % len(methods)
                methods_order = list(methods[rotation:]) + list(methods[:rotation])
                for method in methods_order:
                    if on_start:
                        on_start(task_id, method, repeat)
                    episode = _episode(available[task_id], method, settings, llm, graph, action_budget, seed + repeat - 1)
                    episode["repeat"] = repeat
                    episode["run_order"] = len(run["episodes"]) + 1
                    run["episodes"].append(episode)
                    _write(run, output_path)
                    if on_episode:
                        on_episode(run)
                    if episode["status"] == "error":
                        run["status"] = "error"
                        run["error"] = "Evaluation stopped after an episode error; remaining comparisons were not run."
                        return run
        run["status"] = "completed"
        return run
    except Exception as exc:
        run["status"] = "error"
        run["error"] = f"Evaluation stopped ({type(exc).__name__}); remaining comparisons were not run."
        return run
    finally:
        if graph:
            graph.close()
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write(run, output_path)
