"""Local, step-by-step showcase for the kitchen agent."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from nesyfs import KitchenDemoEnvironment, NeSyFSAgent, Neo4jKnowledgeGraph, NeSyFSLanguageModel, Settings, TSMCPlanner

TASK = "Put the apple in the fridge"
# Same style of rules the comparison-lab scenarios give. Without them a small model
# (and its own reflection votes) treats "put apple in fridge" as matching the goal.
RULES = (" Commands: take apple; open fridge; put apple in fridge. You must be holding the "
         "apple and the fridge must be open before you can put the apple in it. Each new "
         "observation starts with the result of your last action.")
WEB = Path(__file__).resolve().parent / "web"
RESULTS = Path(__file__).resolve().parent / "results"


class UserError(ValueError):
    pass


class EvaluationManager:
    """Single bounded local evaluation worker; the runner persists each episode."""
    def __init__(self, results=RESULTS):
        self.results = results
        self.lock = threading.Lock()
        self.state = {"running": False, "completed": 0, "total": 0,
                      "current_task": None, "current_method": None, "provider": None, "error": None}

    @staticmethod
    def config():
        from nesyfs.eval_env import SCENARIOS
        from nesyfs.config import provider_options
        return {"tasks": [{"id": scenario.id, "task": scenario.task,
                           "label": getattr(scenario, "label", scenario.id.replace("_", " ")),
                           "description": getattr(scenario, "description", ""),
                           "difficulty": getattr(scenario, "difficulty", "standard"),
                           "min_actions": getattr(scenario, "min_actions", None)}
                          for scenario in SCENARIOS],
                "providers": provider_options(), "default_provider": "mlx",
                "limits": {"max_tasks": len(SCENARIOS), "max_repeats": 3, "max_steps": 20}}

    def status(self):
        with self.lock:
            return dict(self.state)

    def history(self):
        runs = []
        if self.results.exists():
            for path in sorted(self.results.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    with path.open(encoding="utf-8") as source:
                        run = json.load(source)
                    if isinstance(run, dict):
                        runs.append(run)
                except (OSError, ValueError):
                    logging.warning("Could not read an evaluation result file")
        runs.sort(key=lambda run: str(run.get("created_at") or ""), reverse=True)
        return {"runs": runs}

    def start(self, payload):
        config = self.config()
        allowed = {item["id"] for item in config["tasks"]}
        ids, repeats, steps = payload.get("task_ids"), payload.get("repeats"), payload.get("max_steps")
        if (not isinstance(ids, list) or not 1 <= len(ids) <= config["limits"]["max_tasks"]
                or any(not isinstance(item, str) or item not in allowed for item in ids)
                or len(set(ids)) != len(ids)):
            raise UserError("Select distinct available tasks.")
        if type(repeats) is not int or not 1 <= repeats <= 3:
            raise UserError("Repeats must be an integer from 1 to 3.")
        if type(steps) is not int or not 1 <= steps <= config["limits"]["max_steps"]:
            raise UserError(f"Maximum steps must be an integer from 1 to {config['limits']['max_steps']}.")
        provider = payload.get("provider")
        if provider not in ("openai", "mlx", "mlx_4b"):
            raise UserError("Choose a supported model provider.")
        providers = config.get("providers", [])
        if providers and not any(p["id"] == provider and p["ready"] for p in providers):
            raise UserError("The selected provider is not configured or available. Check the provider status.")
        try:
            settings = Settings.from_env(provider=provider)
        except (ValueError, TypeError):
            raise UserError("Set the required credentials and valid numeric settings first.") from None
        with self.lock:
            if self.state["running"]:
                raise UserError("An evaluation is already running.")
            self.state = {"running": True, "completed": 0, "total": len(ids) * repeats * 4,
                          "current_task": None, "current_method": None, "provider": provider, "error": None}
        path = self.results / f"{uuid.uuid4()}.json"
        def work():
            try:
                from nesyfs.evaluation import run_comparison
                self.results.mkdir(parents=True, exist_ok=True)
                def on_start(task_id, method, repeat):
                    with self.lock:
                        self.state.update(current_task=task_id, current_method=method)
                def on_episode(run):
                    with self.lock:
                        self.state["completed"] = len(run["episodes"])
                result = run_comparison(settings=settings, task_ids=ids,
                               methods=["reactive", "history", "graph_reflection", "adaptive"],
                               action_budget=steps, output_path=path, seed=42,
                               repeats=repeats, on_start=on_start, on_episode=on_episode)
                if result.get("status") == "error":
                    with self.lock:
                        self.state["error"] = "Evaluation stopped after an error; see saved partial results."
            except Exception as exc:
                logging.error("Evaluation failed: %s", type(exc).__name__)
                with self.lock:
                    self.state["error"] = "Evaluation stopped; check the local server log and saved partial results."
            finally:
                with self.lock:
                    self.state.update(running=False, current_task=None, current_method=None)
        threading.Thread(target=work, name="evaluation", daemon=True).start()
        return self.status()


class Showcase:
    def __init__(self):
        self.lock = threading.Lock()
        self.event_lock = threading.Lock()
        self.session = None
        self.graph = None
        self.running = False
        self.evaluation = EvaluationManager()

    def status(self):
        from nesyfs.config import provider_options
        providers = provider_options()
        ready = bool(os.getenv("NEO4J_PASSWORD"))
        return {"ready": ready, "providers": providers,
                "default_provider": next((p["id"] for p in providers if p["id"] != "openai" and p["ready"]), "openai"),
                "settings": {name: os.getenv(f"NESYFS_{name.upper()}") for name in
                             ("reflection_limit", "reflection_samples", "particle_count", "planning_horizon")},
                "error": None if ready else "Set NEO4J_PASSWORD in .env and start Neo4j."}

    def start(self, task, provider="openai"):
        if task != TASK:
            raise UserError("This showcase supports the apple and fridge task.")
        if provider not in ("openai", "mlx", "mlx_4b"):
            raise UserError("Choose a supported model provider.")
        if not self.lock.acquire(blocking=False):
            raise UserError("An agent step is already running.")
        try:
            try:
                settings = Settings.from_env(provider=provider)
            except (ValueError, TypeError):
                raise UserError("Set the required credentials in .env for this provider.") from None
            graph = Neo4jKnowledgeGraph(settings.neo4j_uri, settings.neo4j_username,
                                         settings.neo4j_password, settings.retrieval_limit)
            try:
                graph.verify_connection()
                graph.setup()
                llm = NeSyFSLanguageModel(settings.model, settings.openai_api_key, provider=provider)
                planner = TSMCPlanner(llm, settings.particle_count, settings.planning_horizon)
                agent = NeSyFSAgent(llm, graph, planner, settings.reflection_limit, settings.reflection_samples)
            except Exception:
                graph.close()
                raise
            if self.graph:
                self.graph.close()
            env = KitchenDemoEnvironment()
            self.graph = graph
            self.session = {"id": str(uuid.uuid4()), "env": env, "agent": agent, "llm": llm,
                            "previous_action": None, "events": []}
            return {"session_id": self.session["id"], "observation": env.observe(),
                    "state": self.kitchen(env), "model": settings.model, "provider": provider}
        finally:
            self.lock.release()

    @staticmethod
    def kitchen(env):
        return {"apple": env.apple_location, "fridge_open": env.fridge_open}

    def current(self, session_id):
        session = self.session
        if not session or session_id != session["id"]:
            raise UserError("Episode not found. Start a new episode.")
        return session

    def snapshot(self, session_id):
        session = self.current(session_id)
        with self.event_lock:
            return {"events": list(session["events"]), "observation": session["env"].observe(),
                    "state": self.kitchen(session["env"]),
                    "completed": session["env"].is_complete, "running": self.running}

    def step(self, session_id):
        if not self.lock.acquire(blocking=False):
            raise UserError("An agent step is already running.")
        try:
            session = self.current(session_id)
            env = session["env"]
            if env.is_complete:
                raise UserError("This episode is complete. Start a new one.")
            with self.event_lock:
                session["events"] = []
                self.running = True
            started = time.perf_counter()
            def trace(stage, message, data=None):
                with self.event_lock:
                    session["events"].append({"stage": stage, "message": message, "data": data,
                                              "t": round((time.perf_counter() - started) * 1000)})
            session["llm"].on_call = lambda record: trace("llm", record["kind"], record)
            class RecordingEnvironment:
                def execute(self, action):
                    observation = env.execute(action)
                    session["previous_action"] = action
                    session["last_outcome"] = observation
                    return observation
            # The agent must see what its last action did, not only the static state;
            # with greedy decoding an unchanged input repeats the same (wrong) action forever.
            observation = env.observe()
            if session["previous_action"] and session.get("last_outcome"):
                observation = (f"Result of your last action '{session['previous_action']}': "
                               f"{session['last_outcome']}\nCurrent state: {observation}")
            result = session["agent"].step(session_id, TASK + RULES, observation, RecordingEnvironment(),
                                           previous_action=session["previous_action"], trace=trace)
            return {"action": result.action, "observation": result.observation, "state": self.kitchen(env),
                    "completed": env.is_complete, "events": list(session["events"])}
        finally:
            self.running = False
            self.lock.release()

    def close(self):
        if self.graph:
            self.graph.close()


def make_handler(showcase):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def allowed(self):
            host = self.headers.get("Host", "").split(":")[0].lower()
            origin = self.headers.get("Origin")
            if host not in ("localhost", "127.0.0.1"):
                return False
            if origin:
                parsed = urlsplit(origin)
                return parsed.scheme == "http" and parsed.netloc == self.headers.get("Host")
            return True

        def send_json(self, code, data):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.allowed():
                self.send_error(403)
                return
            url = urlsplit(self.path)
            if url.path == "/api/status":
                self.send_json(200, showcase.status())
            elif url.path == "/api/evaluation/config":
                self.send_json(200, showcase.evaluation.config())
            elif url.path == "/api/evaluation/status":
                self.send_json(200, showcase.evaluation.status())
            elif url.path == "/api/evaluation/results":
                self.send_json(200, showcase.evaluation.history())
            elif url.path == "/api/events":
                try:
                    self.send_json(200, showcase.snapshot(parse_qs(url.query).get("session_id", [""])[0]))
                except UserError as exc:
                    self.send_json(400, {"error": str(exc)})
            elif url.path in ("/", "/index.html", "/style.css", "/studio.css", "/app.js", "/evaluation.html", "/evaluation.css", "/evaluation.js"):
                name = "index.html" if url.path == "/" else url.path[1:]
                data = (WEB / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", {"html": "text/html", "css": "text/css", "js": "text/javascript"}[name.rsplit(".", 1)[1]] + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.allowed():
                self.send_json(403, {"error": "Forbidden"})
                return
            if self.path not in ("/api/session", "/api/step", "/api/evaluation/run"):
                self.send_json(404, {"error": "Not found"})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise UserError("Expected JSON.")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 4096:
                    raise UserError("Invalid request size.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise UserError("Expected a JSON object.")
                if self.path == "/api/session":
                    data = showcase.start(payload.get("task"), payload.get("provider", "openai"))
                elif self.path == "/api/evaluation/run":
                    data = showcase.evaluation.start(payload)
                else:
                    data = showcase.step(payload.get("session_id"))
                self.send_json(200, data)
            except UserError as exc:
                self.send_json(400, {"error": str(exc)})
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_json(400, {"error": "Invalid JSON."})
            except Exception as exc:
                logging.error("Showcase operation failed: %s", type(exc).__name__)
                data = {"error": "The agent could not complete this operation. Check local configuration and service logs; the environment may have changed."}
                if self.path == "/api/step":
                    try:
                        data.update(showcase.snapshot(payload.get("session_id")))
                    except (ValueError, UnboundLocalError):
                        pass
                self.send_json(500, data)
    return Handler


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the local NeSyFS showcase")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    for name, value in {"NESYFS_MODEL": "gpt-6-luna", "NESYFS_REFLECTION_LIMIT": "1",
                        "NESYFS_REFLECTION_SAMPLES": "1", "NESYFS_PARTICLE_COUNT": "1",
                        "NESYFS_PLANNING_HORIZON": "1"}.items():
        os.environ.setdefault(name, value)
    showcase = Showcase()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(showcase))
    print(f"NeSyFS showcase: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        showcase.close()


if __name__ == "__main__":
    main()
