import argparse

from nesyfs import (
    CallableEnvironment,
    KitchenDemoEnvironment,
    NeSyFSAgent,
    Neo4jKnowledgeGraph,
    NeSyFSLanguageModel,
    Settings,
    TSMCPlanner,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NeSyFS episode")
    parser.add_argument("--session-id", required=True, help="Isolates this episode's graph facts")
    parser.add_argument("--task", required=True)
    parser.add_argument("--environment", choices=("manual", "kitchen-demo"), default="manual")
    parser.add_argument("--observation", help="Required for --environment manual")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    if not args.session_id.strip():
        parser.error("--session-id must be non-empty")
    if not args.task.strip():
        parser.error("--task must be non-empty")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.environment == "manual" and not (args.observation and args.observation.strip()):
        parser.error("--observation is required for --environment manual")

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        parser.error(str(exc))
    graph = Neo4jKnowledgeGraph(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.retrieval_limit,
    )
    try:
        graph.verify_connection()
        graph.setup()
        llm = NeSyFSLanguageModel(settings.model, settings.openai_api_key)
        planner = TSMCPlanner(
            llm, settings.particle_count, settings.planning_horizon
        )
        agent = NeSyFSAgent(
            llm,
            graph,
            planner,
            settings.reflection_limit,
            settings.reflection_samples,
        )

        if args.environment == "kitchen-demo":
            environment = KitchenDemoEnvironment()
            observation = environment.observe()
        else:
            def execute(action: str) -> str:
                print(f"\nProposed action: {action}")
                return input("Run it in your environment, then paste the new observation: ").strip()

            environment = CallableEnvironment(execute)
            observation = args.observation

        previous_action = None
        for step_number in range(1, args.max_steps + 1):
            result = agent.step(
                args.session_id,
                args.task,
                observation,
                environment,
                previous_action=previous_action,
            )
            print(
                f"Step {step_number}: {result.action}\n"
                f"  Planning used: {result.planning_used}; "
                f"reflection attempts: {result.reflection_attempts}\n"
                f"  Observation: {result.observation}"
            )
            observation = result.observation
            previous_action = result.action
            if getattr(environment, "is_complete", False):
                print("Task complete.")
                break
            if args.environment == "manual":
                break
        else:
            print(f"Stopped after {args.max_steps} steps without environment completion.")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
