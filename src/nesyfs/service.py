from __future__ import annotations

from math import ceil
from typing import Callable, Protocol

from nesyfs.knowledge_graph import Neo4jKnowledgeGraph
from nesyfs.llm import NeSyFSLanguageModel
from nesyfs.models import StepResult
from nesyfs.planner import TSMCPlanner


class EnvironmentAdapter(Protocol):
    def execute(self, action: str) -> str:
        """Execute one action and return the environment's new observation."""


class NeSyFSAgent:
    def __init__(
        self,
        llm: NeSyFSLanguageModel,
        graph: Neo4jKnowledgeGraph,
        planner: TSMCPlanner,
        reflection_limit: int = 3,
        reflection_samples: int = 5,
    ):
        self._llm = llm
        self._graph = graph
        self._planner = planner
        self._reflection_limit = reflection_limit
        self._reflection_samples = reflection_samples

    def step(
        self,
        session_id: str,
        task: str,
        observation: str,
        environment: EnvironmentAdapter,
        previous_action: str | None = None,
        trace: Callable[[str, str, dict | None], None] | None = None,
    ) -> StepResult:
        if not session_id.strip() or not task.strip() or not observation.strip():
            raise ValueError("session_id, task, and observation must be non-empty")
        # trace(stage, message, data): data is a JSON-safe dict for UIs, or None.
        def emit(stage: str, message: str, data: dict | None = None) -> None:
            if trace is not None:
                trace(stage, message, data)
        def fact_rows(items) -> list[dict]:
            return [{"id": f.id, "subject": f.subject, "predicate": f.predicate, "object": f.object}
                    for f in items]

        emit("memory", "Extracting entities and updating memory from the observation.")
        entities = self._llm.extract_entities(f"Task: {task}\nObservation: {observation}")
        entity_names = list(dict.fromkeys(name.strip() for name in entities.names if name.strip()))
        existing = self._graph.retrieve(session_id, entity_names)
        delta = self._llm.extract_memory(
            previous_action or "initial observation", observation, existing
        )
        self._graph.upsert(session_id, delta, observation)
        entity_names.extend(
            name
            for fact in delta.facts
            for name in (fact.subject, fact.object)
            if name not in entity_names
        )
        facts = self._graph.retrieve(session_id, entity_names)
        fact_text = "; ".join(f"{f.subject} {f.predicate} {f.object}" for f in facts)
        emit("memory", f"Retrieved {len(facts)} relevant facts: {fact_text or 'none'}",
             {"entities": entity_names, "facts": fact_rows(facts),
              "superseded": list(delta.superseded_fact_ids)})

        feedback = ""
        proposal = None
        planning_used = False
        accepted = False
        attempts = 0
        votes_required = ceil(self._reflection_samples * 0.8)
        for _ in range(self._reflection_limit):
            proposal = self._llm.propose_action(task, observation, facts, feedback)
            attempts += 1
            emit("proposal", f"Proposed: {proposal.action}. Model rationale: {proposal.rationale or 'none provided'}",
                 {"attempt": attempts, "action": proposal.action, "rationale": proposal.rationale,
                  "feedback": feedback})
            votes = [
                self._llm.reflect(task, observation, proposal.action, facts)
                for _ in range(self._reflection_samples)
            ]
            approved = sum(vote.aligned for vote in votes)
            verdict = {"attempt": attempts, "action": proposal.action, "required": votes_required,
                       "approved": approved, "accepted": approved >= votes_required,
                       "votes": [vote.model_dump() for vote in votes]}
            if approved >= votes_required:
                emit("reflection", f"Accepted: {approved}/{len(votes)} votes aligned; {votes_required} required.", verdict)
                accepted = True
                break
            feedback = "; ".join(vote.feedback for vote in votes if not vote.aligned)
            emit("reflection", f"Rejected: {approved}/{len(votes)} votes aligned; {votes_required} required. Feedback: {feedback or 'none provided'}", verdict)

        if not accepted:
            planning_used = True
            emit("planning", "Reflection rejected the proposals; running particle planning.")
            proposal.action = self._planner.choose_action(task, observation, facts)
            proposal.rationale = "Selected from uncertainty-aware particle planning."
            emit("planning", f"Planner selected: {proposal.action}",
                 {"action": proposal.action, "rollouts": getattr(self._planner, "last_rollouts", None)})

        next_observation = environment.execute(proposal.action)
        emit("action", f"Executed: {proposal.action}",
             {"action": proposal.action, "planning_used": planning_used, "rationale": proposal.rationale})
        if not isinstance(next_observation, str) or not next_observation.strip():
            raise ValueError("Environment must return a non-empty observation")
        emit("observation", next_observation)
        outcome_delta = self._llm.extract_memory(proposal.action, next_observation, facts)
        self._graph.upsert(session_id, outcome_delta, next_observation)
        emit("memory", f"Recorded outcome: {len(outcome_delta.facts)} facts extracted, {len(outcome_delta.superseded_fact_ids)} marked superseded.",
             {"new_facts": [f.model_dump() for f in outcome_delta.facts],
              "superseded": list(outcome_delta.superseded_fact_ids)})
        emit("complete", "Agent step completed.")
        return StepResult(
            action=proposal.action,
            observation=next_observation,
            planning_used=planning_used,
            reflection_attempts=attempts,
            rationale=proposal.rationale,
        )


class CallableEnvironment:
    """Small adapter for applications that already expose an action callback."""

    def __init__(self, execute):
        self._execute = execute

    def execute(self, action: str) -> str:
        result = self._execute(action)
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Environment action must return a non-empty observation")
        return result
