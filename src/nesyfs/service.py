from __future__ import annotations

from math import ceil
from typing import Protocol

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
    ) -> StepResult:
        if not session_id.strip() or not task.strip() or not observation.strip():
            raise ValueError("session_id, task, and observation must be non-empty")
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

        feedback = ""
        proposal = None
        planning_used = False
        accepted = False
        attempts = 0
        votes_required = ceil(self._reflection_samples * 0.8)
        for _ in range(self._reflection_limit):
            proposal = self._llm.propose_action(task, observation, facts, feedback)
            attempts += 1
            votes = [
                self._llm.reflect(task, observation, proposal.action, facts)
                for _ in range(self._reflection_samples)
            ]
            if sum(vote.aligned for vote in votes) >= votes_required:
                accepted = True
                break
            feedback = "; ".join(vote.feedback for vote in votes if not vote.aligned)

        if not accepted:
            planning_used = True
            proposal.action = self._planner.choose_action(task, observation, facts)
            proposal.rationale = "Selected from uncertainty-aware particle planning."

        next_observation = environment.execute(proposal.action)
        if not isinstance(next_observation, str) or not next_observation.strip():
            raise ValueError("Environment must return a non-empty observation")
        outcome_delta = self._llm.extract_memory(proposal.action, next_observation, facts)
        self._graph.upsert(session_id, outcome_delta, next_observation)
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
