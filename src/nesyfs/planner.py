from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from uuid import uuid4

from nesyfs.llm import NeSyFSLanguageModel
from nesyfs.models import Fact


@dataclass
class Particle:
    observation: str
    facts: list[Fact]
    actions: list[str] = field(default_factory=list)
    weight: float = 1.0


class TSMCPlanner:
    """Lightweight TSMC-style planner with binary progress weights and resampling."""

    def __init__(
        self,
        llm: NeSyFSLanguageModel,
        particle_count: int = 6,
        horizon: int = 5,
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ):
        self._llm = llm
        self._particle_count = particle_count
        self._horizon = horizon
        self._temperature = temperature
        self._rng = rng or random.Random()

    def choose_action(self, task: str, observation: str, facts: list[Fact]) -> str:
        particles = [Particle(observation=observation, facts=list(facts))
                     for _ in range(self._particle_count)]
        for _ in range(self._horizon):
            propagated: list[Particle] = []
            for particle in particles:
                proposal = self._llm.propose_action(task, particle.observation, particle.facts)
                prediction = self._llm.predict(
                    task, particle.observation, proposal.action, particle.facts
                )
                particle.actions.append(proposal.action)
                particle.observation = prediction.observation
                particle.weight *= math.exp(self._temperature) if prediction.progress else 1.0
                # Candidate states are isolated in memory; planning never writes
                # hypothetical facts into the persistent session graph.
                delta = self._llm.extract_memory(
                    proposal.action, prediction.observation, particle.facts
                )
                stale = set(delta.superseded_fact_ids)
                particle.facts = [fact for fact in particle.facts if fact.id not in stale]
                particle.facts.extend(
                    Fact(
                        id=str(uuid4()),
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                        observed_at="simulated",
                        evidence=prediction.observation[:500],
                    )
                    for fact in delta.facts
                )
                propagated.append(particle)
            particles = self._resample(propagated)

        selected = self._rng.choice(particles)
        return selected.actions[0]

    def _resample(self, particles: list[Particle]) -> list[Particle]:
        return [
            Particle(
                observation=particle.observation,
                facts=list(particle.facts),
                actions=list(particle.actions),
                weight=1.0,
            )
            for particle in self._rng.choices(
                particles, weights=[item.weight for item in particles], k=self._particle_count
            )
        ]
