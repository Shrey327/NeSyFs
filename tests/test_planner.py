import math
import random
import unittest
from unittest.mock import Mock

from nesyfs.models import ActionProposal, ExtractedFact, Fact, MemoryDelta, Prediction
from nesyfs.planner import Particle, TSMCPlanner


class RecordingRandom:
    """Deterministic resampling without asserting stochastic outcomes."""

    def __init__(self):
        self.weights = []

    def choices(self, population, weights, k):
        self.weights.append(list(weights))
        return [population[0] for _ in range(k)]

    def choice(self, population):
        return population[0]


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.llm = Mock()
        self.llm.propose_action.return_value = ActionProposal(action="take apple")
        self.llm.predict.return_value = Prediction(observation="You hold the apple.", progress=True)
        self.llm.extract_memory.return_value = MemoryDelta()
        self.fact = Fact(
            id="observed-1", subject="apple", predicate="on", object="table", observed_at="now"
        )

    def test_returns_first_action_of_selected_trajectory(self):
        self.llm.propose_action.side_effect = [
            ActionProposal(action=action) for action in ("take apple", "open fridge", "put apple in fridge")
        ]
        planner = TSMCPlanner(self.llm, particle_count=1, horizon=3, rng=random.Random(7))
        self.assertEqual(planner.choose_action("store apple", "apple on table", [self.fact]), "take apple")
        self.assertEqual(self.llm.propose_action.call_count, 3)
        self.assertEqual(self.llm.predict.call_count, 3)
        self.assertEqual(self.llm.extract_memory.call_count, 3)

    def test_progress_weights_are_passed_to_resampling(self):
        rng = RecordingRandom()
        self.llm.predict.side_effect = [
            Prediction(observation="progress", progress=True),
            Prediction(observation="no change", progress=False),
        ]
        planner = TSMCPlanner(self.llm, particle_count=2, horizon=1, temperature=0.5, rng=rng)
        planner.choose_action("store apple", "apple on table", [self.fact])
        self.assertEqual(rng.weights, [[math.exp(0.5), 1.0]])

    def test_simulated_memory_updates_do_not_mutate_observed_facts(self):
        original = [self.fact]
        original_value = self.fact.model_dump()
        self.llm.extract_memory.return_value = MemoryDelta(
            superseded_fact_ids=[self.fact.id],
            facts=[ExtractedFact(subject="apple", predicate="location", object="held")],
        )
        planner = TSMCPlanner(self.llm, particle_count=1, horizon=2, rng=random.Random(7))
        planner.choose_action("store apple", "apple on table", original)
        self.assertEqual(original, [self.fact])
        self.assertEqual(self.fact.model_dump(), original_value)
        simulated_facts = self.llm.propose_action.call_args_list[1].args[2]
        self.assertNotIn(self.fact.id, [fact.id for fact in simulated_facts])
        self.assertEqual(simulated_facts[0].object, "held")
        self.assertEqual(simulated_facts[0].observed_at, "simulated")

    def test_resampled_particles_have_independent_containers_and_reset_weights(self):
        source = Particle("observation", [self.fact], ["take apple"], weight=4.0)
        planner = TSMCPlanner(self.llm, particle_count=2, horizon=1, rng=RecordingRandom())
        particles = planner._resample([source])
        self.assertEqual(len(particles), 2)
        self.assertTrue(all(particle.weight == 1.0 for particle in particles))
        particles[0].actions.append("open fridge")
        particles[0].facts.clear()
        self.assertEqual(source.actions, ["take apple"])
        self.assertEqual(particles[1].actions, ["take apple"])
        self.assertEqual(source.facts, [self.fact])
        self.assertEqual(particles[1].facts, [self.fact])
