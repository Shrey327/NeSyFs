import unittest
from unittest.mock import Mock, call

from nesyfs.models import ActionProposal, EntityNames, ExtractedFact, Fact, MemoryDelta, Reflection
from nesyfs.service import NeSyFSAgent


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.llm = Mock()
        self.graph = Mock()
        self.planner = Mock()
        self.environment = Mock()
        self.environment.execute.return_value = "You are holding the apple."
        self.fact = Fact(
            id="fact-1", subject="apple", predicate="on", object="table", observed_at="now"
        )
        self.graph.retrieve.return_value = [self.fact]
        self.llm.extract_entities.return_value = EntityNames(names=["apple", " apple ", ""])
        self.llm.extract_memory.return_value = MemoryDelta()
        self.llm.propose_action.return_value = ActionProposal(action="take apple", rationale="Pick it up.")
        self.llm.reflect.return_value = Reflection(aligned=True)
        self.planner.choose_action.return_value = "open fridge"
        self.agent = NeSyFSAgent(self.llm, self.graph, self.planner)

    def run_step(self, **kwargs):
        return self.agent.step(
            "episode-1", "Put the apple in the fridge", "An apple is on the table.",
            self.environment, **kwargs
        )

    def test_four_of_five_votes_accepts_fast_action(self):
        self.llm.reflect.side_effect = [Reflection(aligned=value) for value in (True, True, True, True, False)]
        result = self.run_step()
        self.assertFalse(result.planning_used)
        self.assertEqual(result.reflection_attempts, 1)
        self.assertEqual(result.action, "take apple")
        self.assertEqual(self.llm.reflect.call_count, 5)
        self.environment.execute.assert_called_once_with("take apple")
        self.planner.choose_action.assert_not_called()

    def test_rejected_action_is_retried_with_only_negative_feedback(self):
        self.llm.propose_action.side_effect = [
            ActionProposal(action="put apple in fridge"), ActionProposal(action="take apple")
        ]
        self.llm.reflect.side_effect = [
            Reflection(aligned=True, feedback="accepted"),
            Reflection(aligned=True), Reflection(aligned=True),
            Reflection(aligned=False, feedback="First pick up the apple"),
            Reflection(aligned=False, feedback="The fridge is closed"),
        ] + [Reflection(aligned=True) for _ in range(5)]
        result = self.run_step()
        self.assertFalse(result.planning_used)
        self.assertEqual(result.reflection_attempts, 2)
        feedback = self.llm.propose_action.call_args_list[1].args[3]
        self.assertEqual(feedback, "First pick up the apple; The fridge is closed")
        self.environment.execute.assert_called_once_with("take apple")

    def test_repeated_rejection_uses_planner_before_executing_once(self):
        self.llm.reflect.return_value = Reflection(aligned=False, feedback="Try another action")
        result = self.run_step()
        self.assertTrue(result.planning_used)
        self.assertEqual(result.reflection_attempts, 3)
        self.assertEqual(self.llm.propose_action.call_count, 3)
        self.assertEqual(self.llm.reflect.call_count, 15)
        self.planner.choose_action.assert_called_once_with(
            "Put the apple in the fridge", "An apple is on the table.", [self.fact]
        )
        self.environment.execute.assert_called_once_with("open fridge")

    def test_nondefault_vote_threshold_rounds_up(self):
        self.agent = NeSyFSAgent(self.llm, self.graph, self.planner, reflection_limit=1, reflection_samples=3)
        self.llm.reflect.side_effect = [Reflection(aligned=value) for value in (True, True, False)]
        self.assertTrue(self.run_step().planning_used)
        self.assertEqual(self.llm.reflect.call_count, 3)

    def test_memory_updates_preserve_session_and_include_actual_outcome(self):
        initial_delta = MemoryDelta(facts=[ExtractedFact(subject="apple", predicate="on", object="table")])
        outcome_delta = MemoryDelta(facts=[ExtractedFact(subject="apple", predicate="location", object="held")])
        self.llm.extract_memory.side_effect = [initial_delta, outcome_delta]
        result = self.run_step(previous_action="look around")
        self.assertEqual(self.llm.extract_memory.call_args_list[0].args[0], "look around")
        self.assertEqual(self.llm.extract_memory.call_args_list[1].args[:2], ("take apple", result.observation))
        self.assertEqual(self.graph.upsert.call_args_list, [
            call("episode-1", initial_delta, "An apple is on the table."),
            call("episode-1", outcome_delta, "You are holding the apple."),
        ])
        self.assertEqual(self.graph.retrieve.call_args_list[0].args[0], "episode-1")
        self.assertIn("table", self.graph.retrieve.call_args_list[1].args[1])

    def test_initial_observation_is_labeled(self):
        self.run_step()
        self.assertEqual(self.llm.extract_memory.call_args_list[0].args[0], "initial observation")

    def test_blank_step_inputs_fail_before_external_work(self):
        for index in range(3):
            values = ["episode-1", "task", "observation"]
            values[index] = "  "
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    self.agent.step(*values, self.environment)
        self.llm.extract_entities.assert_not_called()
        self.graph.retrieve.assert_not_called()
        self.environment.execute.assert_not_called()

    def test_invalid_environment_outcome_is_not_recorded(self):
        for observation in (None, "", "  ", 1):
            with self.subTest(observation=observation):
                self.graph.reset_mock()
                self.llm.extract_memory.reset_mock()
                self.environment.execute.return_value = observation
                with self.assertRaisesRegex(ValueError, "non-empty observation"):
                    self.run_step()
                self.assertEqual(self.graph.upsert.call_count, 1)
                self.assertEqual(self.llm.extract_memory.call_count, 1)
