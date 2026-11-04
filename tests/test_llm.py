import unittest
from unittest.mock import MagicMock, patch

from nesyfs.llm import NeSyFSLanguageModel
from nesyfs.models import ActionProposal, MemoryDelta


class LanguageModelConfigurationTests(unittest.TestCase):
    def test_luna_disables_reasoning_for_function_calls(self):
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent"):
            NeSyFSLanguageModel("gpt-6-luna", "test-key")
        model.assert_called_once_with(
            model="gpt-6-luna", api_key="test-key", temperature=0, reasoning_effort="none"
        )

    def test_other_models_keep_existing_configuration(self):
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent"):
            NeSyFSLanguageModel("gpt-4.1-mini", "test-key")
        model.assert_called_once_with(model="gpt-4.1-mini", api_key="test-key", temperature=0)

    def test_remote_policy_temperature_uses_its_own_structured_model(self):
        greedy, policy = MagicMock(), MagicMock()
        policy.with_structured_output.return_value.invoke.return_value = ActionProposal(action="take apple")
        with patch("nesyfs.llm.ChatOpenAI", side_effect=[greedy, policy]) as models, patch("nesyfs.llm.create_agent") as agent:
            llm = NeSyFSLanguageModel("gpt-4.1-mini", "test-key", policy_temperature=0.4)
            self.assertEqual(llm.propose_action("store apple", "apple on table", []).action, "take apple")
        self.assertEqual([call.kwargs["temperature"] for call in models.call_args_list], [0, 0.4])
        policy.with_structured_output.assert_called_once_with(ActionProposal)
        agent.return_value.invoke.assert_not_called()

    def test_no_effect_observation_keeps_memory_without_a_model_call(self):
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent") as agent:
            llm = NeSyFSLanguageModel("gpt-4.1-mini", "test-key")
            delta = llm.extract_memory("unlock cupboard with key",
                                       "You must hold the key to unlock the cupboard; nothing changed.", [])
        self.assertEqual(delta, MemoryDelta())
        model.return_value.with_structured_output.assert_not_called()
        agent.return_value.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
