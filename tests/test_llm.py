import unittest
from unittest.mock import patch

from nesyfs.llm import NeSyFSLanguageModel


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


if __name__ == "__main__":
    unittest.main()
