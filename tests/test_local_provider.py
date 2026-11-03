import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from pydantic import ValidationError

from nesyfs.config import MLX_BASE_URL, MLX_MODEL, MLX_4B_BASE_URL, MLX_4B_MODEL, Settings, provider_options
from nesyfs.llm import NeSyFSLanguageModel


class LocalProviderTests(unittest.TestCase):
    def test_local_settings_never_require_or_use_cloud_key_or_model(self):
        with patch.dict(os.environ, {"NEO4J_PASSWORD": "test", "OPENAI_API_KEY": "secret", "NESYFS_MODEL": "remote"}, clear=True):
            settings = Settings.from_env(provider="mlx")
        self.assertEqual(settings.provider, "mlx")
        self.assertEqual(settings.openai_api_key, "")
        self.assertEqual(settings.model, MLX_MODEL)
        with self.assertRaises(ValueError):
            Settings.from_env(provider="unknown")

    def test_local_endpoint_is_fixed_no_secret_and_no_agent_loop(self):
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent") as agent:
            NeSyFSLanguageModel(MLX_MODEL, "must-not-be-sent", provider="mlx")
        agent.assert_not_called()
        options = model.call_args.kwargs
        self.assertEqual(options["base_url"], MLX_BASE_URL)
        self.assertEqual(options["api_key"], "local-only")
        self.assertEqual(options["max_retries"], 0)
        self.assertEqual(options["max_tokens"], 512)
        self.assertEqual(options["timeout"], 120)
        self.assertFalse(options["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("must-not-be-sent", str(options))
        with self.assertRaises(ValueError):
            NeSyFSLanguageModel("untrusted-model", "", provider="mlx")

    def test_four_b_uses_its_own_loopback_and_no_cloud_key(self):
        with patch.dict(os.environ, {"NEO4J_PASSWORD": "test", "OPENAI_API_KEY": "secret", "NESYFS_MODEL": "remote"}, clear=True):
            settings = Settings.from_env(provider="mlx_4b")
        self.assertEqual(settings.openai_api_key, "")
        self.assertEqual(settings.model, MLX_4B_MODEL)
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent") as agent:
            NeSyFSLanguageModel(settings.model, "secret", provider=settings.provider)
        agent.assert_not_called()
        self.assertEqual(model.call_args.kwargs["base_url"], MLX_4B_BASE_URL)
        self.assertEqual(model.call_args.kwargs["api_key"], "local-only")
        self.assertNotIn("secret", str(model.call_args.kwargs))
        with self.assertRaises(ValueError):
            NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx_4b")

    def test_all_local_outputs_use_one_call_and_validate_their_schema(self):
        valid = [
            {"names": ["apple"]},
            {"facts": [{"subject": "apple", "predicate": "is on", "object": "table"}], "superseded_fact_ids": []},
            {"action": "take apple", "rationale": "The apple is on the table."},
            {"aligned": True, "feedback": ""},
            {"observation": "You hold the apple.", "progress": True},
        ]
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent"):
            model.return_value.invoke.side_effect = [SimpleNamespace(content=json.dumps(value)) for value in valid]
            llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx")
            self.assertEqual(llm.extract_entities("apple").names, ["apple"])
            self.assertEqual(llm.extract_memory("observe", "apple on table", []).facts[0].subject, "apple")
            self.assertEqual(llm.propose_action("store apple", "apple on table", []).action, "take apple")
            self.assertTrue(llm.reflect("store apple", "apple on table", "take apple", []).aligned)
            self.assertTrue(llm.predict("store apple", "apple on table", "take apple", []).progress)
            self.assertEqual(model.return_value.invoke.call_count, len(valid))

    def test_invalid_local_response_retries_once_then_fails_without_cloud_fallback(self):
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent") as agent:
            model.return_value.invoke.return_value = SimpleNamespace(content="not json")
            llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx")
            with self.assertRaises(ValidationError):
                llm.extract_entities("apple")
            self.assertEqual(model.return_value.invoke.call_count, 2)
            agent.assert_not_called()

    def test_invalid_response_gets_one_corrective_attempt(self):
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent"):
            model.return_value.invoke.side_effect = [SimpleNamespace(content='{"action":""}'),
                                                     SimpleNamespace(content='{"action":"take apple"}')]
            llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx")
            self.assertEqual(llm.propose_action("store apple", "apple on table", []).action, "take apple")
            self.assertEqual(model.return_value.invoke.call_count, 2)
            correction = model.return_value.invoke.call_args.args[0][-1][1]
            self.assertIn("Do not invent facts", correction)
            self.assertIn("string_too_short", correction)

    def test_fenced_json_still_receives_full_schema_validation(self):
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent"):
            model.return_value.invoke.return_value = SimpleNamespace(content='```json\n{"names":["apple"]}\n```')
            llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx")
            self.assertEqual(llm.extract_entities("apple").names, ["apple"])
            self.assertEqual(model.return_value.invoke.call_count, 1)

    def test_identical_local_prompts_reuse_output_and_report_telemetry(self):
        reply = SimpleNamespace(content='<think>\n\n</think>{"aligned": true, "feedback": ""}',
                                usage_metadata={"input_tokens": 300, "output_tokens": 9,
                                                "input_token_details": {"cache_read": 250}})
        calls = []
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent"):
            model.return_value.invoke.return_value = reply
            llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx", on_call=calls.append)
            votes = [llm.reflect("store apple", "apple on table", "take apple", []) for _ in range(5)]
            self.assertTrue(all(vote.aligned for vote in votes))
            self.assertEqual(model.return_value.invoke.call_count, 1)
            self.assertEqual(model.return_value.invoke.call_args.kwargs["max_tokens"], 160)
            votes[0].feedback = "mutated"
            self.assertEqual(llm.reflect("store apple", "apple on table", "take apple", []).feedback, "")
        self.assertEqual([call["memo"] for call in calls], [False, True, True, True, True, True])
        self.assertEqual((calls[0]["prompt_tokens"], calls[0]["cached_tokens"], calls[0]["completion_tokens"]), (300, 250, 9))
        self.assertEqual(calls[0]["kind"], "reflect")

    def test_readiness_checks_expected_model_without_inference(self):
        response = MagicMock()
        response.read.return_value = json.dumps({"data": [{"id": MLX_MODEL}]}).encode()
        opener = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = response
        opener.open.side_effect = [context, URLError("offline")]
        with patch("nesyfs.config.build_opener", return_value=opener):
            options = provider_options()
        self.assertTrue(next(item for item in options if item["id"] == "mlx")["ready"])
        self.assertFalse(next(item for item in options if item["id"] == "mlx_4b")["ready"])
        self.assertEqual(opener.open.call_count, 2)
        opener.open.assert_any_call(f"{MLX_BASE_URL}/models", timeout=1)
        opener.open.assert_any_call(f"{MLX_4B_BASE_URL}/models", timeout=1)


if __name__ == "__main__":
    unittest.main()
