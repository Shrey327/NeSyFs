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

    def test_small_model_shapes_seen_in_real_runs_are_repaired_not_fatal(self):
        outputs = [
            {"facts": [{"subject": "apple", "predicate": "on", "object": "table"},
                       {"subject": "fridge", "predicate": "closed", "object": ""},
                       {"subject": "", "predicate": "open", "object": ""}]},
            {"lesson": {"title": "Storage", "lesson": "Open the fridge first."}},
        ]
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent"):
            model.return_value.invoke.side_effect = [SimpleNamespace(content=json.dumps(value)) for value in outputs]
            llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx")
            facts = llm.extract_memory("observe", "apple on table; fridge closed", []).facts
            self.assertEqual([(f.subject, f.predicate, f.object) for f in facts],
                             [("apple", "on", "table"), ("fridge", "is", "closed")])
            self.assertEqual(llm.lesson("store apple", "attempt").lesson, "Open the fridge first.")
            self.assertEqual(model.return_value.invoke.call_count, len(outputs))

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
        self.assertNotIn("raw", calls[0])

    def test_reasoning_markers_fail_closed_without_emitting_raw_output(self):
        unsafe = [
            '<think>private',
            '{"action":"take apple","rationale":"<TH' + 'INK>private</think>"}',
            r'{"action":"take apple","rationale":"\u003cthink\u003eprivate"}',
            r'{"action":"take apple","rationale":"<th\nink>private"}',
            '{"action":"take apple","rationale":"< / think>private"}',
        ]
        for content in unsafe:
            with self.subTest(content=content), patch("nesyfs.llm.ChatOpenAI") as model:
                model.return_value.invoke.return_value = SimpleNamespace(content=content)
                events = []
                llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx", on_call=events.append)
                with self.assertRaisesRegex(ValueError, "reasoning marker"):
                    llm.propose_action("store apple", "apple on table", [])
                self.assertEqual(events, [])
                self.assertFalse(llm._memo)

    def test_sampled_calls_use_sampling_model_and_are_never_reused(self):
        vote = SimpleNamespace(content='{"aligned": true, "feedback": ""}')
        names = SimpleNamespace(content='{"names": ["apple"]}')
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent"):
            model.return_value.invoke.side_effect = [vote, vote, vote, names]
            llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx", sample_temperature=0.7)
            temperatures = [call.kwargs["temperature"] for call in model.call_args_list]
            for _ in range(3):
                llm.reflect("store apple", "apple on table", "take apple", [])
            llm.extract_entities("apple")
            llm.extract_entities("apple")
        self.assertEqual(temperatures, [0, 0.7])
        self.assertEqual(model.return_value.invoke.call_count, 4)
        with self.assertRaises(ValueError):
            NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx", sample_temperature=5)

    def test_evaluation_can_disable_local_memo_even_for_identical_greedy_prompts(self):
        calls = []
        with patch("nesyfs.llm.ChatOpenAI") as model, patch("nesyfs.llm.create_agent"):
            model.return_value.invoke.return_value = SimpleNamespace(content='{"action":"take apple"}')
            llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx", cache_enabled=False, on_call=calls.append)
            llm.propose_action("store apple", "apple on table", [])
            llm.propose_action("store apple", "apple on table", [])
            self.assertIn("replace angle-bracket placeholders", model.return_value.invoke.call_args.args[0][1][1])
        self.assertEqual(model.return_value.invoke.call_count, 2)
        self.assertEqual([call['memo'] for call in calls], [False, False])

    def test_policy_temperature_is_distinct_from_reflection_and_planning_sampling(self):
        greedy, sampler, policy = MagicMock(), MagicMock(), MagicMock()
        greedy.invoke.return_value = SimpleNamespace(content='{"names":["apple"]}')
        sampler.invoke.side_effect = [SimpleNamespace(content='{"aligned":true,"feedback":""}'),
                                      SimpleNamespace(content='{"action":"open fridge"}')]
        policy.invoke.return_value = SimpleNamespace(content='{"action":"take apple"}')
        events = []
        with patch("nesyfs.llm.ChatOpenAI", side_effect=[greedy, sampler, policy]) as model, patch("nesyfs.llm.create_agent"):
            llm = NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx", sample_temperature=0.7,
                                      policy_temperature=0.4, on_call=events.append)
            llm.propose_action("store apple", "apple on table", [])
            llm.propose_action("store apple", "apple on table", [])
            llm.reflect("store apple", "apple on table", "take apple", [])
            llm.propose_action("store apple", "apple on table", [], sampled=True)
            llm.extract_entities("apple")
        self.assertEqual([c.kwargs['temperature'] for c in model.call_args_list], [0, 0.7, 0.4])
        self.assertEqual(policy.invoke.call_count, 2)
        self.assertEqual(sampler.invoke.call_count, 2)
        self.assertEqual(greedy.invoke.call_count, 1)
        self.assertEqual([event['temperature'] for event in events], [0.4, 0.4, 0.7, 0.7, 0])
        self.assertTrue(all(not event['memo'] for event in events))
        with self.assertRaises(ValueError):
            NeSyFSLanguageModel(MLX_MODEL, "", provider="mlx", policy_temperature=3)

    def test_policy_and_sampling_temperatures_are_configured_independently(self):
        with patch.dict(os.environ, {"NEO4J_PASSWORD": "test", "NESYFS_POLICY_TEMPERATURE": "0.3",
                                     "NESYFS_SAMPLE_TEMPERATURE": "0.8"}, clear=True):
            settings = Settings.from_env(provider="mlx")
        self.assertEqual((settings.policy_temperature, settings.sample_temperature), (0.3, 0.8))
        with patch.dict(os.environ, {"NEO4J_PASSWORD": "test", "NESYFS_POLICY_TEMPERATURE": "2.1"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env(provider="mlx")

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
