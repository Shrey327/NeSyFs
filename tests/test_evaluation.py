import json
from hashlib import sha256
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from nesyfs import evaluation
from nesyfs.eval_env import SCENARIOS
from nesyfs.models import ActionProposal, EntityNames, MemoryDelta, Prediction, Reflection

SETTINGS = SimpleNamespace(model='fake-model', openai_api_key='secret', neo4j_uri='',
    neo4j_username='', neo4j_password='password', retrieval_limit=100,
    reflection_limit=1, reflection_samples=1, particle_count=1, planning_horizon=1)


class FakeGraph:
    def __init__(self, *args): self.sessions = []
    def verify_connection(self): pass
    def setup(self): pass
    def close(self): pass
    def retrieve(self, session, names): self.sessions.append(session); return []
    def upsert(self, session, delta, observation): self.sessions.append(session)


class FakeModel:
    def __init__(self, *args, aligned=True, fail_after_action=False, **kwargs):
        self.aligned, self.fail_after_action = aligned, fail_after_action
        self.memory_calls = 0
        self.contexts = []
    def extract_entities(self, text): return EntityNames(names=['apple'])
    def extract_memory(self, action, observation, facts):
        self.memory_calls += 1
        if self.fail_after_action and self.memory_calls == 2: raise RuntimeError('secret')
        return MemoryDelta()
    def propose_action(self, task, observation, facts, feedback=''):
        self.contexts.append(observation)
        if 'holding' in observation.lower(): action = 'put apple in fridge'
        elif 'opened the fridge' in observation.lower(): action = 'take apple'
        elif 'hands are empty' in observation.lower(): action = 'open fridge'
        else: action = 'take apple'
        return ActionProposal(action=action)
    def reflect(self, task, observation, action, facts): return Reflection(aligned=self.aligned, feedback='no')
    def predict(self, task, observation, action, facts): return Prediction(observation='predicted', progress=True)


class EvaluationTests(unittest.TestCase):
    def test_reactive_and_history_use_same_initial_state_but_history_retains_it(self):
        reactive = FakeModel()
        history = FakeModel()
        for method, model in [('reactive', reactive), ('history', history)]:
            result = evaluation._episode(SCENARIOS[0], method, SETTINGS, model, None, 3, 42)
            self.assertTrue(result['success'])
            self.assertEqual([a['action'] for a in result['actions']],
                             ['open fridge', 'take apple', 'put apple in fridge'])
            self.assertEqual(result['decision_count'], 3)
        self.assertEqual(reactive.contexts[0].split('\n\n')[0], history.contexts[0].split('\n\n')[0])
        self.assertNotIn('Initial observation:', reactive.contexts[1])
        self.assertIn('Initial observation:', history.contexts[1])

    def test_rejected_ablation_consumes_budget_without_action_or_planning(self):
        result = evaluation._episode(SCENARIOS[0], 'graph_reflection', SETTINGS,
                                     FakeModel(aligned=False), FakeGraph(), 2, 42)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['rejected_decisions'], 2)
        self.assertEqual(result['planner_uses'], 0)
        self.assertEqual(result['actions'], [])

    def test_adaptive_runs_planner_after_rejection(self):
        result = evaluation._episode(SCENARIOS[0], 'adaptive', SETTINGS,
                                     FakeModel(aligned=False), FakeGraph(), 1, 42)
        self.assertEqual(result['planner_uses'], 1)
        self.assertEqual(len(result['actions']), 1)
        self.assertTrue(result['actions'][0]['planning_used'])

    def test_action_preserved_when_outcome_memory_fails(self):
        result = evaluation._episode(SCENARIOS[0], 'graph_reflection', SETTINGS,
                                     FakeModel(fail_after_action=True), FakeGraph(), 3, 42)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error'], 'RuntimeError')
        self.assertEqual(result['actions'][0]['action'], 'open fridge')
        self.assertNotIn('secret', json.dumps(result))

    def test_success_state_survives_failure_after_final_action(self):
        model = FakeModel()
        original_extract = model.extract_memory
        def extract(action, observation, facts):
            if model.memory_calls == 5:
                raise RuntimeError('outcome update failed')
            return original_extract(action, observation, facts)
        model.extract_memory = extract
        result = evaluation._episode(SCENARIOS[0], 'graph_reflection', SETTINGS,
                                     model, FakeGraph(), 3, 42)
        self.assertEqual(result['status'], 'error')
        self.assertTrue(result['success'])
        self.assertEqual(result['actions'][-1]['action'], 'put apple in fridge')

    def test_usage_is_unavailable_if_even_one_call_lacks_metadata(self):
        tracker = evaluation.UsageTracker()
        one, two = uuid4(), uuid4()
        tracker.on_chat_model_start({}, [], run_id=one)
        tracker.on_chat_model_start({}, [], run_id=two)
        msg = SimpleNamespace(usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5})
        response = SimpleNamespace(generations=[[SimpleNamespace(message=msg)]])
        tracker.on_llm_end(response, run_id=one)
        self.assertIsNone(tracker.measured())
        tracker.on_llm_end(response, run_id=two)
        self.assertEqual(tracker.measured(), {'input': 6, 'output': 4, 'total': 10})

    def test_matched_repeats_save_results_and_isolated_sessions(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(evaluation, 'NeSyFSLanguageModel', FakeModel), \
             patch.object(evaluation, 'Neo4jKnowledgeGraph', FakeGraph):
            path = Path(directory) / 'run.json'
            starts = []
            run = evaluation.run_comparison(SETTINGS, [SCENARIOS[0].id],
                ['reactive', 'graph_reflection'], 3, path, repeats=2,
                on_start=lambda *args: starts.append(args))
            self.assertEqual(run['status'], 'completed')
            self.assertEqual(len(run['episodes']), 4)
            self.assertEqual(len({e['episode_id'] for e in run['episodes']}), 4)
            self.assertEqual(len({e['initial_observation'] for e in run['episodes']}), 1)
            self.assertEqual(len(starts), 4)
            self.assertEqual(run['provider'], 'openai')
            self.assertEqual(run['episodes'][0]['scenario']['difficulty'], SCENARIOS[0].difficulty)
            self.assertEqual(json.loads(path.read_text())['status'], 'completed')
            self.assertNotIn('secret', path.read_text())

    def test_provider_is_recorded_and_passed_to_model(self):
        for provider in ('mlx', 'mlx_4b'):
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as directory, \
                 patch.object(evaluation, 'NeSyFSLanguageModel', return_value=FakeModel()) as model:
                settings = SimpleNamespace(**vars(SETTINGS), provider=provider)
                run = evaluation.run_comparison(settings, [SCENARIOS[0].id], ['reactive'], 3,
                                                Path(directory) / 'run.json')
                model.assert_called_once_with(settings.model, '', provider=provider)
                self.assertEqual(run['provider'], provider)
                self.assertIn('local_model_provenance', run)

    def test_local_provenance_records_small_configs_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = b'{"model_type":"qwen3","quantization":{"bits":4,"group_size":64}}'
            (path / 'config.json').write_bytes(config)
            (path / 'tokenizer_config.json').write_text('{}')
            (path / '.stillroom-install.json').write_text(json.dumps({
                'repo': 'mlx-community/Qwen3-1.7B-4bit', 'revision': 'a' * 40,
                'secret': 'must-not-be-in-results'}))
            result = evaluation._local_model_provenance(directory)
            self.assertEqual(result['declared_model_id'], 'mlx-community/Qwen3-1.7B-4bit')
            self.assertEqual(result['quantization']['bits'], 4)
            self.assertEqual(result['config_sha256']['config.json'], sha256(config).hexdigest())
            self.assertIn('tokenizer_config.json', result['config_sha256'])
            self.assertIsNone(result['weights_sha256'])
            self.assertIsNone(result['runtime_version'])
            self.assertNotIn('must-not-be-in-results', json.dumps(result))

    def test_local_provenance_tolerates_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluation._local_model_provenance(directory)
        self.assertEqual(result['config_sha256'], {})
        self.assertIsNone(result['declared_model_id'])

if __name__ == '__main__': unittest.main()
