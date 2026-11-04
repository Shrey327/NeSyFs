import json
import threading
from dataclasses import replace
from hashlib import sha256
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from nesyfs import evaluation
from nesyfs.eval_env import SCENARIOS
from nesyfs.models import ActionProposal, EntityNames, Lesson, MemoryDelta, Prediction, Reflection

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
    def propose_action(self, task, observation, facts, feedback='', sampled=False):
        self.contexts.append(observation)
        if 'holding' in observation.lower(): action = 'put apple in fridge'
        elif 'opened the fridge' in observation.lower(): action = 'take apple'
        elif 'hands are empty' in observation.lower(): action = 'open fridge'
        else: action = 'take apple'
        return ActionProposal(action=action)
    def reflect(self, task, observation, action, facts): return Reflection(aligned=self.aligned, feedback='no')
    def predict(self, task, observation, action, facts): return Prediction(observation='predicted', progress=True)
    def lesson(self, task, transcript): return Lesson(lesson='open the fridge first')


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
        self.assertTrue(reactive.contexts[1].startswith("Result of your last action 'open fridge':"))
        self.assertIn('Initial observation:', history.contexts[1])

    def test_react_sees_its_previous_thoughts(self):
        class Thinker(FakeModel):
            def propose_action(self, task, observation, facts, feedback='', sampled=False):
                proposal = super().propose_action(task, observation, facts, feedback)
                return ActionProposal(action=proposal.action, rationale='fridge is closed')
        model = Thinker()
        result = evaluation._episode(SCENARIOS[0], 'react', SETTINGS, model, None, 3, 42)
        self.assertTrue(result['success'])
        self.assertIn('Reason before acting', model.contexts[0])
        self.assertIn('Thought: fridge is closed', model.contexts[1])

    def test_reflexion_retries_fresh_with_lessons(self):
        class Learner(FakeModel):
            def propose_action(self, task, observation, facts, feedback='', sampled=False):
                self.contexts.append(observation)
                if 'Lessons' not in observation: return ActionProposal(action='dance')
                return super().propose_action(task, observation.split('\n\n', 1)[1], facts, feedback)
        model = Learner()
        result = evaluation._episode(SCENARIOS[0], 'reflexion', SETTINGS, model, None, 9, 42)
        self.assertTrue(result['success'])
        self.assertEqual(result['trials'], 2)
        self.assertEqual(result['lessons'], ['open the fridge first'])
        self.assertEqual(result['invalid_actions'], 3)
        self.assertEqual([a['trial'] for a in result['actions']], [1, 1, 1, 2, 2, 2])
        self.assertLessEqual(result['decision_count'], 9)
        self.assertIn('open the fridge first', model.contexts[3])

    def test_reflexion_allocates_trials_to_fit_public_reference(self):
        class Invalid(FakeModel):
            def propose_action(self, *args, **kwargs):
                return ActionProposal(action='dance')
        for budget, reference, expected in [(20, 11, [20]), (20, 6, [7, 7, 6]), (2, 3, [2])]:
            with self.subTest(budget=budget, reference=reference):
                scenario = replace(SCENARIOS[0], min_actions=reference)
                result = evaluation._episode(scenario, 'reflexion', SETTINGS, Invalid(), None, budget, 42)
                self.assertEqual(result['trial_limit'], len(expected))
                self.assertEqual(result['trial_budgets'], expected)
                self.assertEqual(result['decision_count'], budget)
                self.assertEqual(len(result['actions']), budget)

    def test_cancellation_between_proposal_and_action_does_not_mutate_environment(self):
        cancelled = threading.Event()
        class Cancels(FakeModel):
            def propose_action(self, *args, **kwargs):
                cancelled.set()
                return ActionProposal(action='open fridge')
        result = evaluation._episode(SCENARIOS[0], 'reactive', SETTINGS, Cancels(), None, 3, 42,
                                     cancel_event=cancelled)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['actions'], [])
        self.assertFalse(result['success'])

    def test_rejected_ablation_consumes_budget_without_action_or_planning(self):
        result = evaluation._episode(SCENARIOS[0], 'graph_reflection', SETTINGS,
                                     FakeModel(aligned=False), FakeGraph(), 2, 42)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['rejected_decisions'], 2)
        self.assertEqual(result['planner_uses'], 0)
        self.assertEqual(result['actions'], [])

    def test_rejection_feedback_reaches_the_next_decision(self):
        model = FakeModel(aligned=False)
        evaluation._episode(SCENARIOS[0], 'graph_reflection', SETTINGS, model, FakeGraph(), 2, 42)
        self.assertNotIn('was rejected', model.contexts[0])
        self.assertIn('was rejected and not executed: no', model.contexts[1])

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
                model.assert_called_once_with(settings.model, '', provider=provider, sample_temperature=0.0,
                                              policy_temperature=0.0, cache_enabled=False)
                self.assertEqual(run['provider'], provider)
                self.assertIn('local_model_provenance', run)

    def test_six_methods_have_fresh_episodes_and_shared_budget(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(evaluation, 'NeSyFSLanguageModel', FakeModel), \
             patch.object(evaluation, 'Neo4jKnowledgeGraph', FakeGraph):
            path = Path(directory) / 'run.json'
            run = evaluation.run_comparison(SETTINGS, [SCENARIOS[0].id], list(evaluation.METHODS), 6, path)
            self.assertEqual(run['status'], 'completed')
            self.assertEqual(len(run['episodes']), 6)
            self.assertEqual(len({e['episode_id'] for e in run['episodes']}), 6)
            self.assertTrue(all(e['decision_count'] <= 6 for e in run['episodes']))
            self.assertFalse(run['cache_enabled'])
            self.assertEqual(json.loads(path.read_text())['status'], 'completed')

    def test_recoverable_episode_error_does_not_skip_remaining_methods(self):
        def episode(scenario, method, *args, **kwargs):
            return {'status': 'error' if method == 'react' else 'completed',
                    'error': 'ValidationError' if method == 'react' else None}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(evaluation, 'NeSyFSLanguageModel', FakeModel), \
             patch.object(evaluation, 'Neo4jKnowledgeGraph', FakeGraph), \
             patch.object(evaluation, '_episode', side_effect=episode):
            path = Path(directory) / 'run.json'
            run = evaluation.run_comparison(SETTINGS, [SCENARIOS[0].id], list(evaluation.METHODS), 1, path)
            self.assertEqual(run['status'], 'completed_with_errors')
            self.assertEqual(len(run['episodes']), 6)

    def test_fatal_service_error_stops_before_more_calls(self):
        def episode(*args, **kwargs):
            return {'status': 'error', 'error': 'AuthenticationError'}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(evaluation, 'NeSyFSLanguageModel', FakeModel), \
             patch.object(evaluation, 'Neo4jKnowledgeGraph', FakeGraph), \
             patch.object(evaluation, '_episode', side_effect=episode):
            path = Path(directory) / 'run.json'
            run = evaluation.run_comparison(SETTINGS, [SCENARIOS[0].id], list(evaluation.METHODS), 1, path)
            self.assertEqual(run['status'], 'error')
            self.assertEqual(len(run['episodes']), 1)

    def test_cancellation_keeps_actual_decision_and_partial_progress(self):
        cancelled = threading.Event()
        snapshots = []
        def progress(snapshot):
            snapshots.append(snapshot)
            cancelled.set()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(evaluation, 'NeSyFSLanguageModel', FakeModel):
            path = Path(directory) / 'run.json'
            run = evaluation.run_comparison(SETTINGS, [SCENARIOS[0].id], ['reactive', 'history'], 6,
                                            path, cancel_event=cancelled, on_progress=progress)
            self.assertEqual(run['status'], 'cancelled')
            self.assertEqual(len(run['episodes']), 1)
            self.assertEqual(run['episodes'][0]['status'], 'cancelled')
            self.assertEqual(run['episodes'][0]['decision_count'], 1)
            self.assertTrue(snapshots)
            self.assertEqual(json.loads(path.read_text())['status'], 'cancelled')

    def test_live_model_events_are_validated_and_raw_is_never_saved(self):
        class Observable(FakeModel):
            def __init__(self):
                super().__init__()
                self.on_call = None
            def propose_action(self, *args, **kwargs):
                proposal = super().propose_action(*args, **kwargs)
                if self.on_call:
                    self.on_call({'kind': 'propose', 'prompt': 'task', 'output': proposal.model_dump(),
                                  'raw': 'sensitive hidden content', 'ms': 2})
                return proposal
        snapshots = []
        result = evaluation._episode(SCENARIOS[0], 'reactive', SETTINGS, Observable(), None, 3, 42,
                                     on_progress=lambda snapshot, persist: snapshots.append((snapshot, persist)))
        calls = [e for e in result['trace'] if e['stage'] == 'model']
        self.assertEqual(len(calls), 3)
        self.assertNotIn('raw', calls[0]['data'])
        self.assertIn('output', calls[0]['data'])
        self.assertEqual([e['index'] for e in result['trace']], list(range(1, len(result['trace']) + 1)))
        self.assertTrue(any(persist for _, persist in snapshots))

    def test_cancel_after_final_action_preserves_goal_and_reports_cancelled(self):
        cancelled = threading.Event()
        class Observable(FakeModel):
            def __init__(self):
                super().__init__()
                self.on_call = None
            def extract_memory(self, action, observation, facts):
                result = super().extract_memory(action, observation, facts)
                if self.memory_calls == 6 and self.on_call:
                    self.on_call({'kind': 'memory', 'output': {'phase': 'final-outcome'}})
                return result
        def progress(snapshot, persist):
            data = snapshot['trace'][-1].get('data') or {}
            if isinstance(data.get('output'), dict) and data['output'].get('phase') == 'final-outcome':
                cancelled.set()
        result = evaluation._episode(SCENARIOS[0], 'graph_reflection', SETTINGS,
                                     Observable(), FakeGraph(), 3, 42,
                                     cancel_event=cancelled, on_progress=progress)
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue(result['success'])
        self.assertIsNone(result['error'])
        self.assertEqual(result['actions'][-1]['action'], 'put apple in fridge')
        self.assertEqual(result['decision_count'], 3)

    def test_live_model_output_is_checkpointed_before_episode_finishes(self):
        class Observable(FakeModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.on_call = None
            def propose_action(self, *args, **kwargs):
                proposal = super().propose_action(*args, **kwargs)
                if self.on_call:
                    self.on_call({'kind': 'propose', 'prompt': 'synthetic task',
                                  'output': proposal.model_dump(), 'raw': 'do-not-save'})
                return proposal
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(evaluation, 'NeSyFSLanguageModel', Observable):
            path = Path(directory) / 'run.json'
            saved = []
            def progress(snapshot):
                if snapshot['trace'][-1]['stage'] == 'model':
                    saved.append(json.loads(path.read_text()))
            run = evaluation.run_comparison(SETTINGS, [SCENARIOS[0].id], ['reactive'], 3,
                                            path, on_progress=progress)
            self.assertTrue(saved)
            self.assertEqual(saved[0]['status'], 'running')
            self.assertEqual(saved[0]['active_episode']['trace'][-1]['stage'], 'model')
            self.assertIn('output', saved[0]['active_episode']['trace'][-1]['data'])
            self.assertNotIn('do-not-save', json.dumps(saved[0]))
            self.assertEqual(run['status'], 'completed')
            self.assertNotIn('active_episode', json.loads(path.read_text()))

    def test_cancel_before_start_does_not_create_fake_episodes(self):
        cancelled = threading.Event()
        cancelled.set()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(evaluation, 'NeSyFSLanguageModel', FakeModel):
            path = Path(directory) / 'run.json'
            run = evaluation.run_comparison(SETTINGS, [SCENARIOS[0].id], ['reactive'], 3,
                                            path, cancel_event=cancelled)
            self.assertEqual(run['status'], 'cancelled')
            self.assertEqual(run['episodes'], [])
            self.assertEqual(json.loads(path.read_text())['status'], 'cancelled')

    def test_keyboard_interrupt_marks_run_cancelled(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(evaluation, 'NeSyFSLanguageModel', FakeModel), \
             patch.object(evaluation, '_episode', side_effect=KeyboardInterrupt):
            path = Path(directory) / 'run.json'
            run = evaluation.run_comparison(SETTINGS, [SCENARIOS[0].id], ['reactive'], 1, path)
            self.assertEqual(run['status'], 'cancelled')
            self.assertEqual(run['episodes'], [])
            self.assertEqual(json.loads(path.read_text())['status'], 'cancelled')

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
