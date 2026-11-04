import io
import json
import sys
import tempfile
import types
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import showcase


class InlineThread:
    def __init__(self, target, **kwargs):
        self.target = target
    def start(self):
        self.target()


class EvaluationHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.app = showcase.Showcase()
        self.app.evaluation = showcase.EvaluationManager(Path(self.tmp.name))
        self.config_patch = patch.object(showcase.EvaluationManager, 'config', return_value={
            'tasks': [{'id': 'kitchen-1', 'task': 'Place the object.'}],
            'limits': {'max_tasks': 4, 'max_repeats': 3, 'max_steps': 8}})
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.handler = showcase.make_handler(self.app).__new__(showcase.make_handler(self.app))
        self.handler.headers = Message()
        self.handler.headers['Host'] = '127.0.0.1:18765'
        self.responses = []
        self.handler.send_json = lambda status, data: self.responses.append((status, data))
        self.handler.send_error = lambda status: self.responses.append((status, None))

    def post(self, path, payload):
        self.handler.path = path
        self.handler.headers['Content-Type'] = 'application/json'
        body = json.dumps(payload).encode()
        self.handler.headers['Content-Length'] = str(len(body))
        self.handler.rfile = io.BytesIO(body)
        self.handler.do_POST()
        del self.handler.headers['Content-Type']
        del self.handler.headers['Content-Length']
        return self.responses[-1]

    def get(self, path):
        self.handler.path = path
        self.handler.do_GET()
        return self.responses[-1]

    def test_rejects_invalid_bounds_and_cross_origin(self):
        payload = {'task_ids': ['kitchen-1'], 'repeats': 1, 'max_steps': True}
        self.assertEqual(self.post('/api/evaluation/run', payload)[0], 400)
        payload['max_steps'] = 9
        self.assertEqual(self.post('/api/evaluation/run', payload)[0], 400)
        payload['max_steps'] = 4
        payload['task_ids'] = ['kitchen-1', 'kitchen-1']
        self.assertEqual(self.post('/api/evaluation/run', payload)[0], 400)
        payload['task_ids'] = ['kitchen-1']
        self.assertEqual(self.post('/api/evaluation/run', payload)[0], 400)
        payload['provider'] = 'http://example.com'
        self.assertEqual(self.post('/api/evaluation/run', payload)[0], 400)
        self.handler.headers['Origin'] = 'https://example.com'
        self.assertEqual(self.post('/api/evaluation/run', payload)[0], 403)

    def test_async_run_status_and_persisted_results(self):
        observed = []
        def run_comparison(**kwargs):
            observed.append(kwargs)
            kwargs['on_start']('kitchen-1', 'reactive', 0)
            kwargs['on_progress']({'episode_id': 'episode-1', 'decision_count': 1, 'current_trial': 1,
                                   'trace': [{'stage': 'model', 'index': 1, 'message': 'completed',
                                              'data': {'kind': 'propose', 'output': {'action': 'open fridge'}}}]})
            live = self.app.evaluation.status()
            self.assertEqual(live['current_episode_id'], 'episode-1')
            self.assertEqual((live['task_ids'], live['action_budget'], live['repeats']), (['kitchen-1'], 4, 1))
            self.assertEqual(live['current_events'][0]['stage'], 'model')
            run = {'status': 'complete', 'episodes': [{'task_id': 'kitchen-1'}]}
            kwargs['output_path'].write_text(json.dumps(run))
            kwargs['on_episode'](run)
            return run
        module = types.ModuleType('nesyfs.evaluation')
        module.run_comparison = run_comparison
        with patch.dict(sys.modules, {'nesyfs.evaluation': module}), \
             patch.object(showcase.Settings, 'from_env', return_value=object()) as settings_from_env, \
             patch.object(showcase.threading, 'Thread', InlineThread):
            status, body = self.post('/api/evaluation/run',
                {'task_ids': ['kitchen-1'], 'repeats': 1, 'max_steps': 4, 'provider': 'mlx'})
        self.assertEqual(status, 200)
        settings_from_env.assert_called_once_with(provider='mlx')
        self.assertFalse(body['running'])
        self.assertEqual(body['provider'], 'mlx')
        self.assertEqual(body['completed'], 1)
        self.assertEqual(body['total'], len(showcase.METHODS))
        self.assertEqual(observed[0]['methods'], ['reactive', 'history', 'react', 'reflexion', 'graph_reflection', 'adaptive'])
        self.assertEqual(self.get('/api/evaluation/results')[1]['runs'][0]['status'], 'complete')
        self.assertEqual(self.get('/api/evaluation/status')[1]['completed'], 1)

    def test_running_evaluation_rejects_duplicate(self):
        self.app.evaluation.state['running'] = True
        with patch.object(showcase.Settings, 'from_env', return_value=object()):
            status, _ = self.post('/api/evaluation/run',
                {'task_ids': ['kitchen-1'], 'repeats': 1, 'max_steps': 4, 'provider': 'mlx'})
        self.assertEqual(status, 400)

    def test_unavailable_local_provider_does_not_start_or_fall_back(self):
        config = {'tasks': [{'id': 'kitchen-1', 'task': 'Place the object.'}],
                  'limits': {'max_tasks': 1, 'max_repeats': 3, 'max_steps': 20},
                  'providers': [{'id': 'openai', 'ready': True}, {'id': 'mlx', 'ready': False}]}
        with patch.object(showcase.EvaluationManager, 'config', return_value=config), \
             patch.object(showcase.Settings, 'from_env') as settings:
            status, _ = self.post('/api/evaluation/run',
                {'task_ids': ['kitchen-1'], 'repeats': 1, 'max_steps': 12, 'provider': 'mlx'})
        self.assertEqual(status, 400)
        settings.assert_not_called()

    def test_local_4b_provider_is_accepted(self):
        module = types.ModuleType('nesyfs.evaluation')
        module.run_comparison = lambda **kwargs: {'status': 'completed'}
        with patch.dict(sys.modules, {'nesyfs.evaluation': module}), \
             patch.object(showcase.Settings, 'from_env', return_value=object()) as settings, \
             patch.object(showcase.threading, 'Thread', InlineThread):
            status, _ = self.post('/api/evaluation/run',
                {'task_ids': ['kitchen-1'], 'repeats': 1, 'max_steps': 4, 'provider': 'mlx_4b'})
        self.assertEqual(status, 200)
        settings.assert_called_once_with(provider='mlx_4b')

    def test_runner_error_is_visible_in_status(self):
        module = types.ModuleType('nesyfs.evaluation')
        module.run_comparison = lambda **kwargs: {'status': 'error'}
        with patch.dict(sys.modules, {'nesyfs.evaluation': module}), \
             patch.object(showcase.Settings, 'from_env', return_value=object()), \
             patch.object(showcase.threading, 'Thread', InlineThread):
            status, body = self.post('/api/evaluation/run',
                {'task_ids': ['kitchen-1'], 'repeats': 1, 'max_steps': 4, 'provider': 'openai'})
        self.assertEqual(status, 200)
        self.assertFalse(body['running'])
        self.assertIsNotNone(body['error'])

    def test_history_orders_by_run_time_not_copy_time(self):
        directory = Path(self.tmp.name)
        (directory / 'new.json').write_text(json.dumps({'created_at': '2026-11-03T23:00:00+00:00'}))
        (directory / 'old.json').write_text(json.dumps({'created_at': '2026-11-03T22:00:00+00:00'}))
        runs = self.get('/api/evaluation/results')[1]['runs']
        self.assertEqual(runs[0]['created_at'], '2026-11-03T23:00:00+00:00')

    def test_unmonitored_running_record_is_not_rewritten(self):
        path = Path(self.tmp.name) / 'external.json'
        original = json.dumps({'status': 'running', 'created_at': '2026-11-03T23:00:00+00:00'})
        path.write_text(original)
        run = self.get('/api/evaluation/results')[1]['runs'][0]
        self.assertEqual(run['tracking'], 'unmonitored')
        self.assertEqual(path.read_text(), original)
        self.app.evaluation.active_path = path
        self.app.evaluation.state['running'] = True
        self.assertEqual(self.get('/api/evaluation/results')[1]['runs'][0]['tracking'], 'active')

    def test_json_boundary_sanitizes_without_mutating_source(self):
        handler = showcase.make_handler(self.app).__new__(showcase.make_handler(self.app))
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler.wfile = io.BytesIO()
        source = {'created_at': '2026-01-01', 'data': {'raw': 'private',
                  'output': '<think>secret</think>visible', 'action': 'open fridge'}}
        original = json.dumps(source)
        path = Path(self.tmp.name) / 'record.json'
        path.write_text(original)
        handler.send_json(200, self.app.evaluation.history())
        response = json.loads(handler.wfile.getvalue())
        self.assertNotIn('raw', response['runs'][0]['data'])
        self.assertEqual(response['runs'][0]['data']['output'], '[withheld]')
        self.assertEqual(response['runs'][0]['data']['action'], 'open fridge')
        self.assertEqual(path.read_text(), original)

    def test_cancel_sets_cooperative_request_without_faking_completion(self):
        self.app.evaluation.state['running'] = True
        status, body = self.post('/api/evaluation/cancel', {})
        self.assertEqual(status, 200)
        self.assertTrue(body['running'])
        self.assertTrue(body['cancel_requested'])
        self.assertTrue(self.app.evaluation.cancel_event.is_set())
        self.app.evaluation.state['running'] = False
        self.assertEqual(self.post('/api/evaluation/cancel', {})[0], 400)


if __name__ == '__main__':
    unittest.main()
