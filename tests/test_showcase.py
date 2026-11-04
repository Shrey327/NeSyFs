import threading
import unittest
from unittest.mock import patch

import showcase


class Graph:
    def __init__(self, *args): self.closed = False
    def verify_connection(self): pass
    def setup(self): pass
    def close(self): self.closed = True


class Agent:
    fail = False
    def __init__(self, *args): pass
    def step(self, session_id, task, observation, environment, previous_action=None, trace=None):
        trace('action', 'Taking apple')
        environment.execute('take apple')
        if self.fail:
            raise RuntimeError('failure after action')
        return type('Result', (), {'action': 'take apple', 'observation': 'Holding apple'})()


class ShowcaseTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch('showcase.Settings.from_env', return_value=type('Settings', (), {
            'neo4j_uri': '', 'neo4j_username': '', 'neo4j_password': '', 'retrieval_limit': 1,
            'model': '', 'openai_api_key': '', 'particle_count': 1, 'planning_horizon': 1,
            'reflection_limit': 1, 'reflection_samples': 1})()),
            patch('showcase.Neo4jKnowledgeGraph', Graph), patch('showcase.NeSyFSLanguageModel'),
            patch('showcase.TSMCPlanner'), patch('showcase.NeSyFSAgent', Agent)]
        for p in self.patches: p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
        self.app = showcase.Showcase()
        Agent.fail = False

    def test_step_events_and_session_isolation(self):
        first = self.app.start(showcase.TASK)
        result = self.app.step(first['session_id'])
        self.assertEqual(result['action'], 'take apple')
        self.assertEqual(result['events'][0]['stage'], 'action')
        self.assertIn('t', result['events'][0])
        self.assertEqual(result['state'], {'apple': 'held', 'fridge_open': False})
        self.assertIn('holding', self.app.snapshot(first['session_id'])['observation'])
        graph = self.app.graph
        second = self.app.start(showcase.TASK)
        self.assertTrue(graph.closed)
        self.assertNotEqual(first['session_id'], second['session_id'])
        with self.assertRaises(showcase.UserError): self.app.snapshot(first['session_id'])
        self.app.close()
        self.assertTrue(self.app.graph.closed)

    def test_action_is_preserved_after_later_failure(self):
        session = self.app.start(showcase.TASK)
        Agent.fail = True
        with self.assertRaises(RuntimeError): self.app.step(session['session_id'])
        self.assertIn('holding', self.app.snapshot(session['session_id'])['observation'])
        self.assertEqual(self.app.session['previous_action'], 'take apple')
        self.assertFalse(self.app.running)

    def test_busy_and_task_validation(self):
        with self.assertRaises(showcase.UserError): self.app.start('Unsupported task')
        with self.assertRaises(showcase.UserError): self.app.start(showcase.TASK, 'remote-anything')
        self.app.lock.acquire()
        try:
            with self.assertRaises(showcase.UserError): self.app.step('anything')
        finally:
            self.app.lock.release()


if __name__ == '__main__': unittest.main()
