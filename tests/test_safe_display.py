import copy
import json
import unittest
from nesyfs.safe_display import safe_display, WITHHELD


class SafeDisplayTests(unittest.TestCase):
    def test_nested_private_fields_are_dropped_without_mutating_source(self):
        source = {'status': 'completed', 'count': 3, 'ok': True,
                  'trace': [{'stage': 'model', 'data': {'raw': 'secret', 'RAW_text': 'secret2',
                            'Reasoning_Content': 'secret3', 'analysis': {'inner': 'secret4'},
                            'chain-of-thought': 'secret5', 'output': {'rationale': 'Open the fridge.',
                            'action': 'open fridge', 'observation': 'The fridge is open.'}}}]}
        original = copy.deepcopy(source)
        shown = safe_display(source)
        self.assertEqual(source, original)
        self.assertEqual((shown['status'], shown['count'], shown['ok']), ('completed', 3, True))
        self.assertEqual(shown['trace'][0]['data']['output'], source['trace'][0]['data']['output'])
        self.assertEqual(set(shown['trace'][0]['data']), {'output'})

    def test_thinking_tags_withheld_entirely_even_malformed(self):
        samples = ['before <think>secret</think> after', '<THINK>unfinished',
                   'start < t h i n k >secret', 'start </ analysis >',
                   '<thinking>private', '<|analysis|>private', '<think_broken>private',
                   '<t\u200bhink>private', '<reasoning class="hidden">private', r'\u003cthink\u003eprivate',
                   r'\\u003c\\u0074\\u0068\\u0069\\u006e\\u006b\\u003eprivate',
                   '&amp;lt;think&amp;gt;private', r'\x3cThink\x3eprivate']
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(safe_display(sample), WITHHELD)
        self.assertEqual(safe_display('Model explanation: open the fridge.'), 'Model explanation: open the fridge.')

    def test_json_encoded_payloads_also_filter_private_keys(self):
        nested = json.dumps({'action': 'take apple', 'reasoning_content': 'secret',
                             'facts': [{'raw_response': 'secret2', 'name': 'apple'}]})
        shown = safe_display({'output': nested})['output']
        self.assertEqual(json.loads(shown), {'action': 'take apple', 'facts': [{'name': 'apple'}]})
        double_encoded = json.dumps(nested)
        self.assertEqual(json.loads(json.loads(safe_display(double_encoded))),
                         {'action': 'take apple', 'facts': [{'name': 'apple'}]})
        self.assertEqual(safe_display(json.dumps({'output': '<think>private'})), WITHHELD)
        self.assertEqual(safe_display('{"reasoning_content": "secret"'), WITHHELD)


if __name__ == '__main__':
    unittest.main()
