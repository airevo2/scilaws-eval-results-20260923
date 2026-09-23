"""Verify the request boundary that failed in the live n=5 preflight."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import custom_client


class TransportContract(unittest.TestCase):
    def exercise(self, kind, model, n, max_tokens, temperature):
        requests = []
        def create(**request):
            requests.append(request)
            return types.SimpleNamespace(
                id=f'unit-{len(requests)}', model=model, usage=None,
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='response'),
                                               finish_reason='stop')])
        fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        with patch.object(custom_client, '_get_client', return_value=fake):
            outputs = custom_client.complete(model, 'Frozen prompt\nwith exact spacing.',
                                             temperature, n, max_tokens, kind)
        self.assertEqual(len(outputs), n)
        self.assertEqual(len(requests), n)
        for request in requests:
            self.assertEqual(request['n'], 1)
            self.assertEqual(request['messages'], [{'role': 'user', 'content': 'Frozen prompt\nwith exact spacing.'}])
            self.assertNotIn('reasoning_effort', request)
        return requests

    def test_reasoning_samples(self):
        for model in ['gpt-5.6-sol', 'gpt-6-astra']:
            requests = self.exercise('reasoning', model, 5, 900, 0.8)
            for request in requests:
                self.assertEqual(request['max_completion_tokens'], 28800)
                self.assertNotIn('temperature', request)
                self.assertNotIn('max_tokens', request)

    def test_frozen_judge(self):
        requests = self.exercise('chat', 'gpt-4.1', 1, 400, 0.0)
        self.assertEqual(requests[0]['max_tokens'], 400)
        self.assertEqual(requests[0]['temperature'], 0.0)
        self.assertNotIn('max_completion_tokens', requests[0])


if __name__ == '__main__':
    unittest.main()
