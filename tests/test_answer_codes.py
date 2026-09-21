import contextlib
import io
import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from open_cricket import LocalClient
from open_cricket.cli import main
from open_cricket.integrations import as_runnable
from open_cricket.questionnaire import classify
from open_cricket.server import create_app


class CodeBackend:
    def __init__(self):
        self.calls = []

    def prompt_ids(self, system, form):
        self.system, self.form = system, form
        return [99, 98]

    def answer_ids(self, answer):
        return list(answer.encode()) + [0]

    def scorer(self, prompt):
        raise AssertionError('Single-token scoring must not allocate a cache')

    def next_logprobs(self, prompt, prefix, allowed):
        self.calls.append((prompt, prefix, allowed))
        return {token: math.log({65: .2, 66: .6}.get(token, .001)) for token in allowed}


class AnswerCodeTests(unittest.TestCase):
    def test_scores_one_distribution_preserves_labels_and_excludes_eos(self):
        backend = CodeBackend()
        options = [{'label': 'billing "refund"', 'description': 'Payments'}, '技術支援']
        result = classify(backend, 'untrusted\n"message"', 'Which team?', options,
                          mode='answer_codes')
        self.assertEqual(result['answer'], '技術支援')
        self.assertEqual(result['mode'], 'answer_codes')
        self.assertEqual(result['model_calls'], 1)
        self.assertEqual(result['prompt_tokens'], 2)
        self.assertEqual(backend.calls, [([99, 98], (), (65, 66))])
        self.assertAlmostEqual(result['candidate_log_mass'], math.log(.8))
        self.assertAlmostEqual(result['options'][0]['probability'], .75)
        self.assertEqual(result['options'][0]['code'], 'B')
        self.assertEqual(result['options'][0]['token_ids'], [66])
        self.assertEqual(len(result['options'][0]['trace']), 1)
        self.assertIn('Payments', result['form'])
        self.assertIn('billing \\"refund\\"', result['form'])
        self.assertNotIn('as a JSON string', backend.system + backend.form)
        self.assertIn('untrusted', backend.system)
        warmed = classify(backend, 'x', 'q', ['a', 'b'], mode='answer_codes', temperature=2)
        self.assertAlmostEqual(warmed['options'][0]['probability'], math.sqrt(.6) / (math.sqrt(.2) + math.sqrt(.6)))

    def test_tokenization_limits_and_bad_probabilities_fail_before_scoring(self):
        backend = CodeBackend()
        for paths in ([1, 2, 0], [0, 0], [1], []):
            with patch.object(backend, 'answer_ids', return_value=paths):
                with self.assertRaisesRegex(ValueError, 'one token'):
                    classify(backend, 'x', 'q', ['a'], mode='answer_codes')
        with patch.object(backend, 'answer_ids', return_value=[1, 0]):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                classify(backend, 'x', 'q', ['a', 'b'], mode='answer_codes')
        with self.assertRaisesRegex(ValueError, '26 options'):
            classify(backend, 'x', 'q', [str(i) for i in range(27)], mode='answer_codes')
        for temperature in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                classify(backend, 'x', 'q', ['a'], mode='answer_codes', temperature=temperature)
        self.assertFalse(backend.calls)
        for scores in ({}, {65: float('nan')}, {65: .2}):
            with patch.object(backend, 'next_logprobs', return_value=scores):
                with self.assertRaisesRegex(ValueError, 'valid log probabilities'):
                    classify(backend, 'x', 'q', ['a'], mode='answer_codes')

    def test_all_26_codes_and_single_option(self):
        backend = CodeBackend()
        result = classify(backend, 'x', 'q', [str(i) for i in range(26)], mode='answer_codes')
        self.assertEqual(len(result['options']), 26)
        self.assertEqual(result['model_calls'], 1)
        result = classify(backend, 'x', 'q', ['only'], mode='answer_codes')
        self.assertEqual(result['options'][0]['probability'], 1)

    def test_typed_answers_restore_choice_score_and_noul_meanings(self):
        result = LocalClient(backend=CodeBackend(), mode='answer_codes').system_one(
            'message', {
                'route': {'type': 'choice', 'criteria': {'billing': None, 'other': None}},
                'urgency': {'type': 'score', 'criteria': ['Low', 'High']},
                'refund': {'type': 'noul', 'instructions': 'Refund requested?'},
            })
        self.assertEqual(result['answers']['route']['choice'], 'other')
        self.assertAlmostEqual(result['answers']['route']['probabilities']['billing'], .25)
        self.assertAlmostEqual(result['answers']['urgency']['score'], .75)
        self.assertEqual(result['answers']['urgency']['legend'], {'0': 'Low', '1': 'High'})
        self.assertAlmostEqual(result['answers']['refund']['noul'], .25)

    def test_sdk_cli_runnable_and_server_keep_the_response_contract(self):
        path = Path(__file__).resolve().parents[1] / 'examples/support.json'
        payload = json.loads(path.read_text())
        backend = CodeBackend()
        client = LocalClient(model=payload['model'], backend=backend, mode='answer_codes')
        expected = client.invoke(payload)
        self.assertEqual(set(expected), {'model', 'answers', 'usage'})
        self.assertEqual(expected['usage'], {'input_tokens': 6, 'output_tokens': 0})
        runnable = as_runnable(backend, model=client.model, mode='answer_codes')
        self.assertEqual(runnable.invoke(payload), expected)
        output = io.StringIO()
        with patch.object(sys, 'argv', ['open-cricket', '--input', str(path), '--mode', 'answer_codes']), \
             patch('open_cricket.backend.load_backend', return_value=backend), \
             contextlib.redirect_stdout(output):
            main()
        self.assertEqual(json.loads(output.getvalue()), expected)
        with patch.dict('os.environ', {'OPEN_CRICKET_MODE': 'answer_codes', 'OPEN_CRICKET_API_KEY': ''}), \
             patch('open_cricket.backend.load_backend', return_value=backend):
            with TestClient(create_app(model_name=client.model)) as http:
                response = http.post('/v1/systemone', json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json(), expected)

    def test_default_remains_sequence_and_bad_modes_do_not_load_models(self):
        self.assertEqual(LocalClient(backend=CodeBackend()).mode, 'sequence')
        with patch('open_cricket.backend.load_backend') as load:
            with self.assertRaisesRegex(ValueError, 'mode must'):
                LocalClient(mode='typo')
            load.assert_not_called()
        with patch.object(sys, 'argv', ['open-cricket', '--demo', '--mode', 'answer_codes']), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
