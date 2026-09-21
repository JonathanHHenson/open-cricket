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
from open_cricket.questionnaire import _answer_codes, classify
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
        return {
            token: math.log({
                65: .1, 97: .1, 66: .3, 98: .3,
            }.get(token, .001))
            for token in allowed
        }


class BatchCodeBackend(CodeBackend):
    def __init__(self):
        super().__init__()
        self.batches = []

    def batch_next_logprobs(self, requests):
        self.batches.append(requests)
        return [
            {
                token: math.log({65: .1, 97: .1, 66: .3, 98: .3}.get(token, .001))
                for token in allowed
            }
            for _, allowed in requests
        ]

    def next_logprobs(self, prompt, prefix, allowed):
        raise AssertionError('Multiple questions should use the batch scorer')


class AnswerCodeTests(unittest.TestCase):
    def test_client_batches_multiple_questions_and_preserves_question_order(self):
        backend = BatchCodeBackend()
        result = LocalClient(backend=backend).system_one('message', {
            'first': {'type': 'choice', 'criteria': {'a': None, 'b': None}},
            'second': {'type': 'choice', 'criteria': {'x': None, 'y': None}},
            'third': {'type': 'noul'},
        })
        self.assertEqual(list(result['answers']), ['first', 'second', 'third'])
        self.assertEqual([answer.get('choice') for answer in result['answers'].values()][:2],
                         ['b', 'y'])
        self.assertEqual(len(backend.batches), 1)
        self.assertEqual(len(backend.batches[0]), 3)
        self.assertEqual(
            [allowed for _, allowed in backend.batches[0]],
            [(65, 97, 66, 98), (65, 97, 66, 98), (65, 97, 66, 98)],
        )
        self.assertEqual(result['usage'], {'input_tokens': 6, 'output_tokens': 0})

    def test_bad_batch_cardinality_is_rejected(self):
        backend = BatchCodeBackend()
        backend.batch_next_logprobs = lambda requests: requests[:1]
        with self.assertRaisesRegex(ValueError, 'every question'):
            LocalClient(backend=backend).system_one('message', {
                'one': {'type': 'choice', 'criteria': {'a': None, 'b': None}},
                'two': {'type': 'choice', 'criteria': {'a': None, 'b': None}},
            })

    def test_scores_one_distribution_preserves_labels_and_excludes_eos(self):
        backend = CodeBackend()
        options = [{'label': 'billing "refund"', 'description': 'Payments'}, '技術支援']
        result = classify(backend, 'untrusted\n"message"', 'Which team?', options,
                          mode='answer_codes')
        self.assertEqual(result['answer'], '技術支援')
        self.assertEqual(result['mode'], 'answer_codes')
        self.assertEqual(result['model_calls'], 1)
        self.assertEqual(result['prompt_tokens'], 2)
        self.assertEqual(backend.calls, [([99, 98], (), (65, 97, 66, 98))])
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
        for paths in ([0, 0], [1], []):
            with patch.object(backend, 'answer_ids', return_value=paths):
                with self.assertRaisesRegex(ValueError, 'terminator'):
                    classify(backend, 'x', 'q', ['a'], mode='answer_codes')
        with patch.object(backend, 'answer_ids', return_value=[1, 0]):
            with self.assertRaisesRegex(ValueError, 'duplicate answer codes'):
                classify(backend, 'x', 'q', ['a', 'b'], mode='answer_codes')
        for temperature in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                classify(backend, 'x', 'q', ['a'], mode='answer_codes', temperature=temperature)
        self.assertFalse(backend.calls)
        for scores in ({}, {65: float('nan')}, {65: .2}):
            with patch.object(backend, 'next_logprobs', return_value=scores):
                with self.assertRaisesRegex(ValueError, 'valid log probabilities'):
                    classify(backend, 'x', 'q', ['a'], mode='answer_codes')

    def test_all_36_base_codes_and_single_option(self):
        backend = CodeBackend()
        result = classify(backend, 'x', 'q', [str(i) for i in range(36)], mode='answer_codes')
        self.assertEqual(len(result['options']), 36)
        self.assertEqual(result['model_calls'], 1)
        by_label = {row['label']: row['code'] for row in result['options']}
        self.assertEqual(by_label['0'], 'A')
        self.assertEqual(by_label['25'], 'Z')
        self.assertEqual(by_label['26'], '0')
        self.assertEqual(by_label['35'], '9')
        result = classify(backend, 'x', 'q', ['only'], mode='answer_codes')
        self.assertEqual(result['options'][0]['probability'], 1)

    def test_lowercase_predictions_are_aggregated_into_canonical_codes(self):
        backend = CodeBackend()
        scores = {65: .01, 97: .69, 66: .2, 98: .1}
        with patch.object(backend, 'next_logprobs', side_effect=lambda prompt, prefix, allowed: {
            token: math.log(scores[token]) for token in allowed
        }):
            result = classify(backend, 'x', 'q', ['first', 'second'])
        rows = {row['label']: row for row in result['options']}
        self.assertEqual(result['answer'], 'first')
        self.assertAlmostEqual(rows['first']['probability'], .7)
        self.assertEqual(rows['first']['code'], 'A')
        self.assertEqual(rows['first']['matched_alias'], 'a')
        self.assertEqual([alias['code'] for alias in rows['first']['aliases']], ['A', 'a'])

    def test_true_and_false_labels_use_normal_positional_codes(self):
        backend = CodeBackend()
        result = classify(
            backend,
            'A refund was requested.',
            'Was a refund requested?',
            ['true', 'false'],
        )
        rows = {row['label']: row for row in result['options']}
        self.assertEqual(rows['true']['code'], 'A')
        self.assertEqual(rows['false']['code'], 'B')
        self.assertEqual([alias['code'] for alias in rows['true']['aliases']], ['A', 'a'])
        self.assertEqual([alias['code'] for alias in rows['false']['aliases']], ['B', 'b'])
        self.assertIn('A. "true"', result['form'])
        self.assertIn('B. "false"', result['form'])

    def test_unsupported_and_colliding_codes_are_skipped(self):
        backend = CodeBackend()

        def selective_ids(code):
            if code == 'A':
                raise ValueError('unsupported')
            if code == 'B':
                return [67, 0]
            return [ord(code), 0]

        with patch.object(backend, 'answer_ids', side_effect=selective_ids):
            result = classify(backend, 'x', 'q', ['first', 'second'], mode='answer_codes')

        by_label = {row['label']: row['code'] for row in result['options']}
        self.assertEqual(by_label, {'first': 'B', 'second': 'D'})
        self.assertIn('B. "first"', result['form'])
        self.assertIn('D. "second"', result['form'])
        self.assertNotIn('A. "first"', result['form'])

    def test_long_codes_score_complete_paths_and_use_cache(self):
        backend = CodeBackend()
        with patch.object(backend, 'scorer', return_value=lambda prefix, allowed: {
            token: math.log(.5 if token == 0 else .1) for token in allowed
        }) as scorer:
            result = classify(backend, 'x', 'q', [str(i) for i in range(256)])
        scorer.assert_called_once()
        self.assertFalse(backend.calls)
        rows = {row['code']: row for row in result['options']}
        self.assertEqual(len(rows), 256)
        self.assertEqual(rows['A']['token_ids'], [65, 0])
        self.assertEqual(rows['AA']['token_ids'], [65, 65, 0])
        self.assertAlmostEqual(rows['A']['log_likelihood'], math.log(2 * .1 * .5))
        self.assertAlmostEqual(rows['AA']['log_likelihood'], math.log(4 * .1 * .1 * .5))
        self.assertAlmostEqual(sum(row['probability'] for row in rows.values()), 1)

    def test_codes_extend_past_two_characters(self):
        codes = _answer_codes(CodeBackend(), 36 + 36 ** 2 + 1)
        self.assertEqual(codes[-2][0], '99')
        self.assertEqual(codes[-1][0], 'AAA')
        self.assertEqual(codes[-1][1][0], ('AAA', (65, 65, 65, 0)))

    def test_multitoken_codes_support_original_backend_protocol(self):
        backend = CodeBackend()
        backend.scorer = None
        result = classify(backend, 'x', 'q', [str(i) for i in range(37)])
        self.assertEqual(len(result['options']), 37)
        self.assertTrue(any(prefix == (65, 65) for _, prefix, _ in backend.calls))

    def test_single_tokens_precede_multitoken_characters(self):
        backend = CodeBackend()
        original = backend.answer_ids
        with patch.object(backend, 'answer_ids', side_effect=lambda code:
                          [123, 124, 0] if code == 'A' else original(code)):
            small = classify(backend, 'x', 'q', ['first'])
            self.assertEqual(small['options'][0]['code'], 'B')
            with patch.object(backend, 'scorer', return_value=lambda prefix, allowed:
                              {token: math.log(.1) for token in allowed}):
                large = classify(backend, 'x', 'q', [str(i) for i in range(36)])
        rows = {row['label']: row for row in large['options']}
        self.assertEqual(rows['35']['code'], 'A')
        self.assertEqual(rows['35']['matched_alias'], 'a')
        self.assertEqual(rows['35']['aliases'][0]['token_ids'], [123, 124, 0])

    def test_mixed_question_lengths_do_not_use_root_only_batching(self):
        backend = BatchCodeBackend()
        with patch.object(backend, 'scorer', return_value=lambda prefix, allowed:
                          {token: math.log(.1) for token in allowed}), \
             patch.object(backend, 'next_logprobs', return_value={
                 65: math.log(.05), 97: math.log(.05)
             }):
            result = LocalClient(backend=backend).system_one('x', {
                'small': {'type': 'choice', 'criteria': {'only': None}},
                'large': {'type': 'choice', 'criteria': {str(i): None for i in range(256)}},
            })
        self.assertFalse(backend.batches)
        self.assertEqual(len(result['answers']['large']['probabilities']), 256)

    def test_typed_answers_restore_choice_score_and_noul_meanings(self):
        backend = CodeBackend()
        result = LocalClient(backend=backend, mode='answer_codes').system_one(
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
        self.assertIn('A. "true": "Yes"', backend.form)
        self.assertIn('B. "false": "No"', backend.form)

    def test_sdk_cli_runnable_and_server_keep_the_response_contract(self):
        path = Path(__file__).resolve().parents[1] / 'examples/support.json'
        payload = json.loads(path.read_text())
        backend = CodeBackend()
        client = LocalClient(model=payload['model'], backend=backend)
        expected = client.invoke(payload)
        self.assertEqual(set(expected), {'model', 'answers', 'usage'})
        self.assertEqual(expected['usage'], {'input_tokens': 6, 'output_tokens': 0})
        runnable = as_runnable(backend, model=client.model)
        self.assertEqual(runnable.invoke(payload), expected)
        output = io.StringIO()
        with patch.object(sys, 'argv', ['open-cricket', '--input', str(path)]), \
             patch('open_cricket.backend.load_backend', return_value=backend), \
             contextlib.redirect_stdout(output):
            main()
        self.assertEqual(json.loads(output.getvalue()), expected)
        with patch.dict('os.environ', {'OPEN_CRICKET_API_KEY': ''}, clear=True), \
             patch('open_cricket.backend.load_backend', return_value=backend):
            with TestClient(create_app(model_name=client.model)) as http:
                response = http.post('/v1/systemone', json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json(), expected)

    def test_default_is_answer_codes_and_bad_modes_do_not_load_models(self):
        self.assertEqual(LocalClient(backend=CodeBackend()).mode, 'answer_codes')
        self.assertEqual(LocalClient(backend=CodeBackend(), mode='sequence').mode, 'sequence')
        self.assertEqual(classify(CodeBackend(), 'x', 'q', ['a', 'b'])['mode'], 'answer_codes')
        with patch('open_cricket.backend.load_backend') as load:
            with self.assertRaisesRegex(ValueError, 'mode must'):
                LocalClient(mode='typo')
            load.assert_not_called()
        with patch.object(sys, 'argv', ['open-cricket', '--demo', '--mode', 'answer_codes']), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
