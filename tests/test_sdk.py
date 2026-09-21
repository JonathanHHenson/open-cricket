import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from labeljudge import Choice, Client, LocalClient, Noul, Score
from labeljudge.cli import main, format_pretty
from labeljudge.server import create_app
from labeljudge.systemone import SystemOneRequest

ROOT = Path(__file__).resolve().parents[1]


class Tokens:
    def prompt_ids(self, system, form):
        return [99]

    def answer_ids(self, answer):
        return list(answer.encode()) + [0]

    def next_logprobs(self, prompt, prefix, allowed):
        return {token: -5.6 for token in allowed}


class SdkTests(unittest.TestCase):
    def test_examples_share_cli_local_and_http_contract(self):
        for filename in ('support.json',):
            with self.subTest(filename=filename):
                path = ROOT / 'examples' / filename
                payload = json.loads(path.read_text())
                SystemOneRequest.model_validate(payload)
                client = LocalClient(model=payload['model'], backend=Tokens())
                expected = client.invoke(payload)
                self.assertEqual(set(expected), {'model', 'answers', 'usage'})
                self.assertEqual(set(expected['answers']), set(payload['questions']))
                self.assertEqual(asyncio.run(client.ainvoke(payload)), expected)
                with patch.dict('os.environ', {'LABELJUDGE_API_KEY': ''}):
                    with TestClient(create_app(client, model_name=client.model)) as http:
                        response = http.post('/v1/systemone', json=payload)
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(response.json(), expected)
                output = io.StringIO()
                with patch.object(sys, 'argv', ['labeljudge', '--input', str(path)]), \
                     patch('labeljudge.backend.load_backend', return_value=Tokens()) as load, \
                     contextlib.redirect_stdout(output):
                    main()
                load.assert_called_once_with('hf', payload['model'], 'auto', None)
                self.assertEqual(json.loads(output.getvalue()), expected)

    def test_typed_direct_call_and_pretty_mixed_answers(self):
        client = LocalClient(backend=Tokens())
        questions = {'route': Choice(criteria={'billing': None, 'other': None}),
                     'urgency': Score(criteria=['Low', 'High']),
                     'refund': Noul(instructions='Refund requested?')}
        result = client.system_one('Test', questions)
        self.assertEqual(set(result['answers']), set(questions))
        pretty = format_pretty(result, 'Test', questions)
        self.assertIn('Question Type: choice', pretty)
        self.assertIn('Question Type: score', pretty)
        self.assertIn('Question Type: noul', pretty)
        self.assertIn('Answer:', pretty)
        self.assertIn('Score:', pretty)
        self.assertIn('Yes probability:', pretty)
        with self.assertRaisesRegex(ValueError, 'Unknown model'):
            client.system_one('Test', questions, model='labeljudge')

    def test_invalid_shape_rejected_across_clients_and_cli(self):
        payload = {'message': 'test', 'questions': [{'question': 'Which?', 'options': ['a']}]}
        with self.assertRaises(ValueError):
            LocalClient(backend=Tokens()).invoke(payload)
        with patch('labeljudge.client.urlopen') as network:
            with self.assertRaises(ValueError):
                Client().invoke(payload)
        network.assert_not_called()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as source:
            json.dump(payload, source)
            source.flush()
            with patch.object(sys, 'argv', ['labeljudge', '--input', source.name]), \
                 patch('labeljudge.backend.load_backend') as load, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()

    def test_cli_explicit_model_override_loads_and_reports_selected_model(self):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['labeljudge', '--input', str(ROOT / 'examples/support.json'),
                                      '--model', 'custom/checkpoint']), \
             patch('labeljudge.backend.load_backend', return_value=Tokens()) as load, \
             contextlib.redirect_stdout(output):
            main()
        self.assertEqual(load.call_args.args[1], 'custom/checkpoint')
        self.assertEqual(json.loads(output.getvalue())['model'], 'custom/checkpoint')
