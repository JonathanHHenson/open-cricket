import io
import json
import unittest
from unittest.mock import patch

from labeljudge.client import Client


class ClientTests(unittest.TestCase):
    def test_system_one_requests(self):
        client = Client('http://localhost:8000/', api_key='local', timeout=5)
        for method, value, path in (
            (client.invoke, {'state': 'test', 'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'questions': {'q': {'type': 'noul'}}}, '/v1/systemone'),
            (lambda value: client.system_one('test', value), {'q': {'type': 'noul'}}, '/v1/systemone'),
        ):
            with patch('labeljudge.client.urlopen', return_value=io.BytesIO(b'{"ok":true}')) as send:
                self.assertEqual(method(value), {'ok': True})
            request = send.call_args.args[0]
            self.assertEqual(request.full_url, 'http://localhost:8000' + path)
            self.assertEqual(request.get_header('Authorization'), 'Bearer local')
            self.assertEqual(request.method, 'POST')
            if path == '/v1/systemone':
                self.assertEqual(json.loads(request.data)['state'], 'test')
                self.assertEqual(json.loads(request.data)['model'], 'Qwen/Qwen2.5-1.5B-Instruct')
            self.assertEqual(send.call_args.kwargs['timeout'], 5)
        with patch('labeljudge.client.urlopen', return_value=io.BytesIO(b'{"models":[]}')) as send:
            self.assertEqual(client.models(), {'models': []})
        self.assertEqual(send.call_args.args[0].method, 'GET')
        self.assertTrue(send.call_args.args[0].full_url.endswith('/v1/models'))

    def test_rejects_legacy_payload_without_network(self):
        with patch('labeljudge.client.urlopen') as send:
            with self.assertRaises(ValueError):
                Client().invoke({'message': 'test', 'question': 'Which?', 'options': ['a']})
        send.assert_not_called()
