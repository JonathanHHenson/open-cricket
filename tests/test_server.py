import unittest
from fastapi.testclient import TestClient
from labeljudge.server import create_app


class ServerTests(unittest.TestCase):
    def test_public_routes_and_request_validation(self):
        with TestClient(create_app(object())) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            paths = client.get('/openapi.json').json()['paths']
            self.assertEqual(set(paths), {'/health', '/v1/models', '/v1/systemone'})
            self.assertIn('/v1/systemone', paths)
            self.assertEqual(client.post('/v1/systemone', json={
                'message': 'test', 'questions': []}).status_code, 422)
