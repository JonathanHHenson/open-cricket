import importlib.util
import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ["fastapi", "httpx", "langchain_core"]),
                     "Install .[test]")
class ServerTests(unittest.TestCase):
    def test_http_contract_validation_auth_and_runnable(self):
        from fastapi.testclient import TestClient
        from labeljudge.server import create_app
        from labeljudge.integrations import as_runnable
        from test_integrations import ToyBackend, VALUE
        with patch.dict(os.environ, {"LABELJUDGE_API_KEY": "test-only-secret"}):
            with TestClient(create_app(as_runnable(ToyBackend()))) as client:
                self.assertEqual(client.get("/health").json(), {"status": "ready"})
                self.assertEqual(client.post("/classify", json=VALUE).status_code, 401)
                headers = {"Authorization": "Bearer test-only-secret"}
                r = client.post("/classify", json=VALUE, headers=headers)
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["answer"], "billing")
                self.assertEqual(client.post("/classify", json={**VALUE, "options": []}, headers=headers).status_code, 422)
                self.assertEqual(client.post("/classify", json={**VALUE, "options": ["a", "a"]}, headers=headers).status_code, 422)
                self.assertEqual(client.post("/classify", json={**VALUE, "extra": 1}, headers=headers).status_code, 422)
