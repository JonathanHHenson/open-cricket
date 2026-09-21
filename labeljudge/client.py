"""Small stdlib HTTP client; optional Runnable conversion for remote inference."""
import json
from urllib.request import Request, urlopen


class Client:
    def __init__(self, base_url="http://127.0.0.1:8000", *, api_key=None, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/v1/systemone"
        self.api_key = api_key
        self.timeout = timeout

    def invoke(self, value):
        from .systemone import SystemOneRequest
        request = SystemOneRequest.model_validate(value)
        return self._request(self.url, request.model_dump(mode="json"))

    def system_one(self, state, questions, *, model="Qwen/Qwen2.5-1.5B-Instruct"):
        """Evaluate typed questions against state; returns the JSON response as a dict."""
        questions = {name: question.model_dump(mode="json") if hasattr(question, "model_dump") else question
                     for name, question in questions.items()}
        return self.invoke({"state": state, "questions": questions, "model": model})

    def models(self):
        return self._request(self.base_url + "/v1/models")

    def _request(self, url, value=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        data = json.dumps(value, allow_nan=False).encode("utf-8") if value is not None else None
        request = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def as_runnable(self):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(self.invoke, name="labeljudge_http")
