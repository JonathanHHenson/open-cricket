"""Small stdlib HTTP client; optional Runnable conversion for remote inference."""
import json
from urllib.request import Request, urlopen


class Client:
    def __init__(self, base_url="http://127.0.0.1:8000", *, api_key=None, timeout=120):
        self.url = base_url.rstrip("/") + "/classify"
        self.api_key = api_key
        self.timeout = timeout

    def invoke(self, value):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(self.url, data=json.dumps(value).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def as_runnable(self):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(self.invoke, name="labeljudge_http")
