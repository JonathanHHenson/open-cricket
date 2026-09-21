"""Direct local SDK using the same request and response contract as REST."""
import asyncio
import threading

from .questionnaire import classify
from .systemone import SystemOneRequest, classification_input, format_response


class LocalClient:
    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct", *, backend=None,
                 runtime="hf", device="auto", revision=None, mode="sequence", temperature=1.0):
        from .backend import load_backend
        self.model = model
        self.backend = backend if backend is not None else load_backend(runtime, model, device, revision)
        self.mode, self.temperature = mode, temperature
        self._lock = threading.Lock()

    def invoke(self, value):
        request = SystemOneRequest.model_validate(value)
        if request.model != self.model:
            raise ValueError(f"Unknown model; this client serves {self.model}")
        with self._lock:
            results = [classify(self.backend, **classification_input(request.state, question),
                               mode=self.mode, temperature=self.temperature)
                       for question in request.questions.values()]
        return format_response(request, self.model, results).model_dump(mode="json")

    def system_one(self, state, questions, *, model=None):
        questions = {name: question.model_dump(mode="json") if hasattr(question, "model_dump") else question
                     for name, question in questions.items()}
        return self.invoke({"state": state, "model": self.model if model is None else model,
                            "questions": questions})

    async def ainvoke(self, value):
        return await asyncio.to_thread(self.invoke, value)
