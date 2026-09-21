import asyncio
import importlib.util
import math
import unittest
from types import SimpleNamespace
from typing import TypedDict

from labeljudge.integrations import (
    ProbabilityUnavailableError,
    as_runnable,
    chat_runnable,
    graph_node,
)


class ToyBackend:
    def prompt_ids(self, system, form):
        return [99]

    def answer_ids(self, answer):
        return [1 if answer == '"billing"' else 2, 0]

    def next_logprobs(self, prompt, prefix, allowed):
        probabilities = {1: 0.6, 2: 0.3} if not prefix else {0: 0.5}
        return {t: math.log(probabilities[t]) for t in allowed}


VALUE = {"message": "charged twice", "question": "Which team?", "options": ["billing", "other"]}
DESCRIBED_VALUE = {
    **VALUE,
    "options": [
        {"label": "billing", "description": "Charges, payments, and refunds"},
        {"label": "other", "description": "Only when no category applies"},
    ],
}


class FakeChat:
    def __init__(self, missing=False):
        self.missing = missing
        self.config = None

    def bind(self, **kwargs):
        self.kwargs = kwargs
        return self

    def invoke(self, messages, config=None):
        self.config = config
        self.messages = messages
        entries = [{"token": "A", "logprob": math.log(0.6)}]
        if not self.missing:
            entries.append({"token": "B", "logprob": math.log(0.3)})
        return SimpleNamespace(
            response_metadata={"logprobs": {"content": [{"top_logprobs": entries}]}}
        )

    async def ainvoke(self, messages, config=None):
        return self.invoke(messages, config)


@unittest.skipUnless(importlib.util.find_spec("langchain_core"), "Install .[langchain]")
class IntegrationTests(unittest.TestCase):
    def test_invoke_batch_async_and_lcel(self):
        classifier = as_runnable(ToyBackend())
        self.assertEqual(classifier.invoke(VALUE)["answer"], "billing")
        self.assertEqual(len(classifier.batch([VALUE, VALUE])), 2)
        self.assertEqual(asyncio.run(classifier.ainvoke(VALUE))["answer"], "billing")
        self.assertEqual(len(asyncio.run(classifier.abatch([VALUE, VALUE]))), 2)
        self.assertEqual(classifier.invoke(DESCRIBED_VALUE)["answer"], "billing")
        chain = classifier | (lambda r: r["answer"])
        self.assertEqual(chain.invoke(VALUE), "billing")

    def test_chat_codes_probabilities_and_config(self):
        model = FakeChat()
        classifier = chat_runnable(model)
        r = classifier.invoke(VALUE, config={"tags": ["test"], "metadata": {"request": 1}})
        self.assertAlmostEqual(r["options"][0]["probability"], 2 / 3)
        self.assertEqual(model.kwargs["max_tokens"], 1)
        self.assertEqual(model.config["metadata"]["request"], 1)
        self.assertIn('A = "billing"', model.messages[1].content)
        described = classifier.invoke(DESCRIBED_VALUE)
        self.assertEqual(described["answer"], "billing")
        self.assertEqual(described["options"][0]["label"], "billing")
        self.assertIn("Charges, payments, and refunds", model.messages[1].content)
        self.assertEqual(asyncio.run(classifier.ainvoke(VALUE))["answer"], "billing")

    def test_missing_top_k_is_not_zero(self):
        with self.assertRaises(ProbabilityUnavailableError):
            chat_runnable(FakeChat(missing=True)).invoke(VALUE)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "Install .[langgraph]")
    def test_real_langgraph_state_and_conditional_routing(self):
        from langgraph.graph import END, START, StateGraph

        class State(TypedDict, total=False):
            message: str
            question: str
            options: list[str | dict[str, str]]
            classification: dict
            marker: str
            routed: str

        g = StateGraph(State)
        g.add_node("judge", graph_node(as_runnable(ToyBackend())))
        g.add_node("billing", lambda state: {"routed": "billing"})
        g.add_node("other", lambda state: {"routed": "other"})
        g.add_edge(START, "judge")
        g.add_conditional_edges(
            "judge",
            lambda state: state["classification"]["answer"],
            {"billing": "billing", "other": "other"},
        )
        g.add_edge("billing", END)
        g.add_edge("other", END)
        graph = g.compile()
        for result in [
            graph.invoke({**VALUE, "marker": "preserve me"}),
            asyncio.run(graph.ainvoke({**VALUE, "marker": "preserve me"})),
        ]:
            self.assertEqual(result["marker"], "preserve me")
            self.assertEqual(result["routed"], "billing")


if __name__ == "__main__":
    unittest.main()
