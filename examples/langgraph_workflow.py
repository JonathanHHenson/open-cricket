"""Run after uv sync --extra hf --extra langgraph."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from labeljudge.hf import HuggingFaceBackend
from labeljudge.integrations import as_runnable, graph_node


class State(TypedDict, total=False):
    state: str
    model: str
    questions: dict
    classification: dict
    action: str


classifier = as_runnable(HuggingFaceBackend())
builder = StateGraph(State)
builder.add_node("judge", graph_node(classifier))
builder.add_node("billing", lambda state: {"action": "queue_for_billing_review"})
builder.add_node("other", lambda state: {"action": "queue_for_general_review"})
builder.add_edge(START, "judge")
builder.add_conditional_edges("judge", lambda state: state["classification"]["answers"]["route"]["choice"],
                              {"billing": "billing", "other": "other"})
builder.add_edge("billing", END)
builder.add_edge("other", END)
graph = builder.compile()
if __name__ == "__main__":
    print(graph.invoke({"state": "I was charged twice; please refund the extra payment.",
                        "model": "Qwen/Qwen2.5-1.5B-Instruct",
                        "questions": {"route": {"type": "choice", "instructions": "Which team?",
                                                "criteria": {"billing": None, "other": None}}}}))
