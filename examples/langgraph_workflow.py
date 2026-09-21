"""Run after pip install -e '.[hf,langgraph]' from the project directory."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from labeljudge.hf import HuggingFaceBackend
from labeljudge.integrations import as_runnable, graph_node


class State(TypedDict, total=False):
    message: str
    question: str
    options: list[str]
    classification: dict
    action: str


classifier = as_runnable(HuggingFaceBackend())
builder = StateGraph(State)
builder.add_node("classify", graph_node(classifier))
builder.add_node("billing", lambda state: {"action": "queue_for_billing_review"})
builder.add_node("other", lambda state: {"action": "queue_for_general_review"})
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", lambda state: state["classification"]["answer"],
                              {"billing": "billing", "other": "other"})
builder.add_edge("billing", END)
builder.add_edge("other", END)
graph = builder.compile()
if __name__ == "__main__":
    print(graph.invoke({"message": "I was charged twice; please refund the extra payment.",
                        "question": "Which team should handle this?", "options": ["billing", "other"]}))
