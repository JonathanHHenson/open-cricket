"""Requires langchain-openai and a compatible logprobs-enabled model.

Set MODEL_NAME and OPENAI_API_KEY. This example makes a billed API request.
Not every model supports logprobs or the default request parameters.
"""

import importlib
import os

from labeljudge.integrations import chat_runnable

ChatOpenAI = importlib.import_module("langchain_openai").ChatOpenAI

model_name = os.environ["MODEL_NAME"]
classifier = chat_runnable(ChatOpenAI(model=model_name), model_name=model_name)
if __name__ == "__main__":
    print(
        classifier.invoke(
            {
                "state": "I was charged twice.",
                "model": model_name,
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"billing": None, "technical support": None, "other": None},
                    }
                },
            }
        )
    )
