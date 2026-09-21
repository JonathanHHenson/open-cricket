"""Requires langchain-openai and a compatible logprobs-enabled model.

Set MODEL_NAME and OPENAI_API_KEY. This example makes a billed API request.
Not every model supports logprobs or the default request parameters.
"""
import os
from langchain_openai import ChatOpenAI
from labeljudge.integrations import chat_runnable

classifier = chat_runnable(ChatOpenAI(model=os.environ["MODEL_NAME"]))
if __name__ == "__main__":
    print(classifier.invoke({"message": "I was charged twice.",
                             "question": "Which team should handle this?",
                             "options": ["billing", "technical support", "other"]}))
