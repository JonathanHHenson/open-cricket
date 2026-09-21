# Python SDK

[Documentation index](README.md) · [Request and answer reference](api.md)

`LocalClient` loads a model in your process. `Client` sends requests to a running
server. Both return dictionaries with `model`, `answers`, and `usage`.

## Installation

From a checkout, install into your application's environment:

```bash
python -m pip install -e '.[hf]'
```

Use `.[mlx]` for Apple silicon MLX or `.` for HTTP-only usage. With the repository's
`uv` environment, use `uv sync --extra hf` and run scripts with `uv run python`.

## Local inference

```python
from open_cricket import Choice, LocalClient, Noul, Score

client = LocalClient(model="Qwen/Qwen2.5-1.5B-Instruct")
result = client.system_one(
    state={"message": "I was charged twice. Please fix this today."},
    questions={
        "team": Choice(criteria={"billing": "Payments and refunds", "other": None}),
        "urgency": Score(criteria=["Can wait", "Within a few days", "Today"]),
        "billing_issue": Noul(instructions="Does this concern billing?"),
    },
)
print(result["answers"]["team"]["choice"])
print(result["answers"]["urgency"]["score"])
print(result["answers"]["billing_issue"]["noul"])
```

Construction loads and potentially downloads the model. Reuse the client.

| `LocalClient` argument | Default | Purpose |
| --- | --- | --- |
| `model` | `Qwen/Qwen2.5-1.5B-Instruct` | Checkpoint and accepted request model |
| `runtime` | `hf` | `hf` or `mlx` |
| `device` | `auto` | Runtime device selection |
| `revision` | `None` | Optional checkpoint revision |
| `mode` | `answer_codes` | Single-token codes; `sequence` or `constrained` for full label scoring |
| `temperature` | `1.0` | Finite positive temperature on answer scores |
| `backend` | `None` | Custom token backend; bypasses runtime loading |

Only `model` may be positional. An injected backend does not infer the model
name; set it yourself. Read [scoring semantics](technical/scoring.md) before
changing mode or temperature.

For compact answer codes, use `LocalClient(mode="answer_codes")` (or pass
the same mode to `as_runnable` / `classify`). Categories receive `A–Z`, then
`0–9` in criterion order. Codes are case-insensitive: uppercase and lowercase
output probabilities are combined per candidate before temperature is applied.
Single-token alphanumerics take priority; when exhausted, longer codes (`AA`,
`AB`, …) are used and all case variants are combined. There is no fixed candidate
cap; context and memory limits still apply. Responses
retain original labels, including Score levels and Noul's true/false meanings.
Single-token sets score without EOS in one pass. If any code needs multiple tokens,
all codes are scored with EOS using the cached trie. Probabilities and predictions
can differ from label scoring. Evaluate your own data and candidate orders before
switching modes. The default is `answer_codes`; use `mode="sequence"` for full
label scoring. Multi-token codes require additional work and introduce length bias.
Noul uses normal positional answer codes with default descriptions `Yes` and `No`.

### JSON requests and async calls

`invoke` requires the complete request, including `model`. `system_one` accepts
typed helpers or question dictionaries and defaults to the client's model.

```python
import asyncio
import json
from open_cricket import LocalClient

client = LocalClient()
with open("examples/support.json", encoding="utf-8") as source:
    payload = json.load(source)
result = client.invoke(payload)

async def main():
    result = await client.ainvoke(payload)
    print(result["answers"])

asyncio.run(main())
```

Inside an existing event loop, await `client.ainvoke(payload)` directly.
Async calls run synchronous inference in a worker thread. A per-client lock
serializes model access; async scheduling does not enable GPU batching.

## HTTP client

Start the [server](server.md), then:

```python
import os
from open_cricket import Choice, Client

client = Client(
    "http://127.0.0.1:8000",
    api_key=os.environ.get("OPEN_CRICKET_API_KEY"),
    timeout=120,
)
model = client.models()["models"][0]["name"]
result = client.system_one(
    state="I was charged twice.",
    questions={"team": Choice(criteria={"billing": None, "other": None})},
    model=model,
)
print(result["answers"]["team"]["choice"])
```

The base URL excludes `/v1`. Defaults are `http://127.0.0.1:8000`, no key, and a
120-second timeout. The client does not automatically read environment variables,
discover models, retry errors, or expose headers. Call `models()` for discovery.

HTTP `system_one` defaults to `Qwen/Qwen2.5-1.5B-Instruct`, regardless of server
configuration. Pass `model` for other checkpoints. `invoke(payload)` requires
the full request, just like the local client.

`Client` has no direct `ainvoke`. Use `await asyncio.to_thread(client.invoke,
payload)` or the optional LangChain runnable.

### Error handling

Pydantic `ValidationError` indicates invalid requests before network access or
local inference. Local model mismatches and scoring failures can raise
`ValueError`. The HTTP client propagates standard-library errors:

```python
from urllib.error import HTTPError, URLError
from pydantic import ValidationError

try:
    result = client.system_one(
        "I was charged twice.",
        {"billing": {"type": "noul", "instructions": "About billing?"}},
        model=model,
    )
except ValidationError as error:
    print(error.errors())
except HTTPError as error:
    print(error.code, error.read().decode("utf-8"))
except (URLError, TimeoutError) as error:
    print(f"Server unavailable or timed out: {error}")
```

## LangChain and LangGraph

Install `.[langchain]` or `.[langgraph]` alongside any local runtime. HTTP clients
expose `client.as_runnable()`. For a local backend:

```python
import json
from open_cricket.hf import HuggingFaceBackend
from open_cricket.integrations import as_runnable

judge = as_runnable(HuggingFaceBackend(), model="Qwen/Qwen2.5-1.5B-Instruct")
with open("examples/support.json", encoding="utf-8") as source:
    payload = json.load(source)
results = judge.batch([payload, payload])
route = judge | (lambda result: result["answers"]["department"]["choice"])
print(route.invoke(payload))
```

Runnables support `ainvoke` and `abatch` too. `graph_node(judge)` reads `state`,
`model`, and `questions` and returns a partial graph update under `classification`
(or a custom `output_key`). See the [LangGraph example](../examples/langgraph_workflow.py).

`chat_runnable` uses provider first-token answer codes. All candidate code
probabilities must be present or it raises `ProbabilityUnavailableError`. It
supports at most `min(26, top_logprobs)` options and differs from local sequence
scoring. The [chat example](../examples/langchain_chat.py) requires separate
provider dependencies and credentials and makes paid provider calls when run.
