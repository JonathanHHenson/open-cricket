# Architecture

[Technical index](README.md)

## Modules

| Module | Responsibility |
| --- | --- |
| [systemone.py](../../open_cricket/systemone.py) | Strict wire models, question conversion, answer mapping, entropy confidence, usage |
| [sdk.py](../../open_cricket/sdk.py) | Local client, model identity, lock, sequential question evaluation |
| [questionnaire.py](../../open_cricket/questionnaire.py) | Option validation, prompt rendering, canonical answer token paths |
| [core.py](../../open_cricket/core.py) | Dependency-free exhaustive trie scoring |
| [backend.py](../../open_cricket/backend.py) | Token protocol and lazy runtime factory |
| [local.py](../../open_cricket/local.py) | Tokenizer rules and request-local cache traversal |
| [hf.py](../../open_cricket/hf.py), [mlx.py](../../open_cricket/mlx.py) | Runtime loading, forward passes, normalization, cache trimming |
| [server.py](../../open_cricket/server.py) | Lifespan, authentication, routes, model gate, headers |
| [client.py](../../open_cricket/client.py) | Standard-library HTTP client and request validation |
| [integrations.py](../../open_cricket/integrations.py) | Local/chat runnables and graph adapter |
| [cli.py](../../open_cricket/cli.py) | File input, model override, demo, formatting, timing |

Pydantic is the base dependency. Heavy runtimes and frameworks are optional,
imported at use sites. Preserve that separation in public exports and extensions.

## Local request lifecycle

1. Validate `SystemOneRequest` and check the requested model against the client.
2. Acquire the per-client lock and prepare questions in insertion order.
3. `classification_input` renders structured content and maps Choice to its
   labels, Score to numeric-string indices, and Noul to `true`/`false`.
4. `classify` maps labels to case-sensitive alphanumeric codes, preferring
   single-token characters (`A–Z`, `a–z`, then `0–9`) before longer codes.
   Explicit `sequence` / `constrained` modes instead tokenize JSON-quoted labels
   followed by EOS.
5. For multiple questions whose codes are all single-token, a capable backend shares their exact token
   prefix and batches suffix scoring while returning results in request order.
   Otherwise single-token code mode calls `next_logprobs` once per question.
   Multi-token code sets include EOS in every path. They and label modes use
   `backend.scorer(prompt)` when available, otherwise call `next_logprobs`
   with the prompt, prefix, and allowed tokens for each trie node.
6. `score_paths` explores every path and returns detailed scoring rows.
7. After classification, `format_response` checks complete normalized criterion
   probabilities and constructs the typed answer envelope.

Question identifiers are not prompt text. Descriptions guide the prompt but are
not scored answer text. Questions do not share answers or prompt caches. Missing
prompt-token usage from any question makes aggregate input usage null.

## Server and concurrency boundaries

Constructing the module-level app does not load a model. The lifespan loads the
default local client, or stores an injected classifier. POST checks the served
model and awaits `classifier.ainvoke` with a JSON-compatible request dictionary.
The local async method runs inference through `asyncio.to_thread`.

One lock spans all questions in a local request. Multiple clients wrapping the
same backend have separate locks and do not make that backend safe. Direct
`classify` calls have no lock. An injected runnable must supply appropriate
concurrency control.

The endpoint converts classification `ValueError` to 422. FastAPI validates input
and returned `SystemOneResponse` data. Other exceptions are not caught there.
Authentication reads the environment key on each call and uses
`hmac.compare_digest` on the complete bearer header.

## Custom application

`create_app` accepts an object with `ainvoke(payload)` returning the public answer
envelope. This is different from a token backend. Save as `custom_server.py` in
the repository root:

```python
from open_cricket import LocalClient
from open_cricket.server import create_app

model = "Qwen/Qwen2.5-1.5B-Instruct"
classifier = LocalClient(model=model, mode="sequence", temperature=0.8)
app = create_app(classifier, model_name=model)
```

With HF and server extras installed, run:

```bash
uv run uvicorn custom_server:app --host 127.0.0.1 --port 8000
```

This example loads the model at module import. Injection bypasses default
environment-based runtime loading. Without `model_name`, injected classifiers
are advertised as `open-cricket-custom`; supply the name your classifier accepts.

## Framework adapters

`as_runnable` wraps a local client with sync/async entry points.
`Client.as_runnable` wraps HTTP invocation. `graph_node` forwards only the three
request fields and returns a single output-key update, preserving other graph
state through LangGraph's update mechanism.

`chat_runnable` is a separate scoring path. It assigns A–Z in criterion order,
binds provider logprob options, and normalizes first-token code log probabilities.
It forwards runnable configuration and evaluates questions sequentially. Missing
code probabilities raise errors. It supplies no prompt count, so formatted input
usage is null. See [scoring](scoring.md) for the mathematical distinction.
