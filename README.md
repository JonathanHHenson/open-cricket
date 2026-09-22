# Open Cricket

[![Test, build, and publish](https://github.com/JonathanHHenson/open-cricket/actions/workflows/publish.yml/badge.svg?branch=main)](https://github.com/JonathanHHenson/open-cricket/actions/workflows/publish.yml)
[![PyPI version](https://img.shields.io/pypi/v/open-cricket.svg)](https://pypi.org/project/open-cricket/)
[![Python versions](https://img.shields.io/pypi/pyversions/open-cricket.svg)](https://pypi.org/project/open-cricket/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/JonathanHHenson/open-cricket/blob/main/LICENSE)

![Open Cricket mascot](https://raw.githubusercontent.com/JonathanHHenson/open-cricket/main/docs/assets/open-cricket-icon.svg)

Ask a local language model structured questions. Open Cricket can run in your
Python process or behind a small HTTP server. Both interfaces use the same
request shape and return the same `model`, `answers`, and `usage` envelope.

Choice selects a category, Score evaluates an ordered rubric, and Noul returns
the relative probability of a yes/true answer.

## Install from PyPI

Open Cricket requires Python 3.10 or newer. Install the Hugging Face runtime
for local inference—the default and most portable option:

```bash
python -m pip install "open-cricket[hf]"
```

For the HTTP server, install its extra too:

```bash
python -m pip install "open-cricket[hf,server]"
```

On Apple silicon, use the MLX runtime instead:

```bash
python -m pip install "open-cricket[mlx]"
```

The first local use downloads the chosen model. The default is
`Qwen/Qwen2.5-1.5B-Instruct`; remote model code is disabled.

## Use the Python SDK directly

`LocalClient` loads and keeps a model in your Python process. Reuse one client
for multiple requests so the model stays in memory.

```python
from open_cricket import Choice, LocalClient, Noul, Score

client = LocalClient(model="Qwen/Qwen2.5-1.5B-Instruct")
result = client.system_one(
    state="I was charged twice. Please fix this today.",
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

Use `runtime="mlx"` with `LocalClient` after installing the MLX extra. To use
a complete JSON request instead, call `client.invoke(payload)`; it must include
`state`, `model`, and `questions`. `await client.ainvoke(payload)` is available
for async applications.

## Run it as a server

Start a server when several applications should share one loaded model:

```bash
export OPEN_CRICKET_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export OPEN_CRICKET_API_KEY="choose-a-local-server-key"
uvicorn open_cricket.server:app --host 127.0.0.1 --port 8000
```

The API key is optional: leave `OPEN_CRICKET_API_KEY` unset to disable
authentication. Once startup has completed, try a request from another terminal:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $OPEN_CRICKET_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "state": "I was charged twice.",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "questions": {
      "team": {
        "type": "choice",
        "criteria": {"billing": "Payments and refunds", "other": null}
      }
    }
  }'
```

`GET /health` reports readiness, `GET /v1/models` lists the served checkpoint,
and [`/docs`](http://127.0.0.1:8000/docs) provides interactive API documentation.
The server accepts only its configured model name and does not download models
per request.

### Call the server with the SDK

Use `Client` to send the same typed questions to the server:

```python
import os
from open_cricket import Choice, Client

client = Client(
    "http://127.0.0.1:8000",
    api_key=os.environ.get("OPEN_CRICKET_API_KEY"),
)
result = client.system_one(
    state="I was charged twice.",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    questions={"team": Choice(criteria={"billing": None, "other": None})},
)
print(result["answers"]["team"]["choice"])
```

## Request format

Every interface accepts a request with `state`, the exact `model` name, and a
map of typed `questions`. Descriptions belong in `criteria`; use `null`/`None`
when a label needs no description.

```json
{
  "state": "I was charged twice. Please refund the extra payment.",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoices, and refunds",
        "technical": "Product errors and troubleshooting",
        "other": null
      }
    }
  }
}
```

`state`, instructions, and descriptions may be strings, structured JSON
objects, or arrays. Add an optional top-level `images` array (image URLs, data
URLs, or local paths) when using a vision-capable model such as
`Qwen/Qwen3.5-0.8B`. Only use trusted image sources when operating a server.

## Other ways to use Open Cricket

The package also includes a command-line interface:

```bash
open-cricket --input examples/support.json --pretty --time
```

Install `open-cricket[langchain]` or `open-cricket[langgraph]` for framework
integrations. The default `answer_codes` scoring mode is fast; select
`mode="sequence"` (or `OPEN_CRICKET_MODE=sequence` on the server) when you
need full-label likelihood scoring. Evaluate model choice, prompt wording, and
decision thresholds on your own data before automating decisions.

## Documentation

- [Python SDK guide](https://github.com/JonathanHHenson/open-cricket/blob/main/docs/sdk.md) — local, async, and HTTP SDK details
- [Server guide](https://github.com/JonathanHHenson/open-cricket/blob/main/docs/server.md) — configuration, operation, and troubleshooting
- [API reference](https://github.com/JonathanHHenson/open-cricket/blob/main/docs/api.md) — request and response schemas
- [CLI examples](https://github.com/JonathanHHenson/open-cricket/blob/main/examples/README.md) — more ready-to-run request files
- [Technical documentation](https://github.com/JonathanHHenson/open-cricket/blob/main/docs/technical/README.md) — architecture, scoring,
  backend development, and contributor checks
- [Migration from Jev](https://github.com/JonathanHHenson/open-cricket/blob/main/docs/jev-migration.md) — adapting an existing TypeSafe/Jev
  integration

Open Cricket is an independent educational prototype with no affiliation with
TypeSafe.
