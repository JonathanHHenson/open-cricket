# Switch from Jev to LabelJudge

LabelJudge serves the TypeSafe/Jev HTTP request and response format at
`POST /v1/systemone`, plus `GET /v1/models`. The REST call shapes and `system_one(state, questions)` SDK pattern follow Jev.
Use LabelJudge names and your local checkpoint; no Jev model aliases are provided.

## Start the server

```bash
uv sync --extra hf --extra server
export LABELJUDGE_API_KEY="choose-a-local-server-key"
uv run uvicorn labeljudge.server:app --host 127.0.0.1 --port 8000
```

For Apple silicon MLX, use `--extra mlx` instead of `--extra hf` and set
`LABELJUDGE_BACKEND=mlx`. `LABELJUDGE_MODEL`, `LABELJUDGE_DEVICE`, and
`LABELJUDGE_REVISION` select the local checkpoint, device, and revision at startup.

If `LABELJUDGE_API_KEY` is unset, authentication is disabled for local use.
The TypeSafe SDK still requires a key; any nonempty placeholder works in that case.
When authentication is enabled, the client's bearer key must match the server's
`LABELJUDGE_API_KEY`. No TypeSafe account or hosted inference is needed.

## Direct local Python SDK

```python
from labeljudge import LocalClient, Choice

client = LocalClient(model="Qwen/Qwen2.5-1.5B-Instruct")
result = client.system_one(
    state="I was charged twice.",
    questions={"team": Choice(criteria={"billing": None, "other": None})},
)
print(result["answers"]["team"]["choice"])
```

The CLI and both SDKs accept the same `state` / `model` / `questions` payload via
`invoke`. `examples/support.json` can be used directly with any of them.

## HTTP Python SDK

LabelJudge's HTTP client also supports this shape:

```python
from labeljudge.client import Client

client = Client("http://127.0.0.1:8000", api_key="choose-a-local-server-key")
result = client.system_one(
    state="I was charged twice.",
    questions={"billing": {"type": "noul", "instructions": "About billing?"}},
)
print(result["answers"]["billing"]["noul"])
```


## Using an existing TypeSafe Python client

Install `typesafe-sdk` in your client application's environment, then change:

```bash
export TYPESAFE_BASE_URL="http://127.0.0.1:8000"
export TYPESAFE_API_KEY="choose-a-local-server-key"
export TYPESAFE_DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
```

Use the API root, **without `/v1`**; the SDK adds the endpoint path.
Existing `TypeSafeClient()` and `AsyncTypeSafeClient()` calls can stay unchanged.
You can also configure the client explicitly:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(
    base_url="http://127.0.0.1:8000",
    api_key="choose-a-local-server-key",
    model="Qwen/Qwen2.5-1.5B-Instruct",
) as client:
    result = client.system_one(
        state={"message": "I was charged twice. Please fix this today."},
        questions={
            "team": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments and refunds", "other": None},
            ),
            "urgency": Score(criteria=["Can wait", "Within a few days", "Today"]),
            "billing_issue": Noul(instructions="Does this concern billing?"),
        },
    )
    print(result.choices["team"].choice)
    print(result.scores["urgency"].score)
    print(result.nouls["billing_issue"].noul)
    print(result.model)
```

The server is tested with `typesafe-sdk==0.7.0`. For other HTTP clients, change the
host to your LabelJudge server and use the same JSON body and bearer header:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $LABELJUDGE_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @examples/support.json
```


## Supported interface

| Input | Returned answer |
| --- | --- |
| Choice: `criteria` map of labels to descriptions or null | `type`, `choice`, `probabilities`, `confidence` |
| Score: ordered `criteria` array | `type`, `score`, `legend`, `probabilities`, `confidence` |
| Noul: optional `criteria.true` and `criteria.false` | `type`, `noul` |

`state`, `instructions`, and descriptions accept strings, objects, or arrays.
Nested JSON values are preserved; structured content is serialized into the local
model's prompt. Questions are identified by their original map keys; these keys
are not included in model prompts. Instructions may be omitted or null, matching
the official SDK's schema. Empty instructions use a generic question for the type.
Choice accepts 1–255 nonblank labels; Score accepts 1–10 levels, including the
single-level case allowed by SDK 0.7.0 (descriptive rubrics normally need two or more).
Malformed requests return 422 with field details. Missing or incorrect configured
bearer credentials return 401. `/docs` exposes the server's OpenAPI interface.

Use `"model": "Qwen/Qwen2.5-1.5B-Instruct"`, the default local checkpoint.
If you configure `LABELJUDGE_MODEL`, use that exact name in SDK and REST requests;
`/v1/models` lists the configured name. The response identifies this same model.
Generic `labeljudge` and Jev-branded aliases are rejected with 422. Requests do
not download or switch checkpoints; choose the checkpoint at server startup.
Injected classifiers default to `labeljudge-custom` unless
`create_app(..., model_name=...)` supplies a name. Model listing dates refer to
the API adapter, not the checkpoint's release.

## Behavioral differences to account for

This is API compatibility, not equivalent model predictions or calibration.

- Choice probabilities come from LabelJudge's existing exhaustive answer scoring.
  Score uses the probability-weighted mean of zero-based rubric indices. Noul
  scores `true` and `false` and returns the relative probability of `true`.
- `confidence` is **1 − normalized Shannon entropy** of the probabilities
  (single-option confidence is 1). This measures concentration, not calibrated
  correctness. TypeSafe's public docs do not specify its exact confidence formula.
  Retest any thresholds used to automate decisions. Responses identify our method
  in `x-labeljudge-confidence-method: normalized-entropy`.
- Local inference evaluates questions separately and sequentially; it does not
  reproduce Jev's parallel sampler. Context limits come from your local checkpoint.
- `usage.input_tokens` sums each question's prompt length once, including repeated
  state. `usage.output_tokens` is 0 because the scorer generates no text sequence.
  Custom classifiers without prompt counts return null for input usage.
  These counts are not TypeSafe billing units.
- The server emits the standard `x-request-id` header. TypeSafe-specific
  `response.request_id` raises because its branded header is absent; use
  `response.raw_http_response.headers["x-request-id"]` instead. There is no TypeSafe branding, hosted rate-limit policy, billing, or
  account management in the local API.

Sources checked on 2026-09-21: [TypeSafe API reference](https://docs.typesafe.ai/api),
[Python SDK configuration](https://docs.typesafe.ai/sdk/python/usage),
[model discovery](https://docs.typesafe.ai/models), and
[confidence documentation](https://docs.typesafe.ai/confidence).

## Verification

```bash
uv sync --extra test
uv run python -m unittest discover -s tests -v
```

Contract tests route the real SDK through the local FastAPI application using
an in-process HTTP transport. They require no model download, credentials for
TypeSafe, or external inference calls.
