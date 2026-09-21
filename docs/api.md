# API reference

[Documentation index](README.md) · [Server setup](server.md) · [Python SDK](sdk.md)

Use JSON request bodies. When `OPEN_CRICKET_API_KEY` is nonempty, both `/v1`
endpoints require `Authorization: Bearer <key>`.

| Method and path | Response |
| --- | --- |
| `GET /health` | `{"status":"ready"}`; unauthenticated |
| `GET /v1/models` | `{"models":[{"name":"…","description":"…","release_date":"…"}]}` |
| `POST /v1/systemone` | Envelope containing `model`, `answers`, and `usage` |
| `GET /openapi.json` | Generated OpenAPI schema; unauthenticated |
| `GET /docs`, `GET /redoc` | Schema documentation; unauthenticated |

Model listing dates describe the adapter, not the checkpoint release.

## Request body

Default local scoring prefers single-token `A–Z` and `0–9` answer codes and extends
to longer codes as needed. Letter case variants are combined into the same
candidate probability. There is no fixed Choice candidate cap; model context and
memory limits still apply. For full label likelihoods, configure
`OPEN_CRICKET_MODE=sequence` on the server (or `LocalClient(mode="sequence")`).
The local wire schema has no maximum Choice label count.

All three top-level fields are required:

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | String, JSON object, or JSON array | Data to evaluate; no top-level number, boolean, or null |
| `model` | Nonempty string | Exact configured checkpoint name from `/v1/models` |
| `questions` | Nonempty object | User-chosen identifiers mapped to typed questions |

Objects and arrays may contain nested JSON values. Structured content is
serialized into the prompt. Question identifiers become answer keys but are
not included in model prompts. Each question sees the same state independently;
earlier answers do not feed later questions.

Each question requires a `type`. `instructions` accepts a string, object, array,
null, or omission. Omitted, null, or blank-string instructions use a generic
question for the type. Unknown fields are forbidden, types are strict, and
nonfinite numeric values are rejected.

### Choice

`"type":"choice"` requires a nonempty `criteria` map of nonblank labels. Descriptions
accept strings, objects, arrays, or null. They guide the model; the returned
choice is a label. Empty string descriptions are treated as absent.

### Score

`"type":"score"` requires an ordered `criteria` array of 1–10 descriptions
(strings, objects, or arrays, not null). Order from lowest to highest score.
Indices start at zero: three levels correspond to 0, 1, 2. The result is a
probability-weighted mean, can be fractional, and is not a percentage. A single
level always produces score 0.

### Noul

`"type":"noul"` asks a true/false question. `criteria` may be omitted, null, or an
object with optional `true` and `false` descriptions. Each accepts a string,
object, array, or null; missing/null descriptions use generic true/false wording.
Put the proposition in `instructions` or describe both alternatives in `criteria`.

## Complete example

Save as `request.json`:

```json
{
  "state": {"message": "I was charged twice. Please fix this today."},
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payments and refunds", "other": null}
    },
    "urgency": {
      "type": "score",
      "criteria": ["Can wait", "Within a few days", "Today"]
    },
    "billing_issue": {
      "type": "noul",
      "instructions": "Does this concern billing?"
    }
  }
}
```

With the server running and `OPEN_CRICKET_API_KEY` set to its configured key:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $OPEN_CRICKET_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

Illustrative response only; these values are not measured predictions:

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 1.0, "other": 0.0},
      "confidence": 1.0
    },
    "urgency": {
      "type": "score",
      "score": 1.0,
      "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25},
      "legend": {"0": "Can wait", "1": "Within a few days", "2": "Today"},
      "confidence": 0.0536053696428138
    },
    "billing_issue": {"type": "noul", "noul": 0.9}
  },
  "usage": {"input_tokens": 300, "output_tokens": 0}
}
```

## Interpreting answers

Choice returns the highest-probability label and full distribution. Ties select
the first label in request order. Score returns the expected zero-based index,
a distribution with string index keys, and a `legend` preserving descriptions.
Noul returns only the probability of `true`, without confidence or a probability map.

Choice and Score confidence is `1 - entropy(p) / log(number of criteria)`;
single-criterion confidence is 1. Concentrated distributions can still be wrong.
Probabilities are relative to the supplied candidates. Wording, labels, order,
model, and runtime can affect results.

Local `usage.input_tokens` sums each question's full prompt once, including
repeated state. It does not count every trie evaluation. Adapters without prompt
counts return null input usage. `output_tokens` is 0 because scoring generates
no answer text sequence. These are not hosted billing units.

## Errors and unsupported fields

FastAPI errors use `detail`: schema errors normally contain a list with field
locations, while application errors contain a string, for example:

```json
{"detail":"Invalid API key"}
```

```json
{"detail":"Unknown model; see GET /v1/models"}
```

401 means missing/incorrect configured credentials. 422 covers malformed
requests, unknown models, and classification `ValueError`, including context
overflow. Other runtime errors can produce 500 responses.

There are no streaming, batch, sampling, `mode`, or `temperature` request fields.
Send separate requests for multiple states. Configure scoring on a local client
or [injected server classifier](technical/architecture.md#custom-application).
