# Running the server

[Documentation index](README.md) · [API reference](api.md) · [Python SDK](sdk.md)

The service loads one checkpoint at startup and reuses it for requests. Clients
must request its exact name; requests cannot switch or download models.

## Install and start

From the repository root:

```bash
uv sync --extra hf --extra server
export OPEN_CRICKET_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export OPEN_CRICKET_API_KEY="choose-a-local-server-key"
uv run uvicorn open_cricket.server:app --host 127.0.0.1 --port 8000
```

Replace the example key with your own value. First startup may download model
weights and the tokenizer. Startup completes after loading the model; remote
model code is disabled.

In another terminal, set the same key and check the service:

```bash
export OPEN_CRICKET_API_KEY="choose-a-local-server-key"
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $OPEN_CRICKET_API_KEY"
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $OPEN_CRICKET_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @examples/support.json
```

Health returns `{"status":"ready"}`. If you change checkpoints, update the
request's `model` too. Open [interactive docs](http://127.0.0.1:8000/docs) to
inspect the schema. For authenticated calls there, supply the endpoint's
`authorization` header parameter as `Bearer <your-key>`; the application uses
a header parameter rather than an OpenAPI security scheme.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPEN_CRICKET_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Loaded checkpoint and accepted request model |
| `OPEN_CRICKET_BACKEND` | `hf` | Runtime: `hf` or `mlx` |
| `OPEN_CRICKET_DEVICE` | `auto` | HF device; automatic selection tries CUDA, MPS, then CPU |
| `OPEN_CRICKET_REVISION` | Unset | Optional checkpoint revision; use a commit for reproducibility |
| `OPEN_CRICKET_API_KEY` | Unset | Exact bearer secret; unset or empty disables authentication |

Restart after changing runtime configuration. Default service scoring uses
`sequence` mode and temperature `1.0`. These are not environment variables or
request fields; use a [custom application](technical/architecture.md#custom-application)
to change them.

For Apple silicon with accessible Metal:

```bash
uv sync --extra mlx --extra server
export OPEN_CRICKET_BACKEND=mlx
export OPEN_CRICKET_DEVICE=auto
uv run uvicorn open_cricket.server:app --host 127.0.0.1 --port 8000
```

MLX accepts devices `auto` and `mps`. It supports compatible checkpoints,
including MLX-converted quantized models. Changing runtime or precision can
change probabilities; recheck decision thresholds.

## Operation and concurrency

One local client serializes model access. Questions run sequentially, and
concurrent requests do not provide GPU batching. Each Uvicorn worker loads its
own model, increasing memory consumption. Start with one worker and keep it
running to avoid repeated initialization.

`/health` is an unauthenticated static readiness response after startup, not an
inference or GPU health test. `/docs`, `/redoc`, and `/openapi.json` are also
unauthenticated. Bearer authentication covers the two `/v1` routes. The app does
not implement TLS, rate limiting, or request-size limits; configure these at a
reverse proxy if exposing it beyond a trusted local environment.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing runtime or server module | Install both the chosen runtime and `server` extras in the environment running Uvicorn |
| Startup download/load failure | Check checkpoint access, revision, network, disk space, and device memory |
| 401 | Send the configured key with the exact `Bearer ` prefix |
| 422, unknown model | Use the exact `name` from `/v1/models` |
| 422, schema validation | Check [request types and limits](api.md#request-body); extra fields are forbidden |
| Context limit exceeded | Shorten state, instructions, or criteria; there is no silent truncation |
| Client timeout | Increase the client timeout and inspect logs; concurrent calls may be waiting for the model lock |
| 500 | Inspect logs for runtime or response-validation failures; not all exceptions become 422 |

Responses produced through the `/v1/` middleware include `x-request-id` and
`x-open-cricket-confidence-method: normalized-entropy`. Use an HTTP client's
response headers to capture them; the small Python SDK returns only the body.
