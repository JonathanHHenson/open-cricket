# LabelJudge

Local structured decisions using causal language models. The CLI, Python SDK,
REST API, and LangChain integrations share one request contract:
**`state`, `model`, and a map of typed `questions`**. Responses contain `model`,
`answers`, and `usage`. Choice selects a category, Score evaluates a rubric,
and Noul returns the relative probability of a yes/true answer.

The default checkpoint is `Qwen/Qwen2.5-1.5B-Instruct`. LabelJudge is independent
of TypeSafe; its API follows Jev's general call shapes, but model predictions
and confidence calibration differ.

## Quick start: local model, no server

```bash
uv sync --extra hf
uv run labeljudge --input examples/support.json --pretty --time
```

The same input file can be sent to the REST API or passed to `LocalClient.invoke`.
For example:

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

`examples/support.json` demonstrates Choice, Score, and Noul questions about
department routing, urgency, and incorrect charges.
Descriptions belong in `criteria`; use null for a label needing no description.
State, instructions, and descriptions also accept structured JSON objects and arrays.

CLI output defaults to JSON. `--pretty` renders answers and probabilities;
`--time` reports request time excluding model initialization. The file's `model`
selects the checkpoint; `--model` overrides it explicitly. `--backend hf|mlx`,
`--device`, `--revision`, `--mode`, and `--temperature` configure local inference.

The first run downloads the selected model from Hugging Face. Remote model code
is disabled. To explore the scoring mathematics with synthetic probabilities:

```bash
uv run labeljudge --demo --pretty
uv run labeljudge --demo --mode constrained
```

## Direct Python SDK

```python
import json
from labeljudge import LocalClient, Choice, Score, Noul

client = LocalClient(model="Qwen/Qwen2.5-1.5B-Instruct")
result = client.system_one(
    state="I was charged twice. Please fix this today.",
    questions={
        "route": Choice(instructions="Which team?", criteria={"billing": "Payments", "other": None}),
        "urgency": Score(criteria=["Can wait", "Within a few days", "Today"]),
        "billing": Noul(instructions="Does this concern billing?"),
    },
)
print(result["answers"]["route"]["choice"])
print(result["answers"]["urgency"]["score"])
print(result["answers"]["billing"]["noul"])

with open("examples/support.json") as f:
    result = client.invoke(json.load(f))
```

`LocalClient` keeps the model loaded and serializes access to it. `await
client.ainvoke(payload)` runs inference in a worker thread. Use `runtime="mlx"`
for Apple silicon. `backend=...` accepts a custom token backend. The requested
model must match the client's loaded model. There are no branded model aliases.

## REST API and HTTP SDK

```bash
uv sync --extra hf --extra server
export LABELJUDGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export LABELJUDGE_API_KEY="choose-a-local-server-key"
uv run uvicorn labeljudge.server:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $LABELJUDGE_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @examples/support.json
```

`GET /v1/models` lists the configured checkpoint. `GET /health` reports readiness;
`/docs` exposes OpenAPI documentation. Set `LABELJUDGE_BACKEND=mlx` to serve MLX.
Authentication is disabled if `LABELJUDGE_API_KEY` is unset. Requests select the
configured model by its exact name; they do not trigger model downloads.

```python
import json
from labeljudge import Client, Choice

client = Client("http://127.0.0.1:8000", api_key="choose-a-local-server-key")
result = client.system_one(
    state="I was charged twice.",
    questions={"route": Choice(criteria={"billing": None, "other": None})},
)
with open("examples/support.json") as f:
    result = client.invoke(json.load(f))
```

The HTTP and local SDKs return dictionaries with the same answer shape.
The server accepts an injected typed runnable through
`create_app(classifier, model_name="your-checkpoint")`.

## LangChain and LangGraph

```bash
uv sync --extra hf --extra langgraph
```

```python
import json
from labeljudge.hf import HuggingFaceBackend
from labeljudge.integrations import as_runnable

judge = as_runnable(HuggingFaceBackend(), model="Qwen/Qwen2.5-1.5B-Instruct")
with open("examples/support.json") as f:
    payload = json.load(f)
result = judge.invoke(payload)
results = judge.batch([payload, payload])
route = judge | (lambda result: result["answers"]["department"]["choice"])
```

`ainvoke`, `abatch`, and `Client.as_runnable()` use this same contract.
`graph_node(judge)` reads `state`, `model`, and `questions` from graph state and
writes the answer envelope to `classification`, preserving unrelated graph data.
See `examples/langgraph_workflow.py` for conditional routing.

`chat_runnable(chat_model, model_name="provider-model")` also accepts the same
contract. It scores single-token answer codes and requires all requested code
probabilities in provider metadata. Missing probabilities raise an error. It has
a maximum of 26 categories, further limited by `top_logprobs`. See
`examples/langchain_chat.py`; that example calls a paid provider when run.

## Switching from Jev

Use the same state/question structure and `system_one` call pattern with your
local checkpoint name. The [migration guide](docs/jev-migration.md) covers
existing TypeSafe client configuration and the differences in model behavior,
confidence, and token accounting.

## Faster local inference and Apple silicon

Local classification reuses a request-local attention (KV) cache across answer
prefixes. Branch changes trim ordinary KV caches back to their shared prefix;
unsupported sliding-window or recurrent cache formats safely recompute instead.
Hugging Face models that support `logits_to_keep` also compute only the final
position's vocabulary logits. No labels or answer tokens are skipped, and
normalisation still uses the full vocabulary. `model_calls` counts scored trie
nodes, so it does not decrease even though each call does much less work.
Custom backends implementing the original three-method protocol still work.
For Hugging Face cache acceleration, use Transformers 4.57; older supported
versions can fall back to full-prefix scoring.

For Apple silicon, install and select the optional MLX runtime:

```bash
uv sync --extra mlx
uv run labeljudge --backend mlx --input examples/support.json --pretty --time
```

MLX accepts compatible Hugging Face checkpoints and MLX-converted quantized
checkpoints via `--model`. The selected checkpoint determines weight precision;
quantization and lower precision can change probabilities and close rankings.
The MLX extra is restricted to Apple silicon macOS and uses MLX-LM 0.28.x to
remain compatible with this project's Transformers 4.x dependency. It requires
an accessible Metal GPU. Hugging Face remains the default runtime.

In Python, use `from labeljudge.mlx import MLXBackend` and pass `MLXBackend()`
to `classify` or `as_runnable`. For the HTTP service:

```bash
uv sync --extra mlx --extra server
LABELJUDGE_BACKEND=mlx uv run uvicorn labeljudge.server:app --host 127.0.0.1
```

Keep the model loaded between requests (for example through the HTTP service).
Starting the CLI for every message reloads the model. The existing runnable
serializes access to its model; async requests do not provide GPU batching.
Caches are isolated per classification and are not retained across requests.

### Measured performance

A local Qwen2.5-0.5B-Instruct benchmark with four options, 112 prompt tokens and
11 scored prefixes produced these warmed median request times, excluding load:

| Runtime | Full-prefix baseline | Optimized | Speedup | Repeats |
| --- | ---: | ---: | ---: | ---: |
| Hugging Face, CPU, float32 | 1.350 s | 0.259 s | 5.2× | 5 |
| MLX, Metal, checkpoint precision | 0.169 s | 0.057 s | 3.0× | 3 |

These measurements use the 0.5B checkpoint; they do not measure the new 1.5B
default. The benchmark script keeps 0.5B as its default for reproducibility.
These are one local workload, not a general performance guarantee or a comparison
of MLX against Hugging Face MPS. The largest absolute probability difference
between baseline and optimized scoring was 0.0000033 for HF and 0.024 for MLX
(about 2.4 percentage points). MLX's lower-precision model arithmetic can vary
between full-sequence and incremental evaluation. Float32 miniature-model
regression tests verify both runtimes' branch rollback against uncached scoring
to five decimal places. Evaluate your categories before switching runtime or
weight precision.

Reproduce the benchmark with:

```bash
uv run python examples/benchmark_backend.py --device cpu --repeats 5
uv run python examples/benchmark_backend.py --backend mlx --repeats 5
```

The benchmark alternates cached and uncached runs after warmup and reports both
speed and probability differences. Rust has not been introduced: eliminating
repeated model computation provides the demonstrated gain, while rewriting the
small Python trie would leave that model work unchanged.

## What happens mathematically?

The questionnaire renders message data as a quoted JSON string, followed by the
question, allowed options, and `Answer:`. The HF adapter uses the tokenizer's
chat template and assistant-generation boundary where available. The system
instruction asks for one JSON-quoted option and immediate end of turn.

For an option with token sequence `t1 ... tn EOS`:

```text
score(option) = sum_j log P(tj | questionnaire, t1 ... t(j-1))
P(option | supplied candidates) = softmax(score(option) / temperature)
```

Logits become log probabilities using full-vocabulary `log_softmax`. We add
log probabilities, not probabilities, and do not softmax already normalised
probabilities. Labels share a token trie: common prefixes are evaluated once
per question. Every branch is explored; this is not greedy token generation.
An explicit EOS/end-of-turn token distinguishes a complete answer from a label
that merely begins the same way. JSON quoting handles multiword labels and
embedded quotes. Every output is one of the supplied labels by construction.

The exact path scores **one canonical tokenisation** of each JSON answer plus
one tokenizer-defined EOS token, starting at a fixed prompt token boundary.
It does not sum alternative tokenisations, whitespace, synonyms, or other
valid end-of-turn tokens. Thus "exact" refers to these particular token-sequence
events, not all text representations of the underlying category.

`mode="constrained"` instead renormalises at every trie branch. It implements
locally masked generation. This is mathematically different from conditioning
the original model on the complete candidate set and can change the winner.
The default is `sequence`. Final temperature acts on completed scores in both
modes. No length normalisation is used: longer spellings may be penalised.
Code scoring reduces spelling-length effects but introduces code/order bias.

The synthetic demo makes the distinction testable:

| Option | Original joint probability | Sequence mode | Constrained mode |
|---|---:|---:|---:|
| refund | .06 | .105263 | .074074 |
| refund status | .24 | .421053 | .592593 |
| technical | .27 | .473684 | .333333 |

## Limits and verification

Probabilities are relative to the supplied categories, even if none fits well.
Include an `other` category when appropriate. Confidence is 1 minus normalized
Shannon entropy; it is not a calibrated probability of correctness. Score is the
probability-weighted rubric index; Noul returns the relative probability of true.
Question wording, label spelling, and model precision can affect the result.

Local context limits come from the checkpoint. Requests evaluate each question
independently and sequentially. Input usage sums prompt lengths across questions;
local scoring generates no output text, so output usage is zero. An injected
classifier may report null input usage when no count is available.

```bash
uv sync --extra test
uv run python -m unittest discover -s tests -v
```

Tests cover scoring mathematics, cache isolation, typed requests and responses,
CLI validation, local/HTTP SDKs, official TypeSafe SDK interoperability,
LangChain/LangGraph, authentication, and schema validation. Optional runtime tests
require their respective HF or MLX dependencies and hardware.

## Sources

- [TypeSafe's Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev): describes its architecture, parallel sampler and RLCD training.
- [Hugging Face model outputs](https://huggingface.co/docs/transformers/main_classes/output): causal-LM logits.
- [Qwen demonstration model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): loading and chat-template usage.
- [LangChain ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai): chat interface and logprobs metadata.
- [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api): state updates and conditional edges.

Version 0.1.0. An independent educational prototype; no affiliation with TypeSafe.
