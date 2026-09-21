# Open Cricket

[![Test, build, and publish](https://github.com/JonathanHHenson/open-cricket/actions/workflows/publish.yml/badge.svg?branch=main)](https://github.com/JonathanHHenson/open-cricket/actions/workflows/publish.yml)
[![PyPI version](https://img.shields.io/pypi/v/open-cricket.svg)](https://pypi.org/project/open-cricket/)
[![Python versions](https://img.shields.io/pypi/pyversions/open-cricket.svg)](https://pypi.org/project/open-cricket/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![Open Cricket mascot](docs/assets/open-cricket-icon.svg)

Local structured decisions using causal language models. The CLI, Python SDK,
REST API, and LangChain integrations share one request contract:
**`state`, `model`, and a map of typed `questions`**. Responses contain `model`,
`answers`, and `usage`. Choice selects a category, Score evaluates a rubric,
and Noul returns the relative probability of a yes/true answer.

The default checkpoint is `Qwen/Qwen2.5-1.5B-Instruct`. Open Cricket is independent
of TypeSafe; its API follows Jev's general call shapes, but model predictions
and confidence calibration differ.

## Documentation

See the [documentation index](docs/README.md) for the [server guide](docs/server.md),
[API reference](docs/api.md), and [Python SDK guide](docs/sdk.md).
The [technical section](docs/technical/README.md) covers architecture, scoring,
backend development, and testing for contributors.

The package and command are `open-cricket`; Python imports use `open_cricket`.
See the [release guide](docs/technical/releases.md) for GitHub-to-PyPI setup.
The [v0.3.0 release notes](docs/releases/v0.3.0.md) cover extensible
case-insensitive answer codes, multi-question performance, and migration from
v0.2.0.

## Quick start: local model, no server

```bash
uv sync --extra hf
uv run open-cricket --input examples/support.json --pretty --time
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
uv run open-cricket --demo --pretty
uv run open-cricket --demo --mode constrained
```

## Direct Python SDK

```python
import json
from open_cricket import LocalClient, Choice, Score, Noul

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
export OPEN_CRICKET_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export OPEN_CRICKET_API_KEY="choose-a-local-server-key"
uv run uvicorn open_cricket.server:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $OPEN_CRICKET_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @examples/support.json
```

`GET /v1/models` lists the configured checkpoint. `GET /health` reports readiness;
`/docs` exposes OpenAPI documentation. Set `OPEN_CRICKET_BACKEND=mlx` to serve MLX.
Authentication is disabled if `OPEN_CRICKET_API_KEY` is unset. Requests select the
configured model by its exact name; they do not trigger model downloads.

```python
import json
from open_cricket import Client, Choice

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
from open_cricket.hf import HuggingFaceBackend
from open_cricket.integrations import as_runnable

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
Deterministic stretches of the answer trie are scored together in one causal
forward pass, up to 16 positions at a time. This combines opening quotes,
shared label prefixes, and unbranched suffixes without changing tokenization.
This is enabled for MLX and Hugging Face GPU inference; Hugging Face CPU retains
token-by-token cached scoring because chain batching slowed the measured workload.
Hugging Face models that support `logits_to_keep` compute only the scored
positions' vocabulary logits. No labels or answer tokens are skipped, and
normalisation still uses the full vocabulary. `model_calls` counts scored trie
nodes, so it does not decrease even though each call does much less work.
Custom backends implementing the original three-method protocol still work.
For Hugging Face cache acceleration, use Transformers 4.57; older supported
versions can fall back to full-prefix scoring.

For Apple silicon, install and select the optional MLX runtime:

```bash
uv sync --extra mlx
uv run open-cricket --backend mlx --input examples/support.json --pretty --time
```

MLX accepts compatible Hugging Face checkpoints and MLX-converted quantized
checkpoints via `--model`. The selected checkpoint determines weight precision;
quantization and lower precision can change probabilities and close rankings.
The MLX extra is restricted to Apple silicon macOS and uses MLX-LM 0.28.x to
remain compatible with this project's Transformers 4.x dependency. It requires
an accessible Metal GPU. Hugging Face remains the default runtime.

In Python, use `from open_cricket.mlx import MLXBackend` and pass `MLXBackend()`
to `classify` or `as_runnable`. For the HTTP service:

```bash
uv sync --extra mlx --extra server
OPEN_CRICKET_BACKEND=mlx uv run uvicorn open_cricket.server:app --host 127.0.0.1
```

Keep the model loaded between requests (for example through the HTTP service).
Starting the CLI for every message reloads the model. The existing runnable
serializes access to its model; async requests do not provide GPU batching.
Caches are isolated per classification and are not retained across requests.

### One-pass answer-code scoring

The default `answer_codes` mode prefers single-token `A–Z`, then `0–9` codes,
then maps the probabilities back to the original labels:

```bash
uv run open-cricket --backend mlx --input examples/support.json --pretty --time
```

```python
client = LocalClient(runtime="mlx")
```

The server, `classify`, and `as_runnable` also default to `answer_codes`;
request/response bodies stay unchanged. An explicit `OPEN_CRICKET_MODE` overrides
the server default.
It supports Choice, Score, and Noul. Letter codes are case-insensitive: the
probability mass of outputs such as `A` and `a` is combined before candidates are
normalized. After single-token alphanumerics run out, it uses longer codes (`AA`,
`AB`, …) and combines all case variants. There is no fixed Choice candidate cap;
model context and memory remain practical limits.

When all codes are single tokens, one forward scores them without EOS. Otherwise,
the cached trie scores complete codes including EOS, distinguishing `A` from `AA`. It
changes the scored events, so code/order bias can change predictions and
probabilities. Use `mode="sequence"`, CLI `--mode sequence`, or server
`OPEN_CRICKET_MODE=sequence` for full label scoring. Longer codes need additional
model work and introduce code-length bias.

Compare speed and predictions, including reversed candidate order, with:

```bash
uv run python examples/benchmark_answer_codes.py --backend mlx
uv run python examples/benchmark_answer_codes.py --device mps
```

The script uses 12 synthetic routing cases in two option orders; it is a smoke
evaluation, not a representative accuracy benchmark or a held-out dataset.
With the default 1.5B model, three timed repeats per case/order after warmup gave:

| Runtime | Label scoring | Answer codes | Speedup |
| --- | ---: | ---: | ---: |
| MLX Metal | 79 ms | 46 ms | 1.72× |
| Hugging Face MPS | 143 ms | 58 ms | 2.45× |

Both runtimes matched the expected label in 24/24 code evaluations versus 22/24
label evaluations. Code scoring used one scored node per question versus nine
for these labels. Reversing option order changed zero code winners and two label
winners. These same synthetic cases informed prompt development, so the results
are not independent evidence of accuracy. Individual candidate probabilities
differed by as much as 0.84; confidence thresholds need separate evaluation.

When one request contains multiple questions, the Hugging Face GPU backend
reuses the exact shared state prefix and batches the remaining question suffixes.
The default 1.5B model on MPS measured 238 ms → 133 ms for four questions (1.79×)
and 957 ms → 335 ms for sixteen (2.86×), using seven warmed repeats. Maximum
probability differences from independent forwards were 0.000000015 and 0.00000349,
respectively. Single-question requests keep the direct path. CPU, MLX, custom
backends, and full-label scoring continue to evaluate questions sequentially.

```bash
uv run python examples/benchmark_questions.py --device mps
```

### Measured performance

Local Qwen2.5-Instruct benchmarks with four options and 112 prompt tokens produced
these warmed median request times, excluding model load:

| Runtime / checkpoint | Previous cached scorer | Chain scoring | Additional speedup | Repeats |
| --- | ---: | ---: | ---: | ---: |
| Hugging Face MPS, 0.5B, float32 | 141 ms | 92 ms | 1.54× | 7 |
| Hugging Face MPS, default 1.5B, float32 | 271 ms | 172 ms | 1.57× | 5 |
| MLX Metal, 0.5B, checkpoint precision | 53 ms | 36 ms | 1.48× | 7 |
| MLX Metal, default 1.5B, checkpoint precision | 128 ms | 86 ms | 1.49× | 7 |

This is one local workload, not a general performance guarantee. Maximum absolute
probability changes versus the previous cached scorer were 0.00000171 for HF MPS
0.5B, 0.000000687 for HF MPS 1.5B,
0.02377 for MLX 0.5B, and 0.00784 for MLX 1.5B (about 0.78 percentage points).
Versus full-prefix scoring, MLX 1.5B differed by up to 0.04136. MLX's lower-precision
arithmetic varies between full-sequence, incremental, and batched evaluation;
close rankings can change. Float32 miniature-model tests verify both runtimes'
token probabilities to five decimal places and accumulated chain scores to four.

HF CPU chain scoring took 240 ms versus 211 ms for the previous cached scorer
(0.5B, 5 repeats), so CPU keeps the previous path by default. CUDA is supported
but was not benchmarked here. The benchmark keeps 0.5B as its default for
reproducibility; pass `--model Qwen/Qwen2.5-1.5B-Instruct` to measure the default model.

Reproduce the benchmark with:

```bash
uv run python examples/benchmark_backend.py --device cpu --repeats 5
uv run python examples/benchmark_backend.py --device mps --repeats 7
uv run python examples/benchmark_backend.py --backend mlx --repeats 5
```

An additional MLX Qwen2 optimization projects only the scored hidden states to
the vocabulary, avoiding unused prompt logits. Against chain scoring alone,
the default 1.5B checkpoint measured 86 ms → 83 ms for the 112-token prompt
(9 repeats) and 197 ms → 179 ms for a synthetic 546-token prompt (7 repeats).
Candidate probabilities were identical in those two comparisons. This shortcut
supports the exact MLX-LM Qwen2 class; other architectures use their normal path.

The benchmark rotates uncached, token-by-token cached, chain-scored, and fully
optimized runs after warmup and reports speed and probability differences.
`speedup_over_cached` compares all optimizations to token-by-token cache reuse;
`speedup_over_chains` isolates the MLX vocabulary-projection improvement.
Use `--chain-scoring on|off` to override the runtime default for experiments.
Use `--message-repeats 32` to reproduce the longer synthetic prompt.
Rust has not been introduced: eliminating
repeated model computation provides the demonstrated gain, while rewriting the
small Python trie would leave that model work unchanged.

## What happens mathematically?

In explicit `sequence` label-scoring mode, the questionnaire renders message data as a
quoted JSON string, followed by the question, allowed options, and `Answer:`.
The HF adapter uses the tokenizer's
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
The default is `answer_codes`; select `sequence` for full label likelihoods.
Final temperature acts on completed scores in both label-scoring
modes. No length normalisation is used: longer spellings may be penalised.
Code scoring reduces spelling-length effects but introduces code/order bias.

### Would prefilling the opening quote help?

Local scoring already generates zero output text tokens. If every answer starts
with the same quote **token**, its log probability is a common additive term:
omitting it preserves relative candidate probabilities, but changes raw likelihoods
and candidate mass. A quote character is not guaranteed to be a separate token;
retokenizing it as part of the prompt can change the scored events.

The chain optimization keeps the original token boundaries and scores the quote
alongside subsequent known tokens in one pass. It also combines deterministic
suffixes, keeping complete likelihoods, traces, and EOS scoring intact. Floating
point differences can still occur when batching positions, especially with
lower-precision weights; this is mathematical equivalence, not a bitwise guarantee.

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

Version 0.2.0. An independent educational prototype; no affiliation with TypeSafe.
