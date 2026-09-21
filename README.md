# LabelJudge

A basic Jev-inspired classifier using standard causal language models. Input is
a **message, question, and allowed options**. Output is a chosen option and a
distribution over those options. No generated explanation or parsed confidence
estimate is used. Includes Python, CLI, LangChain, LangGraph, and optional HTTP.

This reproduces an interface and an LLM scoring technique, not TypeSafe's Jev
architecture, training, calibrated confidence, performance, or full typed API.

## Quick start: local model, no server

Use Python 3.10+ in a fresh virtual environment. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hf,langgraph]'
python -m labeljudge.cli --input examples/support.json
```

Add `--pretty` for a concise, human-readable view of the input, questions,
answers, and option probabilities:

```bash
python -m labeljudge.cli --input examples/support.json --pretty
```

```text
Input: I was charged twice for my subscription yesterday. Please refund the extra payment; I still want to keep my account.

Result: route
Question: Which team should handle this message?
Answer: billing

Options:
  billing               72.34%
  other                 15.21%
  technical support      9.87%
  account cancellation   2.58%
```

The standard output remains detailed JSON for scripts and integrations. The
probabilities shown above are illustrative; actual values depend on the model.

Add `--time` to either output mode to report how long scoring the request took:

```bash
python -m labeljudge.cli --input examples/support.json --pretty --time
```

Pretty output ends with `Processing time: 1.234 seconds`; detailed JSON includes
`"processing_time_seconds": 1.234`. The timer starts after model initialization
and covers classification of every question in the input.

The first model run downloads `Qwen/Qwen2.5-0.5B-Instruct` through Hugging Face.
It is a small demonstration default, not an accuracy recommendation. Pass
`--model org/model` or a local checkpoint directory to choose another compatible
causal LM. CPU, CUDA, and Apple MPS are supported by the device selector. Loading
uses the model's default dtype and no quantisation; larger models need more RAM.
Use `--revision COMMIT` to pin a model snapshot. Remote model code is disabled.

Try the mathematics without dependencies or a model download:

```bash
python -m labeljudge.cli --demo
python -m labeljudge.cli --demo --pretty
python -m labeljudge.cli --demo --mode constrained
```

These use explicitly synthetic probabilities, not LLM predictions.

## Category descriptions

Each option can be either a label string or an object with a `label` and
`description`. Descriptions give the model classification context but do not
change the response contract: `answer` and each result option's `label` remain
strings.

```json
{
  "message": "I was charged twice. Please refund the extra payment.",
  "question": "Which team should handle this?",
  "options": [
    {
      "label": "billing",
      "description": "Charges, payments, subscriptions, and refunds"
    },
    {
      "label": "technical support",
      "description": "Product errors, outages, and troubleshooting"
    },
    {
      "label": "other",
      "description": "Only when none of the other categories applies"
    }
  ]
}
```

Plain string options remain supported, and strings and described categories may
be mixed in one request. Labels must be unique. When an object is used, both
`label` and a nonempty `description` are required.

## LangChain: direct local inference

```python
from labeljudge.hf import HuggingFaceBackend
from labeljudge.integrations import as_runnable

judge = as_runnable(HuggingFaceBackend("Qwen/Qwen2.5-0.5B-Instruct"))
request = {
    "message": "I was charged twice. Please refund the extra payment.",
    "question": "Which team should handle this?",
    "options": [
        {"label": "billing", "description": "Charges, payments, and refunds"},
        {"label": "technical support", "description": "Product errors and troubleshooting"},
        {"label": "other", "description": "Only when no other category applies"},
    ],
}
result = judge.invoke(request)
print(result["answer"])
print(result["options"])

# Standard Runnable operations:
answers = judge.batch([request, request])
pipeline = judge | (lambda result: result["answer"])
print(pipeline.invoke(request))
# Inside an async function: await judge.ainvoke(request)
```

`invoke`, `ainvoke`, `batch`, `abatch`, LCEL composition, and Runnable config
work. The local backend serialises calls through a lock; async uses a worker
thread and batch is not vectorised GPU batching. Each independent question has
its own prompt and distribution. The CLI example asks three questions about
the same message. Joint or conditional dependencies between answers are not
modelled automatically.

## Bring an existing LangChain chat model

```python
from labeljudge.integrations import chat_runnable

# llm is your configured LangChain chat model.
judge = chat_runnable(llm, top_logprobs=20)
result = judge.invoke(request)
```

This path maps options to answer codes `A`, `B`, etc., asks for a single code,
and reads first-token log probabilities. Long option names are still described
in the prompt, but **this path scores codes, not their multi-token spellings**.
It uses the same result keys needed for workflow routing. See
`examples/langchain_chat.py` for a ChatOpenAI example (separate provider package
and API credentials required).

The provider must support logprobs, return compatible LangChain response
metadata, and expose **every** candidate code at that position. `top_logprobs`
is provider-limited; requesting 20 does not guarantee all options appear.
Missing candidates raise `ProbabilityUnavailableError`; they are never assigned
zero or invented probabilities. A returned exact `A` token is distinct from
` A` or `\nA`. Only exact code-token events are scored. At most
`min(26, top_logprobs)` options are accepted. If a code requires multiple tokens
or the model starts with whitespace/reasoning, use the exact backend instead.

Default bind parameters are `logprobs=True`, `top_logprobs=20`, `max_tokens=1`,
and `temperature=1`. `bind_kwargs` can override provider-specific settings.
Do not add logit bias or token constraints if you want the original model's
distribution. Provider sampling temperature and LabelJudge's final score
temperature are separate. Some models reject these parameters or cannot
produce a visible answer within one output token; no universal compatibility
claim is made. Native provider errors propagate. Config/callbacks are forwarded
to nested calls; the chat path has native asynchronous invocation.

## LangGraph: use as a node

```python
from labeljudge.integrations import graph_node

builder.add_node("classify", graph_node(judge))
# State contains message, question, options, and a classification field.
# The node updates classification and leaves unrelated state untouched.
# Route on state["classification"]["answer"].
```

`examples/langgraph_workflow.py` is a complete StateGraph with conditional
routing. Swap the Runnable for local inference, a compatible chat model, or a
remote LabelJudge client without changing the graph.

## Optional serving endpoint

You do not need a service for Python/LangGraph usage. Use one when multiple
applications share a loaded model or other languages need access:

```bash
python -m pip install -e '.[hf,server]'
export LABELJUDGE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
uvicorn labeljudge.server:app --host 127.0.0.1 --port 8000
```

Model loads once at startup. `LABELJUDGE_DEVICE` and `LABELJUDGE_REVISION` are
optional. Set `LABELJUDGE_API_KEY` to require bearer authentication on
`/classify`; it is unauthenticated if unset. This is a basic local service,
without TLS termination, rate limiting, production batching or autoscaling.
Use one process per model/GPU; extra workers load extra copies.

```bash
curl http://127.0.0.1:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"message":"I was charged twice","question":"Which team?","options":["billing","technical support","other"]}'
```

`GET /health` reports readiness; `/docs` exposes the generated API docs.
For an authenticated server add `Authorization: Bearer YOUR_KEY`.

Remote Python / LangChain client:

```python
from labeljudge.client import Client

client = Client("http://127.0.0.1:8000")  # api_key=... if configured
result = client.invoke(request)
judge = client.as_runnable()
```

To serve an already configured chat or custom backend, define your own
`app = create_app(judge)` using `from labeljudge.server import create_app`.
No backend-specific logic is required in HTTP clients. The prototype serves
one configured classifier per process; it does not dynamically download models
named by incoming requests.

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

## Backend support and limits

| Route | Multi-token label likelihood | Requirement |
|---|---|---|
| Local Hugging Face | Yes, canonical token paths | Causal LM, compatible tokenizer, logits, explicit EOS |
| LangChain chat adapter | Uses single-token codes instead | All code logprobs returned in compatible metadata |
| Custom backend | Yes | Implement `TokenBackend` in `labeljudge/backend.py` |
| Text-only / no-logprobs API | Unavailable | Another model/API capability is needed |

A generic chat API cannot in general score arbitrary forced continuations.
An assistant message is not necessarily an assistant prefill. We do not pretend
that appending partial labels as messages reproduces the same conditional
distribution. Full-logit serving runtimes can be integrated via `TokenBackend`;
no vLLM, llama.cpp, Ollama or MLX adapter is bundled yet.

The simple HF implementation recomputes the prompt at each unique prefix. It
shares trie work but **does not share transformer KV caches**. This favours
clarity over performance and makes it much slower than an optimised server.
Next improvements would be prompt KV reuse, safe branch-cache handling, and
batched teacher-forced sequence scoring. Scoring known sequences can be batched
without sampling one token at a time. None of this recreates Jev's training.

Probabilities sum to one over the candidate list, even when every option is a
poor fit. Include `other`/`unclear` where useful. `candidate_log_mass` reports
the original probability mass of the scored answer paths, but is not a semantic
confidence measure. Evaluate accuracy, NLL, Brier score and calibration on
held-out task data before choosing decision thresholds. Temperature is a knob,
not automatic calibration. Prompt wording, label order and tokenisation affect
results. Quoting message data is not a guarantee against prompt injection.

## Tests and verification

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
```

Tests cover hand-computed multi-token probabilities, shared prefixes,
termination, local-vs-global normalisation, numerical stability, input errors,
actual LangChain sync/async/batch/LCEL operations, actual LangGraph state updates
and conditional routing, missing-provider-probability errors, and the HTTP
contract/authentication. Framework tests use a deterministic model stub so no
model download, API key or paid inference is needed. The HF adapter and a live
cloud model have not been run in the authoring environment; model quality and
latency are unbenchmarked. See `TEST_RESULTS.txt` for the captured run.

## Sources

- [TypeSafe's Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev): describes its architecture, parallel sampler and RLCD training.
- [Hugging Face model outputs](https://huggingface.co/docs/transformers/main_classes/output): causal-LM logits.
- [Qwen demonstration model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): loading and chat-template usage.
- [LangChain ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai): chat interface and logprobs metadata.
- [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api): state updates and conditional edges.

Version 0.1.0. An independent educational prototype; no affiliation with TypeSafe.
