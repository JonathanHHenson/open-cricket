# Backend development and caching

[Technical index](README.md) · [Protocol](../../open_cricket/backend.py)

## Required protocol

A backend implements three methods without needing a particular base class:

```python
def prompt_ids(self, system: str, form: str, images: Sequence[str] = ()) -> Sequence[int]: ...
def answer_ids(self, answer: str) -> list[int]: ...
def next_logprobs(self, prompt, prefix, allowed) -> dict[int, float]: ...
```

`prompt_ids` establishes the prompt boundary. `answer_ids` receives an already
JSON-quoted label and returns canonical tokens plus a terminator. Paths must be
unique and prefix-free. `next_logprobs` conditions on exactly `prompt + prefix`
and returns every requested token ID.

Normalize logits across the **entire vocabulary before selecting allowed IDs**.
Pre-masking changes sequence semantics. Do not sample, prune, use incomplete top-k
lists, or substitute zero for unknown probabilities. Follow the
[numerical contract](scoring.md#numerical-and-output-invariants).

Inject with `LocalClient(model="my-model", backend=backend)` or
`as_runnable(backend, model="my-model")`. Requests must match that name. To add a
selectable runtime, extend `load_backend`, add an optional dependency extra, and
keep imports lazy.

## Optional scoring sessions

`scorer(prompt)` may return a `(prefix, allowed)` callback. `classify` prefers it
over the required-method fallback. Create a new session per classification;
cache state must not survive between questions or requests.

The original two-argument `prompt_ids` method remains valid for custom text-only
backends. It is called with `images` only when a request actually supplies images.
Backends that support images may return `open_cricket.local.ModelPrompt`, which
keeps processor tensors alongside the prompt token IDs for the initial prefill.

MLX selects MLX-VLM when the checkpoint config contains `vision_config`, otherwise
MLX-LM. VLM scoring recomputes complete prefixes with the image tensors for each
candidate, avoiding rollback of recurrent and image-position state. This is slower
than cache reuse but keeps branch scores independent. Text-only MLX-LM models
retain the existing request-local cache optimization.

The callback may additionally provide `score_chain(requests)`, returning one
probability mapping per `(prefix, allowed)` pair in order. Requests are consecutive
prefixes along a deterministic trie stretch, with an optional final branching
node. The scorer submits at most 16 positions together. All original token
probabilities, including EOS, must be returned; full-vocabulary normalization
still applies independently at each position. Callbacks without this method
continue to score one node at a time. `model_calls` retains its historical meaning
of scored trie nodes, not physical forward passes.

A backend may also expose `batch_next_logprobs(requests)` for request-level
answer-code batching. Each item is `(prompt_ids, allowed_token_ids)` and the
result must contain one full-vocabulary-normalized token mapping per item in the
same order. `classify_many` uses this only when a request has multiple questions
in `answer_codes` mode and all selected paths contain one token. Requests with
multi-token codes, other modes, and other backends retain sequential behavior.

The Hugging Face GPU implementation finds the exact common token prefix, leaves
at least one suffix token per question, prefills the prefix once, and deep-copies
and broadcasts its reorderable cache into suffix batches. Right padding is masked
and each row's actual final position supplies its logits. Batches are limited to
32 rows and 32,768 padded suffix tokens. CPU disables question batching because
small matrix batches can cost more than serial evaluation. Input usage retains
the public sum-of-question-prompts definition rather than reporting reduced
physical computation.

The shared [CachedScorer](../../open_cricket/local.py) works as follows:

1. Validate the requested prompt-plus-prefix against the context limit.
2. Find its longest common prefix with the previous evaluation when cached.
3. Retain at most the first scored prefix's length minus one, leaving its final
   token and the rest of the chain to recompute next-token logits.
4. Trim to the retained length. If unsupported, discard the cache and recompute.
5. Forward the uncached suffix, retain cache/tokens, then normalize probabilities
   at each requested position. Backends without `_forward_many` use serial calls.
6. Clear session state and re-raise if forwarding or probability extraction fails.

Traversal can move to siblings, ancestors, and repeated prefixes, so an append-only
cache is insufficient. Every evaluated prompt-plus-prefix is checked; no silent
truncation occurs.

## Shared tokenizer behavior

`LocalBackend` uses the tokenizer chat template and assistant-generation boundary
when available, otherwise a plain-text prompt with special tokens enabled.
Answers are separately encoded without special-token insertion and must round-trip
exactly. Special-token IDs inside labels are rejected before EOS is appended.
Both adapters require an EOS and derive context bounds from model/tokenizer limits.

## Hugging Face implementation

The adapter disables remote model code, loads a causal LM, moves it to the chosen
device, and uses evaluation/inference mode. Automatic device selection prefers
CUDA, then MPS, then CPU. Attention masks cover cached and new tokens; scored-position
logits are converted to float32 before full-vocabulary log-softmax.

If `forward` explicitly supports `logits_to_keep`, the adapter requests just the
scored positions' logits (one for ordinary calls). Chain scoring is enabled on
GPU; CPU retains serial cached scoring because small chain batches regressed
the measured workload. Rollback only supports `DynamicCache` with ordinary
`DynamicLayer` layers. Other formats recompute the full prefix. A cache exposing
a crop method is not necessarily safe: sliding/recurrent state may have lost
history. Older supported Transformers versions may miss the optimized path.

## MLX implementation

MLX requires Apple silicon with accessible Metal and accepts devices `auto`/`mps`.
It loads through MLX-LM, creates a prompt cache per session, and normalizes float32
scored-position logits across the vocabulary. Only exact ordinary `KVCache` layers
are trimmed; unsupported layouts recompute.

For the exact MLX-LM 0.28 `qwen2.Model` class, hidden states are sliced to the
scored positions before the vocabulary projection. This avoids calculating
unused prompt logits while retaining the full vocabulary at every scored
position. Both tied embeddings and separate output heads are supported, including
their quantized forms. Other model classes (including subclasses and wrappers)
use their normal forward method. This shortcut must be reviewed if MLX-LM's
Qwen2 implementation or the supported dependency range changes.

Checkpoint weights determine precision, including quantized models. Float32 logit
normalization does not remove differences from lower-precision forward arithmetic.
Compare cached and full-prefix results before reporting performance improvements.

## Extension checks

Cover shared prefixes, multiword/quoted labels, EOS, sibling/ancestor rollback,
repeated prefixes, cache isolation, unsupported-cache fallback, context overflow,
and missing/nonfinite probabilities. Compare both scoring modes with an uncached
baseline. [Local tests](../../tests/test_local.py) provide toy and miniature real
model checks without downloads.

The backend itself does not acquire the client's lock. Direct callers must
provide synchronization for runtime state that is not thread-safe.
