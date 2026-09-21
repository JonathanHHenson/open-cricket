# Open Cricket versus Simple Jev — 2026-09-21

This compares Open Cricket `2ea8391` with the local Simple Jev checkout
`b02aa81c915a8193759b3cd33fef74721d6e005b`. It is a comparison of these revisions,
not a claim about subsequent upstream releases.

## Method

Both used Qwen/Qwen2.5-1.5B-Instruct, float32, PyTorch 2.14.0, and the same Mac's
MPS GPU. Timings cover in-process validation, prompt construction, model scoring,
and response assembly. They exclude loading, warmup, HTTP transport, server
queues, and concurrent traffic. Each implementation retains its own prompt.

The paired run used the same model instance under Transformers 4.57.6, alternating
execution order across seven warmed repeats. Because Simple Jev declares
Transformers 5 as a dependency, it was also measured separately under its supported
5.17.0 environment with seven warmed repeats. Those measurements were within 1%
of the paired run. GPU benchmarks ran sequentially to avoid mutual interference.

## Four-choice routing latency

| Questions per request | Open Cricket | Simple Jev, supported runtime | Open Cricket speedup |
|---|---:|---:|---:|
| 1 | 60 ms | 169 ms | 2.81× |
| 4 | 131 ms | 347 ms | 2.65× |
| 16 | 338 ms | 1,116 ms | 3.30× |

Both returned `billing` for all routing questions. Both implementations now
reuse shared prefixes and batch question suffixes. Open Cricket's smaller prompt
is a major difference: the single-question request uses 153 tokens versus 478.
It also skips cache setup for single-question, single-token scoring; Simple Jev
uses a prefix prefill plus suffix forward even for one question. These are
application-level results, not an isolated comparison of identical model inputs.

The reported multi-question usage counts are not directly comparable: Open
Cricket sums logical prompt lengths; Simple Jev reports the token-prefix union.

## Scope and tradeoffs

### Candidate scaling

Three warmed repeats, one question, synthetic `candidate N` labels:

| Candidates | Open Cricket | Simple Jev, shared runtime | Longest scored code path |
|---|---:|---:|---:|
| 50 | 146 ms | 639 ms | 1 token |
| 62 | 187 ms | Rejected | 1 token |
| 63 | 190 ms | Rejected | 1 token |
| 128 | 3,427 ms | Rejected | 3 tokens including EOS |
| 256 | 6,930 ms | Rejected | 3 tokens including EOS |

Qwen encodes `AA` as a single token, so 63 candidates remain on the fast path.
The transition depends on actual tokenization, not simply the number of code
characters. These generic labels differ from the four-choice routing workload.
The larger-set measurements validate execution and latency, not classification
quality across hundreds of meaningful categories.

### Small accuracy smoke check

Twelve synthetic routing messages were evaluated in normal and reversed option
order: Open Cricket was correct on 21/24, Simple Jev on 17/24, with 20/24 prediction
agreement. Reversing options changed Open Cricket's prediction on 3/12 messages
and Simple Jev's on 1/12. These cases previously informed Open Cricket prompt
development, so they are not an independent accuracy benchmark. The results
show order sensitivity and do not establish a general accuracy ranking.

### Capability differences

Open Cricket has no fixed local Choice candidate cap, supports complete-label
likelihoods, and offers both Hugging Face and MLX backends. Context and memory
limits still apply. Its provider chat adapter has separate limits.

This Simple Jev revision allows 2–50 Choice candidates and retains one scored
position per question. It also includes a specialized Laya backend and RFDT
training tools; neither was benchmarked here. These results do not rank those
backends, CUDA performance, default bfloat16 serving, or concurrent throughput.

When Open Cricket needs a multi-token code, it includes EOS in every candidate
path and traverses the cached trie. That changes scoring semantics, introduces
length bias, and increases latency. Multi-question root-only batching is disabled
for requests containing such paths. Larger capacity is therefore not free.

## Reproduction and evidence

Run in Open Cricket's HF environment with a separate Simple Jev checkout:

```sh
HF_HUB_OFFLINE=1 python examples/benchmark_simple_jev.py --simple-jev /path/to/simple-jev --repeats 7
```

For the supported Simple Jev runtime check, run its Python interpreter with
`PYTHONPATH` containing the checkout root and its `hf-server` directory:

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=/path/to/simple-jev:/path/to/simple-jev/hf-server /path/to/simple-jev/.venv/bin/python examples/benchmark_simple_jev_supported.py
```

Raw measurements: [paired runtime](simple-jev-shared-runtime.json) and
[supported runtime](simple-jev-supported-runtime.json), plus
[candidate scaling and smoke predictions](simple-jev-expanded.json).
