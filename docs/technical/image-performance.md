# MLX image scoring performance

For the pinned MLX-VLM runtime, open-cricket limits Qwen2.5-VL, Qwen3-VL, and
Qwen3.5 vocabulary projection to the positions being scored. Image embedding,
visual attention, position calculation, and language-model computation still run
on the complete prompt. Both tied embeddings and separate output heads retain
their original weights and quantization. Other language-model classes keep the
original projection behavior.

For a prompt with N tokens and vocabulary size V, default single-token answer
codes now require V output logits instead of N × V. A synthetic Metal benchmark
with 1,024 positions, hidden size 512, and vocabulary size 32,768 measured median
projection times of 3.96 ms before and 0.28 ms after (10 measured runs after two
warmups). This measures only vocabulary projection, not end-to-end latency.

Sequence scoring also combines consecutive answer positions into one full
forward pass, reducing repeated vision encoding. A three-position chain needs
one forward instead of three. Branches still recompute full prefixes without
reusing multimodal KV state. Image preprocessing still runs separately for each question.

`tests/test_mlx_images.py` compares full and selected projections through real,
tiny image models for all three families, with tied/untied and quantized weights.
It also checks restoration after exceptions and image forwarding during chain
scoring. These tests require the optional MLX runtime and Metal access.

## Shared image prefixes across questions

Qwen2.5-VL answer-code requests now reuse the image and common text prefix
across questions. Each question gets a separate copy of the prefix KV cache and
explicit multimodal positions for its text suffix. Caches are released after the
request, including on failure. This path requires identical image tensors and
grids and a shared prefix containing every complete image. Unsupported models,
different images, or incomplete image prefixes use independent scoring.

The unchanged `examples/apple-image.json`, using its specified
`Qwen/Qwen2.5-VL-3B-Instruct` model on MLX, measured:

| Request processing | Median seconds |
| --- | ---: |
| Independent questions (previous implementation) | 1.138 |
| Shared image prefix | 0.497 |

This is **2.29× faster**, or **56% less processing time**. Seven measured runs
alternated baseline and optimized order after one warmup per path; loading is
excluded. The request shares 407 prefix tokens across three questions. The
image, resolution, model weights, prompts, and reported token usage are unchanged.
Versions: MLX 0.32.2, MLX-VLM 0.3.12, Transformers 5.17.0.

Reduced-precision matrix operations can round differently when a prompt is
split into a cached prefix and suffix. The measured maximum probability change
was 0.010516 (1.052 percentage points), on the ripeness question. Object and
colour remained apple and red; ripeness changed from 0.949992 to 0.955355 on its
0–2 scale. This optimization does not promise bit-identical probabilities.
The tiny FP32/quantized regression models agree within 0.00002 in log probability.

Reproduce from the repository root:

```sh
.venv/bin/python -m examples.benchmark_images --repeats 7
```

The benchmark prints all timings and the maximum probability difference, checks
choice labels and token accounting, and fails above its configurable absolute
probability tolerance (default 0.02 for the BF16 checkpoint). For diagnostics,
setting `client.backend._batch_questions = False` restores independent scoring.
