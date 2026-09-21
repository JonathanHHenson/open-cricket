# CLI examples

Run any request from the repository root:

```bash
uv run open-cricket --input examples/product-review.json --pretty --time
uv run open-cricket --input examples/incident-triage.json --pretty --time
uv run open-cricket --input examples/multilingual-support.json --pretty --time
uv run open-cricket --input examples/structured-order.json --pretty --time
uv run open-cricket --input examples/many-candidates.json --pretty --time
```

| File | What it exercises |
| --- | --- |
| `support.json` | Choice, Score, and Noul in one support-routing request |
| `product-review.json` | Mixed sentiment, a five-level score, and custom Noul criteria |
| `incident-triage.json` | Structured JSON state and operational severity |
| `multilingual-support.json` | Unicode input and multilingual routing |
| `structured-order.json` | Nested objects, arrays, and fulfillment decisions |
| `many-candidates.json` | Forty Choice candidates and multi-character answer codes |

Use `--backend mlx` on Apple silicon to try MLX, or `--mode sequence` to compare
full-label scoring with the default answer-code mode:

```bash
uv run open-cricket --backend mlx --input examples/incident-triage.json --pretty --time
uv run open-cricket --input examples/many-candidates.json --mode sequence --pretty --time
```

The first local run may download the configured model. Keep the process alive for
repeated measurements; launching the CLI again reloads the model.
