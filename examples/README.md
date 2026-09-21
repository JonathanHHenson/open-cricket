# CLI examples

Run any request from the repository root:

```bash
uv run open-cricket --input examples/product-review.json --pretty --time
uv run open-cricket --input examples/incident-triage.json --pretty --time
uv run open-cricket --input examples/multilingual-support.json --pretty --time
uv run open-cricket --backend hf --mode sequence --input examples/qwen35-image.json --pretty --time
uv run open-cricket --input examples/structured-order.json --pretty --time
uv run open-cricket --input examples/many-candidates.json --pretty --time
```

| File | What it exercises |
| --- | --- |
| `support.json` | Choice, Score, and Noul in one support-routing request |
| `product-review.json` | Mixed sentiment, a five-level score, and custom Noul criteria |
| `incident-triage.json` | Structured JSON state and operational severity |
| `multilingual-support.json` | Unicode input and multilingual routing |
| `qwen35-image.json` | Qwen 3.5 vision classification using a bundled local PNG |
| `structured-order.json` | Nested objects, arrays, and fulfillment decisions |
| `many-candidates.json` | Forty Choice candidates and multi-character answer codes |

Use `--backend mlx` on Apple silicon to try MLX, or `--mode sequence` to compare
full-label scoring with the default answer-code mode:

```bash
uv run open-cricket --backend mlx --input examples/incident-triage.json --pretty --time
uv run open-cricket --input examples/many-candidates.json --mode sequence --pretty --time
```

For Qwen 3.5 on MLX, prefer full-label scoring for semantic Choice questions.
The small model can strongly prefer the first positional answer code regardless
of the option text:

```bash
uv run open-cricket --backend mlx \
  --model mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
  --mode sequence \
  --input examples/multilingual-support.json \
  --pretty
```

The first local run may download the configured model. Keep the process alive for
repeated measurements; launching the CLI again reloads the model.

`qwen35-image.json` works with Hugging Face or MLX on Apple silicon. Run it from
the repository root so its relative image path resolves:

```bash
uv run open-cricket \
  --backend hf \
  --model Qwen/Qwen3.5-0.8B \
  --mode sequence \
  --input examples/qwen35-image.json \
  --pretty
```

For MLX image inference, install the vision dependencies and select `mlx`:

```bash
uv sync --extra mlx
uv run open-cricket --backend mlx --mode sequence \
  --input examples/qwen35-image.json --pretty --time
```

Use the original Qwen checkpoint or a vision-capable MLX conversion; a conversion
that removes the vision weights cannot accept images.
For Qwen2.5-VL, replace the model with `Qwen/Qwen2.5-VL-3B-Instruct` or
`mlx-community/Qwen2.5-VL-3B-Instruct-4bit` using `--model`.

The example uses `sequence` mode to avoid the compact model's positional answer-code
bias. Its image is stored at `examples/assets/red-apple.png` and can be replaced with
another local path, image URL, or data URL.
