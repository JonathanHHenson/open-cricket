# Contributing and testing

[Technical index](README.md)

## Setup

Use Python 3.10 or later. From the repository root:

```bash
uv sync --extra test
uv run python -m unittest discover -s tests -v
```

Most tests use synthetic backends, in-process HTTP transports, or mocked clients.
They need no checkpoint downloads or hosted inference. Optional runtime tests
skip when their dependencies are unavailable.

When changing a runtime, include its extra:

```bash
uv sync --extra test --extra hf
uv run python -m unittest discover -s tests -p 'test_local.py' -v
```

Use `--extra mlx` alongside `--extra test` on supported Apple silicon. Keep all
needed extras in each `uv sync` command, which can remove unselected optional
dependencies. MLX also needs accessible Metal, potentially unavailable in
restricted execution environments.

## Test coverage

| Module | Behavior protected |
| --- | --- |
| [test_core.py](../../tests/test_core.py) | Scoring mathematics, temperature, underflow, paths, rendering |
| [test_local.py](../../tests/test_local.py) | Cache rollback/isolation/fallback, limits, miniature HF/MLX models |
| [test_sdk.py](../../tests/test_sdk.py) | Shared example contract, typed calls, invalid input, CLI model override |
| [test_client.py](../../tests/test_client.py) | HTTP requests and validation before network access |
| [test_cli.py](../../tests/test_cli.py) | Pretty output, timing, demo, runtime factory |
| [test_server.py](../../tests/test_server.py) | Basic routes and validation |
| [test_systemone.py](../../tests/test_systemone.py) | Mixed questions, auth, models, limits, usage, TypeSafe interoperability |
| [test_integrations.py](../../tests/test_integrations.py) | Runnable sync/async/batch/config, provider probabilities, graph updates |

Narrow discovery using `-p` during iteration, then run the full suite before
submitting behavior changes. Report skipped tests and why, especially for
hardware-dependent paths.

## Changing behavior

Preserve the strict public `state`/`model`/`questions` envelope unless deliberately
changing the contract. Add regression coverage at the failed boundary and update
the relevant [API](../api.md), [SDK](../sdk.md), or [server](../server.md) guide.
New question types require validation, conversion, response formatting, CLI
rendering, and shared local/HTTP examples.

For scoring changes, check hand-computed distributions in both modes. For runtime
changes, compare cached and uncached results and preserve lazy optional imports.
Performance changes must not silently drop candidates or normalize over only
allowed vocabulary entries.

## Debugging classifications

Call `open_cricket.questionnaire.classify(backend, message, question, options)` for
the rendered form, prompt count, token paths, traces, raw likelihoods, and candidate
mass. This lower-level `options` argument is a list of strings or exact
`{label, description}` objects, not the public criteria map. Descriptions here
must be nonblank strings or null. Avoid unnecessary logging of real user prompts.

Inspect rendered questions and criteria, then tokenization/termination, then
original versus mode-specific scores. High confidence alone does not show that
a classification is correct.

## Benchmarks

After installing the relevant runtime:

```bash
uv run python examples/benchmark_backend.py --device cpu --repeats 5
uv run python examples/benchmark_backend.py --backend mlx --repeats 5
```

The benchmark defaults to Qwen2.5-0.5B-Instruct, unlike the application's 1.5B
default. It downloads/loads a real checkpoint, warms it, then alternates cached
and uncached runs. Report checkpoint, device, precision, dependency versions,
workload, repeats, request timing, and probability differences. Load time is
excluded. Faster execution does not justify unexplained probability changes.

For documentation-only edits, check relative links, example schemas, and relevant
offline examples. Label illustrative outputs explicitly; do not run paid provider
examples merely to verify documentation.
