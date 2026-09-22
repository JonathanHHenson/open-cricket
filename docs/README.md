# Open Cricket documentation

Open Cricket answers typed questions about text or structured JSON using a causal
language model. Choice selects a category, Score evaluates an ordered rubric,
and Noul returns the relative probability of a yes/true answer.

## User guides

Read the [v0.4.1 release notes](releases/v0.4.1.md) for faster MLX image scoring
and the Qwen prompt fix. The [v0.4.0 notes](releases/v0.4.0.md) cover image inputs
and vision-language model support. The [v0.3.1 notes](releases/v0.3.1.md) cover
clearer Noul prompts and expanded CLI examples; the [v0.3.0 notes](releases/v0.3.0.md)
cover answer codes and multi-question performance.

| Guide | Use it to |
| --- | --- |
| [Running the server](server.md) | Install, configure, operate, and troubleshoot the service |
| [API reference](api.md) | Build requests and interpret responses and errors |
| [Python SDK](sdk.md) | Run locally, connect over HTTP, and use framework integrations |
| [Migrating from Jev](jev-migration.md) | Adapt an existing TypeSafe/Jev application |

The CLI, SDKs, and HTTP API share the `state`, `model`, and `questions` request.
[The support example](../examples/support.json) demonstrates all three types.
Run repository commands in these guides from the checkout root using Python
3.10 or later and `uv`.

For a first local classification without a server:

```bash
uv sync --extra hf
uv run open-cricket --input examples/support.json --pretty
```

The first run downloads the selected checkpoint. To try synthetic scoring
without downloading a model:

```bash
uv sync
uv run open-cricket --demo --pretty
```

Probabilities compare supplied alternatives; they do not establish correctness.
Include an `other` choice where appropriate and evaluate your own examples before
using thresholds to automate decisions.

## Contributor guides

The [technical section](technical/README.md) explains architecture, scoring
mathematics, backend extension points, caching, and development checks.
