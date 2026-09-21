# Scoring mathematics and invariants

[Technical index](README.md) · [Implementation](../../open_cricket/core.py)

## Scored events

Each label becomes `json.dumps(label, ensure_ascii=False)`, canonically tokenized
and followed by one tokenizer EOS token. Answers are encoded without automatic
special tokens at a fixed prompt boundary. Shared token prefixes form a trie;
paths must be unique and prefix-free.

Scoring exhaustively evaluates these token-sequence events. It does not sum over
alternative tokenizations, whitespace, synonyms, or all possible end-of-turn
tokens. Descriptions appear only in the prompt.

## Sequence mode

For prompt `x` and label path `t(y)` including EOS, the backend supplies original
full-vocabulary-normalized log probabilities:

```text
L(y) = sum_j log P(t_j(y) | x, t_<j(y))
p(y) = exp(L(y) / T - logsumexp_z(L(z) / T))
```

Temperature `T` must be finite and positive. At `T = 1`, this conditions on the
supplied canonical completed-answer events. Other temperatures reshape completed
scores. There is no length normalization; label spelling and token count matter.

## Constrained mode

For prefix `u`, let `A(u)` be its allowed child tokens:

```text
edge(t, u) = log P(t | x, u) - logsumexp_a in A(u)(log P(a | x, u))
S(y) = sum_j edge(t_j(y), t_<j(y))
p(y) = softmax_y(S(y) / T)
```

This renormalizes at every branch. A single-child node contributes zero regardless
of its original probability. It differs from conditioning on complete answer
events and may change the winner. Temperature is applied after path scoring in
both modes.

The synthetic demo illustrates the difference:

| Label | Original mass | Sequence | Constrained |
| --- | ---: | ---: | ---: |
| refund | 0.06 | 0.105263 | 0.074074 |
| refund status | 0.24 | 0.421053 | 0.592593 |
| technical | 0.27 | 0.473684 | 0.333333 |

Run `uv run open-cricket --demo --pretty`, then add `--mode constrained`.
[Core tests](../../tests/test_core.py) verify these values.

## Traversal and diagnostics

An explicit depth-first stack explores all branches. Each non-leaf node causes
one callback, requesting every allowed child's probability. There is no sampling,
pruning, or top-k filtering. Prefix sharing avoids redundant callback evaluations.

| Low-level field | Meaning |
| --- | --- |
| `answer` | Highest-probability label |
| `options[].probability` | Completed score normalized across candidates |
| `options[].log_likelihood` | Sum of original log probabilities |
| `options[].score` | Mode-specific edge sum before temperature |
| `options[].token_ids`, `trace` | Canonical tokens and original/scoring edge values |
| `candidate_log_mass` | Log-sum-exp of original completed-event likelihoods, independent of mode/temperature |
| `model_calls` | Number of scored trie nodes |

`classify` adds `question`, rendered `form`, and `prompt_tokens`. The typed public
API omits these diagnostics. Caching reduces computation per callback, not the
reported `model_calls` count.

## Numerical and output invariants

The scorer rejects empty/overlapping/duplicate token paths, blank labels, invalid
modes, and invalid temperatures. Every requested token must have a finite log
probability no greater than `1e-6` (tolerance around zero). Negative infinity is
rejected, not treated as an acceptable missing branch. Log-sum-exp avoids ordinary
probability underflow during normalization; final exponentiation can round tiny
values to zero.

The typed formatter requires exact criterion coverage, finite probabilities in
`[0, 1]`, and a sum close to 1 using `math.isclose(..., abs_tol=1e-6)`. It restores
request criterion order, so ties pick the first label.

Score is `sum(index * probability)` over zero-based rubric indices. Noul is
`p(true)`. Choice and Score confidence are:

```text
H(p) = -sum_y p(y) * log(p(y))       # omit zero terms
confidence = clamp(1 - H(p) / log(K), 0, 1)
```

For `K = 1`, confidence is 1. Concentration is not calibrated correctness.
Provider answer-code scoring evaluates first-token codes, introduces code/order
bias, and does not score the same events as local canonical answer sequences.
