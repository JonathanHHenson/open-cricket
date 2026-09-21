# Scoring mathematics and invariants

[Technical index](README.md) · [Implementation](../../open_cricket/core.py)

## Scored events

In `sequence` and `constrained` modes, each label becomes
`json.dumps(label, ensure_ascii=False)`, canonically tokenized
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

## Local answer-code mode

`classify(..., mode="answer_codes")` maps categories in input order to `A–Z`,
`a–z`, then `0–9`, rendering each code with its original label and description.
Codes are case-sensitive. Single-token alphanumerics take priority, followed by
valid multi-token characters and longer codes (`AA`, `AB`, …, `AAA`, …).
There is no fixed candidate cap. Invalid or colliding single-character codes are
skipped; incompatible generated codes fail explicitly. Context and memory limits
still apply.

When all selected codes are single tokens, one uncached model forward supplies
full-vocabulary-normalized probabilities for
all code tokens. Only those first-token events are scored:

```text
L(y) = log P(code(y) | code_prompt)
p(y) = softmax_y(L(y) / T)
```

EOS, quotes, and label spellings are not part of the scored output path. The
model's subsequent continuation is unspecified. This differs from completed
label likelihoods and introduces code/order bias. Original labels are restored
in responses; Score and Noul use the resulting distribution as usual.
Low-level results expose `mode="answer_codes"`, one scored node, and each row's
`code`, single token ID, and trace. Candidate mass is the total first-code-token
mass, so it is not comparable to label mode's completed-answer mass.

When any selected code needs multiple tokens, every code includes the backend's
terminal token in its scored path. This distinguishes overlapping codes such as
`A` and `AA`. The cached trie scores joint code-plus-EOS likelihoods, normalizes
them over candidates, and restores original labels. Longer codes need more model
work and can receive lower scores from length bias; adding candidates can also
switch the scoring semantics of existing codes. Question batching applies only
when every code in the request is single-token.

`examples/benchmark_answer_codes.py` compares speed, predictions, probabilities,
and reversed candidate order on a small synthetic smoke dataset. It does not
establish general classification accuracy. Default classification uses
`answer_codes`. Select `sequence` explicitly for completed label likelihoods.
The low-level `score_paths` function and synthetic CLI demo
retain sequence scoring because they operate on explicit token paths.

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
Answer-code scoring introduces code/order bias and scores different events from
canonical label sequences. Provider scoring and local single-token sets score
first tokens; local multi-token sets score completed codes including EOS.
