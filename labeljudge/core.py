"""Dependency-free exhaustive token-trie scoring. No sampling or pruning."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def logsumexp(values):
    values = list(values)
    m = max(values)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in values))


@dataclass
class Node:
    children: dict = field(default_factory=dict)
    label: str | None = None


def score_paths(
    paths: Mapping[str, Sequence[int]],
    next_logprobs: Callable[[tuple[int, ...], tuple[int, ...]], Mapping[int, float]],
    *,
    mode: str = "sequence",
    temperature: float = 1.0,
):
    """Score prefix-free paths including an explicit answer terminator.

    Callback returns ORIGINAL full-vocabulary log-softmax values for requested
    tokens, conditional on the prompt plus the supplied answer prefix.
    sequence: softmax(sum(original token log probabilities) / temperature).
    constrained: normalise allowed children at each node before multiplying.
    Temperature acts on completed-label scores in BOTH modes.
    """
    if mode not in {"sequence", "constrained"}:
        raise ValueError("mode must be sequence or constrained")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not paths:
        raise ValueError("At least one option is required")
    root = Node()
    for label, path in paths.items():
        if not isinstance(label, str) or not label.strip() or not path:
            raise ValueError("Labels and token paths must be nonempty")
        node = root
        for token in path:
            if node.label is not None:
                raise ValueError("Paths must be prefix-free; include a terminator")
            node = node.children.setdefault(token, Node())
        if node.label is not None or node.children:
            raise ValueError("Duplicate or prefix-overlapping token paths")
        node.label = label

    raw, selected, traces = {}, {}, {}
    calls = 0
    stack: list[tuple[Node, tuple[int, ...], float, float, list[dict[str, Any]]]] = [
        (root, (), 0.0, 0.0, [])
    ]
    while stack:
        node, prefix, raw_score, selected_score, trace = stack.pop()
        if node.label is not None:
            raw[node.label] = raw_score
            selected[node.label] = selected_score
            traces[node.label] = trace
            continue
        allowed = tuple(node.children)
        lp = next_logprobs(prefix, allowed)
        calls += 1
        for token in allowed:
            if token not in lp or not math.isfinite(lp[token]) or lp[token] > 1e-6:
                raise ValueError(
                    "Backend must return valid log probabilities for every requested token"
                )
        norm = logsumexp(lp[t] for t in allowed)
        if norm == -math.inf:
            raise ValueError("All allowed continuations have zero probability")
        for token, child in node.children.items():
            edge = lp[token] - norm if mode == "constrained" else lp[token]
            stack.append(
                (
                    child,
                    prefix + (token,),
                    raw_score + lp[token],
                    selected_score + edge,
                    trace
                    + [
                        {
                            "token_id": token,
                            "log_probability": lp[token],
                            "scoring_log_probability": edge,
                        }
                    ],
                )
            )
    scaled = {label: selected[label] / temperature for label in paths}
    norm = logsumexp(scaled.values())
    if norm == -math.inf:
        raise ValueError("All completed answers have zero probability")
    rows = [
        {
            "label": label,
            "probability": math.exp(scaled[label] - norm),
            "log_likelihood": raw[label],
            "score": selected[label],
            "token_ids": list(paths[label]),
            "trace": traces[label],
        }
        for label in paths
    ]
    rows.sort(key=lambda row: row["probability"], reverse=True)
    return {
        "answer": rows[0]["label"],
        "options": rows,
        "mode": mode,
        "temperature": temperature,
        "model_calls": calls,
        "candidate_log_mass": logsumexp(raw.values()),
        "probability_note": "Relative to supplied options; not calibrated correctness confidence.",
    }
