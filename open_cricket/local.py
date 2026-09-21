"""Shared tokenization and request-local cache traversal for local runtimes."""

import math
from typing import Any


class LocalBackend:
    tokenizer: Any
    eos: int
    limit: int
    _last_logits: bool

    def prompt_ids(self, system, form):
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": form}],
                tokenize=True,
                add_generation_prompt=True,
            )
        return self.tokenizer.encode(system + "\n\n" + form + "\n", add_special_tokens=True)

    def answer_ids(self, answer):
        ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if any(t in self.tokenizer.all_special_ids for t in ids):
            raise ValueError("Options may not contain special tokens")
        if self.tokenizer.decode(ids, skip_special_tokens=False) != answer:
            raise ValueError("Option does not round-trip through this tokenizer")
        return ids + [self.eos]

    def _validate(self, ids):
        if not ids:
            raise ValueError("Prompt must contain at least one token")
        if len(ids) > self.limit:
            raise ValueError(f"Input exceeds context limit ({self.limit}); no silent truncation")

    def _forward(self, ids, cache, use_cache):
        raise NotImplementedError

    def _trim(self, cache, length):
        raise NotImplementedError

    def _select(self, logits, allowed):
        raise NotImplementedError

    def next_logprobs(self, prompt, prefix, allowed):
        ids = tuple(prompt) + tuple(prefix)
        self._validate(ids)
        logits, _ = self._forward(ids, None, False)
        return self._probabilities(logits, allowed)

    def _probabilities(self, logits, allowed):
        selected = self._select(logits, allowed)
        if any(not math.isfinite(x) for x in selected):
            raise ValueError("Model returned non-finite token probabilities")
        return dict(zip(allowed, selected))

    def scorer(self, prompt):
        """Create an isolated scoring session; never retain another request's KV state."""
        session = CachedScorer(self, tuple(prompt))
        return session if getattr(self, "_batch_chains", True) else session.__call__


class CachedScorer:
    def __init__(self, backend, prompt):
        self.backend, self.prompt = backend, prompt
        self.cache = None
        self.previous = ()

    def __call__(self, prefix, allowed):
        return self.score_chain([(prefix, allowed)])[0]

    def score_chain(self, requests):
        """Score consecutive known prefixes in one causal forward pass when supported."""
        forward_many = getattr(self.backend, "_forward_many", None)
        if forward_many is None and len(requests) > 1:
            return [self(prefix, allowed) for prefix, allowed in requests]
        ids = self.prompt + tuple(requests[-1][0])
        self.backend._validate(ids)
        # The first scored position must be recomputed, even on a cache hit.
        first_length = len(self.prompt) + len(requests[0][0])
        self.backend._validate(self.prompt + tuple(requests[0][0]))
        common = 0
        if self.cache is not None:
            for left, right in zip(self.previous, ids):
                if left != right:
                    break
                common += 1
            # Leave at least one token to recompute its next-token logits.
            common = min(common, first_length - 1)
            if not self.backend._trim(self.cache, common):
                self.cache, common = None, 0
        try:
            if forward_many is not None and len(requests) > 1:
                logits, self.cache = forward_many(ids[common:], self.cache, True, len(requests))
            else:
                last, self.cache = self.backend._forward(ids[common:], self.cache, True)
                logits = [last]
            self.previous = ids
            return [self.backend._probabilities(row, allowed)
                    for row, (_, allowed) in zip(logits, requests)]
        except Exception:
            self.cache, self.previous = None, ()
            raise
