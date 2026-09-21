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
        return CachedScorer(self, tuple(prompt))


class CachedScorer:
    def __init__(self, backend, prompt):
        self.backend, self.prompt = backend, prompt
        self.cache = None
        self.previous = ()

    def __call__(self, prefix, allowed):
        ids = self.prompt + tuple(prefix)
        self.backend._validate(ids)
        common = 0
        if self.cache is not None:
            for left, right in zip(self.previous, ids):
                if left != right:
                    break
                common += 1
            # Leave at least one token to recompute its next-token logits.
            common = min(common, len(ids) - 1)
            if not self.backend._trim(self.cache, common):
                self.cache, common = None, 0
        try:
            logits, self.cache = self.backend._forward(ids[common:], self.cache, True)
            self.previous = ids
            return self.backend._probabilities(logits, allowed)
        except Exception:
            self.cache, self.previous = None, ()
            raise
