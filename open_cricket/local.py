"""Shared tokenization and request-local cache traversal for local runtimes."""

import math
from collections.abc import Mapping
from typing import Any


class ModelPrompt(tuple):
    """Token IDs plus processor-created tensors needed for a multimodal prefill."""

    model_inputs: dict[str, Any]

    def __new__(cls, ids, model_inputs=None):
        value = super().__new__(cls, ids)
        value.model_inputs = model_inputs or {}
        return value


class LocalBackend:
    tokenizer: Any
    eos: int
    limit: int
    _last_logits: bool
    _batch_chains: bool

    def prompt_ids(self, system, form, images=()):
        if images:
            raise ValueError("This backend does not support image inputs")
        if self.tokenizer.chat_template:
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": form}],
                tokenize=True,
                add_generation_prompt=True,
                # Classification scores the immediate next token, so a reasoning
                # preamble would score <think> rather than an answer candidate.
                enable_thinking=False,
            )
            # Transformers 5 returns BatchEncoding by default; 4.x returned IDs directly.
            return encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
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

    def _forward(self, ids, cache, use_cache, model_inputs=None):
        raise NotImplementedError

    def _trim(self, cache, length):
        raise NotImplementedError

    def _select(self, logits, allowed):
        raise NotImplementedError

    def next_logprobs(self, prompt, prefix, allowed):
        ids = tuple(prompt) + tuple(prefix)
        self._validate(ids)
        model_inputs = getattr(prompt, "model_inputs", None)
        if model_inputs:
            logits, _ = self._forward(ids, None, False, model_inputs=model_inputs)
        else:
            logits, _ = self._forward(ids, None, False)
        return self._probabilities(logits, allowed)

    def _probabilities(self, logits, allowed):
        selected = self._select(logits, allowed)
        if any(not math.isfinite(x) for x in selected):
            raise ValueError("Model returned non-finite token probabilities")
        return dict(zip(allowed, selected))

    def scorer(self, prompt):
        """Create an isolated scoring session; never retain another request's KV state."""
        session = CachedScorer(self, prompt)
        return session if getattr(self, "_batch_chains", True) else session.__call__


class CachedScorer:
    def __init__(self, backend, prompt):
        self.backend, self.prompt = backend, tuple(prompt)
        self.model_inputs = getattr(prompt, "model_inputs", None)
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
                kwargs = (
                    {"model_inputs": self.model_inputs} if common == 0 and self.model_inputs else {}
                )
                logits, self.cache = forward_many(
                    ids[common:], self.cache, True, len(requests), **kwargs
                )
            else:
                kwargs = (
                    {"model_inputs": self.model_inputs} if common == 0 and self.model_inputs else {}
                )
                last, self.cache = self.backend._forward(ids[common:], self.cache, True, **kwargs)
                logits = [last]
            self.previous = ids
            return [
                self.backend._probabilities(row, allowed)
                for row, (_, allowed) in zip(logits, requests)
            ]
        except Exception:
            self.cache, self.previous = None, ()
            raise
