"""Implement this small protocol to add another full-logit model runtime.

Backends may additionally expose scorer(prompt), returning a request-local
(prefix, allowed) callback. classify uses it for cache reuse when available.
"""
from typing import Protocol, Sequence, Mapping


class TokenBackend(Protocol):
    def prompt_ids(self, system: str, form: str) -> Sequence[int]: ...

    def answer_ids(self, answer: str) -> Sequence[int]:
        """Canonical answer tokens followed by an explicit terminal token."""
        ...

    def next_logprobs(self, prompt: Sequence[int], prefix: tuple[int, ...],
                      allowed: tuple[int, ...]) -> Mapping[int, float]:
        """Full-vocabulary-normalised, finite log probabilities for ALL allowed IDs.

        Condition on the exact prompt + prefix. Do not pre-mask the vocabulary,
        sample, prune candidates, or omit requested tokens because of top-k.
        """
        ...


def load_backend(backend="hf", model="Qwen/Qwen2.5-0.5B-Instruct", device="auto", revision=None):
    """Load the selected runtime without importing unselected optional dependencies."""
    if backend == "hf":
        from .hf import HuggingFaceBackend
        return HuggingFaceBackend(model, device, revision)
    if backend == "mlx":
        from .mlx import MLXBackend
        return MLXBackend(model, device, revision)
    raise ValueError("backend must be hf or mlx")
