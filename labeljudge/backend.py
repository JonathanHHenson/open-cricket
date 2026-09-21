"""Implement this small protocol to add another full-logit model runtime."""
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
