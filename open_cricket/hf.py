"""Local Hugging Face causal-LM adapter; no top-k truncation or text generation."""

import copy
import importlib
import inspect

from .local import LocalBackend


class HuggingFaceBackend(LocalBackend):
    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct", device="auto", revision=None):
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        self.torch = torch
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = device
        # Small matrix batches can cost more than serial decoding on CPU.
        self._batch_chains = device != "cpu"
        self._batch_questions = device != "cpu"
        self._max_batch_size = 32
        self._max_batch_tokens = 32768
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=False
        )
        self.model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                model, revision=revision, trust_remote_code=False
            )
            .to(device)
            .eval()
        )
        self._last_logits = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        self.eos = self.tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Model tokenizer must define an EOS/end-of-turn token")
        limits = [
            getattr(self.model.config, "max_position_embeddings", None),
            self.tokenizer.model_max_length,
        ]
        self.limit = min(x for x in limits if isinstance(x, int) and x > 0)

    def _forward(self, ids, cache, use_cache):
        logits, cache = self._forward_many(ids, cache, use_cache, 1)
        return logits[0], cache

    def _forward_many(self, ids, cache, use_cache, count):
        tensor = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        length = len(ids) + (cache.get_seq_length() if cache is not None else 0)
        kwargs = {"logits_to_keep": count} if self._last_logits else {}
        with self.torch.inference_mode():
            output = self.model(
                input_ids=tensor,
                attention_mask=self.torch.ones(
                    (1, length), dtype=self.torch.long, device=self.device
                ),
                past_key_values=cache,
                use_cache=use_cache,
                **kwargs,
            )
            return output.logits[0, -count:].float(), output.past_key_values

    def _trim(self, cache, length):
        # Sliding-window/recurrent caches cannot safely restore arbitrary branches.
        from transformers.cache_utils import DynamicCache

        if not isinstance(cache, DynamicCache):
            return False
        layers = getattr(cache, "layers", None)
        if layers is None or any(type(layer).__name__ != "DynamicLayer" for layer in layers):
            return False
        cache.crop(length)
        return True

    def _select(self, logits, allowed):
        with self.torch.inference_mode():
            return self.torch.log_softmax(logits, dim=-1)[list(allowed)].cpu().tolist()

    def batch_next_logprobs(self, requests):
        """Score answer codes for several prompts using one shared-prefix prefill."""
        if not requests:
            return []
        if not getattr(self, "_batch_questions", True):
            return [self.next_logprobs(prompt, (), allowed) for prompt, allowed in requests]
        sequences = [tuple(prompt) for prompt, _ in requests]
        for sequence in sequences:
            self._validate(sequence)
        prefix_length = 0
        for tokens in zip(*sequences):
            if len(set(tokens)) != 1:
                break
            prefix_length += 1
        # Every row must retain at least one real suffix token to produce logits.
        prefix_length = min(prefix_length, min(map(len, sequences)) - 1)
        prefix = sequences[0][:prefix_length]
        lengths = [len(sequence) - prefix_length for sequence in sequences]
        max_batch_tokens = getattr(self, "_max_batch_tokens", 32768)
        if max(lengths) > max_batch_tokens:
            return [self.next_logprobs(prompt, (), allowed) for prompt, allowed in requests]
        cache = None
        if prefix:
            _, cache = self._forward(prefix, None, True)
            if cache is None or not hasattr(cache, "reorder_cache"):
                return [self.next_logprobs(prompt, (), allowed) for prompt, allowed in requests]

        pending = sorted(range(len(sequences)), key=lambda i: lengths[i], reverse=True)
        results = [None] * len(requests)
        with self.torch.inference_mode():
            while pending:
                width = lengths[pending[0]]
                limit = min(
                    getattr(self, "_max_batch_size", 32),
                    max_batch_tokens // width,
                )
                batch, pending = pending[:limit], pending[limit:]
                branch_cache = copy.deepcopy(cache)
                if branch_cache is not None:
                    branch_cache.reorder_cache(
                        self.torch.zeros(len(batch), dtype=self.torch.long, device=self.device)
                    )
                ids = self.torch.zeros(
                    (len(batch), width), dtype=self.torch.long, device=self.device
                )
                mask = self.torch.zeros(
                    (len(batch), prefix_length + width),
                    dtype=self.torch.long,
                    device=self.device,
                )
                for row, index in enumerate(batch):
                    suffix = sequences[index][prefix_length:]
                    ids[row, :len(suffix)] = self.torch.tensor(
                        suffix, dtype=self.torch.long, device=self.device
                    )
                    mask[row, :prefix_length + len(suffix)] = 1
                positions = self.torch.arange(
                    prefix_length, prefix_length + width, device=self.device
                ).unsqueeze(0).expand(len(batch), -1)
                last = self.torch.tensor(
                    [lengths[index] - 1 for index in batch], device=self.device
                )
                if self._last_logits:
                    keep, inverse = self.torch.unique(last, sorted=True, return_inverse=True)
                    extra = {"logits_to_keep": keep}
                else:
                    inverse, extra = last, {}
                output = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=branch_cache,
                    use_cache=True,
                    **extra,
                )
                selected = output.logits[
                    self.torch.arange(len(batch), device=self.device), inverse
                ].float()
                log_probs = self.torch.log_softmax(selected, dim=-1)
                for row, index in enumerate(batch):
                    allowed = requests[index][1]
                    values = log_probs[row, list(allowed)].cpu().tolist()
                    results[index] = dict(zip(allowed, values))
                del output, branch_cache, selected, log_probs
        return results
