"""Local Hugging Face causal-LM adapter; no top-k truncation or text generation."""
import inspect

from .local import LocalBackend


class HuggingFaceBackend(LocalBackend):
    def __init__(self, model="Qwen/Qwen2.5-0.5B-Instruct", device="auto", revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, trust_remote_code=False).to(device).eval()
        self._last_logits = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        self.eos = self.tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Model tokenizer must define an EOS/end-of-turn token")
        limits = [getattr(self.model.config, "max_position_embeddings", None),
                  self.tokenizer.model_max_length]
        self.limit = min(x for x in limits if isinstance(x, int) and x > 0)

    def _forward(self, ids, cache, use_cache):
        tensor = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        length = len(ids) + (cache.get_seq_length() if cache is not None else 0)
        kwargs = {"logits_to_keep": 1} if self._last_logits else {}
        with self.torch.inference_mode():
            output = self.model(
                input_ids=tensor,
                attention_mask=self.torch.ones((1, length), dtype=self.torch.long, device=self.device),
                past_key_values=cache, use_cache=use_cache, **kwargs)
            return output.logits[0, -1].float(), output.past_key_values

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
