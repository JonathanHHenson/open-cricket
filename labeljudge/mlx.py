"""Optional MLX-LM adapter for Apple silicon, including converted quantized models."""
from .local import LocalBackend


class MLXBackend(LocalBackend):
    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct", device="auto", revision=None):
        if device not in {"auto", "mps"}:
            raise ValueError("MLX backend requires device auto or mps")
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tokenizer, config = load(
            model, revision=revision, return_config=True,
            tokenizer_config={"trust_remote_code": False})
        self.eos = self.tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Model tokenizer must define an EOS/end-of-turn token")
        limits = [config.get("max_position_embeddings"), self.tokenizer.model_max_length]
        self.limit = min(x for x in limits if isinstance(x, int) and x > 0)

    def _forward(self, ids, cache, use_cache):
        from mlx_lm.models.cache import make_prompt_cache
        if cache is None and use_cache:
            cache = make_prompt_cache(self.model)
        logits = self.model(self.mx.array([ids]), cache=cache)[0, -1].astype(self.mx.float32)
        return logits, cache

    def _trim(self, cache, length):
        from mlx_lm.models.cache import KVCache
        # Restrict rollback to ordinary KV caches: recurrent/windowed states
        # may have discarded history even if their API advertises trimming.
        if not cache or any(type(layer) is not KVCache for layer in cache):
            return False
        for layer in cache:
            layer.trim(layer.offset - length)
        return True

    def _select(self, logits, allowed):
        log_probs = logits - self.mx.logsumexp(logits)
        return log_probs[self.mx.array(list(allowed))].tolist()
