"""Optional MLX-LM adapter for Apple silicon, including converted quantized models."""

from .local import LocalBackend


class MLXBackend(LocalBackend):
    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct", device="auto", revision=None):
        if device not in {"auto", "mps"}:
            raise ValueError("MLX backend requires device auto or mps")
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        loaded = load(
            model,
            revision=revision,
            return_config=True,
            tokenizer_config={"trust_remote_code": False},
        )
        self.model, self.tokenizer = loaded[:2]
        config = loaded[2] if len(loaded) == 3 else {}
        self.eos = self.tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Model tokenizer must define an EOS/end-of-turn token")
        limits = [config.get("max_position_embeddings"), self.tokenizer.model_max_length]
        self.limit = min(x for x in limits if isinstance(x, int) and x > 0)

    def _forward(self, ids, cache, use_cache):
        logits, cache = self._forward_many(ids, cache, use_cache, 1)
        return logits[0], cache

    def _forward_many(self, ids, cache, use_cache, count):
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.qwen2 import Model as Qwen2Model

        if cache is None and use_cache:
            cache = make_prompt_cache(self.model)
        inputs = self.mx.array([ids])
        # MLX-LM 0.28 Qwen2 projects every prompt position to the vocabulary.
        # Slice hidden states first; restrict this shortcut to the known class.
        if type(self.model) is Qwen2Model and getattr(self, "_last_logits", True):
            hidden = self.model.model(inputs, cache=cache)[:, -count:]
            if self.model.args.tie_word_embeddings:
                output = self.model.model.embed_tokens.as_linear(hidden)
            else:
                output = self.model.lm_head(hidden)
        else:
            output = self.model(inputs, cache=cache)[:, -count:]
        logits = output[0].astype(self.mx.float32)
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
