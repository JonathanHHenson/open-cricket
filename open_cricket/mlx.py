"""Optional MLX-LM adapter for Apple silicon, including converted quantized models."""

from .local import LocalBackend, ModelPrompt


class MLXBackend(LocalBackend):
    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct", device="auto", revision=None):
        if device not in {"auto", "mps"}:
            raise ValueError("MLX backend requires device auto or mps")
        import mlx.core as mx
        from mlx_lm import load
        from transformers import AutoConfig

        from .hf import _cached_from_pretrained

        self.mx = mx
        config = _cached_from_pretrained(
            AutoConfig, model, revision=revision, trust_remote_code=False
        ).to_dict()
        self._multimodal = config.get("vision_config") is not None
        self.processor = None
        if self._multimodal:
            from mlx_vlm import load as load_vision

            self.model, self.processor = load_vision(
                model, revision=revision, trust_remote_code=False
            )
            if config.get("model_type") in {"qwen3_5", "qwen3_vl", "qwen3_5_moe"}:
                from .mlx_vision import QwenVisionAdapter

                self.model.vision_tower = QwenVisionAdapter(self.model.vision_tower)
            elif config.get("model_type") == "qwen2_5_vl":
                from .mlx_vision import Qwen25VisionAdapter

                self.model.vision_tower = Qwen25VisionAdapter(self.model.vision_tower)
            self.tokenizer = self.processor.tokenizer
        else:
            loaded = load(
                model,
                revision=revision,
                return_config=True,
                tokenizer_config={"trust_remote_code": False},
            )
            self.model, self.tokenizer = loaded[:2]
            if len(loaded) == 3:
                config = loaded[2]
        self.config = config
        self.eos = self.tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Model tokenizer must define an EOS/end-of-turn token")
        limits = [
            config.get("max_position_embeddings"),
            config.get("text_config", {}).get("max_position_embeddings"),
            self.tokenizer.model_max_length,
        ]
        self.limit = min(x for x in limits if isinstance(x, int) and x > 0)

    def prompt_ids(self, system, form, images=()):
        if not self._multimodal:
            return super().prompt_ids(system, form, images)
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        prompt = apply_chat_template(
            self.processor,
            self.config,
            [{"role": "system", "content": system}, {"role": "user", "content": form}],
            num_images=len(images),
            enable_thinking=False,
        )
        encoded = prepare_inputs(
            self.processor,
            images=list(images) or None,
            prompts=prompt,
            image_token_index=self.config.get("image_token_index"),
        )
        return ModelPrompt(
            encoded["input_ids"][0].tolist(),
            {
                key: value
                for key, value in encoded.items()
                if key not in {"input_ids", "attention_mask"}
            },
        )

    def scorer(self, prompt):
        if getattr(self, "_multimodal", False):
            # VLMs can retain image-position/recurrent state outside their KV cache.
            # Recompute complete prefixes so candidate branches stay independent.
            return lambda prefix, allowed: self.next_logprobs(prompt, prefix, allowed)
        return super().scorer(prompt)

    def _forward(self, ids, cache, use_cache, model_inputs=None):
        logits, cache = self._forward_many(ids, cache, use_cache, 1, model_inputs)
        return logits[0], cache

    def _forward_many(self, ids, cache, use_cache, count, model_inputs=None):
        if getattr(self, "_multimodal", False):
            output = self.model(self.mx.array([ids]), cache=None, **(model_inputs or {}))
            logits = output.logits if hasattr(output, "logits") else output
            return logits[0, -count:].astype(self.mx.float32), None
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
