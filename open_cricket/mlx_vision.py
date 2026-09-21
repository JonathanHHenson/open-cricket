"""Compatibility adapter for the pinned MLX-VLM Qwen vision encoder."""

from contextlib import contextmanager
import copy

import mlx.core as mx
from mlx import nn


def score_shared_image_prefix(backend, requests):
    """Reuse a complete image prefix for Qwen2.5-VL answer-code questions.

    Return None when the model or inputs cannot use this path. Caches live only
    for this batch, and every question receives an independent cache branch.
    """
    from mlx_vlm.models.qwen2_5_vl.qwen2_5_vl import Model
    from mlx_vlm.models.qwen2_5_vl.language import LanguageModel
    from mlx_lm.models.cache import KVCache

    model = backend.model
    if (len(requests) < 2 or type(model) is not Model
            or type(model.language_model) is not LanguageModel):
        return None
    # This pinned model ignores processor mm_token_type_ids; they describe
    # the complete prompt and are neither image data nor cache inputs.
    def image_inputs(prompt):
        return {key: value for key, value in getattr(prompt, "model_inputs", {}).items()
                if key != "mm_token_type_ids"}

    inputs = image_inputs(requests[0][0])
    if set(inputs) != {"pixel_values", "image_grid_thw"}:
        return None
    for prompt, _ in requests:
        backend._validate(prompt)
        other = image_inputs(prompt)
        if set(other) != set(inputs):
            return None
        for key, value in inputs.items():
            candidate = other[key]
            if not isinstance(value, mx.array) or not isinstance(candidate, mx.array):
                return None
            if (value.shape != candidate.shape or value.dtype != candidate.dtype
                    or not mx.array_equal(value, candidate).item()):
                return None
    sequences = [tuple(prompt) for prompt, _ in requests]
    common = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:
            break
        common += 1
    common = min(common, min(map(len, sequences)) - 1)
    prefix = sequences[0][:common]
    config = model.config
    # The shared prefix must contain every complete image; suffixes are text only.
    visual_tokens = {config.image_token_id, config.video_token_id, config.vision_start_token_id}
    if config.image_token_id not in prefix or config.vision_end_token_id not in prefix:
        return None
    last_image = max(i for i, token in enumerate(prefix) if token in visual_tokens)
    if config.vision_end_token_id not in prefix[last_image + 1:]:
        return None
    if any(token in visual_tokens for sequence in sequences for token in sequence[common:]):
        return None
    language = model.language_model
    cache = [KVCache() for _ in language.layers]
    try:
        with selected_logits(model, 1):
            model(mx.array([prefix]), cache=cache, **inputs)
        mx.eval([layer.state for layer in cache])
        delta = language._rope_deltas
        results = []
        for sequence, (_, allowed) in zip(sequences, requests):
            branch = copy.deepcopy(cache)
            positions = mx.arange(common, len(sequence))[None, :] + delta
            positions = mx.broadcast_to(positions[None, ...], (3, 1, len(sequence) - common))
            with selected_logits(model, 1):
                output = language(
                    mx.array([sequence[common:]]), cache=branch, position_ids=positions
                )
            results.append(backend._probabilities(output.logits[0, -1].astype(mx.float32), allowed))
        return results
    finally:
        language._position_ids = None
        language._rope_deltas = None


class _SelectedProjection(nn.Module):
    """Project only scored positions, preserving the original quantized layer."""

    def __init__(self, layer, count, tied):
        super().__init__()
        self.layer, self.count, self.tied = layer, count, tied

    def __call__(self, inputs):
        return self.layer(inputs if self.tied else inputs[:, -self.count :])

    def as_linear(self, hidden):
        return self.layer.as_linear(hidden[:, -self.count :])


@contextmanager
def selected_logits(model, count):
    """Limit the pinned Qwen VLMs' final projection without bypassing image/RoPE setup.

    Unknown or customized language models retain their original forward path.
    The temporary layer is restored even when image processing or inference fails.
    """
    from mlx_vlm.models.qwen2_5_vl.language import LanguageModel as Qwen25
    from mlx_vlm.models.qwen3_5.language import LanguageModel as Qwen35
    from mlx_vlm.models.qwen3_vl.language import LanguageModel as Qwen3

    language = getattr(model, "language_model", None)
    if type(language) not in {Qwen25, Qwen3, Qwen35}:
        yield
        return
    tied = language.args.tie_word_embeddings
    owner, name = (language.model, "embed_tokens") if tied else (language, "lm_head")
    original = getattr(owner, name)
    setattr(owner, name, _SelectedProjection(original, count, tied))
    try:
        yield
    finally:
        setattr(owner, name, original)


class QwenVisionAdapter(nn.Module):
    """Run Qwen's encoder with integer grid sizes for MLX repeat counts.

    MLX-VLM 0.3.12 passes MLX scalar arrays to mx.repeat(repeats=...), which
    requires a Python integer. Keep the fix local to this loaded encoder.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    @property
    def patch_embed(self):
        return self.encoder.patch_embed

    def __call__(self, pixels, grid, **kwargs):
        encoder = self.encoder
        hidden = encoder.patch_embed(pixels)
        hidden = hidden + encoder.fast_pos_embed_interpolate(grid)
        rotary = encoder.rot_pos_emb(grid)
        length = hidden.shape[0]
        hidden = hidden.reshape(length, -1)
        rotary = rotary.reshape(length, -1)
        lengths = [h * w for t, h, w in grid.tolist() for _ in range(t)]
        offsets = mx.pad(mx.cumsum(mx.array(lengths, dtype=mx.int32)), (1, 0))
        features = []
        for index, block in enumerate(encoder.blocks):
            hidden = block(hidden, cu_seqlens=offsets, rotary_pos_emb=rotary)
            if index in encoder.deepstack_visual_indexes:
                merger = encoder.deepstack_merger_list[
                    encoder.deepstack_visual_indexes.index(index)
                ]
                features.append(merger(hidden))
        return encoder.merger(hidden), features


class Qwen25VisionAdapter(nn.Module):
    """Keep Qwen2.5-VL's window layout while fixing MLX repeat counts."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    @property
    def patch_embed(self):
        return self.encoder.patch_embed

    def __call__(self, pixels, grid, output_hidden_states=None):
        encoder = self.encoder
        hidden = encoder.patch_embed(pixels)
        rotary = encoder.rot_pos_emb(grid)
        window_index, window_offsets = encoder.get_window_index(grid)
        window_offsets = mx.array(list(dict.fromkeys(window_offsets.tolist())), dtype=mx.int32)

        length = hidden.shape[0]
        unit = encoder.spatial_merge_unit
        hidden = hidden.reshape(length // unit, unit, -1)[window_index].reshape(length, -1)
        rotary = rotary.reshape(length // unit, unit, -1)[window_index].reshape(length, -1)
        lengths = [h * w for t, h, w in grid.tolist() for _ in range(t)]
        offsets = mx.pad(mx.cumsum(mx.array(lengths, dtype=mx.int32)), (1, 0))

        for index, block in enumerate(encoder.blocks):
            hidden = block(
                hidden,
                cu_seqlens=offsets if index in encoder.fullatt_block_indexes else window_offsets,
                rotary_pos_emb=rotary,
            )
        hidden = encoder.merger(hidden)
        return hidden[mx.argsort(window_index, axis=0)]
