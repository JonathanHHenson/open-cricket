"""Compatibility adapter for the pinned MLX-VLM Qwen vision encoder."""

import mlx.core as mx
from mlx import nn


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
