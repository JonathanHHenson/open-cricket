"""Regression checks for image scoring's selected vocabulary projection."""
import importlib
import inspect
import unittest
from types import SimpleNamespace


class ImageProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core as mx
            mx.eval(mx.ones(1))
        except Exception as error:
            raise unittest.SkipTest(f'MLX runtime unavailable: {error}') from error

    def test_real_qwen_projections_match_full_and_restore_layers(self):
        import mlx.core as mx
        import mlx.nn as nn
        from open_cricket.mlx_vision import selected_logits

        for family in ('qwen2_5_vl', 'qwen3_vl', 'qwen3_5'):
            config = importlib.import_module(f'mlx_vlm.models.{family}.config')
            language = importlib.import_module(f'mlx_vlm.models.{family}.language')
            for tied in (False, True):
                for quantized in (False, True):
                    with self.subTest(family=family, tied=tied, quantized=quantized):
                        values = dict(
                            model_type=family, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                            head_dim=32, rms_norm_eps=1e-6, vocab_size=64,
                            max_position_embeddings=128, tie_word_embeddings=tied,
                            rope_theta=10000, rope_scaling={'type': 'default', 'mrope_section': [4, 6, 6]},
                            rope_parameters={'type': 'default', 'mrope_section': [4, 6, 6],
                                             'rope_theta': 10000, 'partial_rotary_factor': 1.0},
                            linear_num_value_heads=2, linear_num_key_heads=2,
                            linear_key_head_dim=32, linear_value_head_dim=32,
                            linear_conv_kernel_dim=4, full_attention_interval=2,
                        )
                        args = config.TextConfig(**{
                            k: v for k, v in values.items()
                            if k in inspect.signature(config.TextConfig).parameters
                        })
                        model = language.LanguageModel(args, None)
                        if quantized:
                            nn.quantize(model, group_size=32, bits=4)
                        outer = SimpleNamespace(language_model=model)
                        ids = mx.array([[1, 2, 3, 4]])
                        positions = mx.broadcast_to(mx.arange(4)[None, None, :], (3, 1, 4))
                        expected = model(ids, position_ids=positions).logits
                        mx.eval(expected)
                        owner, name = (model.model, 'embed_tokens') if tied else (model, 'lm_head')
                        original = getattr(owner, name)
                        for count in (1, 3):
                            with selected_logits(outer, count):
                                actual = model(ids, position_ids=positions).logits
                                mx.eval(actual)
                            self.assertEqual(actual.shape, (1, count, 64))
                            self.assertTrue(mx.allclose(actual, expected[:, -count:], atol=1e-5, rtol=1e-5).item())
                            self.assertIs(getattr(owner, name), original)
                        # Exercise image embedding, grid positions and the full model call too.
                        from open_cricket.mlx import MLXBackend
                        from open_cricket.mlx_vision import QwenVisionAdapter, Qwen25VisionAdapter
                        vision = config.VisionConfig(
                            depth=1, hidden_size=64, intermediate_size=128, out_hidden_size=64,
                            num_heads=2, patch_size=2, spatial_patch_size=2,
                            temporal_patch_size=1, spatial_merge_size=2, window_size=4,
                            fullatt_block_indexes=[0],
                        )
                        full_config = config.ModelConfig(
                            text_config=args, vision_config=vision, model_type=family,
                            image_token_id=61, video_token_id=63, vision_start_token_id=60,
                            vision_end_token_id=62,
                        )
                        module = importlib.import_module(f'mlx_vlm.models.{family}.{family}')
                        full = module.Model(full_config)
                        adapter = Qwen25VisionAdapter if family == 'qwen2_5_vl' else QwenVisionAdapter
                        full.vision_tower = adapter(full.vision_tower)
                        if quantized:
                            nn.quantize(full.language_model, group_size=32, bits=4)
                        backend = MLXBackend.__new__(MLXBackend)
                        backend.mx, backend.model, backend._multimodal = mx, full, True
                        image_inputs = {'pixel_values': mx.ones((4, 12)),
                                        'image_grid_thw': mx.array([[1, 2, 2]])}
                        image_ids = [60, 61, 62, 1, 2]
                        for count in (1, 3):
                            backend._last_logits = False
                            baseline, _ = backend._forward_many(image_ids, None, False, count, image_inputs)
                            mx.eval(baseline)
                            backend._last_logits = True
                            selected, cache = backend._forward_many(image_ids, None, False, count, image_inputs)
                            self.assertIsNone(cache)
                            self.assertTrue(mx.allclose(selected, baseline, atol=1e-5, rtol=1e-5).item(), f'max error={mx.max(mx.abs(selected-baseline)).item()}, baseline finite={mx.all(mx.isfinite(baseline)).item()}')
                        if family == 'qwen2_5_vl':
                            from open_cricket.local import ModelPrompt
                            from unittest.mock import patch
                            backend.limit = 128
                            requests = [
                                (ModelPrompt(image_ids + suffix, {
                                    **image_inputs,
                                    'mm_token_type_ids': mx.zeros((1, len(image_ids) + len(suffix))),
                                }), (3, 4, 5))
                                for suffix in ([3], [4, 5, 6], [5, 2])
                            ]
                            expected_rows = [backend.next_logprobs(p, (), a) for p, a in requests]
                            calls = []
                            original_vision = adapter.__call__

                            def counted_vision(instance, *a, **kw):
                                calls.append(1)
                                return original_vision(instance, *a, **kw)

                            with patch.object(adapter, '__call__', counted_vision):
                                actual_rows = backend.batch_next_logprobs(requests)
                            self.assertEqual(len(calls), 1)
                            for expected_row, actual_row in zip(expected_rows, actual_rows):
                                for token in expected_row:
                                    self.assertAlmostEqual(expected_row[token], actual_row[token], delta=2e-5)
                            self.assertIsNone(full.language_model._position_ids)
                            self.assertIsNone(full.language_model._rope_deltas)
                            # Different pixels must never reuse image features or KV state.
                            changed = ModelPrompt(requests[-1][0], {
                                **image_inputs, 'pixel_values': mx.zeros((4, 12)),
                            })
                            mixed = [requests[0], (changed, (3, 4, 5))]
                            with patch.object(adapter, '__call__', counted_vision):
                                calls.clear()
                                mixed_rows = backend.batch_next_logprobs(mixed)
                            self.assertEqual(len(calls), 2)
                            self.assertEqual(mixed_rows[-1], backend.next_logprobs(changed, (), (3, 4, 5)))
                            # Identical requests still leave a suffix and preserve order.
                            duplicates = backend.batch_next_logprobs([requests[0], requests[0]])
                            self.assertEqual(duplicates[0], duplicates[1])
                            with patch.object(backend, '_probabilities', side_effect=RuntimeError('failure')):
                                with self.assertRaisesRegex(RuntimeError, 'failure'):
                                    backend.batch_next_logprobs(requests)
                            self.assertIsNone(full.language_model._position_ids)
                            self.assertIsNone(full.language_model._rope_deltas)
                        with self.assertRaisesRegex(RuntimeError, 'failure'):
                            with selected_logits(outer, 1):
                                raise RuntimeError('failure')
                        self.assertIs(getattr(owner, name), original)

    def test_image_chains_keep_pixels_and_reduce_full_forwards(self):
        import mlx.core as mx
        from open_cricket.local import ModelPrompt
        from open_cricket.mlx import MLXBackend

        calls = []

        class VisionModel:
            def __call__(self, ids, cache=None, pixel_values=None):
                calls.append((ids.tolist(), cache, pixel_values.item()))
                return mx.broadcast_to(mx.arange(16), (1, ids.shape[1], 16))

        backend = MLXBackend.__new__(MLXBackend)
        backend.mx, backend.limit, backend._multimodal = mx, 128, True
        backend.model = VisionModel()
        prompt = ModelPrompt([7, 8], {'pixel_values': mx.array(3)})
        requests = [((), (9, 10)), ((1,), (9, 10)), ((1, 2), (9, 10))]
        expected = [backend.next_logprobs(prompt, prefix, allowed) for prefix, allowed in requests]
        self.assertEqual(len(calls), 3)
        calls.clear()
        scorer = backend.scorer(prompt)
        self.assertEqual(scorer.score_chain(requests), expected)
        self.assertEqual(calls, [([[7, 8, 1, 2]], None, 3)])
        scorer((4,), (9, 10))
        self.assertEqual(calls[-1], ([[7, 8, 4]], None, 3))


if __name__ == '__main__':
    unittest.main()
