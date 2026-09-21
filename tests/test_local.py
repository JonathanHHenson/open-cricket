"""Cache regression tests run without model downloads or optional runtimes."""
import unittest

from open_cricket.core import score_paths
from open_cricket.local import LocalBackend, ModelPrompt


def check_real_chains(test, backend):
    paths = {'a': [1, 3, 4, 9], 'b': [1, 3, 5, 9], 'c': [2, 6, 9],
             'long': [2] + [7] * 20 + [9]}
    for mode in ('sequence', 'constrained'):
        expected = score_paths(paths, lambda p, a: backend.next_logprobs([7, 8, 6], p, a), mode=mode)
        actual = score_paths(paths, backend.scorer([7, 8, 6]), mode=mode)
        test.assertEqual(actual['answer'], expected['answer'])
        test.assertEqual(actual['model_calls'], expected['model_calls'])
        test.assertAlmostEqual(actual['candidate_log_mass'], expected['candidate_log_mass'], places=4)
        for row, reference in zip(actual['options'], expected['options']):
            test.assertEqual(row['label'], reference['label'])
            test.assertEqual(row['token_ids'], reference['token_ids'])
            for key in ('probability', 'score', 'log_likelihood'):
                test.assertAlmostEqual(row[key], reference[key], places=4)
            for edge, ref_edge in zip(row['trace'], reference['trace']):
                test.assertEqual(edge['token_id'], ref_edge['token_id'])
                test.assertAlmostEqual(edge['log_probability'], ref_edge['log_probability'], places=5)


class ToyBackend(LocalBackend):
    limit = 64

    def __init__(self, trimmable=True):
        self.work = []
        self.trimmable = trimmable

    def _forward(self, ids, cache, use_cache):
        self.work.append(tuple(ids))
        cache = [] if cache is None else cache
        cache.extend(ids)
        # Depends on the entire prefix, so sibling contamination changes scores.
        value = sum((i + 1) * token for i, token in enumerate(cache)) % 11
        return value, cache if use_cache else None

    def _trim(self, cache, length):
        if not self.trimmable:
            return False
        del cache[length:]
        return True

    def _select(self, logits, allowed):
        return [-1 - (logits + token) / 20 for token in allowed]


class CacheTests(unittest.TestCase):
    paths = {'a': [1, 9], 'ab': [1, 3, 9], 'ac': [1, 4, 5, 9], 'b': [2, 9]}

    def test_branch_rollback_matches_uncached_in_both_modes(self):
        for mode in ('sequence', 'constrained'):
            backend = ToyBackend()
            expected = score_paths(self.paths, lambda p, a: backend.next_logprobs([7, 8], p, a), mode=mode)
            old_work = sum(map(len, backend.work))
            backend.work.clear()
            actual = score_paths(self.paths, backend.scorer([7, 8]), mode=mode)
            self.assertEqual(actual, expected)
            self.assertEqual(backend.work[0], (7, 8))
            self.assertTrue(all(len(ids) == 1 for ids in backend.work[1:]))
            self.assertLess(sum(map(len, backend.work)), old_work)

    def test_untrimmable_cache_falls_back(self):
        backend = ToyBackend(False)
        expected = score_paths(self.paths, lambda p, a: backend.next_logprobs([7], p, a))
        actual = score_paths(self.paths, backend.scorer([7]))
        self.assertEqual(actual, expected)

    def test_sessions_are_isolated_and_allow_repeated_or_ancestor_prefixes(self):
        backend = ToyBackend()
        a, b = backend.scorer([7]), backend.scorer([8])
        for prefix in [(), (1,), (1, 2), (1,), (), (3,)]:
            self.assertEqual(a(prefix, (9,)), backend.next_logprobs([7], prefix, (9,)))
            self.assertEqual(b(prefix, (9,)), backend.next_logprobs([8], prefix, (9,)))

    def test_limit_and_nonfinite_validation(self):
        backend = ToyBackend()
        with self.assertRaisesRegex(ValueError, 'context limit'):
            backend.scorer([7] * 64)((1,), (9,))
        with self.assertRaisesRegex(ValueError, 'at least one token'):
            backend.scorer([])((), (9,))
        backend._select = lambda *args: [float('nan')]
        session = backend.scorer([7])
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            session((), (9,))
        self.assertIsNone(session.cache)

    def test_multimodal_prompt_metadata_reaches_prefill(self):
        class MultimodalBackend(ToyBackend):
            def _forward(self, ids, cache, use_cache, model_inputs=None):
                self.inputs = model_inputs
                return super()._forward(ids, cache, use_cache)

        backend = MultimodalBackend()
        prompt = ModelPrompt([7, 8], {'pixel_values': object()})
        backend.next_logprobs(prompt, (), (9,))
        self.assertIn('pixel_values', backend.inputs)
        backend.scorer(prompt)((), (9,))
        self.assertIn('pixel_values', backend.inputs)

    def test_text_chat_template_accepts_transformers_4_and_5_shapes(self):
        class Tokenizer:
            chat_template = 'template'

            def __init__(self, encoded):
                self.encoded = encoded
                self.kwargs = None

            def apply_chat_template(self, *args, **kwargs):
                self.kwargs = kwargs
                return self.encoded

        backend = ToyBackend()
        for encoded in ([1, 2, 3], {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}):
            backend.tokenizer = Tokenizer(encoded)
            self.assertEqual(backend.prompt_ids('system', 'form'), [1, 2, 3])
            self.assertIs(backend.tokenizer.kwargs['enable_thinking'], False)


class ChainBackend(ToyBackend):
    def _forward_many(self, ids, cache, use_cache, count):
        # A causal model produces a next-token distribution at every position.
        self.work.append(tuple(ids))
        cache = [] if cache is None else cache
        values = []
        for token in ids:
            cache.append(token)
            values.append(sum((i + 1) * t for i, t in enumerate(cache)) % 11)
        return values[-count:], cache if use_cache else None


class ChainTests(unittest.TestCase):
    def test_disabled_batching_uses_serial_scoring(self):
        backend = ChainBackend()
        backend._batch_chains = False
        result = score_paths({'a': [1, 2, 3, 9]}, backend.scorer([7]))
        self.assertEqual(len(backend.work), result['model_calls'])

    def test_chains_preserve_complete_results_and_reduce_forward_calls(self):
        for trimmable in (True, False):
            for mode in ('sequence', 'constrained'):
                for paths in (CacheTests.paths, {'only': [1] * 40 + [9]},
                              {'a': [1, 2, 3, 9], 'b': [1, 2, 4, 9]}):
                    backend = ChainBackend(trimmable)
                    session = backend.scorer([7, 8])
                    expected = score_paths(paths, lambda p, a: session(p, a), mode=mode)
                    old_calls = len(backend.work)
                    backend.work.clear()
                    actual = score_paths(paths, backend.scorer([7, 8]), mode=mode)
                    self.assertEqual(actual, expected)
                    self.assertLess(len(backend.work), old_calls)
                    if trimmable:
                        self.assertTrue(all(len(ids) <= 18 for ids in backend.work))

    def test_chain_context_limit_and_nonfinite_reset(self):
        backend = ChainBackend()
        with self.assertRaisesRegex(ValueError, 'context limit'):
            score_paths({'long': [1] * 65}, backend.scorer([7]))
        backend._select = lambda *args: [float('nan')]
        session = backend.scorer([7])
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            score_paths({'a': [1, 2, 9]}, session)
        self.assertIsNone(session.cache)


try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional HF dependencies not installed')
class HuggingFaceCacheTests(unittest.TestCase):
    def test_pretrained_loader_uses_local_cache_before_network(self):
        from open_cricket.hf import _cached_from_pretrained

        class Loader:
            calls = []

            @classmethod
            def from_pretrained(cls, model, **kwargs):
                cls.calls.append((model, kwargs))
                return 'loaded'

        self.assertEqual(_cached_from_pretrained(Loader, 'model', revision='abc'), 'loaded')
        self.assertEqual(Loader.calls, [
            ('model', {'local_files_only': True, 'revision': 'abc'})
        ])

        Loader.calls = []

        class MissingLoader(Loader):
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                cls.calls.append((model, kwargs))
                if kwargs.get('local_files_only'):
                    raise OSError('not cached')
                return 'downloaded'

        self.assertEqual(_cached_from_pretrained(MissingLoader, 'new-model'), 'downloaded')
        self.assertEqual(MissingLoader.calls, [
            ('new-model', {'local_files_only': True}),
            ('new-model', {}),
        ])

    def test_question_batch_matches_independent_full_forwards(self):
        from open_cricket.hf import HuggingFaceBackend
        torch.manual_seed(42)
        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend.torch, backend.device, backend.limit = torch, 'cpu', 128
        backend.model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128)).eval()
        backend._batch_questions = True
        backend._max_batch_size, backend._max_batch_tokens = 2, 128
        requests = [
            ([7, 8, 6], (9, 10)),
            ([7, 8, 5, 4], (9, 11, 12)),
            ([7, 8, 3, 2, 1], (10, 12)),
            ([7, 8, 6, 4], (8, 13)),
        ]
        for last_logits in (True, False):
            backend._last_logits = last_logits
            expected = [backend.next_logprobs(prompt, (), allowed)
                        for prompt, allowed in requests]
            actual = backend.batch_next_logprobs(requests)
            for row, reference in zip(actual, expected):
                self.assertEqual(set(row), set(reference))
                for token in reference:
                    self.assertAlmostEqual(row[token], reference[token], places=5)

    def test_disabled_question_batch_uses_independent_forwards(self):
        from open_cricket.hf import HuggingFaceBackend
        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend._batch_questions = False
        backend.next_logprobs = lambda prompt, prefix, allowed: {
            token: -float(token) for token in allowed
        }
        self.assertEqual(
            backend.batch_next_logprobs([([1], (2, 3)), ([4], (5,))]),
            [{2: -2.0, 3: -3.0}, {5: -5.0}],
        )
        backend._batch_questions = True
        backend._max_batch_tokens = 1
        backend.limit = 128
        self.assertEqual(
            backend.batch_next_logprobs([([1, 2, 3], (4,)), ([1, 4, 5], (5,))]),
            [{4: -4.0}, {5: -5.0}],
        )

    def test_real_transformer_cache_and_final_logits_match_full_forward(self):
        from open_cricket.hf import HuggingFaceBackend
        torch.manual_seed(42)
        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend.torch, backend.device, backend.limit = torch, 'cpu', 128
        backend.model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128)).eval()
        backend._last_logits = True
        session = backend.scorer([7, 8, 6])
        for prefix in [(), (1,), (1, 3), (2,), (), (1, 4)]:
            cached = session(prefix, (9, 10))
            backend._last_logits = False
            reference = backend.next_logprobs([7, 8, 6], prefix, (9, 10))
            backend._last_logits = True
            for token in reference:
                self.assertAlmostEqual(cached[token], reference[token], places=5)
            self.assertIsNotNone(session.cache)
        check_real_chains(self, backend)
        backend._last_logits = False
        check_real_chains(self, backend)


class MLXCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core  # noqa: F401
            from mlx_lm.models.qwen2 import Model  # noqa: F401
        except Exception as error:
            raise unittest.SkipTest(f'MLX runtime unavailable: {error}') from error

    def test_selected_projection_matches_full_for_tied_untied_and_quantized_models(self):
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm.models.qwen2 import Model, ModelArgs
        except ImportError as error:
            self.skipTest(f'MLX runtime unavailable: {error}')
        from open_cricket.mlx import MLXBackend
        for tied in (True, False):
            for quantized in (True, False):
                backend = MLXBackend.__new__(MLXBackend)
                backend.mx, backend.limit = mx, 128
                backend.model = Model(ModelArgs(
                    model_type='qwen2', vocab_size=32, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=128, rms_norm_eps=1e-6,
                    tie_word_embeddings=tied))
                if quantized:
                    nn.quantize(backend.model, group_size=32, bits=4)
                for count in (1, 3):
                    backend._last_logits = False
                    expected, _ = backend._forward_many([7, 8, 6, 4], None, False, count)
                    backend._last_logits = True
                    actual, _ = backend._forward_many([7, 8, 6, 4], None, False, count)
                    self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())
                check_real_chains(self, backend)

        # A custom model's __call__ must never be bypassed based on attributes alone.
        original = backend.model

        class WrappedModel:
            def __call__(self, inputs, cache=None):
                return original(inputs, cache=cache) * 0 + 3

        backend.model = WrappedModel()
        actual, _ = backend._forward([7, 8], None, False)
        self.assertTrue(mx.all(actual == 3).item())

    def test_vision_branches_recompute_images_without_cache(self):
        import mlx.core as mx
        from open_cricket.mlx import MLXBackend

        calls = []

        class VisionModel:
            def __call__(self, ids, cache=None, pixel_values=None):
                calls.append((ids.tolist(), cache, pixel_values.item()))
                return mx.broadcast_to(mx.arange(16), (1, ids.shape[1], 16))

        backend = MLXBackend.__new__(MLXBackend)
        backend.mx, backend.limit = mx, 128
        backend._multimodal = True
        backend.model = VisionModel()
        prompt = ModelPrompt([7, 8], {'pixel_values': mx.array(3)})
        scorer = backend.scorer(prompt)
        for prefix in [(), (1,), (2,), ()]:
            actual = scorer(prefix, (9, 10))
            self.assertAlmostEqual(actual[10] - actual[9], 1.0, places=5)
        self.assertEqual(calls, [
            ([[7, 8]], None, 3), ([[7, 8, 1]], None, 3),
            ([[7, 8, 2]], None, 3), ([[7, 8]], None, 3),
        ])

    def test_mlx_vision_template_disables_thinking_and_keeps_images(self):
        from unittest.mock import patch
        import mlx.core as mx
        from open_cricket.mlx import MLXBackend

        class Processor:
            chat_template = 'reasoning template'

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return 'instant' if kwargs.get('enable_thinking') is False else 'thinking'

        backend = MLXBackend.__new__(MLXBackend)
        backend._multimodal = True
        backend.processor = Processor()
        backend.config = {'model_type': 'qwen3_5'}
        pixels = mx.ones((1, 3))
        with patch('mlx_vlm.utils.prepare_inputs', return_value={
            'input_ids': mx.array([[7, 8]]), 'pixel_values': pixels,
            'attention_mask': mx.array([[1, 1]]),
        }) as prepare:
            for images in ((), ('apple.png', 'second.png')):
                prompt = backend.prompt_ids('system', 'question', images)
                self.assertEqual(prepare.call_args.kwargs['prompts'], 'instant')
                self.assertEqual(prepare.call_args.kwargs['images'], list(images) or None)
                content = backend.processor.messages[-1]['content']
                self.assertEqual(sum(item['type'] == 'image' for item in content), len(images))
                self.assertEqual(tuple(prompt), (7, 8))
                self.assertIs(prompt.model_inputs['pixel_values'], pixels)

    def test_qwen_vision_grid_uses_integer_frame_counts(self):
        import mlx.core as mx
        from open_cricket.mlx_vision import QwenVisionAdapter

        class Encoder:
            deepstack_visual_indexes = []
            patch_embed = staticmethod(lambda x: x)
            merger = staticmethod(lambda x: x)
            fast_pos_embed_interpolate = staticmethod(lambda grid: mx.zeros((6, 2)))
            rot_pos_emb = staticmethod(lambda grid: mx.zeros((6, 2)))

            def __init__(self):
                def block(hidden, cu_seqlens, rotary_pos_emb):
                    self.offsets = cu_seqlens.tolist()
                    return hidden
                self.blocks = [block]

        encoder = Encoder()
        output, features = QwenVisionAdapter(encoder)(
            mx.ones((6, 2)), mx.array([[2, 1, 2], [1, 1, 2]])
        )
        self.assertEqual(encoder.offsets, [0, 2, 4, 6])
        self.assertEqual(output.shape, (6, 2))
        self.assertEqual(features, [])

    def test_qwen25_vision_preserves_window_offsets(self):
        import mlx.core as mx
        from open_cricket.mlx_vision import Qwen25VisionAdapter

        class Encoder:
            spatial_merge_unit = 1
            fullatt_block_indexes = [1]
            patch_embed = staticmethod(lambda x: x)
            merger = staticmethod(lambda x: x)
            rot_pos_emb = staticmethod(lambda grid: mx.zeros((6, 2)))
            get_window_index = staticmethod(
                lambda grid: (mx.arange(6), mx.array([0, 1, 1, 3, 6]))
            )

            def __init__(self):
                self.offsets = []

                def block(hidden, cu_seqlens, rotary_pos_emb):
                    self.offsets.append(cu_seqlens.tolist())
                    return hidden

                self.blocks = [block, block]

        encoder = Encoder()
        output = Qwen25VisionAdapter(encoder)(
            mx.ones((6, 2)), mx.array([[2, 1, 2], [1, 1, 2]])
        )
        self.assertEqual(encoder.offsets, [[0, 1, 3, 6], [0, 2, 4, 6]])
        self.assertEqual(output.shape, (6, 2))

    def test_real_mlx_cache_matches_full_forward(self):
        try:
            import mlx.core as mx
            from mlx_lm.models.qwen2 import Model, ModelArgs
        except ImportError as error:
            self.skipTest(f'MLX runtime unavailable: {error}')
        from open_cricket.mlx import MLXBackend
        mx.random.seed(42)
        backend = MLXBackend.__new__(MLXBackend)
        backend.mx, backend.limit = mx, 128
        backend.model = Model(ModelArgs(
            model_type='qwen2', vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, rms_norm_eps=1e-6))
        session = backend.scorer([7, 8, 6])
        for prefix in [(), (1,), (1, 3), (2,), (), (1, 4)]:
            cached = session(prefix, (9, 10))
            reference = backend.next_logprobs([7, 8, 6], prefix, (9, 10))
            for token in reference:
                self.assertAlmostEqual(cached[token], reference[token], places=5)
        check_real_chains(self, backend)
