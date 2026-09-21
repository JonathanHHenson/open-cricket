"""Cache regression tests run without model downloads or optional runtimes."""
import unittest

from open_cricket.core import score_paths
from open_cricket.local import LocalBackend


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
