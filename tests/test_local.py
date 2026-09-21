"""Cache regression tests run without model downloads or optional runtimes."""
import unittest

from open_cricket.core import score_paths
from open_cricket.local import LocalBackend


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


class MLXCacheTests(unittest.TestCase):
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
