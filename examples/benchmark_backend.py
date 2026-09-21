"""Compare warmed cached scoring with the original full-prefix implementation.

Run: python examples/benchmark_backend.py --device cpu --repeats 5
For MLX: python examples/benchmark_backend.py --backend mlx
Model loading and one warmup per implementation are excluded from timing.
"""
import argparse
import json
import statistics
from time import perf_counter

from labeljudge.backend import load_backend
from labeljudge.questionnaire import classify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['hf', 'mlx'], default='hf')
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    backend = load_backend(args.backend, args.model, args.device)

    class Uncached:
        prompt_ids = backend.prompt_ids
        answer_ids = backend.answer_ids
        next_logprobs = backend.next_logprobs

    message = 'I was charged twice for my subscription. Please refund the duplicate payment.'
    options = ['billing', 'billing refund', 'technical support', 'other']
    timings = {'original': [], 'optimized': []}
    results = {}
    last_logits = getattr(backend, '_last_logits', False)
    # Alternate order to reduce warmup/thermal bias; discard the first pair.
    for repeat in range(args.repeats + 1):
        order = ['original', 'optimized'] if repeat % 2 else ['optimized', 'original']
        for name in order:
            if args.backend == 'hf':
                backend._last_logits = last_logits if name == 'optimized' else False
            start = perf_counter()
            results[name] = classify(
                backend if name == 'optimized' else Uncached(),
                message, 'Which team should handle this?', options)
            elapsed = perf_counter() - start
            if repeat:
                timings[name].append(elapsed)
    medians = {k: statistics.median(v) for k, v in timings.items()}
    original = {row['label']: row['probability'] for row in results['original']['options']}
    difference = max(abs(row['probability'] - original[row['label']])
                     for row in results['optimized']['options'])
    print(json.dumps({
        'backend': args.backend, 'model': args.model, 'device': args.device,
        'repeats': args.repeats, 'median_seconds': medians,
        'speedup': medians['original'] / medians['optimized'],
        'max_probability_difference': difference,
        'prompt_tokens': results['optimized']['prompt_tokens'],
        'model_calls': results['optimized']['model_calls'],
    }, indent=2))


if __name__ == '__main__':
    main()
