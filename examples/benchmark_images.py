"""Compare independent and shared image-prefix scoring, excluding model loading.

Run from the repository root: python -m examples.benchmark_images
"""
import argparse
import json
import statistics
from time import perf_counter

from open_cricket import LocalClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='examples/apple-image.json')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--probability-tolerance', type=float, default=0.02,
                        help='Maximum absolute probability drift; BF16 chunking can change rounding')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    with open(args.input, encoding='utf-8') as source:
        payload = json.load(source)
    client = LocalClient(model=payload['model'], runtime='mlx')
    backend = client.backend
    timings = {'independent': [], 'shared_prefix': []}
    results = {}
    for enabled in (False, True):
        backend._batch_questions = enabled
        client.invoke(payload)
    max_difference = 0.0
    for repeat in range(args.repeats):
        order = [(False, 'independent'), (True, 'shared_prefix')]
        if repeat % 2:
            order.reverse()
        for enabled, name in order:
            backend._batch_questions = enabled
            start = perf_counter()
            results[name] = client.invoke(payload)
            timings[name].append(perf_counter() - start)
        for question, expected in results['independent']['answers'].items():
            actual = results['shared_prefix']['answers'][question]
            if expected.get('choice') != actual.get('choice'):
                raise AssertionError(f'Prediction changed for {question}')
            for label, probability in expected['probabilities'].items():
                max_difference = max(max_difference, abs(probability - actual['probabilities'][label]))
        if results['independent']['usage'] != results['shared_prefix']['usage']:
            raise AssertionError('Token accounting changed')
    medians = {name: statistics.median(values) for name, values in timings.items()}
    print(json.dumps({
        'input': args.input, 'model': payload['model'], 'runtime': 'mlx',
        'repeats': args.repeats, 'median_seconds': medians,
        'speedup': medians['independent'] / medians['shared_prefix'],
        'max_probability_difference': max_difference,
        'answers': results['shared_prefix']['answers'],
        'timings_seconds': timings,
    }, indent=2))
    if max_difference > args.probability_tolerance:
        raise AssertionError(f'Probability difference exceeds tolerance: {max_difference}')


if __name__ == '__main__':
    main()
