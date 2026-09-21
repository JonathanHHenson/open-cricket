"""Compare sequential and shared-prefix question scoring on Hugging Face.

Model loading and one warmup per path are excluded. Answer-code mode is used
because request-level batching applies to its single scored position per question.
"""

import argparse
import copy
import json
import statistics
from time import perf_counter

from open_cricket import LocalClient
from open_cricket.backend import load_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--questions', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 1 for count in args.questions):
        parser.error('--repeats and every --questions value must be positive')

    backend = load_backend('hf', args.model, args.device)
    client = LocalClient(model=args.model, backend=backend)
    state = 'I was charged twice for my subscription. Please refund the duplicate payment.'
    criteria = {
        'billing': 'Payments, invoices, and refunds',
        'technical': 'Product errors and troubleshooting',
        'sales': 'Pricing and purchases',
        'other': 'Anything else',
    }
    rows = []
    for count in args.questions:
        payload = {
            'model': args.model,
            'state': state,
            'questions': {
                f'route_{i}': {
                    'type': 'choice',
                    'instructions': f'Which team should handle this message? Request {i + 1}.',
                    'criteria': copy.deepcopy(criteria),
                }
                for i in range(count)
            },
        }
        timings = {'sequential': [], 'batched': []}
        results = {}
        for enabled, name in ((False, 'sequential'), (True, 'batched')):
            backend._batch_questions = enabled
            client.invoke(payload)
        for repeat in range(args.repeats):
            order = [(False, 'sequential'), (True, 'batched')]
            if repeat % 2:
                order.reverse()
            for enabled, name in order:
                backend._batch_questions = enabled
                start = perf_counter()
                results[name] = client.invoke(payload)
                timings[name].append(perf_counter() - start)
        medians = {name: statistics.median(values) for name, values in timings.items()}
        difference = max(
            abs(results['sequential']['answers'][question]['probabilities'][label]
                - results['batched']['answers'][question]['probabilities'][label])
            for question in payload['questions']
            for label in criteria
        )
        rows.append({
            'questions': count,
            'median_seconds': medians,
            'speedup': medians['sequential'] / medians['batched'],
            'max_probability_difference': difference,
        })
    print(json.dumps({
        'model': args.model,
        'device': backend.device,
        'repeats': args.repeats,
        'rows': rows,
    }, indent=2))


if __name__ == '__main__':
    main()
