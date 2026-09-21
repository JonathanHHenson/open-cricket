"""Compare local label and answer-code scoring on a small synthetic routing set.

Includes reversed candidate order to expose code/order sensitivity. This smoke
evaluation is not a representative accuracy benchmark. Loading and warmup are
excluded. Run with --backend mlx or --device mps on Apple silicon.
"""

import argparse
import json
import statistics
from time import perf_counter

from open_cricket.backend import load_backend
from open_cricket.questionnaire import classify


CASES = [
    ("I was charged twice. Please refund the duplicate payment.", "billing"),
    ("Where can I download my latest invoice?", "billing"),
    ("My card expired and I need to update my payment details.", "billing"),
    ("Please explain the extra charge on this month's bill.", "billing"),
    ("The app crashes every time I open the settings page.", "technical support"),
    ("I cannot log in even after resetting my password.", "technical support"),
    ("File uploads fail with error 500.", "technical support"),
    ("The screen is blank after the latest update.", "technical support"),
    ("Are you hiring designers?", "other"),
    ("I would like to discuss a marketing partnership.", "other"),
    ("Where is your company's head office?", "other"),
    ("Thank you for a great product!", "other"),
]
OPTIONS = [
    {"label": "billing", "description": "Charges, invoices, refunds, and payment methods"},
    {"label": "technical support", "description": "Software errors, crashes, login and upload problems"},
    {"label": "other", "description": "Anything outside billing and technical support"},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['hf', 'mlx'], default='hf')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    backend = load_backend(args.backend, args.model, args.device)
    modes = ['sequence', 'answer_codes']
    timings = {mode: [] for mode in modes}
    question = 'Which team should handle this message?'
    for mode in modes:
        classify(backend, CASES[0][0], question, OPTIONS, mode=mode)
    results = []
    for index, (message, expected) in enumerate(CASES):
        for reverse in (False, True):
            options = list(reversed(OPTIONS)) if reverse else OPTIONS
            predictions = {}
            for repeat in range(args.repeats):
                order = modes if (index + repeat + reverse) % 2 else list(reversed(modes))
                for mode in order:
                    start = perf_counter()
                    result = classify(backend, message, question, options, mode=mode)
                    timings[mode].append(perf_counter() - start)
                    predictions[mode] = {
                        'answer': result['answer'],
                        'probabilities': {row['label']: row['probability'] for row in result['options']},
                        'model_calls': result['model_calls'],
                        'prompt_tokens': result['prompt_tokens'],
                    }
            results.append({'message': message, 'expected': expected,
                            'reversed_order': reverse, **predictions})
    medians = {mode: statistics.median(times) for mode, times in timings.items()}
    print(json.dumps({
        'note': '12 synthetic cases, each in two candidate orders; not a general accuracy estimate.',
        'backend': args.backend, 'device': args.device, 'model': args.model,
        'repeats': args.repeats, 'median_seconds': medians,
        'speedup': medians['sequence'] / medians['answer_codes'],
        'correct_out_of_24': {mode: sum(row[mode]['answer'] == row['expected'] for row in results)
                              for mode in modes},
        'agreement_out_of_24': sum(row['sequence']['answer'] == row['answer_codes']['answer']
                                  for row in results),
        'order_flips_out_of_12': {
            mode: sum(results[i][mode]['answer'] != results[i + 1][mode]['answer']
                      for i in range(0, len(results), 2)) for mode in modes},
        'max_probability_difference': max(
            abs(row['sequence']['probabilities'][label] - row['answer_codes']['probabilities'][label])
            for row in results for label in row['sequence']['probabilities']),
        'results': results,
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
