"""Compare warmed cached scoring with the original full-prefix implementation.

Run: python examples/benchmark_backend.py --device cpu --repeats 5
For MLX: python examples/benchmark_backend.py --backend mlx
Model loading and one warmup per implementation are excluded from timing.
"""

import argparse
import json
import statistics
from time import perf_counter

from open_cricket import LocalClient
from open_cricket.backend import load_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hf", "mlx"], default="hf")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chain-scoring", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--message-repeats", type=int, default=1,
                        help="Repeat the benchmark message to measure longer prompts")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.message_repeats < 1:
        parser.error("--message-repeats must be positive")
    backend = load_backend(args.backend, args.model, args.device)
    if args.chain_scoring != "auto":
        backend._batch_chains = args.chain_scoring == "on"

    class Uncached:
        prompt_ids = backend.prompt_ids
        answer_ids = backend.answer_ids
        next_logprobs = backend.next_logprobs

    class Cached(Uncached):
        def scorer(self, prompt):
            session = backend.scorer(prompt)
            # Hide the optional chain API to measure the previous cached scorer.
            return lambda prefix, allowed: session(prefix, allowed)

    message = "I was charged twice for my subscription. Please refund the duplicate payment."
    message = " ".join([message] * args.message_repeats)
    options = ["billing", "billing refund", "technical support", "other"]
    clients = {
        "original": LocalClient(model=args.model, backend=Uncached(), mode="sequence"),
        "cached": LocalClient(model=args.model, backend=Cached(), mode="sequence"),
        "chains": LocalClient(model=args.model, backend=backend, mode="sequence"),
        "optimized": LocalClient(model=args.model, backend=backend, mode="sequence"),
    }
    timings = {name: [] for name in clients}
    results = {}
    last_logits = getattr(backend, "_last_logits", True)
    # Rotate order to reduce warmup/thermal bias; discard the first round.
    for repeat in range(args.repeats + 1):
        order = list(clients)
        order = order[repeat % len(order):] + order[:repeat % len(order)]
        for name in order:
            keep_last = name != "original" if args.backend == "hf" else name == "optimized"
            backend._last_logits = last_logits if keep_last else False
            start = perf_counter()
            results[name] = clients[name].system_one(
                state=message,
                questions={
                    "route": {
                        "type": "choice",
                        "instructions": "Which team should handle this?",
                        "criteria": {label: None for label in options},
                    }
                },
            )
            elapsed = perf_counter() - start
            if repeat:
                timings[name].append(elapsed)
    medians = {k: statistics.median(v) for k, v in timings.items()}
    original = results["original"]["answers"]["route"]["probabilities"]
    difference = max(
        abs(p - original[label])
        for label, p in results["optimized"]["answers"]["route"]["probabilities"].items()
    )
    print(
        json.dumps(
            {
                "backend": args.backend,
                "model": args.model,
                "device": args.device,
                "chain_scoring": getattr(backend, "_batch_chains", True),
                "repeats": args.repeats,
                "median_seconds": medians,
                "speedup": medians["original"] / medians["optimized"],
                "speedup_over_cached": medians["cached"] / medians["optimized"],
                "speedup_over_chains": medians["chains"] / medians["optimized"],
                "max_probability_difference_from_chains": max(
                    abs(p - results["chains"]["answers"]["route"]["probabilities"][label])
                    for label, p in results["optimized"]["answers"]["route"]["probabilities"].items()
                ),
                "max_probability_difference_from_cached": max(
                    abs(p - results["cached"]["answers"]["route"]["probabilities"][label])
                    for label, p in results["optimized"]["answers"]["route"]["probabilities"].items()
                ),
                "max_probability_difference": difference,
                "prompt_tokens": results["optimized"]["usage"]["input_tokens"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
