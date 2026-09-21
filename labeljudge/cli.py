import argparse
import json
from time import perf_counter

from .core import score_paths
from .questionnaire import build_form, classify


def format_pretty(result, input_message=None):
    """Render the input, questions, selected answers, and option probabilities."""
    results = result.get("results", [result])
    sections = []
    if input_message is not None:
        sections.append(f"Input: {input_message}")
    for index, item in enumerate(results):
        lines = []
        identifier = item.get("id")
        if identifier is not None or len(results) > 1:
            lines.append(f"Result: {identifier if identifier is not None else index}")
        if "question" in item:
            lines.append(f"Question: {item['question']}")
        lines.append(f"Answer: {item['answer']}")
        lines.extend(("", "Options:"))
        label_width = max(len(option["label"]) for option in item["options"])
        lines.extend(
            f"  {option['label']:<{label_width}}  {option['probability']:.2%}"
            for option in item["options"]
        )
        sections.append("\n".join(lines))
    if "processing_time_seconds" in result:
        sections.append(f"Processing time: {result['processing_time_seconds']:.3f} seconds")
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Score questionnaire options using LLM token probabilities"
    )
    parser.add_argument("--input", help="JSON file containing message and questions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", help="Optional Hugging Face commit revision")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mode", default="sequence", choices=["sequence", "constrained"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--demo", action="store_true", help="Synthetic math demo; no LLM")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Show input, questions, answers, and human-readable option probabilities",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Show request processing time, excluding model initialization",
    )
    args = parser.parse_args()
    input_message = None
    request_started_at = 0.0
    if args.demo:
        import math

        if args.time:
            request_started_at = perf_counter()
        # Actual vocabulary has additional tokens that are not allowed answers.
        table = {(): {1: 0.6, 2: 0.3}, (1,): {9: 0.1, 3: 0.8}, (1, 3): {9: 0.5}, (2,): {9: 0.9}}
        result = score_paths(
            {"refund": [1, 9], "refund status": [1, 3, 9], "technical": [2, 9]},
            lambda p, a: {t: math.log(table[p][t]) for t in a},
            mode=args.mode,
            temperature=args.temperature,
        )
        result["demo_note"] = (
            "SYNTHETIC probabilities: demonstrates mathematics, not classification quality"
        )
    else:
        if not args.input:
            parser.error("--input is required unless --demo is supplied")
        with open(args.input, encoding="utf-8") as f:
            payload = json.load(f)
        input_message = payload.get("message")
        questions = payload.get("questions")
        if not isinstance(questions, list) or not questions:
            parser.error("input must contain a nonempty questions list")
        for q in questions:
            build_form(payload["message"], q["question"], q["options"])
        from .hf import HuggingFaceBackend

        backend = HuggingFaceBackend(args.model, args.device, args.revision)
        if args.time:
            request_started_at = perf_counter()
        result = {
            "model": args.model,
            "results": [
                {
                    "id": q.get("id", str(i)),
                    **classify(
                        backend,
                        payload["message"],
                        q["question"],
                        q["options"],
                        mode=args.mode,
                        temperature=args.temperature,
                    ),
                }
                for i, q in enumerate(questions)
            ],
        }
    if args.time:
        result["processing_time_seconds"] = round(perf_counter() - request_started_at, 6)
    if args.pretty:
        print(format_pretty(result, input_message))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
