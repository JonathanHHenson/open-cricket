import argparse
import json
from .questionnaire import classify, build_form
from .core import score_paths


def main():
    parser = argparse.ArgumentParser(description="Score questionnaire options using LLM token probabilities")
    parser.add_argument("--input", help="JSON file containing message and questions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", help="Optional Hugging Face commit revision")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mode", default="sequence", choices=["sequence", "constrained"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--demo", action="store_true", help="Synthetic math demo; no LLM")
    args = parser.parse_args()
    if args.demo:
        import math
        # Actual vocabulary has additional tokens that are not allowed answers.
        table = {(): {1: .6, 2: .3}, (1,): {9: .1, 3: .8},
                 (1, 3): {9: .5}, (2,): {9: .9}}
        result = score_paths({"refund": [1, 9], "refund status": [1, 3, 9], "technical": [2, 9]},
                             lambda p, a: {t: math.log(table[p][t]) for t in a},
                             mode=args.mode, temperature=args.temperature)
        result["demo_note"] = "SYNTHETIC probabilities: demonstrates mathematics, not classification quality"
    else:
        if not args.input:
            parser.error("--input is required unless --demo is supplied")
        with open(args.input, encoding="utf-8") as f:
            payload = json.load(f)
        questions = payload.get("questions")
        if not isinstance(questions, list) or not questions:
            parser.error("input must contain a nonempty questions list")
        for q in questions:
            build_form(payload["message"], q["question"], q["options"])
        from .hf import HuggingFaceBackend
        backend = HuggingFaceBackend(args.model, args.device, args.revision)
        result = {"model": args.model, "results": [
            {"id": q.get("id", str(i)), **classify(backend, payload["message"],
                q["question"], q["options"], mode=args.mode, temperature=args.temperature)}
            for i, q in enumerate(questions)]}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
