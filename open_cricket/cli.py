import argparse
import json
from time import perf_counter

from .core import score_paths
from .systemone import SystemOneRequest, confidence, render


def format_pretty(result, input_message=None, questions=None):
    """Render typed answers and their probability distributions."""
    sections = []
    if input_message is not None:
        sections.append(f"Input: {render(input_message)}")
    for identifier, item in result["answers"].items():
        lines = [
            f"Question ID: {identifier}",
            f"Question Type: {item['type']}",
        ]
        if questions and questions[identifier].instructions is not None:
            lines.append(f"Question: {render(questions[identifier].instructions)}")
        if item["type"] == "choice":
            lines.append(f"Answer: {item['choice']}")
        elif item["type"] == "score":
            lines.append(f"Score: {item['score']:.3f}")
        else:
            lines.append(f"Yes probability: {item['noul']:.2%}")
        if "confidence" in item:
            lines.append(f"Confidence: {item['confidence']:.2%}")
        if "probabilities" in item:
            lines.extend(("", "Options:"))
            probabilities = item["probabilities"]
            width = max(map(len, probabilities))
            lines.extend(f"  {label:<{width}}  {p:.2%}" for label, p in probabilities.items())
        sections.append("\n".join(lines))
    if "processing_time_seconds" in result:
        sections.append(f"Processing time: {result['processing_time_seconds']:.3f} seconds")
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Score questionnaire options using LLM token probabilities"
    )
    parser.add_argument("--input", help="JSON file containing state, model, and typed questions")
    parser.add_argument("--model", help="Override the model specified in the input JSON")
    parser.add_argument("--backend", default="hf", choices=["hf", "mlx"])
    parser.add_argument("--revision", help="Optional Hugging Face commit revision")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mode", choices=["sequence", "constrained", "answer_codes"],
                        help="Scoring mode (default: answer_codes; sequence for --demo)")
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
    if args.mode is None:
        args.mode = "sequence" if args.demo else "answer_codes"
    if args.demo and args.mode == "answer_codes":
        parser.error("--demo supports sequence or constrained; answer_codes requires --input")
    input_message = None
    questions = None
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
        probabilities = {row["label"]: row["probability"] for row in result["options"]}
        result = {
            "model": "synthetic-demo",
            "answers": {
                "demo": {
                    "type": "choice",
                    "choice": result["answer"],
                    "probabilities": probabilities,
                    "confidence": confidence(probabilities),
                }
            },
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        result["demo_note"] = (
            "SYNTHETIC probabilities: demonstrates mathematics, not classification quality"
        )
    else:
        if not args.input:
            parser.error("--input is required unless --demo is supplied")
        with open(args.input, encoding="utf-8") as f:
            payload = json.load(f)
        from pydantic import ValidationError

        from .sdk import LocalClient

        try:
            request = SystemOneRequest.model_validate(payload)
        except ValidationError as error:
            parser.error(f"Expected state/model/questions request: {error}")
        if args.model:
            request.model = args.model
        input_message, questions = request.state, request.questions
        client = LocalClient(
            model=request.model,
            runtime=args.backend,
            device=args.device,
            revision=args.revision,
            mode=args.mode,
            temperature=args.temperature,
        )
        if args.time:
            request_started_at = perf_counter()
        result = client.invoke(request.model_dump(mode="json"))

    if args.time:
        result["processing_time_seconds"] = round(perf_counter() - request_started_at, 6)
    if args.pretty:
        print(format_pretty(result, input_message, questions))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
