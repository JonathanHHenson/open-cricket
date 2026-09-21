import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from labeljudge.cli import format_pretty, main


class CliTests(unittest.TestCase):
    def test_format_pretty_shows_input_questions_answers_and_probabilities(self):
        result = {
            "model": "test-model",
            "processing_time_seconds": 1.23456,
            "results": [
                {
                    "id": "route",
                    "answer": "billing",
                    "question": "Which team?",
                    "prompt_tokens": 42,
                    "options": [
                        {"label": "billing", "probability": 0.75, "score": -1.0},
                        {"label": "other", "probability": 0.25, "score": -2.0},
                    ],
                }
            ],
        }

        self.assertEqual(
            format_pretty(result, "I was charged twice."),
            "Input: I was charged twice.\n\n"
            "Result: route\n"
            "Question: Which team?\n"
            "Answer: billing\n\n"
            "Options:\n"
            "  billing  75.00%\n"
            "  other    25.00%\n\n"
            "Processing time: 1.235 seconds",
        )

    def test_demo_accepts_pretty_and_time_flags(self):
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["labeljudge", "--demo", "--pretty", "--time"]),
            patch("labeljudge.cli.perf_counter", side_effect=[10.0, 12.3456]),
            contextlib.redirect_stdout(output),
        ):
            main()

        rendered = output.getvalue()
        self.assertIn("Answer: technical", rendered)
        self.assertIn("technical      47.37%", rendered)
        self.assertIn("Processing time: 2.346 seconds", rendered)
        self.assertNotIn("candidate_log_mass", rendered)
        self.assertNotIn("demo_note", rendered)

    def test_time_is_included_in_json_output(self):
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["labeljudge", "--demo", "--time"]),
            patch("labeljudge.cli.perf_counter", side_effect=[10.0, 12.3456789]),
            contextlib.redirect_stdout(output),
        ):
            main()

        result = json.loads(output.getvalue())
        self.assertEqual(result["processing_time_seconds"], 2.345679)

    def test_request_timer_starts_after_model_initialization(self):
        events = []
        times = iter([100.0, 101.5])

        class Backend:
            def __init__(self, model, device, revision):
                events.append("model initialized")

            def prompt_ids(self, system, form):
                return [99]

            def answer_ids(self, answer):
                return [1 if answer == '"billing"' else 2, 0]

            def next_logprobs(self, prompt, prefix, allowed):
                scores = {0: -0.1, 1: -0.5, 2: -1.5}
                return {token: scores[token] for token in allowed}

        def clock():
            events.append("clock read")
            return next(times)

        payload = {
            "message": "charged twice",
            "questions": [
                {
                    "question": "Which team?",
                    "options": ["billing", "other"],
                }
            ],
        }
        output = io.StringIO()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as input_file:
            json.dump(payload, input_file)
            input_file.flush()
            with (
                patch.object(
                    sys,
                    "argv",
                    ["labeljudge", "--input", input_file.name, "--time"],
                ),
                patch("labeljudge.hf.HuggingFaceBackend", Backend),
                patch("labeljudge.cli.perf_counter", side_effect=clock),
                contextlib.redirect_stdout(output),
            ):
                main()

        self.assertEqual(events, ["model initialized", "clock read", "clock read"])
        result = json.loads(output.getvalue())
        self.assertEqual(result["processing_time_seconds"], 1.5)


if __name__ == "__main__":
    unittest.main()


class BackendSelectionTests(unittest.TestCase):
    def test_mlx_factory_is_lazy_and_forwards_configuration(self):
        from labeljudge.backend import load_backend
        with patch('labeljudge.mlx.MLXBackend') as factory:
            result = load_backend('mlx', 'local/model', 'auto', 'revision')
        factory.assert_called_once_with('local/model', 'auto', 'revision')
        self.assertIs(result, factory.return_value)
        with self.assertRaisesRegex(ValueError, 'backend must'):
            load_backend('unknown')
