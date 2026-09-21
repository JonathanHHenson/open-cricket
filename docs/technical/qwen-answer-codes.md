# Qwen answer-code validation

[Technical index](README.md)

## Prompt fix

MLX-VLM 0.3.12's `apply_chat_template` helper accepts extra keyword arguments
but does not forward them to its final template rendering call. Consequently,
passing `enable_thinking=False` to that helper left Qwen3.5-4B in thinking mode.
Open Cricket now uses the helper to construct messages, then explicitly passes
`enable_thinking=False` to `get_chat_template`. Image placeholders and the
non-chat model input convention are preserved.

Before the fix, the Japanese-language example on
`mlx-community/Qwen3.5-4B-MLX-4bit` assigned only 0.0059% of its next-token
probability to the allowed answer letters. The model's preferred continuation
was `Thinking`, and normalizing the tiny answer-code mass produced misleading
category scores. With thinking disabled, allowed codes received 99.49% of the
probability and Japanese received 98.57% of the normalized category probability.

## Example evaluation

Validation used MLX 0.32.2, MLX-LM 0.31.3, MLX-VLM 0.3.12, Transformers 5.17.0,
and Apple Metal. Both models below used the default `answer_codes` mode and
temperature 1.0. Each model was loaded once and reused across requests.

For each model, 35 question outputs were inspected: 19 questions across all seven
bundled example requests, 10 additional Choice questions with their criterion
order reversed, four additional language inputs (English, Spanish, French,
German), and two negative Noul controls. Negative controls explicitly stated that
the bill was correct and the product worked. Scores and recommendations involve
judgment, so this is a smoke evaluation rather than a labeled accuracy benchmark.

| Check | Qwen3.5-4B-MLX-4bit | Qwen3-VL-2B-Instruct |
| --- | --- | --- |
| Billing, database owner, account access, replacement request, damaged/missing items | Expected choices in both orders | Expected choices in both orders |
| Japanese identification | 98.57%; reversed 91.52% | Approximately 100% in both orders |
| Additional four languages | All identified correctly | All identified correctly |
| Product review sentiment | Mixed: 97.70%; reversed 98.27% | Negative/mixed near 50/50; reversed selects mixed |
| Apple object and colour | Apple and red in both orders | Apple and red in both orders |
| Incorrect-charge/product-failure negative controls | True probabilities 14.36% / 10.35% | True probabilities 0.055% / 0.020% |
| 40-category routing | Data residency; reversed selects compliance | Compliance in both orders |

The 40-category request asks about both a data-processing agreement and backup
storage location. Privacy, compliance, legal, and data residency overlap, so
there is no single unambiguous expected category. The changed winner on Qwen3.5
shows remaining order sensitivity even after the template fix.

Qwen3-VL-2B's original-order product sentiment selects negative over mixed by
less than 1e-10 in probability. Its apple-ripeness distribution also looks
unreliable: 60.75% on the first rubric level, "Not Ripe: Don't Eat". Both outputs
were identical before and after the prompt fix. Sequence mode resolved product
sentiment (mixed: 99.72%) but did not resolve ripeness. A photograph alone also
cannot establish actual edibility; the example's object and colour questions
provide firmer correctness checks.

All 71 automated tests passed, including a regression that exercises MLX-VLM's
real message formatting helper and verifies that non-thinking mode reaches the
template for both text-only and two-image requests. No general guarantee of
correct predictions follows from these checks. The fix removes accidental
reasoning-mode scoring; model uncertainty and option-order effects remain.

To rerun an original example with either checkpoint:

```bash
uv run open-cricket --backend mlx \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --mode answer_codes --input examples/multilingual-support.json --pretty
```

Replace the model with `Qwen/Qwen3-VL-2B-Instruct` for the second model, or change
the input file to test another example. For the order check, reverse the entries
of each Choice question's `criteria` object, keeping labels and descriptions
together and leaving the state and instructions unchanged.
