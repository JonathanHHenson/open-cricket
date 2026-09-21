"""Structured decision API over Open Cricket's local probability scorer.

This matches the public question/answer shapes, not Jev's model calibration.
"""

import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

Content = str | dict[str, JsonValue] | list[JsonValue]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class QuestionBase(WireModel):
    # The official Python SDK accepts missing/null instructions, despite the
    # prose API reference describing this field as required.
    instructions: Content | None = None


class Choice(QuestionBase):
    type: Literal["choice"] = "choice"
    criteria: dict[str, Content | None] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def nonempty_labels(cls, value):
        if any(not label.strip() for label in value):
            raise ValueError("Choice labels must be nonempty")
        return value


class Score(QuestionBase):
    type: Literal["score"] = "score"
    # One level is accepted by the official SDK wire schema.
    criteria: list[Content] = Field(min_length=1, max_length=10)


class NoulCriteria(WireModel):
    true: Content | None = None
    false: Content | None = None


class Noul(QuestionBase):
    type: Literal["noul"] = "noul"
    criteria: NoulCriteria | None = None


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]


class SystemOneRequest(WireModel):
    state: Content
    model: str = Field(min_length=1)
    questions: dict[str, Question] = Field(min_length=1)


Probability = Annotated[float, Field(ge=0, le=1)]


class ChoiceAnswer(WireModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(WireModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, Probability]
    legend: dict[str, Content]
    confidence: Probability


class NoulAnswer(WireModel):
    type: Literal["noul"] = "noul"
    noul: Probability


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(WireModel):
    input_tokens: int | None
    output_tokens: int


class SystemOneResponse(WireModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(WireModel):
    name: str
    description: str
    release_date: str


class ModelsResponse(WireModel):
    models: list[ModelMetadata]


def render(value):
    return (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
    )


def confidence(probabilities):
    """Normalized entropy concentration, NOT a calibrated correctness estimate.

    Jev's exact confidence formula is not specified in its public API docs.
    """
    values = list(probabilities.values())
    if len(values) == 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in values if p > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(values))))


def classification_input(state, question):
    if isinstance(question, Choice):
        descriptions = question.criteria
        default = "Which category best describes the supplied state?"
    elif isinstance(question, Score):
        descriptions = {str(i): value for i, value in enumerate(question.criteria)}
        default = "Which rubric level best describes the supplied state?"
    else:
        criteria = question.criteria or NoulCriteria()
        descriptions = {
            "true": criteria.true if criteria.true is not None else "Yes",
            "false": criteria.false
            if criteria.false is not None
            else "No",
        }
        default = (
            "Does the supplied state match the true criterion rather than the false criterion?"
        )
    options = []
    for label, description in descriptions.items():
        # Empty descriptions carry no information; keep a valid canonical label.
        if description is None or (isinstance(description, str) and not description.strip()):
            options.append(label)
        else:
            options.append({"label": label, "description": render(description)})
    instructions = render(question.instructions) if question.instructions is not None else default
    return {
        "message": render(state),
        "question": instructions if instructions.strip() else default,
        "options": options,
    }
def format_response(request, model, results):
    answers = {}
    input_tokens = 0
    for (name, question), result in zip(request.questions.items(), results, strict=True):
        value = classification_input(request.state, question)
        probabilities = {row["label"]: row["probability"] for row in result["options"]}
        labels = [
            option if isinstance(option, str) else option["label"] for option in value["options"]
        ]
        if (
            set(probabilities) != set(labels)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("Classifier must return a normalized probability for every criterion")
        probabilities = {label: probabilities[label] for label in labels}
        count = result.get("prompt_tokens")
        input_tokens = (
            input_tokens + count if input_tokens is not None and count is not None else None
        )
        if isinstance(question, Noul):
            answers[name] = NoulAnswer(noul=probabilities["true"])
        elif isinstance(question, Score):
            answers[name] = ScoreAnswer(
                score=sum(int(level) * p for level, p in probabilities.items()),
                probabilities=probabilities,
                confidence=confidence(probabilities),
                legend={str(i): description for i, description in enumerate(question.criteria)},
            )
        else:
            answers[name] = ChoiceAnswer(
                choice=max(probabilities, key=lambda label: probabilities[label]),
                probabilities=probabilities,
                confidence=confidence(probabilities),
            )
    # Scoring does not generate an output token sequence. Input usage counts
    # each question's full prompt once, not repeated cache/branch evaluations.
    return SystemOneResponse(
        model=model, answers=answers, usage=Usage(input_tokens=input_tokens, output_tokens=0)
    )
