"""SQuAD/MuSiQue-style deterministic answer metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Mapping, Sequence


def normalize_answer(text: str) -> str:
    """Lowercase text and remove punctuation, English articles, and extra spaces."""
    lowered = text.lower()
    without_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Return 1.0 when normalized answers match, otherwise 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute character multiset F1 after MuSiQue/SQuAD answer normalization."""
    prediction_tokens = list(normalize_answer(prediction).replace(" ", ""))
    ground_truth_tokens = list(normalize_answer(ground_truth).replace(" ", ""))
    if not prediction_tokens or not ground_truth_tokens:
        # MuSiQue treats two normalized no-answer strings as a correct match.
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    ground_truths: Sequence[str],
) -> float:
    """Return the best score for a prediction across its acceptable answers."""
    if not ground_truths:
        raise ValueError("ground_truths must contain at least one answer")
    return max(metric_fn(prediction, answer) for answer in ground_truths)


def calculate_corpus_metrics(
    predictions: list[dict], ground_truths: list[dict]
) -> dict[str, float]:
    """Calculate macro-average answer EM and F1, pairing records by ``id``.

    Predictions use ``id`` and ``predicted_answer``. Ground truths use ``id``,
    ``answer``, and an optional ``answer_aliases`` list. Missing or mismatched
    IDs, duplicate IDs, and malformed answers raise ``ValueError``.
    """
    prediction_by_id = _index_records(predictions, "predictions")
    truth_by_id = _index_records(ground_truths, "ground_truths")
    if prediction_by_id.keys() != truth_by_id.keys():
        missing_predictions = sorted(truth_by_id.keys() - prediction_by_id.keys())
        missing_ground_truths = sorted(prediction_by_id.keys() - truth_by_id.keys())
        raise ValueError(
            "prediction and ground-truth IDs do not match "
            f"(missing predictions: {missing_predictions}; "
            f"missing ground truths: {missing_ground_truths})"
        )
    if not prediction_by_id:
        return {"answer_em": 0.0, "answer_f1": 0.0}

    em_scores: list[float] = []
    f1_scores: list[float] = []
    for question_id, prediction_record in prediction_by_id.items():
        truth_record = truth_by_id[question_id]
        predicted_answer = prediction_record.get("predicted_answer")
        answer = truth_record.get("answer")
        aliases = truth_record.get("answer_aliases", [])
        if not isinstance(predicted_answer, str):
            raise ValueError(f"prediction {question_id!r} must have a string predicted_answer")
        if not isinstance(answer, str):
            raise ValueError(f"ground truth {question_id!r} must have a string answer")
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError(f"ground truth {question_id!r} answer_aliases must be a list of strings")
        acceptable_answers = [answer, *aliases]
        em_scores.append(metric_max_over_ground_truths(exact_match_score, predicted_answer, acceptable_answers))
        f1_scores.append(metric_max_over_ground_truths(f1_score, predicted_answer, acceptable_answers))

    count = len(prediction_by_id)
    return {
        "answer_em": sum(em_scores) / count,
        "answer_f1": sum(f1_scores) / count,
    }


def _index_records(records: list[dict], label: str) -> dict[str, Mapping]:
    indexed: dict[str, Mapping] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"each {label} item must be an object")
        question_id = record.get("id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"each {label} item must have a non-empty string id")
        if question_id in indexed:
            raise ValueError(f"duplicate id {question_id!r} in {label}")
        indexed[question_id] = record
    return indexed
