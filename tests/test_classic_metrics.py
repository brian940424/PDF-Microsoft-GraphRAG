import pytest

from automotive_graphrag.evaluation.classic_metrics import (
    calculate_corpus_metrics,
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
    normalize_answer,
)


def test_normalize_answer_matches_musique_rules():
    assert normalize_answer("The QUICK, brown fox!") == "quick brown fox"
    assert normalize_answer("  An   engine; and a belt. ") == "engine and belt"


def test_exact_match_uses_normalized_strings():
    assert exact_match_score("The answer!", "answer") == 1.0
    assert exact_match_score("answer one", "answer two") == 0.0


def test_f1_uses_character_multiset_overlap():
    assert f1_score("the red red car", "red car") == pytest.approx(0.8)
    assert f1_score("xyz", "123") == 0.0


def test_f1_scores_chinese_partial_overlap_by_character():
    assert f1_score("答案是定影器", "定影器很燙") == pytest.approx(6 / 11)
    assert f1_score("定影 器", "定影器很燙") == pytest.approx(0.75)


def test_empty_answers_follow_musique_no_answer_behavior():
    assert f1_score("", "the") == 1.0
    assert f1_score("", "answer") == 0.0
    assert exact_match_score("", "the") == 1.0


def test_metric_max_over_ground_truths_selects_best_alias():
    aliases = ["automobile", "car"]
    assert metric_max_over_ground_truths(exact_match_score, "CAR", aliases) == 1.0
    assert metric_max_over_ground_truths(f1_score, "red car", ["blue truck", "car"]) == pytest.approx(2 / 3)


def test_metric_max_over_ground_truths_rejects_empty_answers():
    with pytest.raises(ValueError, match="at least one"):
        metric_max_over_ground_truths(exact_match_score, "answer", [])


def test_corpus_metrics_pairs_by_id_and_uses_aliases():
    predictions = [
        {"id": "q2", "predicted_answer": "A red car"},
        {"id": "q1", "predicted_answer": "automobile"},
    ]
    ground_truths = [
        {"id": "q1", "answer": "vehicle", "answer_aliases": ["automobile"]},
        {"id": "q2", "answer": "car", "answer_aliases": []},
    ]
    assert calculate_corpus_metrics(predictions, ground_truths) == {
        "answer_em": 0.5,
        "answer_f1": pytest.approx(5 / 6),
    }


def test_corpus_metrics_returns_zero_for_empty_corpus():
    assert calculate_corpus_metrics([], []) == {"answer_em": 0.0, "answer_f1": 0.0}


@pytest.mark.parametrize(
    ("predictions", "ground_truths", "message"),
    [
        ([{"id": "q1", "predicted_answer": "x"}], [], "IDs do not match"),
        ([{"id": "q1", "predicted_answer": "x"}], [{"id": "q2", "answer": "x"}], "IDs do not match"),
        ([{"id": "q1", "predicted_answer": "x"}, {"id": "q1", "predicted_answer": "y"}], [{"id": "q1", "answer": "x"}], "duplicate id"),
    ],
)
def test_corpus_metrics_rejects_count_id_and_duplicate_mismatches(predictions, ground_truths, message):
    with pytest.raises(ValueError, match=message):
        calculate_corpus_metrics(predictions, ground_truths)


def test_corpus_metrics_validates_answer_fields():
    with pytest.raises(ValueError, match="predicted_answer"):
        calculate_corpus_metrics([{"id": "q", "predicted_answer": None}], [{"id": "q", "answer": "x"}])
    with pytest.raises(ValueError, match="answer_aliases"):
        calculate_corpus_metrics([{"id": "q", "predicted_answer": "x"}], [{"id": "q", "answer": "x", "answer_aliases": "y"}])
