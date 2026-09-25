"""Prompt building, few-shot draws, and the index arithmetic of label scoring (no model needed)."""
import pandas as pd
import pytest
import torch

from bangla_sentiment.prompts import LABEL_WORDS, build_messages, draw_shots, label_words
from bangla_sentiment.scoring import label_logprobs, parse_answer, predictions_from_scores

CLASSES = ["neutral", "positive", "negative"]


def test_zero_shot_prompt_is_one_user_turn_with_the_instruction():
    messages = build_messages("ভালো লাগলো", "P-bn", CLASSES)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert "ভালো লাগলো" in messages[0]["content"]
    for word in label_words("P-bn", CLASSES):
        assert word in messages[0]["content"]  # every option is named in the instruction


def test_few_shot_prompt_alternates_user_and_assistant_turns():
    shots = [("a", "positive"), ("b", "negative")]
    messages = build_messages("c", "P-en", CLASSES, shots=shots)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == "positive" and messages[3]["content"] == "negative"
    assert messages[0]["content"].startswith("Classify")  # instruction only on the first turn
    assert not messages[2]["content"].startswith("Classify")


def test_draw_shots_is_class_balanced_and_seeded():
    frame = pd.DataFrame({"text": [f"t{i}" for i in range(60)], "label": CLASSES * 20})
    shots = draw_shots(frame, CLASSES, seed=0, per_class=2)
    assert len(shots) == 6
    assert sorted(c for _, c in shots) == sorted(CLASSES * 2)
    assert shots == draw_shots(frame, CLASSES, seed=0, per_class=2)
    assert shots != draw_shots(frame, CLASSES, seed=1, per_class=2)


def test_label_logprobs_reads_the_positions_that_predict_the_label():
    # Two sequences: prompt of length 2 then a 2-token label, and prompt of length 3 then a 1-token label.
    logits = torch.full((2, 5, 7), -100.0)
    logits[0, 1, 3] = 0.0   # position 1 predicts token at index 2, the label's first token
    logits[0, 2, 4] = 0.0   # position 2 predicts token at index 3, the label's second token
    logits[1, 2, 5] = 0.0   # position 2 predicts token at index 3
    summed, first = label_logprobs(logits, [2, 3], [[3, 4], [5]])
    assert summed[0] == pytest.approx(first[0] * 2, abs=1e-3)  # both label tokens scored, both certain
    assert float(first[0]) == pytest.approx(0.0, abs=1e-3)     # log(1) for a certain token
    assert float(summed[1]) == pytest.approx(0.0, abs=1e-3)


def test_predictions_take_the_highest_scoring_class():
    scores = [[-1.0, -0.1, -5.0], [-0.2, -3.0, -0.1]]
    pred, probs = predictions_from_scores(scores, CLASSES)
    assert pred == ["positive", "negative"]
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("answer,expected", [
    ("positive", "positive"),
    ("  Negative. ", "negative"),
    ("**neutral**", "neutral"),
    ("I think this comment is", None),
    ("", None),
])
def test_parse_answer_accepts_only_a_label_word(answer, expected):
    assert parse_answer(answer, "P-en", CLASSES) == expected


def test_bangla_label_words_are_distinct():
    assert len(set(LABEL_WORDS["P-bn"].values())) == 3
