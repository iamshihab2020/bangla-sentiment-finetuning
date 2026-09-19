"""Metrics on a hand-checked example, and the E0 models (Bangla word splitting, majority class)."""
import re

import yaml

from bangla_sentiment.baselines import BANGLA_WORD, build_model
from bangla_sentiment.evaluate import classification_metrics
from bangla_sentiment.utils import CONFIGS_DIR

CLASSES = ["neutral", "positive", "negative"]


def test_metrics_on_hand_checked_example():
    gold = ["positive", "positive", "negative", "neutral"]
    pred = ["positive", "negative", "negative", "positive"]
    m = classification_metrics(gold, pred, CLASSES)
    # positive: precision 1/2, recall 1/2 -> F1 0.5. negative: precision 1/2, recall 1 -> F1 2/3.
    # neutral is never predicted -> F1 0, and it still counts in the macro average.
    assert m["per_class"]["positive"]["f1"] == 50.0
    assert m["per_class"]["negative"]["f1"] == 66.67
    assert m["per_class"]["neutral"]["f1"] == 0.0
    assert m["macro_f1"] == 38.89
    assert m["accuracy"] == m["micro_f1"] == 50.0
    assert m["confusion_matrix"]["values"] == [[0, 1, 0], [0, 1, 1], [0, 0, 1]]


def test_word_pattern_keeps_bangla_vowel_signs():
    assert re.findall(BANGLA_WORD, "ভালো লাগলো, খুব!") == ["ভালো", "লাগলো", "খুব"]


def test_tfidf_model_uses_bangla_word_pattern():
    cfg = yaml.safe_load((CONFIGS_DIR / "e0.yaml").read_text(encoding="utf-8"))["models"]["tfidf_lr"]
    features = build_model("tfidf_lr", cfg, seed=0).steps[0][1]
    words = features.transformer_list[0][1].build_analyzer()("ভালো লাগলো")
    assert words == ["ভালো", "লাগলো", "ভালো লাগলো"]


def test_majority_predicts_most_frequent_class():
    model = build_model("majority", {}, seed=0).fit(["a", "b", "c"], ["positive", "positive", "neutral"])
    assert list(model.predict(["x", "y"])) == ["positive", "positive"]
