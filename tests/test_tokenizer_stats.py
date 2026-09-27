"""A1: which words count as Bangla, and the three fertility measurements."""
from bangla_sentiment.tokenizer_stats import bangla_only, is_bangla_word, tokenizer_stats


class WordTokenizer:
    """One token per character, so token counts are predictable without model files."""

    unk_token_id = None

    def __init__(self):
        self.vocab = {}

    def __len__(self):
        return len(self.vocab)

    def __call__(self, texts, add_special_tokens=False):
        ids = [[self.vocab.setdefault(ch, len(self.vocab)) for ch in text] for text in texts]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        back = {i: ch for ch, i in self.vocab.items()}
        return [back[i] for i in ids]


def test_bangla_word_ignores_punctuation_but_rejects_mixed_script():
    assert is_bangla_word("ভালো")
    assert is_bangla_word("ভালো!")          # trailing punctuation does not count
    assert is_bangla_word("৫টা")            # Bengali digits are Bengali
    assert not is_bangla_word("nice")
    assert not is_bangla_word("ভালোnice")   # mixed script is not a Bangla word
    assert not is_bangla_word("5")
    assert not is_bangla_word("!!!")        # no letters, marks or digits at all


def test_bangla_only_keeps_bangla_words_and_lists_each_type_once():
    texts = ["ভালো nice ভালো", "খুব 123", "!!!"]
    kept, type_lines = bangla_only(texts)
    assert kept == ["ভালো ভালো", "খুব"]                      # rows left with nothing are dropped
    assert sorted(" ".join(type_lines).split()) == ["খুব", "ভালো"]  # each distinct word once


def test_fertility_reports_all_three_measurements():
    stats = tokenizer_stats(WordTokenizer(), ["ভালো nice", "ভালো"])
    fertility = stats["fertility"]
    # 3 words in total, 2 of them Bangla, 1 distinct Bangla word.
    assert fertility["bangla_word_share"] == round(2 / 3, 4)
    assert fertility["distinct_bangla_words_count"] == 1
    # One token per character, and "ভালো" and "nice" are 4 characters each.
    assert fertility["all_words"] == stats["tokens_per_word"] == round(13 / 3, 3)  # "ভালো nice" + "ভালো"
    assert fertility["bangla_words_only"] == 4.0                                   # "ভালো" + "ভালো"
    assert fertility["distinct_bangla_words"] == 4.0                               # "ভালো" once
