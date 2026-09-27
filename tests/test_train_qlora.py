"""E3 training data: the loss must cover the label word and nothing else, padding included."""
import pandas as pd

from bangla_sentiment.prompts import LABEL_WORDS
from bangla_sentiment.train_qlora import IGNORE, build_example, make_loader

CLASSES = ["neutral", "positive", "negative"]
CFG_MODEL = {"max_text_tokens": 64}


class FakeTokenizer:
    """Whitespace tokenizer with a stable vocabulary, so no model files are needed."""

    pad_token_id = 0

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        ids = [self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()]
        return {"input_ids": ids}

    def decode(self, ids):
        back = {i: word for word, i in self.vocab.items()}
        return " ".join(back[i] for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return " ".join(m["content"] for m in messages)


def test_loss_covers_the_label_word_and_no_prompt_token():
    tok = FakeTokenizer()
    ids, labels = build_example(tok, "khub bhalo", "positive", "P-en", CLASSES, 64)
    answer = tok("positive")["input_ids"]

    assert len(ids) == len(labels)
    assert ids[-len(answer):] == answer                      # the example ends with the gold label
    assert labels[-len(answer):] == answer                   # and only those positions carry a loss
    assert set(labels[:-len(answer)]) == {IGNORE}
    assert sum(lab != IGNORE for lab in labels) == len(answer)


def test_each_class_trains_towards_its_own_label_word():
    tok = FakeTokenizer()
    for label in CLASSES:
        ids, labels = build_example(tok, "text here", label, "P-bn", CLASSES, 64)
        assert labels[-1] == tok(LABEL_WORDS["P-bn"][label])["input_ids"][-1]


def test_padding_is_masked_out_of_attention_and_loss():
    tok = FakeTokenizer()
    frame = pd.DataFrame({"text": ["one", "a much longer comment than the first one"],
                          "label": ["neutral", "negative"]})
    batch = next(iter(make_loader(frame, tok, "P-en", CLASSES, CFG_MODEL, batch_size=2, shuffle=False)))

    lengths = batch["attention_mask"].sum(dim=1).tolist()
    assert lengths[0] < lengths[1] == batch["input_ids"].shape[1]  # the shorter row is padded
    padding = batch["attention_mask"] == 0
    assert (batch["input_ids"][padding] == tok.pad_token_id).all()
    assert (batch["labels"][padding] == IGNORE).all()
    # Rows keep their own label tokens: two rows, two labels, nothing else scored.
    assert [int((row != IGNORE).sum()) for row in batch["labels"]] == [
        len(tok(LABEL_WORDS["P-en"]["neutral"])["input_ids"]),
        len(tok(LABEL_WORDS["P-en"]["negative"])["input_ids"]),
    ]
