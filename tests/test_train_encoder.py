"""E1 training helpers: class order, label indexing in batches, and the weight-decay split."""
import pandas as pd
import torch
from torch import nn

from bangla_sentiment.data import DATASETS
from bangla_sentiment.train_encoder import make_loader, optimizer_groups

CLASSES = ["neutral", "positive", "negative"]


class FakeTokenizer:
    """Stands in for a real tokenizer so the test needs no model files."""

    def __call__(self, texts, truncation, max_length, padding, return_tensors):
        return {"input_ids": torch.ones(len(texts), 4, dtype=torch.long)}


def test_class_order_matches_the_dataset_label_names():
    assert list(DATASETS["sentnob"].label_names.values()) == CLASSES


def test_loader_maps_label_names_to_their_class_index():
    frame = pd.DataFrame({"text": ["a", "b", "c"], "label": ["negative", "neutral", "positive"]})
    loader = make_loader(frame, CLASSES, FakeTokenizer(), {"max_seq_len": 128}, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert batch["labels"].tolist() == [2, 0, 1]  # positions of negative, neutral, positive in CLASSES


def test_no_weight_decay_on_bias_and_layernorm():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(4, 4)
            self.LayerNorm = nn.LayerNorm(4)

    decay, no_decay = optimizer_groups(Tiny(), 0.01)
    assert decay["weight_decay"] == 0.01 and len(decay["params"]) == 1  # dense.weight only
    assert no_decay["weight_decay"] == 0.0 and len(no_decay["params"]) == 3
