"""Prompts for every generative LLM experiment (E2, E3, E4, E5, E6).

One template in two language variants. The variant is piloted on validation in M3 and then frozen.
The instruction goes in the first user turn, not a system turn, because not every small chat model
supports a system role.
"""
import random

LABEL_WORDS = {
    "P-en": {"neutral": "neutral", "positive": "positive", "negative": "negative"},
    "P-bn": {"neutral": "নিরপেক্ষ", "positive": "ইতিবাচক", "negative": "নেতিবাচক"},
}

INSTRUCTION = {
    "P-en": "Classify the sentiment of the Bangla comment. Answer with exactly one word: {labels}.",
    "P-bn": "বাংলা মন্তব্যটির অনুভূতি নির্ণয় করুন। ঠিক একটি শব্দে উত্তর দিন: {labels}।",
}

COMMENT = {"P-en": "Comment: {text}", "P-bn": "মন্তব্য: {text}"}


def label_words(variant, classes):
    """The answer word for each class, in the order the classes are given."""
    return [LABEL_WORDS[variant][c] for c in classes]


def build_messages(text, variant, classes, shots=()):
    """Chat messages for one comment: instruction, optional worked examples, then the comment.

    `shots` is a sequence of (text, class) pairs used as example turns for few-shot prompting.
    """
    header = INSTRUCTION[variant].format(labels=", ".join(label_words(variant, classes)))
    messages = []
    for shot_text, shot_class in shots:
        messages.append({"role": "user", "content": COMMENT[variant].format(text=shot_text)})
        messages.append({"role": "assistant", "content": LABEL_WORDS[variant][shot_class]})
    messages.append({"role": "user", "content": COMMENT[variant].format(text=text)})
    messages[0]["content"] = f"{header}\n\n{messages[0]['content']}"  # instruction on the first turn only
    return messages


def draw_shots(frame, classes, seed, per_class=2):
    """A class-balanced few-shot draw: `per_class` examples of every class, shuffled, seeded."""
    rng = random.Random(seed)
    shots = []
    for c in classes:
        rows = [(t, c) for t, lab in zip(frame["text"], frame["label"]) if lab == c]
        shots += rng.sample(rows, per_class)
    rng.shuffle(shots)
    return shots
