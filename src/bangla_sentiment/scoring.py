"""Label-probability scoring for generative LLMs, plus a generation path for the invalid-output rate.

The prediction is the class whose label word the model finds most likely as its answer, so there are
no invalid predictions and every class gets a probability. Two rules are computed from the same
forward pass (PRD section 7):
  first  the log-probability of the label's first token, valid only when the labels differ there
  full   the summed log-probability of all the label's tokens
Free generation is measured separately, only to report how often it produces an invalid answer.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from bangla_sentiment.prompts import build_messages, label_words

DEVICE = "cuda"


def load_4bit(model_id, dtype=torch.bfloat16):
    """The same 4-bit NF4 load that was measured in M0: double quantization, bf16 compute."""
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 device_map={"": 0}, dtype=dtype)
    model.eval()
    return tok, model


def first_tokens_differ(tok, words):
    """True when every label word starts with a different token, which is what the fast rule needs."""
    firsts = [tok(w, add_special_tokens=False)["input_ids"][0] for w in words]
    return len(set(firsts)) == len(firsts)


def truncate(tok, text, max_tokens):
    """Cut a comment to at most max_tokens tokens of this tokenizer, so prompts stay bounded."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= max_tokens else tok.decode(ids[:max_tokens])


def label_logprobs(logits, label_ids):
    """Summed and first-token log-probabilities of each label continuation.

    Sequences are left-padded and end with their label, and `logits` holds only the last K positions
    (see `logits_to_keep` in score_labels). Position t predicts token t+1, so token j of a label of
    length L sits at window position K - L + j - 1. Only those few positions are normalized, which
    keeps memory small: a full log-softmax over a 262k vocabulary would be about 1 GB per batch.
    """
    window = logits.shape[1]
    summed, first = [], []
    for row, ids in enumerate(label_ids):
        start = window - len(ids) - 1
        scores = torch.log_softmax(logits[row, start:start + len(ids)].float(), dim=-1)
        picked = scores[torch.arange(len(ids)), torch.tensor(ids, device=scores.device)]
        summed.append(picked.sum())
        first.append(picked[0])
    return torch.stack(summed), torch.stack(first)


@torch.no_grad()
def score_labels(model, tok, texts, variant, classes, batch_size=8, max_text_tokens=128, shots=()):
    """Log-probability of every class label for every text. Returns (summed, first) arrays [n, classes].

    `shots` are (text, class) pairs shown as worked examples before the comment, for few-shot prompting.
    """
    words = label_words(variant, classes)
    shots = [(truncate(tok, t, max_text_tokens), c) for t, c in shots]
    sequences, label_ids = [], []
    for text in texts:
        messages = build_messages(truncate(tok, text, max_text_tokens), variant, classes, shots=shots)
        # Render first, then encode without extra specials: the template already carries BOS
        rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt = tok(rendered, add_special_tokens=False)["input_ids"]
        for word in words:
            ids = tok(word, add_special_tokens=False)["input_ids"]
            sequences.append(prompt + ids)
            label_ids.append(ids)
    keep = max(len(ids) for ids in label_ids) + 1  # positions needed to score the longest label

    summed, first = [], []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i + batch_size]
        width = max(len(s) for s in chunk)
        # Left padding, so every sequence ends with its label and the kept logits line up
        input_ids = torch.tensor([[pad] * (width - len(s)) + s for s in chunk], device=DEVICE)
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in chunk], device=DEVICE)
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)  # padding must not shift positions
        logits = model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                       logits_to_keep=keep).logits
        s, f = label_logprobs(logits, label_ids[i:i + batch_size])
        summed.append(s.cpu())
        first.append(f.cpu())
    n_classes = len(classes)
    return (torch.cat(summed).view(-1, n_classes).numpy(), torch.cat(first).view(-1, n_classes).numpy())


def predictions_from_scores(scores, classes):
    """Class with the highest score for each row, plus probabilities from a softmax over the classes."""
    tensor = torch.tensor(scores)
    probs = torch.softmax(tensor, dim=-1).numpy()
    return [classes[i] for i in tensor.argmax(dim=-1).tolist()], probs


@torch.no_grad()
def generate_answers(model, tok, texts, variant, classes, batch_size=8, max_new_tokens=8,
                     max_text_tokens=128):
    """Greedy generation, used only to report how often free generation gives an invalid answer."""
    tok.padding_side = "left"  # generation needs the prompt to end at the last position
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    answers = []
    for i in range(0, len(texts), batch_size):
        chunk = [build_messages(truncate(tok, t, max_text_tokens), variant, classes)
                 for t in texts[i:i + batch_size]]
        batch = tok.apply_chat_template(chunk, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, padding=True, return_tensors="pt").to(DEVICE)
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, batch["input_ids"].shape[1]:]
        answers += tok.batch_decode(new, skip_special_tokens=True)
    return answers


def echoes_input(answer, text, min_run=8):
    """True when the generated answer quotes the comment back, in a run of `min_run` characters.

    Generated samples are saved in result files to illustrate the invalid-output rate, and a model
    that repeats the comment would put dataset text in the repository. SentNoB is CC BY-ND, so those
    samples are redacted instead (PRD section 5).
    """
    return any(text[i:i + min_run] in answer for i in range(max(1, len(text) - min_run + 1)))


def parse_answer(answer, variant, classes):
    """The class whose label word the generated text starts with, or None if it is not a valid answer."""
    cleaned = answer.strip().lower().lstrip("*# ").strip()
    for cls, word in zip(classes, label_words(variant, classes)):
        if cleaned.startswith(word.lower()):
            return cls
    return None
