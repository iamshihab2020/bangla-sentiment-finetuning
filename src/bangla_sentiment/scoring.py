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


def label_logprobs(logits, prompt_lens, label_ids):
    """Summed and first-token log-probabilities of each label continuation.

    logits: [batch, time, vocab] for sequences built as prompt + label.
    Position t of the logits predicts token t+1, so a label token at index i is scored at index i-1.
    """
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    summed, first = [], []
    for row, (start, ids) in enumerate(zip(prompt_lens, label_ids)):
        scores = [logprobs[row, start + k - 1, token] for k, token in enumerate(ids)]
        summed.append(torch.stack(scores).sum())
        first.append(scores[0])
    return torch.stack(summed), torch.stack(first)


@torch.no_grad()
def score_labels(model, tok, texts, variant, classes, batch_size=8, max_text_tokens=128):
    """Log-probability of every class label for every text. Returns (summed, first) arrays [n, classes]."""
    words = label_words(variant, classes)
    sequences, prompt_lens, label_ids = [], [], []
    for text in texts:
        messages = build_messages(truncate(tok, text, max_text_tokens), variant, classes)
        # Render first, then encode without extra specials: the template already carries BOS
        rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt = tok(rendered, add_special_tokens=False)["input_ids"]
        for word in words:
            ids = tok(word, add_special_tokens=False)["input_ids"]
            sequences.append(prompt + ids)
            prompt_lens.append(len(prompt))
            label_ids.append(ids)

    summed, first = [], []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i + batch_size]
        width = max(len(s) for s in chunk)
        input_ids = torch.tensor([s + [pad] * (width - len(s)) for s in chunk], device=DEVICE)
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in chunk], device=DEVICE)
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        s, f = label_logprobs(logits, prompt_lens[i:i + batch_size], label_ids[i:i + batch_size])
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


def parse_answer(answer, variant, classes):
    """The class whose label word the generated text starts with, or None if it is not a valid answer."""
    cleaned = answer.strip().lower().lstrip("*# ").strip()
    for cls, word in zip(classes, label_words(variant, classes)):
        if cleaned.startswith(word.lower()):
            return cls
    return None
