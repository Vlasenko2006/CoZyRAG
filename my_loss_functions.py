import torch
import torch.nn.functional as F


def differentiable_ngram_repeat_penalty(logits, n=3, weight=0.5):
    # logits: (batch, seq_len, vocab)
    tokens = logits.argmax(dim=-1)  # (batch, seq_len)
    penalty = torch.zeros(1, device=logits.device, dtype=logits.dtype)
    for seq in tokens:
        ngrams = {}
        for i in range(seq.size(0) - n + 1):
            ngram = tuple(seq[i:i+n].tolist())
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        penalty = penalty + sum([count - 1 for count in ngrams.values() if count > 1])
    penalty = penalty / tokens.size(0)
    return weight * penalty

def differentiable_keyword_overlap_penalty(logits, prompt_input_ids, weight=0.4):
    # logits: (batch, seq_len, vocab), prompt_input_ids: (batch, seq_len)
    tokens = logits.argmax(dim=-1)
    penalty = torch.zeros(1, device=logits.device, dtype=logits.dtype)
    for pred, prompt in zip(tokens, prompt_input_ids):
        pred_set = set(pred.tolist())
        prompt_set = set(prompt.tolist())
        pred_set.discard(0)      # Remove pad token
        prompt_set.discard(0)
        overlap = len(pred_set & prompt_set) / (len(prompt_set) + 1e-8)
        penalty = penalty + (1 - overlap)
    penalty = penalty / tokens.size(0)
    return weight * penalty

def differentiable_short_answer_penalty(logits, min_words=4, weight=0.01, pad_token_id=0):
    tokens = logits.argmax(dim=-1)
    penalty = torch.zeros(1, device=logits.device, dtype=logits.dtype)
    for seq in tokens:
        num_words = (seq != pad_token_id).sum()
        if num_words < min_words:
            penalty = penalty + (min_words - num_words)
    penalty = penalty / tokens.size(0)
    return weight * penalty


def contrastive_loss(q_embeds, p_embeds, temperature=0.1):
    q_embeds = F.normalize(q_embeds, dim=1)
    p_embeds = F.normalize(p_embeds, dim=1)
    logits = torch.matmul(q_embeds, p_embeds.T) / temperature
    labels = torch.arange(q_embeds.size(0)).to(q_embeds.device)
    loss = F.cross_entropy(logits, labels)
    return loss