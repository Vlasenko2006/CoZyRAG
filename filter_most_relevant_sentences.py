#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 22:44:37 2025

@author: andreyvlasenko
"""

import re
import util 

def filter_most_relevant_sentences(context, question, st_model, max_sentences=10, debug=False):
    sentences = re.split(r'(?<=[.!?])\s+', context)
    sentences = [s.strip() for s in sentences if s.strip()]
    if debug and not sentences:
        print("[DEBUG] No sentences found in context after split.")
        return ""
    question_emb = st_model.encode([question], convert_to_tensor=True)
    sentence_embs = st_model.encode(sentences, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_emb, sentence_embs)[0].cpu().numpy()
    sent_score_pairs = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    selected = [s for s, _ in sent_score_pairs[:max_sentences]]
    if debug:
        print("[DEBUG] FILTERED MOST RELEVANT SENTENCES:")
        for i, s in enumerate(selected):
            print(f"  [{i+1}] {s}")
        print("="*60)
    return "\n".join(selected)