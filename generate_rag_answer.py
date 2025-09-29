import torch
import faiss
import numpy as np
import json
from filter_most_relevant_sentences import filter_most_relevant_sentences
from postprocess_generated_answer import postprocess_generated_answer


def generate_rag_answer(question, model, tokenizer, retriever, faiss_index_path, wiki_passages_path, top_k=1, max_length=384, device=None):
    passages = []
    with open(wiki_passages_path, encoding="utf-8") as fin:
        for line in fin:
            passages.append(json.loads(line))
    index = faiss.read_index(faiss_index_path)
    q_embed = retriever.encode([question])
    D, I = index.search(np.array(q_embed).astype('float32'), top_k)
    retrieved = [passages[i]['text'] for i in I[0]]
    context_raw = "\n".join(retrieved)
    context = filter_most_relevant_sentences(context_raw, question, retriever, max_sentences=8)
    prompt = f"Q: {question}\nContext:\n{context}\nA:"

    batch = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.6,
            top_k=70,
            top_p=0.95
        )
    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = postprocess_generated_answer(prompt, generated_answer, question, context)
    return question, context, answer