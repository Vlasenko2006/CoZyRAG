from torch.utils.data import Dataset
import numpy as np
import json
from filter_most_relevant_sentences import filter_most_relevant_sentences
import faiss



class RAGQADataset(Dataset):
    def __init__(self, qa_path, wiki_passages_path, retriever_model, faiss_index_path, top_k=1, tokenizer=None, max_length=384, filter_sentences=True, max_sentences=10, debug=False):
        self.samples = []
        with open(qa_path, encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        q, a = parts
                        self.samples.append((q, a))
                    elif len(parts) == 1:
                        self.samples.append((parts[0], ""))
        self.passages = []
        with open(wiki_passages_path, encoding="utf-8") as fin:
            for line in fin:
                self.passages.append(json.loads(line))
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.retriever_model = retriever_model
        self.index = faiss.read_index(faiss_index_path)
        self.passage_embeds = np.memmap(
            faiss_index_path + ".npy", dtype="float32", mode="r", shape=(self.index.ntotal, self.index.d))
        self.filter_sentences = filter_sentences
        self.max_sentences = max_sentences
        self.debug = debug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, answer = self.samples[idx]
        q_embed = self.retriever_model.encode([question])
        D, I = self.index.search(np.array(q_embed).astype('float32'), self.top_k)
        retrieved = [self.passages[i]['text'] for i in I[0]]

        max_context_tokens = 128
        context_raw = "\n".join(retrieved)
        # Filter most relevant sentences
        context = context_raw
        if self.filter_sentences:
            context = filter_most_relevant_sentences(context_raw, question, self.retriever_model, max_sentences=self.max_sentences, debug=self.debug)

        if self.tokenizer is not None:
            context_tokens = self.tokenizer.tokenize(context)
            if len(context_tokens) > max_context_tokens:
                context_tokens = context_tokens[:max_context_tokens]
            context = self.tokenizer.convert_tokens_to_string(context_tokens)
        prompt = f"Q: {question}\nContext:\n{context}\nA:"
        if self.tokenizer is not None:
            full_input = prompt + " " + answer
            encoded = self.tokenizer(
                full_input,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"]
            labels = encoded["input_ids"].clone()
            answer_start = (prompt_ids != self.tokenizer.pad_token_id).sum().item()
            if answer_start >= self.max_length:
                labels[:] = -100
            else:
                labels[0, :answer_start] = -100
            encoded["labels"] = labels[0]
            encoded["retrieved_context"] = context
            encoded["original_answer"] = answer
            encoded["question"] = question
            return {k: v.squeeze(0) if hasattr(v, "squeeze") else v for k, v in encoded.items()}
        else:
            return prompt, answer
