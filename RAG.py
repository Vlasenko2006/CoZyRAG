import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
import os
from tqdm import tqdm
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util as st_util
import faiss
import numpy as np
import json
import re
import argparse
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

try:
    import mlflow
    mlflow_available = True
except ImportError:
    mlflow_available = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ngrams(tokens, n):
    ngram_set = set()
    for i in range(len(tokens) - n + 1):
        ngram_set.add(tuple(tokens[i:i+n]))
    return ngram_set

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


def filter_most_relevant_sentences(context, question, st_model, max_sentences=10, debug=False):
    sentences = re.split(r'(?<=[.!?])\s+', context)
    sentences = [s.strip() for s in sentences if s.strip()]
    if debug and not sentences:
        print("[DEBUG] No sentences found in context after split.")
        return ""
    question_emb = st_model.encode([question], convert_to_tensor=True)
    sentence_embs = st_model.encode(sentences, convert_to_tensor=True)
    scores = st_util.pytorch_cos_sim(question_emb, sentence_embs)[0].cpu().numpy()
    sent_score_pairs = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    selected = [s for s, _ in sent_score_pairs[:max_sentences]]
    if debug:
        print("[DEBUG] FILTERED MOST RELEVANT SENTENCES:")
        for i, s in enumerate(selected):
            print(f"  [{i+1}] {s}")
        print("="*60)
    return "\n".join(selected)

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

def build_or_load_faiss_index(passages_path, retriever_model_name, faiss_index_path, embedding_path):
    if os.path.exists(faiss_index_path) and os.path.exists(embedding_path):
        print("Loading FAISS index and passage embeddings...")
        return
    passages = []
    with open(passages_path, encoding="utf-8") as fin:
        for line in fin:
            passages.append(json.loads(line))
    retriever = SentenceTransformer(retriever_model_name)
    texts = [p['text'] for p in passages]
    print("Encoding passages for retrieval...")
    embeds = retriever.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    np.save(embedding_path, embeds)
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)
    faiss.write_index(index, faiss_index_path)
    print("FAISS index built and saved.")

def contrastive_loss(q_embeds, p_embeds, temperature=0.1):
    q_embeds = F.normalize(q_embeds, dim=1)
    p_embeds = F.normalize(p_embeds, dim=1)
    logits = torch.matmul(q_embeds, p_embeds.T) / temperature
    labels = torch.arange(q_embeds.size(0)).to(q_embeds.device)
    loss = F.cross_entropy(logits, labels)
    return loss






def postprocess_generated_answer(prompt, answer, question, retrieved_context):
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    answer_lines = answer.split('\n')
    filtered_lines = [line for line in answer_lines if question.lower() not in line.lower()]
    context_lines = set([l.strip() for l in retrieved_context.split('\n') if l.strip()])
    filtered_lines = [line for line in filtered_lines if line.strip() not in context_lines]
    answer = '\n'.join(filtered_lines).strip()
    answer = re.sub(r"^A:\s*", "", answer)
    return answer

def generate_rag_answer(question, model, tokenizer, retriever, faiss_index_path, wiki_passages_path, top_k=1, max_length=384, device=None):
    passages = []
    with open(wiki_passages_path, encoding="utf-8") as fin:
        for line in fin:
            passages.append(json.loads(line))
    index = faiss.read_index(faiss_index_path)
    passage_embeds = np.memmap(
        faiss_index_path + ".npy", dtype="float32", mode="r", shape=(index.ntotal, index.d))
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



def fine_tune_rag_ddp(rank, 
                      world_size, 
                      local_rank,
                      ddp_enabled,
                      qa_path, 
                      wiki_passages_path,
                      faiss_index_path, 
                      retriever_model_name,
                      output_dir='./fine_tuned_model',
                      num_train_epochs=1,
                      batch_size=2,
                      top_k=1,
                      resume_epoch=None,
                      max_length=384,
                      device="cpu",
                      gpt_lr=5e-5, 
                      retriever_lr=5e-5,
                      ngram_n=3,
                      ngram_weight=0.1,
                      keyword_weight=0.4,
                      short_min_words=4,
                      short_weight=0.01,
                      validation_questions=None
):
    if ddp_enabled:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    logger.info(
        f"[DDP SETUP] rank={rank}, world_size={world_size}, local_rank={local_rank}, ddp_enabled={ddp_enabled}")

    if mlflow_available and (rank == 0 or not ddp_enabled):
        mlflow.set_experiment("distilgpt2-rag-finetune")
        mlflow.start_run()
        mlflow.log_param("num_train_epochs", num_train_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("ddp_enabled", ddp_enabled)

    retriever = SentenceTransformer(retriever_model_name)
    retriever_trainable = hasattr(retriever, 'forward') or hasattr(retriever, 'parameters')
    if retriever_trainable:
        retriever_optimizer = torch.optim.AdamW(retriever.parameters(), lr=retriever_lr)

    model_name = 'distilgpt2'
    if resume_epoch is not None:
        checkpoint_dir = f"{output_dir}/epoch_{resume_epoch}"
        if os.path.exists(checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {checkpoint_dir}")
            model = GPT2LMHeadModel.from_pretrained(checkpoint_dir).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)
            start_epoch = resume_epoch            
            if resume_epoch >= 2: 
                retriever = SentenceTransformer(f"fine_tuned_model/retriever_epoch_{resume_epoch-1}")
        else:
            model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            start_epoch = 0
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        start_epoch = 0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = RAGQADataset(
        qa_path, wiki_passages_path, retriever, faiss_index_path,
        top_k=top_k, tokenizer=tokenizer, max_length=max_length,
        filter_sentences=True, max_sentences=10
    )
    if ddp_enabled:
        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"[DEBUG] Dataset size: {len(dataset)}")
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle, drop_last=False
    )
    print(f"[DEBUG] Number of batches per epoch (expected): {len(train_loader)}")

    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=gpt_lr)

    logger.info("Starting RAG alternate training...")

    retrieved_mem = []

    for epoch in range(start_epoch, num_train_epochs):
        penalty_vals = []
        keyword_penalty_vals = []
        short_answ_penalty_vals = []
        base_loss = []
        retriever_losses = []
        if ddp_enabled:
            train_sampler.set_epoch(epoch)
        model.train()
        batch_losses = []
        num_batches = 0
        num_skipped = 0

        retriever_losses = []
        if epoch % 2 == 1:
            print(f"[DEBUG] Epoch {epoch+1}: retriever training epoch")
            if len(retrieved_mem) == 0:
                print("[WARNING] No retrieved memory from previous epoch. Skipping retriever training.")
                continue
            for batch_idx, (questions_text, contexts) in enumerate(retrieved_mem):
                retriever_optimizer.zero_grad()
                q_features = retriever.tokenizer(
                    questions_text, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                c_features = retriever.tokenizer(
                    contexts, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                q_embeds = retriever(q_features)['sentence_embedding']
                p_embeds = retriever(c_features)['sentence_embedding']
                assert q_embeds.requires_grad, "Query embeddings do not require grad!"
                assert p_embeds.requires_grad, "Context embeddings do not require grad!"
                loss = contrastive_loss(q_embeds, p_embeds)
                loss.backward()
                retriever_optimizer.step()
                retriever_losses.append(loss.item())
                if (batch_idx + 1) % 100 == 0:
                    mean_retriever_loss = sum(retriever_losses[-100:]) / 100
                    print(f"[DEBUG] Epoch {epoch+1} after {batch_idx+1} retriever batches: mean retriever loss = {mean_retriever_loss:.4f}")
            retrieved_mem.clear()
        else:
            print(f"[DEBUG] Epoch {epoch+1}: GPT training epoch")
            retrieved_mem.clear()
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Batches", ncols=100)):
                if rank == 0 or not ddp_enabled: print(f"[DEBUG] Processing batch {batch_idx+1}/{len(train_loader)}")
                if (batch['labels'] == -100).all():
                    num_skipped += 1
                    continue
                num_batches += 1
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                ngram_penalty = differentiable_ngram_repeat_penalty(logits, n=ngram_n, weight=ngram_weight)
                keyword_penalty = differentiable_keyword_overlap_penalty(logits, input_ids, weight=keyword_weight)
                short_answ_penalty = differentiable_short_answer_penalty(
                    logits, min_words=short_min_words, weight=short_weight, pad_token_id=tokenizer.pad_token_id
                )

                loss = outputs.loss + ngram_penalty + keyword_penalty + short_answ_penalty
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

                penalty_vals.append(ngram_penalty.item())
                keyword_penalty_vals.append(keyword_penalty.item())
                short_answ_penalty_vals.append(short_answ_penalty.item())
                base_loss.append(outputs.loss.item())

                questions_text = batch['question'] if isinstance(batch['question'], list) else [batch['question']]
                contexts = batch['retrieved_context'] if isinstance(batch['retrieved_context'], list) else [batch['retrieved_context']]
                retrieved_mem.append((questions_text, contexts))

                if (rank == 0 or not ddp_enabled) and ((batch_idx + 1) % 100 == 0):
                    avg_loss1 = sum(batch_losses) / len(batch_losses) if batch_losses else float('nan')
                    mean_ngram = sum(penalty_vals) / len(penalty_vals) if penalty_vals else 0.0
                    mean_keyword = sum(keyword_penalty_vals) / len(keyword_penalty_vals) if keyword_penalty_vals else 0.0
                    mean_shortansw = sum(short_answ_penalty_vals) / len(short_answ_penalty_vals) if short_answ_penalty_vals else 0.0
                    mean_base_loss = sum(base_loss) / len(base_loss) if base_loss else 0.0
                    print(f"[DEBUG] Epoch {epoch+1} average loss: {avg_loss1:.4f}")
                    print(f"[DEBUG] Epoch {epoch+1} after {batch_idx+1} batches: mean ngram penalty = {mean_ngram:.4f}")
                    print(f"[DEBUG] Epoch {epoch+1} after {batch_idx+1} batches: mean keyword penalty = {mean_keyword:.4f}")
                    print(f"[DEBUG] Epoch {epoch+1} after {batch_idx+1} batches: mean short answer penalty = {mean_shortansw:.4f}")
                    print(f"[DEBUG] Epoch {epoch+1} after {batch_idx+1} batches: mean base loss = {mean_base_loss:.4f}")
                    penalty_vals.clear()
                    keyword_penalty_vals.clear()
                    short_answ_penalty_vals.clear()
                    base_loss.clear()

        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else float('nan')
        if rank == 0 or not ddp_enabled:
            if epoch % 2 == 1:
                if retriever_losses:
                    mean_retriever_loss = sum(retriever_losses) / len(retriever_losses)
                    print(f"[DEBUG] Epoch {epoch+1} mean retriever loss: {mean_retriever_loss:.4f}")
            else:
                logger.info(f"[DEBUG] Epoch {epoch+1} finished: {num_batches} GPT batches, {num_skipped} skipped.")
                logger.info(f"Epoch {epoch+1} average GPT loss: {avg_loss:.4f}")
            if mlflow_available:
                if epoch % 2 == 1 and retriever_losses:
                    mlflow.log_metric("epoch_mean_retriever_loss", mean_retriever_loss, step=epoch+1)
                else:
                    mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch+1)

            logger.info(f"Saving model/tokenizer for epoch {epoch + 1}")
            if rank == 0 or not ddp_enabled:
                logger.info(f"Saving model/tokenizer for epoch {epoch + 1}")
                save_path = f"{output_dir}/epoch_{epoch + 1}"
                if ddp_enabled:
                    model.module.save_pretrained(save_path)
                else:
                    model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                retriever_save_path = f"{output_dir}/retriever_epoch_{epoch + 1}"
                os.makedirs(retriever_save_path, exist_ok=True)
                try:
                    retriever.save(retriever_save_path)
                    print(f"Retriever saved to {retriever_save_path}")
                except Exception as e:
                    print(f"Failed to save retriever: {e}")

            # --------- VALIDATION: Use validation_questions from config ---------
            if epoch % 2 == 0 and validation_questions:
                retriever_for_gen = SentenceTransformer(retriever_model_name)
                tokenizer_for_gen = GPT2Tokenizer.from_pretrained(f"{output_dir}/epoch_{epoch + 1}")
                tokenizer_for_gen.pad_token = tokenizer_for_gen.eos_token
                model_for_gen = GPT2LMHeadModel.from_pretrained(f"{output_dir}/epoch_{epoch + 1}").to(device)
                print(f"Prompt answers after epoch {epoch+1}:")
                for q in validation_questions:
                    q_text, retrieved_text, ans = generate_rag_answer(
                        q, model_for_gen, tokenizer_for_gen, retriever_for_gen,
                        faiss_index_path, wiki_passages_path, top_k=top_k,
                        max_length=max_length,
                        device=device
                    )
                    print("="*40)
                    print(f"QUESTION:\n{q_text}")
                    print(f"RETRIEVED CONTEXT:\n{retrieved_text}")
                    print(f"ANSWER:\n{ans}")
                    print("="*40)

    if mlflow_available and (rank == 0 or not ddp_enabled):
        mlflow.end_run()
    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel (multi-GPU)")
    parser.add_argument("--resume_epoch", type=int, default=None, help="Epoch to resume training from (loads checkpoint if exists)")
    parser.add_argument("--build_index", action="store_true", help="Build FAISS index and embeddings")
    args = parser.parse_args()

    config = load_config(args.config)

    qa_path = config.get('qa_path')
    wiki_passages_path = config.get('wiki_passages_path')
    retriever_model_name = config.get('retriever_model_name')
    faiss_index_path = config.get('faiss_index_path')
    embedding_path = config.get('embedding_path')
    num_train_epochs = config.get('num_train_epochs', 60)
    batch_size = config.get('batch_size', 4)
    top_k = config.get('top_k', 1)
    max_length = config.get('max_length', 384)
    checkpoint_epoch = config.get('checkpoint_epoch', None)
    gpt_lr = config.get('learning_rate_gpt', 5e-5)
    retriever_lr = config.get('learning_rate_retriever', 5e-5)
    output_dir = config.get('output_dir', './fine_tuned_model')
    validation_questions = config.get('validation_questions', [])

    penalty_config = config.get("penalties", {})
    ngram_n = penalty_config.get("ngram_repeat", {}).get("n", 3)
    ngram_weight = penalty_config.get("ngram_repeat", {}).get("weight", 0.1)
    keyword_weight = penalty_config.get("keyword_overlap", {}).get("weight", 0.4)
    short_min_words = penalty_config.get("short_answer", {}).get("min_words", 4)
    short_weight = penalty_config.get("short_answer", {}).get("weight", 0.01)

    ddp_enabled = args.ddp
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    print(f"STARTUP: SLURM_PROCID={os.environ.get('SLURM_PROCID')}, "
          f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}, "
          f"SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}, "
          f"ddp_enabled={ddp_enabled}, rank={rank}, world_size={world_size}")

    if ddp_enabled:
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ.setdefault('MASTER_ADDR', os.environ.get('MASTER_ADDR', 'localhost'))
        os.environ.setdefault('MASTER_PORT', os.environ.get('MASTER_PORT', '12355'))
        dist.init_process_group(backend="nccl", init_method="env://")

    if args.build_index and (rank == 0 or not ddp_enabled):
        print("Building FAISS index and embeddings...")
        build_or_load_faiss_index(wiki_passages_path, retriever_model_name, faiss_index_path, embedding_path)

    if ddp_enabled:
        dist.barrier()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fine_tune_rag_ddp(
        rank, world_size, local_rank, ddp_enabled,
        qa_path, wiki_passages_path, faiss_index_path, retriever_model_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        top_k=top_k,
        resume_epoch=checkpoint_epoch,
        max_length=max_length,
        device=device,
        gpt_lr=gpt_lr,
        retriever_lr=retriever_lr,
        ngram_n=ngram_n,
        ngram_weight=ngram_weight,
        keyword_weight=keyword_weight,
        short_min_words=short_min_words,
        short_weight=short_weight,
        validation_questions=validation_questions
    )

    if rank == 0 or not ddp_enabled:
        retriever = SentenceTransformer(retriever_model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(
            f'{output_dir}/epoch_{num_train_epochs}')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(f'{output_dir}/epoch_{num_train_epochs}').to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for sample_question in validation_questions:
            q_text, retrieved_text, answer = generate_rag_answer(
                sample_question, model, tokenizer, retriever,
                faiss_index_path, wiki_passages_path, top_k=top_k,
                max_length=max_length,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            print(f"QUESTION:\n{q_text}\n")
            print(f"RETRIEVED CONTEXT:\n{retrieved_text}\n")
            print(f"ANSWER:\n{answer}\n")