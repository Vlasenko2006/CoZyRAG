import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data.distributed import DistributedSampler
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from RAGQADataset import RAGQADataset
import mlflow
from torch.utils.data import DataLoader
from my_loss_functions import contrastive_loss
from my_loss_functions import differentiable_ngram_repeat_penalty, \
    differentiable_keyword_overlap_penalty, \
    differentiable_short_answer_penalty

from generate_rag_answer  import generate_rag_answer


try:
    mlflow_available = True
except ImportError:
    mlflow_available = False
    




def fine_tune_rag_ddp(rank, 
                      world_size, 
                      local_rank,
                      ddp_enabled,
                      qa_path, 
                      wiki_passages_path,
                      faiss_index_path, 
                      retriever_model_name,
                      logger,
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
                      validation_questions=None,
                      model_name = 'distilgpt2'
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
