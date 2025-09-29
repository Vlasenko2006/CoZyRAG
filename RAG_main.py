import torch
import torch.distributed as dist
import os
import argparse
import yaml
import logging
from fine_tune_rag_ddp import fine_tune_rag_ddp
from build_or_load_faiss_index import build_or_load_faiss_index



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



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
    model_name = config.get('model_name','distilgpt2')

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
        rank, world_size,
        local_rank,
        ddp_enabled,
        qa_path, 
        wiki_passages_path,
        faiss_index_path,
        retriever_model_name,
        logger,
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
        validation_questions=validation_questions, 
        model_name = model_name
        )

