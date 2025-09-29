#!/bin/bash

#SBATCH --job-name=rag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=07:59:00
#SBATCH --account=gg0302
#SBATCH --partition=gpu
#SBATCH --error=e-wiki_RAG9.out
#SBATCH --output=wiki_RAG9.out
#SBATCH --exclusive
#SBATCH --mem=0

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 50000))

module load git
export PATH=/sw/spack-levante/git-2.43.7-2ofazl/bin:$PATH

# Build index (serial, single rank)
echo "Building FAISS index..."
# /work/gg0302/g260141/RAG/rag_env/bin/python3.10 RAG.py --build_index
if [ $? -ne 0 ]; then
  echo "Index building failed! Aborting job."
  exit 1
fi
echo "Building indexes completed ==="

# Export variables for distributed run
export MASTER_ADDR MASTER_PORT

# Start distributed training
srun /work/gg0302/g260141/RAG/rag_env/bin/python3.10 RAG.py --ddp