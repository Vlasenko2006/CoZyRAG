import os
import re
import json
import torch
import torch.distributed as dist
from tqdm import tqdm

def extract_passages_from_wiki_folder_ddp(
    root_dir, output_file, min_passage_length=30, ddp_enabled=False, rank=0, world_size=1
):
    """
    DDP-enabled: each process works on a subset of folders; results are written to separate files and merged by rank 0.

    Args:
        root_dir (str): Path to root "WIKI/wikiextractor/extracted/"
        output_file (str): Final output file (JSONL, merged by rank 0)
        min_passage_length (int): Minimum passage length to keep
        ddp_enabled (bool): Whether to use DDP
        rank (int): This process rank
        world_size (int): Total DDP processes
    """
    folders = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    # DDP: split folders between ranks (even split)
    assigned_folders = [f for i, f in enumerate(folders) if i % world_size == rank]

    passage_id = 0
    temp_output = f"{output_file}.part{rank}"

    with open(temp_output, "w", encoding="utf-8") as outf:
        for folder in tqdm(assigned_folders, desc=f"Rank {rank} folders"):
            files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("wiki_")])
            for file in tqdm(files, desc=os.path.basename(folder), leave=False):
                with open(file, encoding="utf-8") as fin:
                    doc_buffer = []
                    in_doc = False
                    doc_id, doc_title = None, None
                    for line in fin:
                        if line.startswith("<doc "):
                            in_doc = True
                            doc_buffer = [line]
                            doc_id = re.search(r'id="(\d+)"', line)
                            doc_id = doc_id.group(1) if doc_id else None
                            doc_title = re.search(r'title="([^"]+)"', line)
                            doc_title = doc_title.group(1) if doc_title else None
                        elif line.startswith("</doc>"):
                            doc_buffer.append(line)
                            in_doc = False
                            # Process doc_buffer
                            doc_content = "".join(doc_buffer)
                            paragraphs = re.sub(r"<.*?>", "", doc_content).split('\n\n')
                            for para in paragraphs:
                                para = para.strip()
                                if len(para) >= min_passage_length:
                                    passage = {
                                        "id": f"{doc_id}_{passage_id}_r{rank}",
                                        "title": doc_title,
                                        "text": para
                                    }
                                    outf.write(json.dumps(passage, ensure_ascii=False) + "\n")
                                    passage_id += 1
                        elif in_doc:
                            doc_buffer.append(line)
    # Synchronize before merging
    if ddp_enabled:
        dist.barrier()
        if rank == 0:
            with open(output_file, "w", encoding="utf-8") as outf:
                for r in range(world_size):
                    temp_part = f"{output_file}.part{r}"
                    with open(temp_part, "r", encoding="utf-8") as inf:
                        for line in inf:
                            outf.write(line)
                    os.remove(temp_part)
            print(f"Merged all parts to {output_file}")
        dist.barrier()
    else:
        print(f"Extraction finished. Saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true", help="Enable DDP parallel processing")
    parser.add_argument("--output", type=str, default="wiki_passages.jsonl", help="Output JSONL file")
    parser.add_argument("--min_length", type=int, default=30, help="Minimum passage length")
    args = parser.parse_args()

    WIKI_ROOT = "WIKI/wikiextractor/extracted"

    ddp_enabled = args.ddp
    if ddp_enabled:
        os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    extract_passages_from_wiki_folder_ddp(
        WIKI_ROOT,
        args.output,
        min_passage_length=args.min_length,
        ddp_enabled=ddp_enabled,
        rank=rank,
        world_size=world_size
    )

    if ddp_enabled:
        dist.destroy_process_group()
