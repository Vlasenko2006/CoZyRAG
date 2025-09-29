
import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer





def build_or_load_faiss_index(passages_path, 
                              retriever_model_name,
                              faiss_index_path,
                              embedding_path
                              ):
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