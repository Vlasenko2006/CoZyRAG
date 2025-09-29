# 🛋️ CoZyRAG: Contextual Zero-Yawn Retrieval Augmented Generation

Welcome to my homemade Contextual Zero-Yawn Retrieval Augmented Generation **CoZyRAG**, the only AI model brave enough to answer questions, retrieve context, and still have time for a nap on the digital couch! 

### What is CoZyRAG?

CoZyRAG stands for **Contextual Zero-Yawn Retrieval Augmented Generation**. It's not just another RAG model—it's the RAG model that brings together distributed deep learning, sentence transformers, FAISS-powered passage search, and enough penalties to make your answers feel guilty for being short, repetitive, or too keyword-happy.

- **Distributed Data Parallel**: Because teamwork makes the dream work! (And your GPU cluster is tired of being lonely.)
- **FAISS Retrieval**: Finds the most relevant passages faster than your dog finds snacks.
- **GPT2LMHeadModel**: Generates answers smoother than grandma's mashed potatoes.
- **Penalties**: N-gram repeats? Keyword overlaps? Short answers? Not in this house!
- **MLflow**: Optional experiment tracking—because you know you love those metrics.
- **Sentence Transformers**: For context filtering so sharp, it could slice bread.

### Features

- Alternates training between retriever and generator like a relay race, but with more existential dread.
- Custom penalties for bad behavior. (No, really, it keeps your model honest.)
- Multi-GPU distributed training. Because why suffer alone?
- Easy to run, easier to break (please don’t).

### How to Get Cozy

1. Prepare your environment (see below).
2. Bring your own QA data and wiki passages.
3. Build your FAISS index.
4. Train like a champion, sleep like a winner.

---

### Installation

#### Conda

```bash
conda env create -f environment.yml
conda activate cozyrag
```

#### Pip

```bash
python -m venv cozyrag
source cozyrag/bin/activate
pip install -r requirements.txt
```

---

### Usage

See `python your_cozyrag_script.py --help` for options.

---

### Disclaimer

CoZyRAG is not responsible for:
- Sudden naps induced by long training epochs.
- Existential crises after reading its answers.
- Your GPU catching on fire (it probably won’t, but you’ve been warned).

Enjoy your stay. 🛋️
