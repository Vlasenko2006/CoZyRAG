# 🛋️ CoZyRAG: Contextual Zero-Yawn Retrieval Augmented Generation

Welcome to my homemade cozzy Contextual Zero-Yawn Retrieval Augmented Generation **CoZyRAG**, the only AI model brave enough to answer questions, retrieve context, and still have time for a nap on the digital couch! 

### What is CoZyRAG?

CoZyRAG stands for **Contextual Zero-Yawn Retrieval Augmented Generation**. It's not just another RAG model—it's the RAG model that brings together distributed deep learning, sentence transformers, FAISS-powered passage search, and enough penalties to make your answers feel guilty for being short, repetitive, or too keyword-happy.

- **Distributed Data Parallel**: Because teamwork makes the dream work! (And your GPU cluster is tired of being lonely.)
- **FAISS Retrieval**: Finds the most relevant passages faster than your dog finds snacks.
- **GPT2LMHeadModel**: Generates answers smoother than grandma's mashed potatoes.
- **Penalties**: N-gram repeats? Keyword overlaps? Short answers? Not in this house!
- **MLflow**: Optional experiment tracking—because you know you love those metrics.
- **Sentence Transformers**: For context filtering so sharp, it could slice bread.
### Minimalistic Model and Data Sets

**Model Set**  
To minimize computational resources while maintaining strong performance, we use the pretrained `GPT2 Distilled` model as the LLM and `all-MiniLM-L6-v2` as the retriever.  
The `GPT2 Distilled` model, with 84M parameters, is at least twice as compact as the smallest original GPT-2. Although its text generation and answering capabilities are somewhat reduced compared to the full GPT-2, the distillation process allows it to inherit most of the essential functionality of the original model.  
The `all-MiniLM-L6-v2` is a BERT-style retriever with 6 layers and just 22M parameters. Due to its compact size, it can process requests up to 384 tokens (about 128 words), yet it reliably retrieves relevant texts from the provided corpus.  
Both neural networks are pretrained and further fine-tuned during training:  
- The LLM (`GPT2 Distilled`) answers questions based on retrieved texts.  
- The retriever (`all-MiniLM-L6-v2`) generates valuable context for the LLM.

**Data Set**  
The dataset consists of simple Q/A pairs: each question is one sentence, each answer is one word or one sentence. The retrieval corpus is based on Wikipedia text.

- **Pros:**  
  The `GPT2 Distilled` - `all-MiniLM-L6-v2` pair is lightweight, easy to understand and train, requires minimal hardware, and forms a minimal working RAG (Retrieval-Augmented Generation) infrastructure.

- **Cons:**  
  It typically generates answers of one full sentence, rarely two. Achieving optimal accuracy requires careful selection of hyperparameters.

---

### Can It Run on a CPU Home Desktop?

Yes, it’s true that running an LLM on a standard home desktop CPU is uncommon, but CoZyRAG’s minimalistic setup makes it possible. Training for one epoch on a dataset of 20K Q/A pairs may take more than a day, so it's feasible, but patience is required—you may need to wait a couple of days for the first results.


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
