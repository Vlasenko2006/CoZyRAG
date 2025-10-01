# 🛋️ CoZyRAG: Contextual Zero-Yawn Retrieval Augmented Generation

Welcome to my homemade cozzy Contextual Zero-Yawn Retrieval Augmented Generation **CoZyRAG**, the only AI model brave enough to answer questions, retrieve context, and still have time for a nap on the digital couch! 

### What is CoZyRAG?

CoZyRAG stands for **Contextual Zero-Yawn Retrieval Augmented Generation**. It's not just another RAG model—it's the RAG model that brings together distributed deep learning, sentence transformers, FAISS-powered passage search, and enough penalties to make your answers feel guilty for being short, repetitive, or too keyword-happy.

- **Distributed Data Parallel**: Because teamwork makes the dream work! (And your GPU cluster is tired of being lonely.)
- **FAISS Retrieval**: Finds the most relevant passages faster than your dog finds snacks.
- **GPT2LMHeadModel**: Generates answers smoother than grandma's mashed potatoes.
- **Penalties**: N-gram repeats? Keyword overlaps? Not in this house!
- **MLflow**: Optional experiment tracking—because you know you love those metrics.
- **Sentence Transformers**: For context filtering so sharp, it could slice bread.
### Minimalistic Model and Data Sets

## Model Set

CoZyRAG offers flexibility in choosing any LLM and retriever model. However, to minimize computational resources while maintaining strong performance, we use the pretrained `GPT2 Distilled` model as the LLM and `all-MiniLM-L6-v2` as the retriever. 
The `GPT2 Distilled` model, with 84M parameters, is at least twice as compact as the smallest original GPT-2. Although its text generation and answering capabilities are somewhat reduced compared to the full GPT-2, the distillation process allows it to inherit most essential functionality from the original model.  
The `all-MiniLM-L6-v2` is a BERT-style retriever with 6 layers and just 22M parameters. Thanks to its compact size, it can process requests up to 384 tokens (about 128 words), yet it reliably retrieves relevant texts from the provided corpus.  
Both neural networks are pretrained and further fine-tuned during training:
- The LLM (`GPT2 Distilled`) answers questions based on retrieved texts.
- The retriever (`all-MiniLM-L6-v2`) provides valuable context for the LLM.

### Data Set

The dataset consists of 89K simple Q/A pairs: each question is one sentence, each answer is one word, a group of words, or one sentence. The retrieval corpus is comprised of Wikipedia articles. Here is an example of the Q/A in the dataset:

**Q:** To whom did the Virgin Mary allegedly appear in 1858 in Lourdes, France?  
**A:** Saint Bernadette Soubirous

**Q:** What is in front of the Notre Dame Main Building?  
**A:** A copper statue of Christ

**Q:** How often is Notre Dame's "The Juggler" published?  
**A:** Twice

### Generalization Capabilities Without RAG

Trained on a simple dataset without RAG, `GPT2 Distilled` generalizes well to unseen data, provided that similar questions exist in the dataset. For instance, if the dataset contains various questions like `What is the capital of X?` (where X is a country), the model learns to associate the question pattern with typical answers. During validation, if you ask `What is the capital of France?` and this exact question was not in the training set, but the model has seen sentences where `France`, `capital`, and `Paris` appear together, it may successfully infer the correct answer. However, if these words do not appear together—or at all—in the training set, the model is likely to fail, even though it was pretrained on a large corpus. This is precisely where RAG becomes important.

### Generalization Capabilities With RAG

RAG (Retrieval-Augmented Generation) is designed to supply the LLM with required, previously unseen information in the prompt, thereby avoiding the need for retraining. The original LLM remains focused on answer generation and is not modified. Instead, a smaller retrieval model is trained to fetch necessary information from memory or the internet and provide it to the LLM along with the question. This approach avoids the heavy computational cost of retraining or fine-tuning large LLMs.

### My Case

Both `GPT2 Distilled` and `all-MiniLM-L6-v2` are pretrained, but both still require fine-tuning and are alternately tuned together.

- **Pros:**  
  The `GPT2 Distilled` and `all-MiniLM-L6-v2` pair is lightweight, easy to understand and train, requires minimal hardware, and forms a minimal working RAG (Retrieval-Augmented Generation) infrastructure. It can generalize context from the retriever and generate coherent answers. One-sentence, one-word answers simplify training.

- **Cons:**  
  Due to the simplicity of the dataset, the system typically generates answers of one word or one full sentence, rarely two. Achieving optimal accuracy requires careful selection of hyperparameters.
---
### Examples

Below are examples of answers obtained during the validation stage. None of the questions (or corresponding information) appeared in the training set.

**Q: Who was Albert Einstein?**  
A1: German physicist  
A2: American physicist  
A3: The first Nobel Prize winner  
A4: Pioneer in the theory of general relativity

**Q: What is machine learning?**  
A1: Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical models

**Q: What are black holes?**  
A1: The black hole is a class of astronomical objects that have undergone gravitational collapse  
A2: A class of astronomical objects with no net positive charge

**Q: Describe the function of mitochondria.**  
A1: The production of ATP

### Comment

The answers to the first question are generally correct, except for A3: the **first** Nobel Prize winner was Wilhelm Conrad Röntgen. A1 for the second question and A2 for the third question are not entirely accurate, but notably, these responses reflect exactly what the retriever fetched from the Wikipedia corpus at the time the model was trained. In this case, both the LLM and retriever demonstrated coherent and expected performance—the inaccuracies highlight the need for corrections in the Wikipedia articles themselves.

Summarizing the validation results:  
The LLM-Retriever pair in CoZyRAG works as intended, correctly answering most questions, though it cannot generate large or highly detailed responses. Using a different dataset with more comprehensive answers and a larger LLM may improve this limitation.

### Can It Run on a home desktop with CPU?

It’s true that running an LLM on a standard home desktop CPU is uncommon, but CoZyRAG’s minimalistic setup still makes it possible. Training for one epoch on a dataset of 20K Q/A pairs may take more than a day, so it's feasible, but patience is required—you may need to wait a couple of days for the first results.


## Features

- Alternates training between retriever and generator like a relay race, but with more existential dread.
- Custom penalties for bad behavior. (No, really, it keeps your model honest.)
- Multi-GPU distributed training. Because why suffer alone?
- Easy to run, easier to break (please don’t).

# Module Overview

1. **`Rag_main.py`**  
   The main driver program. Loads configurations from `conf/config.yaml` and starts the RAG pipeline.

2. **`RAGQADataset.py`**  
   The Dataset class is responsible for creating the training dataset, including QA pairs and RAG prompts.

3. **`build_or_load_faiss_index.py`**  
   Builds or loads Facebook AI Similarity Search (FAISS) indexes required by the retriever.

4. **`fine_tune_rag_ddp.py`**  
   Training subroutine that fine-tunes and saves checkpoints for both the GPT model and the retriever.

5. **`generate_rag_answer.py`**  
   Generates human-readable GPT answers by converting tokens to words and removing internal tokens.

6. **`postprocess_generated_answer.py`**  
   Used by `generate_rag_answer.py` to filter internal tokens such as `<SEP>`, etc.

7. **`my_loss_functions.py`**  
   Custom loss functions for training—self-explanatory.


## How to Get Cozy

1. Prepare your environment (see below).
2. Bring your own QA data and wiki passages(see below).
3. Build your FAISS index(see below).
4. Train like a champion, sleep like a winner.

---

# Installation

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
# Quick Start

1. **Prepare the Q/A Dataset**  
   Get and prepare the Q/A dataset as described in [QA-bot](https://github.com/Vlasenko2006/QA_bot).  
   Run the script `convert_qa_format.py` to convert Q/A pairs from the QA-bot format to the format used by CoZyRAG.  
   Ensure that the paths and names of the input and output directories match those of your QA files and the ones you intend to use in your model.

2. **Prepare the Wikipedia Corpus**  
   Download and extract the Wikipedia corpus as described in [wikiextractor](https://github.com/attardi/wikiextractor).  
   Run the script `chunking_and_indexcing.py` to split the extracted Wikipedia corpus into small chunks with unique ID, Title, and content (minimum 30 characters). This step simplifies the retriever's work.

3. **Create Indexes for the Retriever**  
   Create indexes from these chunks for the retriever.  
   You only need to do this once.  
   Run the `exec.bash` script, adjusting it for your HPC environment.

4. **Start the RAG Pipeline**  
   Adjust the `exec.bash` script for your HPC system, and run it to start the RAG pipeline.

5. **Enjoy the Results!**

# Disclaimer

CoZyRAG is not responsible for:
- Sudden naps induced by long training epochs.
- Existential crises after reading its answers.
- Your GPU catching on fire (it probably won’t, but you’ve been warned).

Enjoy your stay. 🛋️
