# 🚀 RAG End-to-End Pipeline

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline in Python. RAG improves LLM responses by fetching relevant, up-to-date information from a custom knowledge base and using it as context for the generative model.

## 📖 Overview

The pipeline handles all stages from raw document ingestion to final query generation. It is designed to be **modular** and **local-first**, using a locally managed vector store and a Hugging Face model for the LLM.

| Stage | Tool/Method |
| :--- | :--- |
| **Data Ingestion** | Custom `loader.py` for documents |
| **Text Chunking** | Configurable `chunk_size` and `chunk_overlap` |
| **Embeddings** | **SentenceTransformer** (e.g., `all-MiniLM-L6-v2`) |
| **Vector Store** | **FAISS** (for fast similarity search and persistence) |
| **Generation** | Local Hugging Face LLM (e.g., `google/flan-t5-small`) |

### 🏗️ Architecture

1.  **Ingestion Pipeline:** Documents → Ingestion → Chunking → Embeddings → **Vector Store (FAISS)**
2.  **RAG Query:** Query → Similarity Search in FAISS → Retrieve Top-k Chunks → Prompt LLM → Generated Result

---

## ✅ Features

* **Modular Pipeline:** Each stage (`loader`, `chunker`, `embedding`, `vector store`, `query`) is separate.
* **Local Inference Support:** Integrates local LLM models (via `langchain-community`).
* **Persistent Index:** Embeddings and metadata are stored in a persistent directory (`faiss_store/`).
* **Summarization:** Provides context-aware summarization based on retrieved documents.

---

## 🛠️ Getting Started

### Prerequisites

* **Python 3.10 / 3.11** (recommended)
* Conda or venv (for dependency isolation).
* Basic familiarity with Conda and `pip`.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/sanathbn27/RAG_end_to_end_pipeline.git](https://github.com/sanathbn27/RAG_end_to_end_pipeline.git)
    cd RAG_end_to_end_pipeline
    ```
2.  **Create & Activate Environment:**
    ```bash
    conda create -n rag_env python=3.11
    conda activate rag_env
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install langchain-community # For local LLM integration
    ```

---

## ⚙️ Usage

### 1. Ingest, Chunk, and Embed

Place your raw documents (e.g., PDF, TXT files) in the **`data/`** directory. Run the following command once to process them and build the vector store:

```bash
python -m src.embedding
```
This will create the necessary index and metadata files in the faiss_store/ directory.

### 2. Run Search and Summerize

Execute the search script to run a default query and see the RAG process in action:

```bash
python -m src.search
```

### 3. Run the full pipeline

Execute the `main.py` script to build the pipeline:

```bash
python main.py
```

## Project Structure
```bash
RAG_end_to_end_pipeline/
├─ data/                     # 📂 Raw documents for indexing
├─ faiss_store/              # 💾 Persistent FAISS index and metadata
├─ src/
│   ├─ embedding.py          # Script for ingestion, chunking, and embedding
│   ├─ loader.py             # Logic for loading documents from 'data/'
│   ├─ vectorstore.py        # FAISS implementation wrapper
│   └─ search.py             # RAGSearch class (query, retrieval, summarization)
├─ main.py                   # 🚀 Unified entry point for demonstration
├─ requirements.txt
└─ README.md
```