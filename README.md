# Research Copilot

A local, privacy-first Retrieval-Augmented Generation (RAG) assistant for academic literature synthesis. Built with **FastAPI**, **ChromaDB**, **Ollama (`llama3.2` & `nomic-embed-text`)**, and a vanilla JS dashboard. All data, embeddings, and inference remain 100% on your machine.

---
<img width="1012" height="849" alt="image" src="https://github.com/user-attachments/assets/0cf1e663-1f0a-40a1-a02c-614916a4cd26" />

## Features

* **Local & Private:** Zero API costs; papers and queries never leave your hardware.
* **Batch PDF Ingestion:** Upload and index multiple research papers directly from the UI.
* **Precise Metadata Filtering:** Query across all documents or target specific papers.
* **Traceable Inline Citations:** Context-mapped bracketed citations (e.g., `[1]`) with automated bibliographies.
* **Comparative & Timeline Analysis:** Dynamic $N$-paper contrastive matrices, chronological evolution tracking, and multi-hop deep dives.
* **Persistent Memory:** Browser local storage retains Q&A history without an external database.

---

## Quick Start

### 1. Prerequisites

* **Python 3.10+**
* **Ollama**:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text

```



### 2. Setup

```bash
# Clone the repository
git clone https://github.com/SuvarnaSatvika/Research_buddy.git
cd Research-copilot

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # macOS/Linux

# Install dependencies
pip install fastapi uvicorn langchain langchain-community langchain-chroma langchain-ollama pypdf python-multipart

```

### 3. Run

1. **Start Backend:**
```bash
python -m uvicorn server:app --reload --port 8000

```


2. **Open Frontend:**
Double-click `frontend/index.html` in your browser.

---

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/ingest` | Upload and vectorize one or more PDFs |
| `GET` | `/api/papers` | List all indexed paper titles |
| `POST` | `/api/ask` | Grounded Q&A with inline citations |
| `POST` | `/api/analyze/compare` | Dynamic $N$-paper contrastive matrix |
| `POST` | `/api/analyze/timeline` | Chronological topic synthesis |
| `POST` | `/api/research/deep-dive` | Multi-hop reasoning chain |
| `DELETE` | `/api/papers/delete` | Remove a paper and its vectors |
