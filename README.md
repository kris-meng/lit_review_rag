# LitRAG — Literature Review Assistant

A research-grade RAG system for academic literature review. Drop in PDFs, chat with your papers, and retrieve figures, tables, and equations with fully source-traced answers.

---

## What it does

- **Structured PDF ingestion** — extracts section-aware text chunks, figure descriptions (via Qwen2.5-VL), and LaTeX formulas (via pix2tex), treating each as a distinct retrieval target
- **Hybrid retrieval** — combines dense semantic search (Nomic embeddings + Qdrant) with keyword retrieval and automatic query expansion
- **Direct asset lookup** — detects queries about specific figures, tables, or equations and fetches them directly, bypassing vector search
- **Answer grounding** — every response is scored for groundedness and relevance; low-scoring answers trigger automatic retry before falling back
- **Adaptive search scope** — pin specific papers for focused queries, or search across the full library; scope is also inferred automatically from the query
- **Source traceability** — every answer links back to the exact chunk, section, page, and paper, with a UI panel showing the retrieved passage or figure

---

## Stack

| Layer | Tools |
|---|---|
| Document parsing | Docling, pix2tex, PyMuPDF |
| Vision model | Qwen2.5-VL (via Ollama) |
| Embeddings | Nomic `nomic-embed-text-v2-moe` (via Ollama) |
| Vector store | Qdrant |
| Retrieval & indexing | LlamaIndex |
| LLM | Qwen2.5:7b (via Ollama) |
| Backend | FastAPI |
| Frontend | React + Vite |

---

## Project structure
├── app/
│   ├── app.py          # FastAPI server
│   ├── chunk.py        # PDF parsing and node construction
│   ├── ingest.py       # Ingestion pipeline and registry
│   ├── retrieval.py    # Hybrid retrieval logic
│   ├── generate.py     # Query handling and answer generation
│   └── db.py           # Qdrant client and vector store setup
│   └── requirements.txt
│   └── rag_gui/        # Frontend
│       └── src/
│           └── App.jsx     # React UI
├── documents/          # PDFs mounted into the container
├── qdrant_db/          # Vector database
└── docker-compose.yml

---

## Getting started

### Prerequisites

- [Docker](https://www.docker.com/)
- [Ollama](https://ollama.com/) running locally

### 1. Pull required models

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
ollama pull nomic-embed-text-v2-moe
```

### 2. Clone the repo

```bash
git clone https://github.com/kris-meng/lit_review_rag
cd lit_review_rag
```

### 3. Start the backend

```bash
docker compose up --build
```

The FastAPI server will be available at `http://localhost:8000`.

### 4. Start the frontend

```bash
cd rag_gui
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`.

---

## Usage

1. Drop a PDF into the sidebar — it will be ingested and indexed automatically
2. Click a paper thumbnail to focus queries on that paper, or leave all unselected to search across everything
3. Ask questions in natural language — answers include clickable source citations linked to the exact page and section
4. Click any source chip to open the retrieved passage or figure in the side panel

---

## Future work

- Cross-encoder reranking for improved retrieval precision
- Qdrant server mode for concurrent multi-user access and native full-text search
