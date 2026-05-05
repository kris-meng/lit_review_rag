# LitRAG 🔬

A research-grade RAG (Retrieval-Augmented Generation) system for academic literature review. Drop in your PDFs and chat with your papers — ask questions, compare findings across studies, and retrieve specific figures, tables, and equations with source-linked answers.

> **Status:** Work in progress. Currently runs via Docker.

---

## Features

- **Multi-paper retrieval** — semantic + keyword hybrid search across your entire library
- **Focused querying** — pin specific papers to scope your questions
- **Direct asset lookup** — ask about Figure 3 or Table 2 and get the exact item, not just nearby text
- **Figure understanding** — vision model (Qwen2.5-VL) describes figures at ingestion time and re-queries them on demand
- **Formula OCR** — extracts and summarises LaTeX equations using pix2tex
- **Source hyperlinks** — every answer links back to the exact chunk, page, and section it came from (IPR)
- **Expand-query fallback** — if the first retrieval scores poorly on groundedness/relevance, automatically retries with expanded queries before giving up
- **Web UI** — React frontend with drag-and-drop PDF ingestion, paper focus toggling, and a source detail modal (IPR)

---

## Stack

| Component | Tool |
|---|---|
| PDF parsing | Docling |
| Embeddings | Nomic `nomic-embed-text-v2-moe` via Ollama |
| Vector store | Qdrant (local) |
| LLM | Qwen2.5:7b via Ollama |
| Vision model | Qwen2.5-VL:7b via Ollama |
| Formula OCR | pix2tex |
| Backend | FastAPI |
| Frontend | React + Vite |

---

## Requirements

- Docker
- [Ollama](https://ollama.com) running on your host machine with the following models pulled:

```bash
ollama pull nomic-embed-text-v2-moe
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/kris-meng/lit_review_rag
cd lit_review_rag
```

### 2. Build and run the Docker container

```bash
docker build -t litrag .
docker run -p 8000:8000 -v $(pwd)/documents:/app/documents -v $(pwd)/qdrant_db:/app/qdrant_db litrag
```

### 3. Start the API

Inside the container:

```bash
cd /app/app
uvicorn app:app --port 8000
```

### 4. Start the frontend

On your host machine:

```bash
cd rag_gui
npm install
npm run dev
```

Then open `http://localhost:5173`.

---

## Usage

### Ingesting papers

Drop PDFs into the web UI or place them in the `documents/` folder and run:

```bash
cd /app/app
python ingest.py
```

### Chatting

- **Click a paper** in the left panel to focus your query on it
- **Click multiple papers** to search across a specific subset
- **Leave all unselected** to search your entire library
- **Click any source chip** in an answer to view the exact retrieved chunk

### Commands (CLI mode)

```
/paper <name>        filter to a specific paper
/papers <n1>, <n2>   filter to multiple papers
/clear               search all papers
/reset               reset conversation history
/quit                exit
```

---

## Project Structure

```
/app/app/
├── app.py          # FastAPI server
├── ingest.py       # PDF registry + ingestion pipeline
├── embedding.py    # Docling parsing, chunking, figure/formula processing
├── retrieval.py    # Vector + keyword retrieval, hybrid search
├── generate.py     # Query contextualization, answer generation, relevancy scoring
├── db.py           # Shared Qdrant client
└── rag_gui/        # React frontend
```

---

## Known Limitations

- Qdrant runs in local file mode — only one process can access it at a time (no `--reload` with uvicorn)
- pix2tex requires `timm==0.5.4` which conflicts with newer `sentence-transformers` — reranking is disabled for now
- Formula extraction is best-effort; image-only formulas in scanned PDFs may be skipped
- `resolve_paper_title` scrolls only 100 points to find titles — may miss papers in very large libraries

---

## Roadmap

- [ ] PDF viewer with bbox highlighting for source chunks
- [ ] Qdrant server mode for concurrent access and native full-text search
- [ ] Cross-encoder reranking (blocked by timm conflict)
- [ ] Supplementary figure/table support
- [ ] Multi-modal answers (inline figure rendering in chat)
