"""
FastAPI backend for the Research RAG GUI.
Run with: uvicorn app:app --reload --port 8000

Install extras:
    pip install fastapi uvicorn python-multipart aiofiles
"""

import asyncio
import base64
import io
import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Your existing modules ──────────────────────────────────────────────────────
from ingest import (
    delete_pdf,
    is_already_ingested,
    load_registry,
    process_and_ingest,   # see note below *
)
from retrieval import resolve_paper_title
from generate import chat

# * If you don't have a `process_and_ingest` helper yet, add this to ingest.py:
#
#   def process_and_ingest(pdf_path: Path):
#       from llama_index.core import VectorStoreIndex
#       doc_nodes = process_research_paper(pdf_path)
#       paper_title = doc_nodes[0].metadata.get("paper_title", "Unknown") if doc_nodes else "Unknown"
#       if Path(BACKUP_PATH).exists():
#           existing = load_nodes(BACKUP_PATH)
#           all_nodes = existing + doc_nodes
#       else:
#           all_nodes = doc_nodes
#       save_nodes(all_nodes, BACKUP_PATH)
#       index = VectorStoreIndex(doc_nodes, storage_context=storage_context,
#                                embed_model=embed_model, show_progress=False)
#       register_pdf(pdf_path.name, paper_title, len(doc_nodes))
#       return paper_title, len(doc_nodes)

PDF_DIR = Path("/app/documents")
PDF_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Research RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory conversation store (keyed by session_id) ────────────────────────
sessions: dict[str, list] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    query: str
    paper_titles: list[str] = []   # empty = search all


class SourceRef(BaseModel):
    paper: str
    section: str
    page: int | None
    type: str
    score: float | str
    scope: str
    # populated for figures/tables
    node_text: str | None = None
    figure_b64: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# PDF management
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/pdfs")
def list_pdfs():
    """Return all ingested PDFs from the registry."""
    registry = load_registry()
    return [
        {
            "filename": fname,
            "paper_title": meta["paper_title"],
            "node_count": meta["node_count"],
            "ingested_at": meta["ingested_at"],
        }
        for fname, meta in registry.items()
    ]


@app.post("/pdfs/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Save a PDF, run ingestion, return the paper title."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    dest = PDF_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    if is_already_ingested(file.filename):
        return {"status": "already_ingested", "filename": file.filename}

    # Run ingestion in a thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    try:
        paper_title, node_count = await loop.run_in_executor(
            None, process_and_ingest, dest
        )
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"Ingestion failed: {e}")

    return {
        "status": "ingested",
        "filename": file.filename,
        "paper_title": paper_title,
        "node_count": node_count,
    }

@app.get("/pdfs/view/{filename:path}")
def serve_pdf(filename: str):
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", headers={"Access-Control-Allow-Origin": "*"})

@app.delete("/pdfs/{filename:path}")
def remove_pdf(filename: str):
    """Delete a PDF from the vector store and registry."""
    try:
        delete_pdf(filename)
    except Exception as e:
        raise HTTPException(500, str(e))

    pdf_path = PDF_DIR / filename
    pdf_path.unlink(missing_ok=True)
    return {"status": "deleted", "filename": filename}


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    history = sessions.get(req.session_id, [])

    resolved_titles = (
        [resolve_paper_title(t) for t in req.paper_titles]
        if req.paper_titles else None
    )

    result = chat(
        query=req.query,
        history=history,
        paper_title=resolved_titles[0] if resolved_titles and len(resolved_titles) == 1 else None,
        paper_titles=resolved_titles if resolved_titles and len(resolved_titles) > 1 else None,
    )

    sessions[req.session_id] = result["history"]

    rich_sources = []
    for s in result["sources"]:
        entry = dict(s)
        entry["node_text"] = s.get("node_text", "")
        entry["bbox"] = {
            "l": s.get("bbox_l"),
            "t": s.get("bbox_t"),
            "r": s.get("bbox_r"),
            "b": s.get("bbox_b"),
        }
        entry["source_pdf"] = s.get("source_pdf")

        pil_image = s.get("pil_image")
        if pil_image is not None:
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            entry["figure_b64"] = base64.b64encode(buf.getvalue()).decode()
        else:
            entry["figure_b64"] = None

        entry.pop("pil_image", None)
        rich_sources.append(entry)

    return {
        "answer": result["answer"],
        "sources": rich_sources,
        "scores": result["scores"],
        "session_id": req.session_id,
    }


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "reset"}