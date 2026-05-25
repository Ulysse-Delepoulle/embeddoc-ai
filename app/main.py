import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.database import (
    init_db, delete_document_chunks,
    save_document, list_documents, delete_document_record,
)
from app.services.retriever import retrieve
from app.services.llm import answer

os.makedirs("documents", exist_ok=True)

# In-memory job store  {job_id: {...}}
_jobs: dict = {}
_jobs_lock = threading.Lock()

# Conservative CPU embedding speed estimate (chunks/sec) for ETA
_EMBED_SPEED = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    filter_docs: list[str] = []
    history: list[dict] = []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_ingestion(job_id: str, file_path: str, filename: str) -> None:
    """Background thread: parse → chunk → embed, updating job state throughout."""
    from app.utils.pdf_parser import parse_pdf, chunk_pages
    from app.services.ingestion import embed_chunks

    try:
        # Phase 1 — parse PDF and split into chunks
        pages = parse_pdf(file_path, filename)
        chunks = chunk_pages(pages)
        chunk_count = len(chunks)

        with _jobs_lock:
            _jobs[job_id]["chunks_total"] = chunk_count
            _jobs[job_id]["estimated_seconds"] = max(5, chunk_count // _EMBED_SPEED)

        # Phase 2 — embed chunks and store in pgvector
        def on_progress(done: int) -> None:
            with _jobs_lock:
                _jobs[job_id]["chunks_done"] = done

        embed_chunks(chunks, on_progress)

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["chunks_done"] = chunk_count

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/docs")
def list_docs():
    """Return documents with their display names. Auto-registers any untracked PDFs on disk."""
    disk_files = sorted(f for f in os.listdir("documents") if f.lower().endswith(".pdf"))
    db_filenames = {d["filename"] for d in list_documents()}
    for f in disk_files:
        if f not in db_filenames:
            save_document(f, os.path.splitext(f)[0])
    return {"files": list_documents()}


@app.post("/ingest")
def ingest(file: UploadFile = File(...), document_name: str = ""):
    effective_name = document_name.strip() or os.path.splitext(file.filename)[0]

    # Save file and persist display name — return immediately so the UI can show progress
    file_path = f"documents/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    save_document(file.filename, effective_name)

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "display_name": effective_name,
            "filename": file.filename,
            "chunks_total": 0,   # unknown until parsing completes in the background
            "chunks_done": 0,
            "started_at": time.time(),
            "estimated_seconds": None,
        }

    threading.Thread(
        target=_run_ingestion, args=(job_id, file_path, file.filename), daemon=True
    ).start()

    return {
        "job_id": job_id,
        "display_name": effective_name,
        "filename": file.filename,
    }


@app.get("/status/{job_id}")
def get_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    elapsed = time.time() - job["started_at"]
    done = job["chunks_done"]
    total = job["chunks_total"]
    progress = round(done / total * 100) if total else 0

    # Refine ETA from actual embedding rate once we have enough data
    if done > 0 and elapsed > 0 and job["status"] == "processing":
        rate = done / elapsed
        remaining = max(0, round((total - done) / rate))
    else:
        remaining = job["estimated_seconds"]

    return {
        "status": job["status"],
        "display_name": job["display_name"],
        "filename": job["filename"],
        "chunks_done": done,
        "chunks_total": total,
        "progress": progress,
        "elapsed_seconds": round(elapsed),
        "remaining_seconds": remaining,
        "error": job.get("error"),
    }


@app.post("/ask")
def ask(request: AskRequest):
    chunks = retrieve(request.question, request.filter_docs or None)
    low_confidence = not chunks
    seen = set()
    sources = []
    for c in chunks:
        if not c["document_name"]:
            continue
        key = (c["document_name"], c["page_num"])
        if key not in seen:
            seen.add(key)
            sources.append({"document": c["document_name"], "page": c["page_num"]})

    return {"answer": answer(request.question, chunks, low_confidence, request.history), "sources": sources}


@app.delete("/ingest")
def delete_document(filename: str):
    deleted = delete_document_chunks(filename)
    file_path = f"documents/{filename}"
    if os.path.exists(file_path):
        os.remove(file_path)
    delete_document_record(filename)
    return {"message": f"Deleted {deleted} chunks for '{filename}'."}


app.mount("/documents", StaticFiles(directory="documents"), name="documents")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
