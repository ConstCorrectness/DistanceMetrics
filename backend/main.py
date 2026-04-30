import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.jobs import create_job, get_job, update_job
from backend.parser import parse_file, row_to_text, SUPPORTED_EXTENSIONS
from backend.embedder import embed_texts, BATCH_SIZE
from backend.vectordb import ensure_collection, upsert_points, search, list_source_files, get_all_vectors
from backend.intent_classifier import classify

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    yield


app = FastAPI(title="SemanticSearch API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _ingest(job_id: str, filename: str, content: bytes) -> None:
    try:
        update_job(job_id, status="running", message="Parsing file...")
        rows = parse_file(filename, content)
        total = len(rows)
        update_job(job_id, total_rows=total, message=f"Parsed {total} rows. Embedding...")

        texts = [row_to_text(r) for r in rows]

        all_vectors: list[list[float]] = []
        for i in range(0, total, BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_vectors = embed_texts(batch_texts)
            all_vectors.extend(batch_vectors)
            update_job(
                job_id,
                processed_rows=min(i + BATCH_SIZE, total),
                message=f"Embedded {min(i + BATCH_SIZE, total)}/{total} rows...",
            )

        update_job(job_id, message="Storing vectors...")
        upsert_points(all_vectors, rows, source_file=filename)
        update_job(job_id, status="done", progress=100, message="Done!")

    except Exception as exc:
        update_job(job_id, status="error", message="Failed.", error=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}",
        )
    content = await file.read()
    job = create_job()
    background_tasks.add_task(_ingest, job.id, file.filename, content)
    return {"job_id": job.id}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "total_rows": job.total_rows,
        "processed_rows": job.processed_rows,
    }


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/search")
def semantic_search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    vectors = embed_texts([req.query])
    results = search(vectors[0], top_k=req.top_k)
    return {"results": results}


@app.get("/collections")
def get_collections():
    return {"files": list_source_files()}


@app.get("/vectors")
def all_vectors():
    points = get_all_vectors()
    return {"points": points, "count": len(points)}


class EmbedRequest(BaseModel):
    query: str


@app.post("/embed")
def embed_query(req: EmbedRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    vectors = embed_texts([req.query])
    return {"vector": vectors[0]}


class ClassifyRequest(BaseModel):
    utterance: str


@app.post("/classify")
def classify_intent(req: ClassifyRequest):
    if not req.utterance.strip():
        raise HTTPException(status_code=400, detail="Utterance cannot be empty")
    return classify(req.utterance)


@app.get("/health")
def health():
    return {"status": "ok"}
