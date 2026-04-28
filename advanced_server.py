import time
import os
import uuid
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional, List
from pathlib import Path

app = FastAPI(title="Advanced RAG API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
from advanced_rag_ind import AdvancedRAGPipeline, RagResult

pipeline: Optional[AdvancedRAGPipeline] = None
history:  List[dict] = []
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Schemas ────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question:         str
    rewritten_query:  str
    hyde_doc:         str
    answer:           str
    sources:          list
    scores:           list
    technique_log:    list
    latency_ms:       float
    hallucination_ok: bool

@app.get("/status")
def status():
    ready = pipeline is not None and pipeline._ready
    return {
        "status": "ready" if ready else "no_index",
        "techniques": [
            "T1: Query Rewriting",
            "T2: HyDE (Hypothetical Doc Embeddings)",
            "T3: Hybrid Search (BM25 + FAISS)",
            "T4: Cross-Encoder Re-Ranking",
            "T5: Parent-Child Chunking",
            "T6: Self-RAG Hallucination Guard",
        ],
        "llm":            "llama3-8b-8192 (Groq FREE)",
        "embeddings":     "all-MiniLM-L6-v2 (local)",
        "reranker":       "ms-marco-MiniLM-L-6-v2 (local)",
        "vectorstore":    "FAISS (local)",
        "queries_served": len(history),
    }

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    global pipeline
    ext = Path(file.filename).suffix.lower()    # FIX 10: was `.suffix.lower` — missing ()
    if ext not in {".pdf", ".txt"}:
        raise HTTPException(400, f"Only PDF and TXT supported. Got: {ext}")
    fid       = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{fid}_{file.filename}"
    save_path.write_bytes(await file.read())

    try:
        pipeline = AdvancedRAGPipeline()
        result   = pipeline.ingest(str(save_path))

        return {
            "status":   "success",
            "filename": file.filename,
            **result,
            "message": f"Indexed {result['child_chunks']} child chunks from {result['parent_chunks']} parent chunks"
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    global pipeline
    if not pipeline or not pipeline._ready:
        raise HTTPException(400, "Upload a document first via /ingest")

    try:
        result: RagResult = pipeline.ask(body.question)
    except Exception as e:
        raise HTTPException(500, str(e))

    entry = {
        "id":              str(uuid.uuid4())[:8],
        "timestamp":       datetime.now().isoformat(),
        "question":        result.question,
        "rewritten_query": result.rewritten_query,
        "answer":          result.answer,
        "latency_ms":      result.latency_ms,
        "hallucination_ok":result.hallucination_ok,
        "num_sources":     len(result.sources),
    }
    history.append(entry)

    return AskResponse(
        question=result.question,
        rewritten_query=result.rewritten_query,
        hyde_doc=result.hyde_doc,
        answer=result.answer,
        sources=result.sources,
        scores=result.scores,
        technique_log=result.technique_log,
        latency_ms=result.latency_ms,
        hallucination_ok=result.hallucination_ok,
    )

@app.get("/history")
def get_history():
    return {"history": history[-20:][::-1]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("advanced_server:app", host="0.0.0.0", port=8000, reload=True)