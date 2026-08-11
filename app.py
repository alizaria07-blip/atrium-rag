"""Local RAG web app.

Exposes a small API over an OpenAI-compatible backend (vLLM, Groq, Together,
OpenAI, ...). Documents are chunked and embedded locally; answers are streamed
back to the browser as server-sent events.

The API key, base URL and model names are supplied per-request by the UI and
are never stored server-side.
"""
import os
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

import docparse
from store import RAGStore

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB

app = FastAPI(title="Local RAG")
store = RAGStore()

MAX_CHUNKS = 2000  # safety cap on embedding calls per file


# --------------------------------------------------------------------- helpers
def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=300)


def embed(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batched, via the OpenAI-compatible endpoint."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        resp = client.embeddings.create(model=model, input=batch)
        # Preserve order as returned by the API.
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend([item.embedding for item in ordered])
    return vectors


class Settings(BaseModel):
    base_url: str = Field(..., description="e.g. https://api.groq.com/openai/v1")
    api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    top_k: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.3, ge=0.0, le=2.0)


# ----------------------------------------------------------------- api routes
@app.get("/api/health")
def health():
    return {"status": "ok", "documents": len(store.list_documents()), "chunks": store.count_chunks()}


@app.get("/api/documents")
def list_documents():
    return {"documents": store.list_documents()}


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    if not store.remove_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True}


class UploadSettings(BaseModel):
    base_url: str
    api_key: str = ""
    embed_model: str = "text-embedding-3-small"


@app.post("/api/documents")
async def add_document(
    file: UploadFile = File(...),
    settings: str = Form(...),  # JSON-encoded UploadSettings
):
    try:
        opts = UploadSettings.model_validate_json(settings)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid settings: {exc}")

    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    try:
        text = docparse.extract_text(file.filename or "document", data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from this file")

    chunks = docparse.chunk_text(text)[:MAX_CHUNKS]
    if not chunks:
        raise HTTPException(status_code=400, detail="No usable text after chunking")

    try:
        client = make_client(opts.base_url, opts.api_key)
        vectors = embed(client, opts.embed_model, chunks)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}")

    meta = store.add_document(file.filename or "document", chunks, vectors)
    return {"document": meta}


class ChatRequest(BaseModel):
    question: str
    settings: Settings
    history: list[dict] = Field(default_factory=list)


SYSTEM_PROMPT = """You are a precise retrieval assistant. Answer the user's question using ONLY the provided context excerpts. If the context does not contain the answer, say "I don't have that information in the uploaded documents." rather than guessing. When you use a specific excerpt, cite it as [1], [2], etc. Keep answers concise and grounded."""


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        client = make_client(req.settings.base_url, req.settings.api_key)
        qvec = embed(client, req.settings.embed_model, [req.question])[0]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not connect to API: {exc}")

    hits = store.retrieve(qvec, top_k=req.settings.top_k)

    if hits:
        context_block = "\n\n".join(
            f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits)
        )
        system = SYSTEM_PROMPT + "\n\nCONTEXT:\n" + context_block
    else:
        system = SYSTEM_PROMPT + "\n\n(No documents have been uploaded yet.)"
        hits = []

    messages = [{"role": "system", "content": system}]
    messages += req.history[-10:]  # keep recent turns only
    messages.append({"role": "user", "content": req.question})

    def sse():
        def emit(obj):
            yield f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        try:
            stream = client.chat.completions.create(
                model=req.settings.chat_model,
                messages=messages,
                temperature=req.settings.temperature,
                stream=True,
            )
            for event in stream:
                if not getattr(event, "choices", None):
                    continue
                delta = event.choices[0].delta
                if delta and getattr(delta, "content", None):
                    yield from emit({"type": "delta", "content": delta.content})
            yield from emit(
                {
                    "type": "sources",
                    "sources": [{"source": h["source"], "score": h["score"]} for h in hits],
                }
            )
            yield from emit({"type": "done"})
        except Exception as exc:
            yield from emit({"type": "error", "message": str(exc)})

    return StreamingResponse(sse(), media_type="text/event-stream")


# ------------------------------------------------------------------ frontend
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")