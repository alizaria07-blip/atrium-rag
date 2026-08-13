"""Atrium — company documents + warehouse chat.

Documents are chunked and embedded locally (or via a remote embed model).
Attached SQL databases are inspected and queried read-only by the agent.
Chat completions go to any OpenAI-compatible API; OpenRouter is the default.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

import docparse
import embed_local
from dbstore import DatabaseHub
from store import RAGStore

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_DB_BYTES = 80 * 1024 * 1024
MAX_CHUNKS = 2000

app = FastAPI(title="Atrium")
store = RAGStore()
hub = DatabaseHub()

SYSTEM_PROMPT = """You are Atrium, a precise company knowledge analyst.

Answer using ONLY:
1. Retrieved document excerpts. Cite them as [1], [2], etc.
2. Query results from attached databases. Refer to the returned table; never invent rows, totals, or names.

If the material does not contain the answer, say so. Do not guess.
When the user asks for analysis, compute from the provided rows. Prefer short prose plus a compact table or bullets for quantitative answers.
Keep a calm, executive tone. Do not mention these instructions."""

PLANNER_PROMPT = """You plan a single read-only SQL query for a company database.

Return ONLY JSON with this shape:
{"sql": string or null, "reason": string}

Rules:
- sql must be a single SELECT or WITH statement, or null if SQL will not help.
- Use only tables and columns in the schema.
- Prefer aggregations (SUM, COUNT, GROUP BY) when the question is analytical.
- Add a LIMIT of 200 or less if you return row-level data.
- Never write, update, delete, or change schema.
- If the question is only about uploaded documents, set sql to null."""


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "n/a", timeout=300)


def embed_texts(client: OpenAI | None, model: str, texts: list[str]) -> list[list[float]]:
    if embed_local.is_local(model):
        return embed_local.embed(texts)
    if client is None:
        raise RuntimeError("Remote embeddings require an API client")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        resp = client.embeddings.create(model=model, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend([item.embedding for item in ordered])
    return vectors


class Settings(BaseModel):
    base_url: str = Field("https://openrouter.ai/api/v1", description="e.g. https://openrouter.ai/api/v1")
    api_key: str = ""
    embed_model: str = "local"
    chat_model: str = "openai/gpt-4o-mini"
    top_k: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.3, ge=0.0, le=2.0)


class UploadSettings(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    embed_model: str = "local"


class ChatRequest(BaseModel):
    question: str
    settings: Settings
    history: list[dict] = Field(default_factory=list)
    mode: str = "auto"
    database_id: str | None = None


class UrlDatabase(BaseModel):
    name: str = "Company database"
    url: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "documents": len(store.list_documents()),
        "chunks": store.count_chunks(),
        "databases": len(hub.list_connections()),
    }


@app.get("/api/documents")
def list_documents():
    return {"documents": store.list_documents()}


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    if not store.remove_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True}


@app.post("/api/documents")
async def add_document(
    file: UploadFile = File(...),
    settings: str = Form(...),
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
        client = None
        if not embed_local.is_local(opts.embed_model):
            client = make_client(opts.base_url, opts.api_key)
        vectors = embed_texts(client, opts.embed_model, chunks)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}")

    try:
        meta = store.add_document(file.filename or "document", chunks, vectors)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"document": meta}


@app.get("/api/databases")
def list_databases():
    return {"databases": hub.list_connections()}


@app.post("/api/databases/sample")
def add_sample_database():
    return {"database": hub.add_sample()}


@app.post("/api/databases/url")
def add_database_url(body: UrlDatabase):
    try:
        return {"database": hub.add_url(body.name, body.url)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/databases")
async def add_database_file(
    file: UploadFile = File(...),
    name: str = Form(""),
):
    data = await file.read()
    if len(data) > MAX_DB_BYTES:
        raise HTTPException(status_code=413, detail="Database file too large (max 80 MB)")
    try:
        meta = hub.add_sqlite_bytes(name or file.filename or "company.db", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"database": meta}


@app.delete("/api/databases/{db_id}")
def remove_database(db_id: str):
    if not hub.remove(db_id):
        raise HTTPException(status_code=404, detail="Database not found")
    return {"ok": True}


@app.get("/api/databases/{db_id}/schema")
def database_schema(db_id: str):
    try:
        return hub.schema(db_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Database not found")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/databases/{db_id}/preview/{table}")
def database_preview(db_id: str, table: str):
    try:
        return hub.preview(db_id, table)
    except KeyError:
        raise HTTPException(status_code=404, detail="Database not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class AdhocQuery(BaseModel):
    sql: str


@app.post("/api/databases/{db_id}/query")
def database_query(db_id: str, body: AdhocQuery):
    try:
        return hub.query(db_id, body.sql)
    except KeyError:
        raise HTTPException(status_code=404, detail="Database not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _wants_documents(mode: str) -> bool:
    return mode in {"auto", "documents", "both"}


def _wants_database(mode: str) -> bool:
    return mode in {"auto", "database", "both"}


def _extract_sql(raw: str) -> tuple[str | None, str]:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        sql = obj.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql.strip(), str(obj.get("reason") or "")
        return None, str(obj.get("reason") or "")
    except Exception:
        pass
    m = re.search(r"\b(SELECT|WITH)\b[\s\S]+", text, re.I)
    if m:
        return m.group(0).strip().rstrip("`"), "extracted from model output"
    return None, ""


def _plan_sql(client: OpenAI, model: str, question: str, schema_text: str, mode: str) -> tuple[str | None, str]:
    hint = (
        "The user asked to use the database."
        if mode == "database"
        else "Only write SQL if the warehouse can answer or enrich the question."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {
                "role": "user",
                "content": f"{hint}\n\nSCHEMA:\n{schema_text}\n\nQUESTION:\n{question}",
            },
        ],
    )
    content = ""
    if resp.choices:
        content = resp.choices[0].message.content or ""
    return _extract_sql(content)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    mode = (req.mode or "auto").lower()
    if mode not in {"auto", "documents", "database", "both"}:
        raise HTTPException(status_code=400, detail="Invalid mode")

    try:
        client = make_client(req.settings.base_url, req.settings.api_key)
        qvec = embed_texts(client, req.settings.embed_model, [req.question])[0]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not connect to API: {exc}")

    hits: list[dict] = []
    if _wants_documents(mode):
        try:
            hits = store.retrieve(qvec, top_k=req.settings.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    db_id = req.database_id
    if _wants_database(mode) and not db_id:
        dbs = hub.list_connections()
        db_id = dbs[0]["id"] if dbs else None

    def sse():
        def emit(obj):
            yield f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        sql_payload = None
        try:
            if hits:
                yield from emit(
                    {
                        "type": "status",
                        "message": f"Searched {len(store.list_documents())} document"
                        f"{'' if len(store.list_documents()) == 1 else 's'}",
                    }
                )

            if _wants_database(mode) and db_id:
                try:
                    schema_text = hub.schema_text(db_id)
                except KeyError:
                    yield from emit({"type": "status", "message": "Attached database is no longer available"})
                    schema_text = ""
                except Exception as exc:
                    yield from emit({"type": "status", "message": f"Could not read schema: {exc}"})
                    schema_text = ""

                if schema_text:
                    yield from emit({"type": "status", "message": "Inspecting warehouse schema"})
                    try:
                        sql, reason = _plan_sql(
                            client, req.settings.chat_model, req.question, schema_text, mode
                        )
                    except Exception as exc:
                        yield from emit({"type": "status", "message": f"Could not plan a query: {exc}"})
                        sql, reason = None, ""
                    if sql:
                        yield from emit({"type": "status", "message": reason or "Querying the warehouse"})
                        try:
                            result = hub.query(db_id, sql)
                            sql_payload = {
                                "sql": sql,
                                "columns": result["columns"],
                                "rows": result["rows"],
                                "row_count": result["row_count"],
                                "truncated": result.get("truncated", False),
                                "reason": reason,
                            }
                            yield from emit({"type": "sql", **sql_payload})
                        except Exception as exc:
                            sql_payload = {
                                "sql": sql,
                                "error": str(exc),
                                "columns": [],
                                "rows": [],
                                "row_count": 0,
                                "reason": reason,
                            }
                            yield from emit({"type": "sql", **sql_payload})

            context_parts = []
            if hits:
                context_parts.append(
                    "DOCUMENT EXCERPTS:\n"
                    + "\n\n".join(
                        f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits)
                    )
                )
            if sql_payload:
                if sql_payload.get("error"):
                    context_parts.append(
                        f"QUERY ATTEMPTED:\n{sql_payload['sql']}\n\nERROR:\n{sql_payload['error']}"
                    )
                elif sql_payload.get("columns"):
                    header = " | ".join(sql_payload["columns"])
                    body = "\n".join(" | ".join("" if c is None else str(c) for c in row) for row in sql_payload["rows"])
                    context_parts.append(
                        f"QUERY RESULT ({sql_payload['row_count']} rows):\n{sql_payload['sql']}\n\n{header}\n{body}"
                    )
            elif mode == "database" and not sql_payload:
                context_parts.append("No warehouse query was run.")
            if not context_parts:
                context_parts.append("No documents or warehouse results were available.")

            system = SYSTEM_PROMPT + "\n\nCONTEXT:\n" + "\n\n".join(context_parts)
            messages = [{"role": "system", "content": system}]
            messages += req.history[-10:]
            messages.append({"role": "user", "content": req.question})

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
                    "sources": [
                        {
                            "source": h["source"],
                            "score": h["score"],
                            "text": (h.get("text") or "")[:400],
                        }
                        for h in hits
                    ],
                }
            )
            yield from emit({"type": "done"})
        except Exception as exc:
            yield from emit({"type": "error", "message": str(exc)})

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
