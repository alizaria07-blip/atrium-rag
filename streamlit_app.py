"""Atrium — Company Documents & Warehouse Intelligence (Streamlit UI).

Runs on Streamlit Community Cloud (or locally).
Reuses the same document extraction, chunking, vector store, and SQL database hub.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from typing import Any

from openai import OpenAI
import streamlit as st

from dbstore import DatabaseHub
import docparse
import embed_local
from store import RAGStore

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


# ------------------------------------------------------------------ Page Config & Styling
st.set_page_config(
    page_title="Atrium — Company Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Atrium Modern Dark Luxury Styling */
:root {
    --bg: #141210;
    --surface: #1d1916;
    --surface-2: #26211d;
    --copper: #c98a56;
    --copper-light: #e2a370;
    --ink: #f5f2ed;
    --muted: #a69e95;
    --line: #38312b;
}

[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--line);
}

.stButton>button {
    border-radius: 8px;
    transition: all 0.15s ease;
}

.stButton>button:hover {
    border-color: var(--copper);
    color: var(--copper-light);
}

.thread-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 10px;
    border-radius: 6px;
    margin-bottom: 4px;
    background: var(--surface-2);
}

.sql-badge {
    display: inline-block;
    background: rgba(201, 138, 86, 0.15);
    color: #e2a370;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-family: monospace;
    margin-bottom: 6px;
}

.pill {
    display: inline-block;
    background: #2b2520;
    color: #ded7ce;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    margin-right: 6px;
    border: 1px solid #423932;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ Config & Secrets
def _secret(key: str, default: Any) -> Any:
    """Robustly fetch secrets from st.secrets (flat or nested) or os.environ."""
    aliases = [key, key.lower(), key.upper()]
    if key == "api_key":
        aliases.extend(["OPENAI_API_KEY", "OPENROUTER_API_KEY", "API_KEY"])
    elif key == "base_url":
        aliases.extend(["OPENAI_BASE_URL", "OPENROUTER_BASE_URL", "BASE_URL"])
    elif key == "chat_model":
        aliases.extend(["CHAT_MODEL", "MODEL"])
    elif key == "embed_model":
        aliases.extend(["EMBED_MODEL"])

    # 1. Direct top-level st.secrets lookup
    try:
        for k in aliases:
            if k in st.secrets:
                val = st.secrets[k]
                if val is not None and str(val).strip() != "":
                    return val
    except Exception:
        pass

    # 2. Nested section lookup (e.g. [openai] or [openrouter] sections)
    try:
        for sec_name in ("openai", "openrouter", "general", "app", "secrets"):
            if sec_name in st.secrets and hasattr(st.secrets[sec_name], "get"):
                sec_dict = st.secrets[sec_name]
                for k in aliases:
                    val = sec_dict.get(k)
                    if val is not None and str(val).strip() != "":
                        return val
    except Exception:
        pass

    # 3. Environment variables lookup
    for k in aliases:
        val = os.environ.get(k)
        if val is not None and str(val).strip() != "":
            return val

    return default


def get_default_config() -> dict:
    return {
        "base_url": _secret("base_url", "https://openrouter.ai/api/v1"),
        "api_key": _secret("api_key", ""),
        "chat_model": _secret("chat_model", "openai/gpt-4o-mini"),
        "embed_model": _secret("embed_model", "local"),
        "top_k": int(_secret("top_k", 5)),
        "temperature": float(_secret("temperature", 0.3)),
    }


# ------------------------------------------------------------------ Stores (Session-Isolated)
def get_store() -> RAGStore:
    if "store_dir" not in st.session_state:
        st.session_state.store_dir = tempfile.mkdtemp(prefix="atrium_store_")
    return RAGStore(st.session_state.store_dir)


def get_hub() -> DatabaseHub:
    if "hub_dir" not in st.session_state:
        st.session_state.hub_dir = tempfile.mkdtemp(prefix="atrium_hub_")
    return DatabaseHub(st.session_state.hub_dir)


# ------------------------------------------------------------------ Session State Initialization
if "cfg" not in st.session_state:
    st.session_state.cfg = get_default_config()

if "threads" not in st.session_state:
    default_id = uuid.uuid4().hex
    st.session_state.threads = [
        {
            "id": default_id,
            "title": "New conversation",
            "created": time.time(),
            "messages": [],
        }
    ]
    st.session_state.active_thread_id = default_id

if "active_thread_id" not in st.session_state or not any(t["id"] == st.session_state.active_thread_id for t in st.session_state.threads):
    if st.session_state.threads:
        st.session_state.active_thread_id = st.session_state.threads[0]["id"]
    else:
        new_id = uuid.uuid4().hex
        st.session_state.threads = [{"id": new_id, "title": "New conversation", "created": time.time(), "messages": []}]
        st.session_state.active_thread_id = new_id

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "doc_uploader_nonce" not in st.session_state:
    st.session_state.doc_uploader_nonce = 0

if "uploaded_db_files" not in st.session_state:
    st.session_state.uploaded_db_files = set()

if "selected_db_id" not in st.session_state:
    st.session_state.selected_db_id = None

if "mode" not in st.session_state:
    st.session_state.mode = "auto"


# ------------------------------------------------------------------ Helpers
def current_thread() -> dict:
    for t in st.session_state.threads:
        if t["id"] == st.session_state.active_thread_id:
            return t
    return st.session_state.threads[0]


def make_client(cfg: dict) -> OpenAI:
    base_url = (cfg.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/")
    headers = {}
    if "openrouter.ai" in base_url.lower():
        headers = {
            "HTTP-Referer": "https://atrium-rag.streamlit.app",
            "X-Title": "Atrium RAG",
        }
    return OpenAI(
        base_url=base_url,
        api_key=cfg.get("api_key") or "n/a",
        default_headers=headers if headers else None,
        timeout=300,
    )


def embed_texts(client: OpenAI | None, model: str, texts: list[str]) -> list[list[float]]:
    if embed_local.is_local(model):
        return embed_local.embed(texts)
    if client is None:
        return embed_local.embed(texts)
    try:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            resp = client.embeddings.create(model=model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend(x.embedding for x in ordered)
        return vectors
    except Exception as exc:
        st.warning(f"Remote embedding model '{model}' failed ({exc}). Falling back to built-in local embedding.")
        return embed_local.embed(texts)


def extract_planner_sql(raw: str) -> tuple[str | None, str]:
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


def plan_sql(client: OpenAI, model: str, question: str, schema_text: str, mode: str) -> tuple[str | None, str]:
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
    content = resp.choices[0].message.content if resp.choices else ""
    return extract_planner_sql(content or "")


def render_sql_table(sql_data: dict) -> None:
    if not sql_data:
        return
    if sql_data.get("error"):
        st.error(f"SQL Error: {sql_data['error']}")
        return
    st.markdown(f"<div class='sql-badge'>Warehouse Query · {sql_data.get('row_count', 0)} rows</div>", unsafe_allow_html=True)
    st.code(sql_data.get("sql", ""), language="sql")
    rows = sql_data.get("rows") or []
    cols = sql_data.get("columns") or []
    if rows and cols:
        table_data = [dict(zip(cols, r)) for r in rows]
        st.dataframe(table_data, use_container_width=True)


# ------------------------------------------------------------------ Sidebar: Navigation & Sources
store = get_store()
hub = get_hub()

with st.sidebar:
    st.title("🏛️ Atrium")
    st.caption("Company intelligence across documents and attached warehouse data.")

    # 1. New Chat & Conversation Management
    col_new, col_cnt = st.columns([3, 2])
    with col_new:
        if st.button("＋ New Chat", use_container_width=True):
            new_id = uuid.uuid4().hex
            st.session_state.threads.insert(0, {
                "id": new_id,
                "title": "New conversation",
                "created": time.time(),
                "messages": [],
            })
            st.session_state.active_thread_id = new_id
            st.rerun()

    with col_cnt:
        st.caption(f"{len(st.session_state.threads)} chat(s)")

    st.subheader("Conversations")
    for t in list(st.session_state.threads):
        is_active = t["id"] == st.session_state.active_thread_id
        col_t, col_del = st.columns([5, 1])
        with col_t:
            label = f"💬 {t['title'][:24]}..." if len(t["title"]) > 24 else f"💬 {t['title']}"
            if st.button(label, key=f"thread_{t['id']}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.active_thread_id = t["id"]
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{t['id']}", help="Delete this conversation"):
                st.session_state.threads = [x for x in st.session_state.threads if x["id"] != t["id"]]
                if not st.session_state.threads:
                    fresh_id = uuid.uuid4().hex
                    st.session_state.threads = [{"id": fresh_id, "title": "New conversation", "created": time.time(), "messages": []}]
                    st.session_state.active_thread_id = fresh_id
                elif st.session_state.active_thread_id == t["id"]:
                    st.session_state.active_thread_id = st.session_state.threads[0]["id"]
                st.rerun()

    st.divider()

    # 2. Documents Section
    st.subheader("📚 Documents")
    uploader_key = f"doc_uploader_{st.session_state.doc_uploader_nonce}"
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / TXT / MD",
        type=["pdf", "docx", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=uploader_key,
    )

    if uploaded_files:
        for f in uploaded_files:
            if f.name in st.session_state.indexed_files:
                continue
            try:
                with st.spinner(f"Indexing {f.name}..."):
                    text = docparse.extract_text(f.name, f.getvalue())
                    chunks = docparse.chunk_text(text)
                    if not chunks:
                        st.error(f"{f.name}: No text extracted.")
                        continue
                    client = None if embed_local.is_local(st.session_state.cfg["embed_model"]) else make_client(st.session_state.cfg)
                    vectors = embed_texts(client, st.session_state.cfg["embed_model"], chunks)
                    store.add_document(f.name, chunks, vectors)
                    st.session_state.indexed_files.add(f.name)
                    st.success(f"Indexed {f.name} ({len(chunks)} passages)")
            except Exception as exc:
                st.error(f"Failed to index {f.name}: {exc}")

    doc_list = store.list_documents()
    if doc_list:
        st.caption(f"{len(doc_list)} document(s) · {store.count_chunks()} chunks")
        for d in doc_list:
            st.caption(f"• **{d['name']}** ({d['chunks']} passages)")
        if st.button("Clear all documents", key="clear_docs_btn"):
            store.clear()
            st.session_state.indexed_files.clear()
            st.session_state.doc_uploader_nonce += 1
            st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # 3. Database Section
    st.subheader("🗄️ Databases")
    dbs = hub.list_connections()

    db_options = {d["id"]: f"{d['name']} ({d['engine']})" for d in dbs}
    if dbs:
        if st.session_state.selected_db_id not in db_options:
            st.session_state.selected_db_id = dbs[0]["id"]
        selected_db = st.selectbox(
            "Active Warehouse",
            options=list(db_options.keys()),
            format_func=lambda x: db_options[x],
            key="db_select_widget",
        )
        st.session_state.selected_db_id = selected_db
    else:
        st.caption("No databases attached.")
        st.session_state.selected_db_id = None

    with st.expander("＋ Attach a database"):
        if st.button("Load Northline sample database", key="load_sample_db_btn"):
            try:
                new_db = hub.add_sample()
                st.session_state.selected_db_id = new_db["id"]
                st.success("Loaded Northline sample database.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")

        db_file = st.file_uploader("Upload SQLite file", type=["db", "sqlite", "sqlite3"], key="db_file_uploader")
        if db_file:
            file_sig = f"{db_file.name}_{db_file.size}"
            if file_sig not in st.session_state.uploaded_db_files:
                try:
                    new_db = hub.add_sqlite_bytes(db_file.name, db_file.getvalue())
                    st.session_state.selected_db_id = new_db["id"]
                    st.session_state.uploaded_db_files.add(file_sig)
                    st.success(f"Attached {db_file.name}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error attaching SQLite: {exc}")

        with st.form("url_db_form"):
            db_name_in = st.text_input("Name", placeholder="Production Replica")
            db_url_in = st.text_input("URL", type="password", placeholder="postgresql://user:pass@host/db")
            if st.form_submit_button("Connect via URL"):
                if db_url_in.strip():
                    try:
                        new_db = hub.add_url(db_name_in or "Remote DB", db_url_in)
                        st.session_state.selected_db_id = new_db["id"]
                        st.success("Connected.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Connection failed: {exc}")

    if st.session_state.selected_db_id:
        with st.expander("🔍 Inspect Schema & Tables"):
            try:
                schema_spec = hub.schema(st.session_state.selected_db_id)
                for t in schema_spec.get("tables", []):
                    st.write(f"**Table:** `{t['name']}` ({t.get('rows', 0)} rows)")
                    cols_desc = ", ".join(f"{c['name']} ({c['type']})" for c in t.get("columns", []))
                    st.caption(cols_desc)
            except Exception as exc:
                st.caption(f"Could not inspect schema: {exc}")

    st.divider()

    # 4. Connection & Model Settings
    st.subheader("⚙️ API Connection")
    st.session_state.cfg["base_url"] = st.text_input("API Base URL", value=st.session_state.cfg["base_url"], key="cfg_base_url")
    st.session_state.cfg["api_key"] = st.text_input("API Key", value=st.session_state.cfg["api_key"], type="password", key="cfg_api_key")
    st.session_state.cfg["chat_model"] = st.text_input("Chat Model", value=st.session_state.cfg["chat_model"], key="cfg_chat_model")
    st.session_state.cfg["embed_model"] = st.text_input("Embedding Model", value=st.session_state.cfg["embed_model"], key="cfg_embed_model")
    st.session_state.cfg["top_k"] = st.number_input("Top K chunks", min_value=1, max_value=20, value=st.session_state.cfg["top_k"], key="cfg_top_k")
    st.session_state.cfg["temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state.cfg["temperature"]), key="cfg_temp")

    if not st.session_state.cfg["api_key"]:
        st.warning("No API key set. Add it above to ask questions.")
    else:
        st.success("API key ready.")


# ------------------------------------------------------------------ Main Area: Conversation & Chat
st.title("📚 Local RAG")
st.caption("Atrium · Company intelligence across documents and attached warehouse data.")

thread = current_thread()

# Header status pills & mode selection
active_db_meta = hub.get(st.session_state.selected_db_id) if st.session_state.selected_db_id else None
db_label = active_db_meta["name"] if active_db_meta else "none"
num_docs = len(store.list_documents())

col_title, col_mode = st.columns([3, 2])
with col_title:
    st.markdown(
        f"#### {thread['title']}  \n"
        f"<span class='pill'>📚 Documents · {num_docs}</span>"
        f"<span class='pill'>🗄️ Database · {db_label}</span>",
        unsafe_allow_html=True,
    )

with col_mode:
    mode_choice = st.radio(
        "Search in:",
        options=["auto", "documents", "database", "both"],
        horizontal=True,
        index=["auto", "documents", "database", "both"].index(st.session_state.mode),
        key="mode_radio",
    )
    st.session_state.mode = mode_choice

st.markdown("---")

# Render conversation turns
for msg in thread.get("messages", []):
    with st.chat_message(msg["role"]):
        if msg.get("steps"):
            for step in msg["steps"]:
                st.caption(f"⚡ {step}")
        if msg.get("sql"):
            render_sql_table(msg["sql"])
        st.markdown(msg.get("content", ""))
        if msg.get("sources"):
            with st.expander(f"Grounding Sources ({len(msg['sources'])} retrieved)"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(f"**[{i+1}] {src.get('source', 'Document')}** (score: {src.get('score', 0)})")
                    st.caption(src.get("text", "")[:400])

# Chat input
prompt = st.chat_input("Ask a question about your documents or connected database...")

if prompt:
    if not st.session_state.cfg["api_key"].strip() and not (st.session_state.cfg["base_url"].startswith("http://127.0.0.1") or st.session_state.cfg["base_url"].startswith("http://localhost")):
        st.error("Please provide an API Key in the sidebar.")
        st.stop()

    # Update thread title if this is the first question
    if not thread["messages"]:
        thread["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    # Append user question
    thread["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Assistant Turn
    with st.chat_message("assistant"):
        status_holder = st.empty()
        status_steps = []

        hits = []
        sql_payload = None

        # 1. Document Retrieval
        if st.session_state.mode in {"auto", "documents", "both"}:
            try:
                status_holder.caption("⚡ Embedding question and searching documents...")
                client = None if embed_local.is_local(st.session_state.cfg["embed_model"]) else make_client(st.session_state.cfg)
                qvec = embed_texts(client, st.session_state.cfg["embed_model"], [prompt])[0]
                hits = store.retrieve(qvec, top_k=int(st.session_state.cfg["top_k"]))
                if hits:
                    status_steps.append(f"Retrieved {len(hits)} document passages")
            except Exception as exc:
                st.error(f"Document search error: {exc}")

        # 2. Database Planning & Querying
        db_id = st.session_state.selected_db_id
        if st.session_state.mode in {"auto", "database", "both"} and db_id:
            try:
                status_holder.caption("⚡ Inspecting warehouse schema and planning query...")
                schema_text = hub.schema_text(db_id)
                if schema_text:
                    api_client = make_client(st.session_state.cfg)
                    sql, reason = plan_sql(api_client, st.session_state.cfg["chat_model"], prompt, schema_text, st.session_state.mode)
                    if sql:
                        status_steps.append(reason or "Executed warehouse query")
                        status_holder.caption("⚡ Running SQL query...")
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
                            render_sql_table(sql_payload)
                        except Exception as exc:
                            sql_payload = {
                                "sql": sql,
                                "error": str(exc),
                                "columns": [],
                                "rows": [],
                                "row_count": 0,
                                "reason": reason,
                            }
                            render_sql_table(sql_payload)
            except Exception as exc:
                status_steps.append(f"Warehouse plan error: {exc}")

        # 3. Assemble Prompt Context
        context_parts = []
        if hits:
            context_parts.append(
                "DOCUMENT EXCERPTS:\n"
                + "\n\n".join(f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits))
            )
        if sql_payload:
            if sql_payload.get("error"):
                context_parts.append(f"QUERY ATTEMPTED:\n{sql_payload['sql']}\n\nERROR:\n{sql_payload['error']}")
            elif sql_payload.get("columns"):
                header = " | ".join(sql_payload["columns"])
                body = "\n".join(" | ".join("" if c is None else str(c) for c in row) for row in sql_payload["rows"])
                context_parts.append(f"QUERY RESULT ({sql_payload['row_count']} rows):\n{sql_payload['sql']}\n\n{header}\n{body}")
        elif st.session_state.mode == "database" and not sql_payload:
            context_parts.append("No warehouse query was run.")
        if not context_parts:
            context_parts.append("No documents or warehouse results were available.")

        system = SYSTEM_PROMPT + "\n\nCONTEXT:\n" + "\n\n".join(context_parts)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in thread["messages"][-11:]
            if m.get("content")
        ]
        messages = [{"role": "system", "content": system}] + chat_history

        status_holder.empty()

        # 4. Stream Answer
        try:
            api_client = make_client(st.session_state.cfg)
            resp_stream = api_client.chat.completions.create(
                model=st.session_state.cfg["chat_model"],
                messages=messages,
                temperature=float(st.session_state.cfg["temperature"]),
                stream=True,
            )

            def gen_stream():
                for chunk in resp_stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            answer_text = st.write_stream(gen_stream())

            # Save Turn into Active Thread
            assistant_turn = {
                "role": "assistant",
                "content": answer_text,
                "steps": status_steps,
                "sql": sql_payload,
                "sources": [
                    {
                        "source": h["source"],
                        "score": h["score"],
                        "text": h.get("text", "")[:400],
                    }
                    for h in hits
                ],
            }
            thread["messages"].append(assistant_turn)

            if hits:
                with st.expander(f"Grounding Sources ({len(hits)} retrieved)"):
                    for i, src in enumerate(hits):
                        st.markdown(f"**[{i+1}] {src['source']}** (score: {src['score']})")
                        st.caption(src.get("text", "")[:400])

        except Exception as exc:
            st.error(f"Completion failed: {exc}")