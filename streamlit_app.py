"""Streamlit UI for the local RAG app.

Runs on Streamlit Community Cloud (or locally). Reuses the same document
extraction, chunking and vector store as the FastAPI version via `docparse`
and `store`.

Configuration is read from Streamlit secrets (`.streamlit/secrets.toml`
locally, or Settings -> Secrets in Streamlit Cloud). The API key is supplied
through secrets — it is never written to disk by this app.

secrets keys (lowercase):
    base_url     ->  default https://openrouter.ai/api/v1
    api_key      ->  your OpenRouter (or other OpenAI-compatible) key
    chat_model   ->  default openai/gpt-4o-mini
    embed_model  ->  default openai/text-embedding-3-small
    top_k        ->  default 5
"""
import os
import tempfile
from openai import OpenAI

import streamlit as st

import docparse
import embed_local
from store import RAGStore

SYSTEM_PROMPT = """You are a precise retrieval assistant. Answer the user's question using ONLY the provided context excerpts. If the context does not contain the answer, say "I don't have that information in the uploaded documents." rather than guessing. When you use a specific excerpt, cite it as [1], [2], etc. Keep answers concise and grounded."""


# ------------------------------------------------------------------ config
def _secret(key, default):
    try:
        val = st.secrets.get(key)
        return val if val else default
    except Exception:
        return default


def get_config() -> dict:
    return {
        "base_url": _secret("base_url", "https://openrouter.ai/api/v1"),
        "api_key": _secret("api_key", ""),
        "chat_model": _secret("chat_model", "openai/gpt-4o-mini"),
        "embed_model": _secret("embed_model", "local"),
        "top_k": int(_secret("top_k", 5)),
    }


# ------------------------------------------------------------------ store
def get_store() -> RAGStore:
    # A per-session store in a temp dir. Documents persist for the session's
    # lifetime; each user has their own isolated document store.
    if "store_dir" not in st.session_state:
        st.session_state.store_dir = tempfile.mkdtemp(prefix="rag_session_")
    return RAGStore(st.session_state.store_dir)


# ------------------------------------------------------------------ client
def get_client(cfg: dict) -> OpenAI:
    return OpenAI(base_url=cfg["base_url"].rstrip("/"), api_key=cfg["api_key"] or "n/a", timeout=300)


def embed_texts(client: OpenAI | None, model: str, texts: list[str]) -> list[list[float]]:
    if embed_local.is_local(model):
        return embed_local.embed(texts)
    if client is None:
        raise RuntimeError("Remote embeddings require an API client and key")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        resp = client.embeddings.create(model=model, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend(x.embedding for x in ordered)
    return vectors


def stream_answer(client: OpenAI, model: str, messages: list[dict]):
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3, stream=True)
    for event in resp:
        if getattr(event, "choices", None):
            delta = event.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content


# ------------------------------------------------------------------ sidebar
st.set_page_config(page_title="Local RAG", page_icon="📚", layout="centered")
st.title("📚 Local RAG")
st.caption("Retrieval-augmented chat over your own documents — any OpenAI-compatible API.")

cfg = get_config()

with st.sidebar:
    st.header("Connection")
    # Allow overriding secrets per-session (handy when testing locally).
    cfg["base_url"] = st.text_input("API Base URL", value=cfg["base_url"])
    cfg["api_key"] = st.text_input("API Key", value=cfg["api_key"], type="password")
    cfg["chat_model"] = st.text_input("Chat model", value=cfg["chat_model"])
    cfg["embed_model"] = st.text_input("Embedding model", value=cfg["embed_model"])
    cfg["top_k"] = st.number_input("Top K chunks", min_value=1, max_value=20, value=cfg["top_k"], step=1)

    if not cfg["api_key"]:
        st.warning("No API key set. Add it below for this session, or put it in `.streamlit/secrets.toml` / Streamlit Cloud Secrets.")
    else:
        st.success("API key configured.")

    st.divider()
    st.header("Documents")
    store = get_store()

    if "indexed" not in st.session_state:
        st.session_state.indexed = set()

    uploads = st.file_uploader(
        "Upload PDF / DOCX / TXT / MD",
        type=["pdf", "docx", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )

    indexed_now = False
    for f in uploads or []:
        if f.name in st.session_state.indexed:
            continue
        try:
            with st.spinner(f"Indexing {f.name}…"):
                text = docparse.extract_text(f.name, f.getvalue())
                chunks = docparse.chunk_text(text)
                if not chunks:
                    st.error(f"{f.name}: no usable text found.")
                    continue
                client = None if embed_local.is_local(cfg["embed_model"]) else get_client(cfg)
                vecs = embed_texts(client, cfg["embed_model"], chunks)
                store.add_document(f.name, chunks, vecs)
                st.session_state.indexed.add(f.name)
                indexed_now = True
        except Exception as exc:  # noqa: BLE001
            st.error(f"{f.name}: failed to index — {exc}")

    docs = store.list_documents()
    if docs:
        st.caption(f"{len(docs)} document(s), {store.count_chunks()} chunks total:")
        for d in docs:
            st.caption(f"• {d['name']} — {d['chunks']} chunks")
        if st.button("Clear all documents"):
            store.clear()
            st.session_state.indexed.clear()
            st.rerun()
    else:
        st.caption("No documents indexed yet.")

# ------------------------------------------------------------------ main chat
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your documents…")

if prompt:
    if not cfg["api_key"].strip():
        st.error("Set an API key first in the sidebar to chat.")
        st.stop()

    st.session_state.setdefault("messages", []).append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        client = None if embed_local.is_local(cfg["embed_model"]) else get_client(cfg)
        qvec = embed_texts(client, cfg["embed_model"], [prompt])[0]
        hits = store.retrieve(qvec, top_k=int(cfg["top_k"]))

        if hits:
            context = "\n\n".join(f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits))
            system = SYSTEM_PROMPT + "\n\nCONTEXT:\n" + context
        else:
            system = SYSTEM_PROMPT + "\n\n(No matching document excerpts were found.)"
            hits = []

        chat_history = [
            m for m in st.session_state["messages"][-11:]
            if m.get("content")
        ]
        messages = [{"role": "system", "content": system}] + chat_history

        with st.chat_message("assistant"):
            chat_client = get_client(cfg)
            answer = st.write_stream(stream_answer(chat_client, cfg["chat_model"], messages))

        st.session_state["messages"].append({"role": "assistant", "content": answer})

        if hits:
            with st.expander(f"Sources ({len(hits)} retrieved)"):
                for i, h in enumerate(hits):
                    st.markdown(f"**[{i + 1}] {h['source']}** · score {h['score']}")
                    st.caption(h["text"][:400])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error: {exc}")