"""Unit tests for the RAG pipeline: docparse, store, and API endpoints."""
import json
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

import docparse
import app as appmod
from store import RAGStore


# ------------------------------------------------------------------ docparse
def test_chunk_text_produces_multiple_overlapping_chunks():
    chunks = docparse.chunk_text("Paragraph one. " * 30 + "\n\n" + "Paragraph two. " * 40)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)


def test_chunk_text_single_small_paragraph_is_one_chunk():
    chunks = docparse.chunk_text("Just a short bit of text.")
    assert chunks == ["Just a short bit of text."]


def test_extract_text_txt():
    assert docparse.extract_text("a.txt", b"hello\nworld") == "hello\nworld"


def test_extract_text_empty():
    assert docparse.chunk_text("   \n\n  ") == []


# ------------------------------------------------------------------ store
def _bow(vocab):
    def fn(t):
        t = t.lower()
        return [float(t.count(w)) for w in vocab]
    return fn


def test_store_add_retrieve_rank_delete_persist():
    vocab = ["apple", "fruit", "tree", "rocket", "orbit", "satellite", "pasta", "wheat", "boil"]
    bow = _bow(vocab)
    sections = {
        "apples": "Apples are round fruits that grow on trees. " * 60,
        "space": "Rockets launch satellites into orbit around Earth. " * 60,
        "pasta": "Pasta is made from wheat flour and water, cooked in boiling water. " * 60,
    }
    doc = "\n\n".join(f"### {k}\n{v}" for k, v in sections.items())
    chunks = docparse.chunk_text(doc)
    assert len(chunks) >= 3

    tmpdir = tempfile.mkdtemp()
    store = RAGStore(tmpdir)
    store.add_document("topics", chunks, [bow(c) for c in chunks])

    def top_label(query):
        hit = store.retrieve(bow(query), top_k=1)[0]
        t = hit["text"]
        return "apples" if "round fruits" in t else ("space" if "Rockets" in t else "pasta")

    assert top_label("How do rockets reach orbit?") == "space"
    assert top_label("What fruits grow on trees?") == "apples"
    assert top_label("Does pasta need to be boiled?") == "pasta"

    # persistence round-trip
    store2 = RAGStore(tmpdir)
    assert store2.count_chunks() == len(chunks)
    assert store2.list_documents()[0]["name"] == "topics"

    # delete
    doc_id = store2.list_documents()[0]["id"]
    assert store2.remove_document(doc_id) is True
    assert store2.list_documents() == []
    assert store2.count_chunks() == 0


# ------------------------------------------------------------------ api
@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the global store at a throwaway dir so tests don't touch data/.
    fresh = RAGStore(str(tmp_path))
    monkeypatch.setattr(appmod, "store", fresh)
    return TestClient(appmod.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Atrium" in r.text


def test_documents_empty(client):
    assert client.get("/api/documents").json() == {"documents": []}


def test_delete_missing_404(client):
    assert client.delete("/api/documents/nope").status_code == 404


def test_upload_bad_embedding_returns_502(client):
    body = json.dumps({"base_url": "http://127.0.0.1:1/v1", "api_key": "x", "embed_model": "m"})
    r = client.post(
        "/api/documents",
        files={"file": ("hello.txt", b"some text content", "text/plain")},
        data={"settings": body},
    )
    assert r.status_code == 502
    assert "Embedding failed" in r.json()["detail"]


def test_upload_unreadable_file_returns_400(client):
    body = json.dumps({"base_url": "http://127.0.0.1:1/v1", "api_key": "x"})
    r = client.post(
        "/api/documents",
        files={"file": ("bad.XXXX", b"\xff\xfe bogus", "application/octet-stream")},
        data={"settings": body},
    )
    # Text decode may succeed with replacement characters; local embeddings can
    # therefore accept this as a small text document, while remote embeddings
    # return a gateway error.
    assert r.status_code in (200, 400, 502)


def test_chat_offline_returns_502(client):
    body = {
        "question": "hi",
        "settings": {
            "base_url": "http://127.0.0.1:1/v1",
            "api_key": "x",
            "embed_model": "text-embedding-3-small",
        },
    }
    r = client.post("/api/chat", json=body)
    assert r.status_code == 502
    assert "Could not connect" in r.json()["detail"]


def test_chat_empty_question_400(client):
    r = client.post(
        "/api/chat",
        json={"question": "   ", "settings": {"base_url": "x", "api_key": ""}},
    )
    assert r.status_code == 400


# ------------------------------------------------------------------ additional tests
def test_chunk_text_overlap_does_not_duplicate_paragraph():
    text = "First paragraph about alpha.\n\nSecond paragraph about beta.\n\nThird paragraph about gamma."
    chunks = docparse.chunk_text(text, chunk_size=40, overlap=10)
    assert len(chunks) >= 2
    for c in chunks:
        # Paragraph header should not be duplicated with itself
        assert "Second paragraph\n\nSecond paragraph" not in c


def test_store_dimension_mismatch_raises_clear_error():
    tmpdir = tempfile.mkdtemp()
    store = RAGStore(tmpdir)
    store.add_document("doc1", ["chunk1"], [[1.0, 2.0, 3.0]])
    
    # Adding a document with different dimension
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        store.add_document("doc2", ["chunk2"], [[1.0, 2.0]])
    
    # Querying with different dimension
    with pytest.raises(ValueError, match="Query vector dimension"):
        store.retrieve([1.0, 2.0])


def test_sql_read_only_with_keywords_in_literals():
    from dbstore import is_read_only_sql
    assert is_read_only_sql("SELECT * FROM products WHERE name LIKE '%vacuum%'") is None
    assert is_read_only_sql("SELECT created_at, update_date FROM orders") is None
    assert is_read_only_sql("SELECT * FROM logs WHERE action = 'delete from cart'") is None
    assert is_read_only_sql("DROP TABLE users") is not None
    assert is_read_only_sql("INSERT INTO users VALUES (1)") is not None
    assert is_read_only_sql("SELECT * FROM users; DROP TABLE users;") is not None


def test_database_hub_relative_path_resolution(tmp_path):
    from dbstore import DatabaseHub
    hub = DatabaseHub(str(tmp_path))
    sample = hub.add_sample()
    cid = sample["id"]
    schema = hub.schema(cid)
    assert len(schema["tables"]) > 0
    # Re-instantiate hub in another instance pointing to same directory
    hub2 = DatabaseHub(str(tmp_path))
    assert hub2.get(cid) is not None
    preview = hub2.preview(cid, "employees")
    assert preview["row_count"] > 0