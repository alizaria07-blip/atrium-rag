"""End-to-end test: real app against a real mock OpenAI-compatible server.

Spins up mock_api in a subprocess, uploads a document through the app, and
verifies streaming chat + source citation. This exercises the entire pipeline
(parse -> embed -> retrieve -> stream) exactly as a user would.
"""
import json
import socket
import subprocess
import sys
import textwrap
import time

import pytest
from fastapi.testclient import TestClient

import app as appmod
from store import RAGStore

MOCK_SRC = "mock_api.py"  # served from cwd (rag-app)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_base():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "mock_api:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=".", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}/v1"
    # wait for readiness
    import urllib.request
    for _ in range(50):
        try:
            with urllib.request.urlopen(base + "/models", timeout=1):
                break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("mock API did not come up")
    base = base.rstrip("/")
    yield base
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "store", RAGStore(str(tmp_path)))
    return TestClient(appmod.app)


def test_full_rag_flow(client, mock_base):
    settings = {
        "base_url": mock_base,
        "api_key": "x",
        "embed_model": "mock-embed",
        "chat_model": "mock-chat",
    }

    text = textwrap.dedent("""\
        Winterfell was built by Brandon the Builder.
        Sansa Stark is the eldest daughter of Eddard Stark, Lord of Winterfell.
        The Wall is made of ice and is seven hundred feet high.
    """)
    r = client.post(
        "/api/documents",
        files={"file": ("winterfell.txt", text.encode(), "text/plain")},
        data={"settings": json.dumps(settings)},
    )
    assert r.status_code == 200, r.text
    doc = r.json()["document"]
    assert doc["chunks"] >= 1

    # streaming chat
    r = client.post(
        "/api/chat",
        json={"question": "Who built Winterfell?", "settings": settings, "history": []},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    frames = [line[5:].strip() for line in r.text.splitlines() if line.startswith("data:")]
    events = [json.loads(f) for f in frames if f]
    kinds = [e["type"] for e in events]
    assert "delta" in kinds and "sources" in kinds and "done" in kinds
    # the mock echoes sources back; make sure it cited the uploaded doc
    sources = next(e["sources"] for e in events if e["type"] == "sources")
    assert any(s["source"] == "winterfell.txt" for s in sources)
    body = "".join(e["content"] for e in events if e["type"] == "delta")
    assert body


def test_database_flow_plans_and_runs_sql(client, mock_base, tmp_path, monkeypatch):
    """Attach a warehouse, ask in database mode: the planner must emit SQL,
    the store must run it, and the answer must stream with the result in context."""
    from dbstore import DatabaseHub

    monkeypatch.setattr(appmod, "hub", DatabaseHub(str(tmp_path)))
    r = client.post("/api/databases/sample")
    assert r.status_code == 200, r.text
    db = r.json()["database"]
    assert db["engine"] == "sqlite"

    settings = {
        "base_url": mock_base,
        "api_key": "x",
        "embed_model": "mock-embed",
        "chat_model": "mock-chat",
    }
    r = client.post(
        "/api/chat",
        json={
            "question": "Which categories drove the most revenue?",
            "settings": settings,
            "history": [],
            "mode": "database",
            "database_id": db["id"],
        },
    )
    assert r.status_code == 200, r.text
    events = [json.loads(f) for f in (line[5:].strip() for line in r.text.splitlines() if line.startswith("data:")) if f]
    kinds = [e["type"] for e in events]
    assert "delta" in kinds and "done" in kinds

    sql_events = [e for e in events if e["type"] == "sql"]
    assert sql_events, "expected a sql event"
    ev = sql_events[0]
    assert ev["row_count"] > 0
    assert ev["columns"]
    assert "order_items" in ev["sql"]

    schema = client.get(f"/api/databases/{db['id']}/schema").json()
    assert {t["name"] for t in schema["tables"]} >= {"orders", "customers", "refunds"}

    preview = client.get(f"/api/databases/{db['id']}/preview/employees").json()
    assert preview["row_count"] > 0

    r = client.delete(f"/api/databases/{db['id']}")
    assert r.status_code == 200