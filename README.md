# Local RAG

A local, self-contained Retrieval-Augmented Generation web app. It points at
**any OpenAI-compatible API** (vLLM, Groq, Together, Fireworks, DeepSeek,
OpenAI, …), indexes your local documents, and lets you chat with them in a
browser.

- All documents are chunked and embedded *locally* and stored on disk in `data/`.
- Only the embedding + chat requests go out to the API you configure.
- Your API key lives in your browser's `localStorage` — it is **never stored
  on the server**.

## What works
- Upload **PDF, DOCX, TXT, Markdown** files (drag & drop or click).
- Ask questions with streaming (token-by-token) answers.
- Answers cite the source chunks they were retrieved from.
- Documents persist across restarts.

## Setup

```bash
cd rag-app
. .venv/bin/activate          # activate the virtualenv
pip install -r requirements.txt

uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 in your browser.

> The `.venv` is already created for you, but re-running `pip install` after a
> fresh clone is always safe.

## Configure in the browser

In the left panel, enter:

| Field | What to put |
|-------|-------------|
| **API Base URL** | Your OpenAI-compatible endpoint. e.g. `https://api.groq.com/openai/v1`, `https://api.together.xyz/v1`, `https://api.openai.com/v1`, or `http://localhost:8000/v1` for a local vLLM server. |
| **API Key** | Your key (blank works for local vLLM with no auth). |
| **Chat model** | e.g. `gpt-4o-mini`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`, `llama-3.3-70b-versatile`. |
| **Embedding model** | e.g. `text-embedding-3-small`, `BAAI/bge-large-en-v1.5`, or any embedding model your server exposes. |
| **Top K** | Number of chunks retrieved per question (default 5). |

Click **Save & Test** to check the connection, then upload documents and chat.

## How it works
1. `docparse.py` — extracts plain text from PDFs (pypdf), DOCX (python-docx),
   and TXT/MD, then splits it into overlapping chunks (~900 chars).
2. `store.py` — embeds each chunk using the configured embedding model and
   stores the vectors locally (numpy), persisted to `data/`.
3. `app.py` — on each question it embeds the query, finds the top-K most
   similar chunks (cosine similarity), injects them into the system prompt,
   and streams the answer back over SSE.

## Endpoints
- `GET  /api/health` — status + counts
- `GET  /api/documents` — list indexed docs
- `POST /api/documents` — upload + index a file (multipart with a `settings` JSON field)
- `DELETE /api/documents/{id}` — remove a doc
- `POST /api/chat` — stream a grounded answer (SSE)

## Optional: run against a local mock API (no key needed to smoke-test)
A tiny mock OpenAI-compatible server is included for testing without any real
API. In one terminal:

```bash
uvicorn mock_api:app --host 127.0.0.1 --port 59998
```

Then in the app set Base URL to `http://127.0.0.1:59998/v1`, chat model
`mock-chat`, embedding model `mock-embed`, leave the key blank. It returns
deterministic fake embeddings and a canned streaming answer — enough to verify
the RAG pipeline end-to-end. When you ask in Database mode it also answers the
planner with a read-only SQL statement, so the warehouse path (schema →
query → grounded answer) can be exercised too. (Not for real use.)