# Deploying to Streamlit Community Cloud

This repo contains **two** front-ends for the same RAG backend:

- `streamlit_app.py` — the **Streamlit** app (what you deploy to Streamlit Cloud)
- `app.py` — the original FastAPI app with a browser UI (run locally)

Streamlit Cloud runs the Streamlit version. The core logic (`store.py`,
`docparse.py`) is shared.

---

## 1. Put the code on GitHub (done)

Streamlit Cloud deploys straight from a public GitHub repo.

- Repo: `alizaria07-blip/local-rag`
- The app entrypoint is **`streamlit_app.py`** at the repo root.

## 2. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io (or https://streamlit.io/cloud) and
   **Sign in with GitHub** — authorise the Streamlit app to access your repos.
2. Click **New app** → **Deploy from a repo**.
3. Choose `alizaria07-blip/local-rag`.
   - Branch: `main`
   - Main file: `streamlit_app.py`
   - Leave "Advanced settings" for a moment — we'll set secrets next.
4. Click **Deploy**. Streamlit installs `requirements.txt` (includes
   `streamlit`, `openai`, `pypdf`, `python-docx`, `numpy`) and starts the app.

## 3. Add your secrets

The app reads your API key from Streamlit secrets:

1. In your app's dashboard, click **⋮ (kebab menu) → Settings** (or
   **Settings → Secrets**).
2. Open the **Secrets** tab and add a file with the same keys as
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example):

   ```toml
   base_url = "https://openrouter.ai/api/v1"
   api_key = "sk-or-..."
   chat_model = "openai/gpt-4o-mini"
   embed_model = "local"
   top_k = 5
   ```

3. Save. Streamlit restarts the app with the secrets available.

The app supports standard keys (e.g. `api_key`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `base_url`, `OPENAI_BASE_URL`).

## 4. Use it

- **Sidebar** → upload PDF/DOCX/TXT/MD files to index them, or attach sample/custom SQL databases.
- **Main area** → ask questions across documents, SQL database, or both.
- **Clear all documents** button in the sidebar wipes the current index.

---

## Notes & recommendations

- **Embeddings:** When using OpenRouter (`base_url = "https://openrouter.ai/api/v1"`), keep `embed_model = "local"`. Local embeddings run built-in deterministic hash vectors locally with zero extra API costs or external embedding dependencies. If using direct OpenAI (`https://api.openai.com/v1`), you can set `embed_model = "openai/text-embedding-3-small"`.
- **Ephemeral storage.** On Cloud, the vector index and uploaded databases live in the app process's temporary directory. It resets whenever the app is redeployed or scaled to zero.
- **Security.** The API key is stored in Streamlit's secrets vault, never in the repo. Never commit a real `secrets.toml`.
- **Free-tier limits.** The free Community Cloud tier sleeps after 3 days of
  inactivity and is single-instance. That's fine for personal RAG.

---

## Local development

```bash
cd rag-app
. .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

It opens at http://localhost:8501.