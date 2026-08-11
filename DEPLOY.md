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
   embed_model = "openai/text-embedding-3-small"
   top_k = 5
   ```

3. Save. Streamlit restarts the app with the secrets available.

The app will still load without secrets, but you'll get a warning in the
sidebar and uploads/chat will refuse until an API key is set.

## 4. Use it

- **Sidebar** → upload PDF/DOCX/TXT/MD files to index them.
- **Main area** → ask questions; answers stream in and cite their source
  chunks (view them under **Sources**).
- **Clear all documents** button in the sidebar wipes the current index.

---

## Notes & limitations

- **Ephemeral storage.** On Cloud, the vector index lives in the app process's
  temp directory. It resets whenever the app is redeployed or scaled to zero,
  so treat it as "upload per session" (fine for personal use).
- **OpenRouter keys.** `openai/text-embedding-3-small` is the recommended
  embedding model. Any OpenAI-compatible embedding endpoint works as long as
  the chosen provider exposes `/v1/embeddings`.
- **Security.** The API key is stored in Streamlit's secrets vault, never in
  the repo. Never commit a real `secrets.toml`.
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