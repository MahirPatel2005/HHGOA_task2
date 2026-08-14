# Deployment plan — Vaani Voice RAG

## Recommended student-friendly architecture

Use **Vercel for the frontend** and a separate Python host for the FastAPI backend:

```text
Vercel Hobby frontend
        │
        ▼
Render/Railway/Hugging Face backend
        │
        ├── Sarvam Saaras
        ├── Ollama Cloud (`gpt-oss:120b`)
        └── Qdrant Cloud index
```

A Vercel-only deployment is not recommended for the full RAG backend because serverless cold starts, audio uploads, Python model loading, and persistent vector storage make the latency and storage behavior unreliable.

## 1. Build the demo index in the cloud

The final answer generator is configured for Ollama Cloud. Ollama's official cloud API uses `https://ollama.com`, a Bearer `OLLAMA_API_KEY`, and the native `/api/chat` endpoint. Set the key only on the backend host.

Do not build the index on the laptop. Push the repository to GitHub, add `QDRANT_URL`, `QDRANT_API_KEY`, and optionally `HF_TOKEN` as GitHub Actions secrets, then run the **Build cloud demo index** workflow from the repository's Actions tab. Full instructions are in [`CLOUD_INDEX.md`](CLOUD_INDEX.md).

The workflow builds a representative index from all 14 MSMARCO-XI language configurations and writes directly to Qdrant Cloud:

```text
collection: voice_rag_demo
embedding model: intfloat/multilingual-e5-small
rows per language: 100
```

The laptop can be shut down after the workflow starts. Do not try to build the index during a Vercel request.

## 2. Deploy the backend

The included `render.yaml` is a starting point for Render. In the Render dashboard:

1. Create a new Web Service from this repository.
2. Use the Python runtime.
3. Add the secret environment variables from `render.yaml`:

```env
OLLAMA_API_KEY=your_ollama_cloud_key
SARVAM_API_KEY=your_sarvam_key
HF_TOKEN=your_huggingface_read_token
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_key
```

`OLLAMA_API_KEY`, `SARVAM_API_KEY`, `HF_TOKEN`, and `QDRANT_API_KEY` belong only in the backend host's secret manager.

4. Set `CORS_ORIGINS` to the eventual Vercel URL.
5. Deploy and copy the resulting backend URL, for example:

```text
https://vaani-rag-api.onrender.com
```

Render's free plan can sleep when idle. That is fine for a demo, but it can violate a strict latency target on the first request. Keep the service warm or use a paid/always-on instance for final latency measurements.

## 3. Point the Vercel frontend at the backend

Edit `frontend/runtime-config.js`:

```js
window.VAANI_API_BASE = "https://vaani-rag-api.onrender.com";
```

The frontend now uses this base URL for `/api/query`, `/api/voice`, and `/api/config`.

## 4. Deploy the frontend to Vercel

Option A — Vercel dashboard:

1. Import the GitHub repository.
2. Keep the project root as the repository root.
3. Deploy with the included `vercel.json`.

Option B — Vercel CLI:

```bash
npm install -g vercel
vercel login
vercel --prod
```

The `vercel.json` file serves `frontend/index.html` and `frontend/runtime-config.js` as a static site.

## Public security checklist

- Keep Sarvam, Ollama, Groq, Hugging Face, and Qdrant credentials only in backend environment variables.
- Keep `frontend/runtime-config.js` limited to the public backend URL; it must never contain a secret.
- Set `CORS_ORIGINS` to the exact Vercel origin, not `*`.
- Set `ALLOWED_HOSTS` to the exact backend host.
- Keep the built-in query/voice rate limits enabled; replace the in-process limiter with Redis/KV for multiple backend instances.
- Keep the 10 MB audio limit and four-request concurrency cap unless you have measured a safe change.
- API responses expose stable error codes only; provider exception details stay server-side.
- Rotate any key that has appeared in chat, screenshots, logs, or Git history.
- Check that `/api/config` reveals only provider names and limits, never credentials.
- Use HTTPS for both Vercel and the backend.

## 5. Final checks

After deployment, verify:

```bash
curl https://YOUR_BACKEND/health
curl https://YOUR_VERCEL_APP/
```

Then use the browser microphone and test:

```text
What is the capital of Goa?
```

Set `CORS_ORIGINS` to the exact Vercel origin, without a trailing slash. Never commit `.env`, API keys, Hugging Face tokens, or Qdrant keys.

## Free-tier limitation

Vercel Hobby plus a sleeping free backend is appropriate for a student demo link, but it is not a reliable way to promise a sub-200ms full voice-to-answer latency. Measure the deployed system honestly and report cold-start, warm-request, STT, retrieval, generation, and total P50/P70/P100 values separately.
