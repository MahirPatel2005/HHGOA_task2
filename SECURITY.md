# Public deployment security

This project uses a split deployment:

- Vercel serves only the browser UI.
- FastAPI runs on a backend host.
- Provider credentials and vector-database credentials stay on the backend.

## Never publish these values

```text
SARVAM_API_KEY
OLLAMA_API_KEY
GROQ_API_KEY
HF_TOKEN
QDRANT_API_KEY
GENERATION_API_KEY
OPENCODE_GO_API_KEY
```

Store them as backend environment variables. Do not put them in `frontend/runtime-config.js`, Vercel project files, screenshots, videos, GitHub, or browser code.

## Backend protections enabled

- exact-origin CORS configuration;
- optional trusted-host allowlist;
- security response headers and Content Security Policy;
- HTTPS HSTS when served over HTTPS;
- per-client query and voice rate limits;
- maximum concurrent pipeline slots;
- 10 MB audio upload limit;
- audio MIME-type validation;
- query length and language-code limits;
- rejection of unexpected JSON fields;
- prompt-injection, unsafe-input, off-topic, and ungrounded-answer guardrails;
- stable public error codes instead of provider exception details;
- no credentials returned by `/api/config`.

## Required production environment

Set these on the backend host:

```env
CORS_ORIGINS=https://your-project.vercel.app
ALLOWED_HOSTS=your-backend-host.example.com
RATE_LIMIT_QUERY_PER_MINUTE=30
RATE_LIMIT_VOICE_PER_MINUTE=10
MAX_CONCURRENT_REQUESTS=4
MAX_AUDIO_BYTES=10485760
```

Use a shared Redis/KV rate limiter if the backend is scaled to multiple instances. The included limiter is appropriate for a single free-tier demo instance.

## If a key was exposed

Immediately revoke it at the provider, create a replacement, update only the backend environment variable, and redeploy. Treat a key as compromised if it appeared in chat, a commit, a log, an uploaded ZIP, a screenshot, or a client-side JavaScript file.
