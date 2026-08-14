# Vaani — Voice-Enabled Multilingual RAG

Vaani is a voice-enabled Retrieval-Augmented Generation system for **HH Goa 2026 Shortlisting Task 2**. A user speaks or types a question, the system transcribes voice input, retrieves evidence from MSMARCO-XI, generates a grounded answer, and displays citations and latency.

## Live demo

**Website:** <https://vaani-voice-rag.vercel.app>

**Backend health:** <https://vaani-voice-rag.onrender.com/health>

## Organizer requirements implemented

| Requirement | Implementation |
|---|---|
| Voice input and speech-to-text | Browser `MediaRecorder` with Sarvam Saaras (`saaras:v4`) |
| MSMARCO-XI corpus | Hugging Face ingestion with all 14 Indic language configurations |
| Broad chunking design | Paragraph, overlapping sentence-window, overlapping token, semantic-boundary, and metadata-aware views |
| Vector retrieval | Qdrant Cloud adapter with persistent collection support; local vector-store adapter for development |
| Answer generation | Ollama Cloud native `/api/chat` adapter (`gpt-oss:120b` by default) |
| Harness | Typed requests/responses, request IDs, retries, timeouts, fallbacks, stage timings, and structured errors |
| Guardrails | Unsafe-input and prompt-injection blocking, retrieval sufficiency checks, citation checks, grounding checks, and abstention |
| Latency analytics | Reproducible benchmark reporting P50/P70/P100 and raw per-query rows |

## Architecture

```text
Browser microphone or typed question
        │
        ▼
Sarvam Saaras speech-to-text
        │
        ▼
Input guardrails and typed orchestration
        │
        ├── Dense multilingual vector search → Qdrant Cloud
        ├── Lexical retrieval and metadata-aware views
        └── Reciprocal-rank fusion and lightweight reranking
        │
        ▼
Ollama Cloud grounded answer generation
        │
        ▼
Citation and grounding validation
        │
        ├── Answer with sources and latency
        └── Abstention/refusal when evidence is insufficient
```

## Repository layout

```text
app/                         FastAPI app, orchestration, retrieval, guardrails
scripts/                     Dataset ingestion and benchmarking
frontend/                    Browser voice/text interface
.github/workflows/           Cloud index workflow
render.yaml                  Render backend configuration
vercel-drop/                 Static frontend package for Vercel Drop
CLOUD_INDEX.md               Cloud indexing instructions
DEPLOYMENT.md                Backend/frontend deployment instructions
SECURITY.md                  Public deployment security rules
tests/                       Automated tests
```

## Local setup

Requirements: Python 3.11 or newer.

```bash
python -m pip install -e ".[production]"
python -m pytest -q
python -m uvicorn app.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The local development configuration can use the deterministic generator and local vector store. Never commit `.env`.

## Environment configuration

Copy the example file:

```bash
cp .env.example .env
```

For the cloud deployment, configure these values in Render's private environment settings:

```env
SARVAM_API_KEY=your_sarvam_key
OLLAMA_API_KEY=your_ollama_cloud_key
HF_TOKEN=your_huggingface_read_token
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key

SARVAM_MODEL=saaras:v4
GENERATION_PROVIDER=ollama
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_BASE_URL=https://ollama.com
VECTOR_BACKEND=qdrant
QDRANT_COLLECTION=voice_rag_demo
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

Provider keys are backend secrets. They must not appear in `frontend/runtime-config.js`, Vercel files, GitHub source files, screenshots, or videos.

## Cloud index

The repository includes a GitHub Actions workflow named **Build cloud demo index**. It builds a representative hackathon index from all 14 MSMARCO-XI language configurations and writes vectors directly to Qdrant Cloud. The default workflow scope is 100 rows per language, which keeps the student demo reproducible and cloud-friendly.

To run it:

1. Add `QDRANT_URL`, `QDRANT_API_KEY`, and optionally `HF_TOKEN` as GitHub Actions repository secrets.
2. Open the GitHub **Actions** tab.
3. Select **Build cloud demo index**.
4. Choose `rows_per_language=100`.
5. Run the workflow.

The workflow uses:

```text
Collection: voice_rag_demo
Embedding model: intfloat/multilingual-e5-small
Languages: all 14 MSMARCO-XI configurations
Evaluation queries: up to 500
```

The laptop is not part of the deployed runtime after the cloud index and backend are live. See [`CLOUD_INDEX.md`](CLOUD_INDEX.md) for details.

## Chunking and retrieval

Each parent passage receives multiple retrieval views:

1. Natural paragraph boundaries.
2. Three-sentence windows with one-sentence overlap.
3. Token windows with configurable overlap.
4. Semantic boundaries based on adjacent sentence embeddings.
5. Metadata-aware text containing title, language, source, and other fields.
6. Stable parent/child IDs and citation metadata.

Dense and lexical evidence are fused before lightweight reranking. The response preserves chunk IDs, parent IDs, strategy names, scores, and metadata.

## Harness and guardrails

Every request returns a structured response containing:

- request ID;
- transcript;
- answer or refusal;
- citations;
- stage latency and total latency;
- retry attempts;
- machine-readable status and refusal reason.

The harness provides bounded retries, per-step timeouts, a grounded fallback generator, and safe public error codes.

Guardrails cover:

- empty or overly long questions;
- unsafe or inappropriate instructions;
- prompt-injection attempts;
- insufficient or off-topic retrieval;
- answers without citations;
- answers with insufficient overlap with retrieved evidence.

## Benchmarking

Run the benchmark after the final index and provider configuration are available:

```bash
python scripts/benchmark.py \
  --queries benchmarks/msmarco_xi_eval.jsonl \
  --count 500 \
  --warmup 25 \
  --output benchmarks/final-results.json
```

The report includes:

- P50, P70, and P100 total latency;
- P50, P70, and P100 stage latency;
- unique query count;
- answered, abstained, and error counts;
- retries and per-query raw rows;
- provider, embedding, vector backend, and platform metadata.

Report retrieval, generation, STT, and complete voice latency separately. Do not reuse local demo measurements as final claims. Remote Sarvam and Ollama calls must be measured on the deployed configuration.

## API

```text
GET  /health
GET  /api/config
POST /api/query
POST /api/voice
GET  /
```

Text query example:

```bash
curl -X POST https://YOUR_BACKEND/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of Goa?","language_code":"unknown"}'
```

## Deployment

Recommended deployment:

```text
Vercel frontend
      ↓
Render FastAPI backend
      ↓
Qdrant Cloud + Sarvam + Ollama Cloud
```

### Backend

Deploy the repository to Render using `render.yaml` or the included Dockerfile. Add the backend secrets and set:

```env
CORS_ORIGINS=https://YOUR-VERCEL-DOMAIN
ALLOWED_HOSTS=YOUR-BACKEND-DOMAIN
```

### Frontend

Edit `frontend/runtime-config.js`:

```js
window.VAANI_API_BASE = "https://YOUR-BACKEND-DOMAIN";
```

For Vercel Drop, use the contents of `vercel-drop/`. The frontend contains only the public backend URL; it never contains provider credentials.

See [`DEPLOYMENT.md`](DEPLOYMENT.md) and [`SECURITY.md`](SECURITY.md) for the complete deployment and security details.

## Data and provider references

- [HH Goa 2026 Task 2 brief](https://docs.google.com/document/d/1gzPyuYMaJGnv7mjPZ7Z_e20VxP0j5PMOPi8WmBi8rFk/edit?tab=t.0)
- [AI4Bharat MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
- [Ollama Cloud API](https://docs.ollama.com/cloud)

Check the applicable dataset, model, and provider terms before publishing a final deployment.
