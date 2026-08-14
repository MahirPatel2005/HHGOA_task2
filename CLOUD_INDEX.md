# Cloud index build (laptop can be off)

The GitHub Actions workflow builds the representative hackathon index in the cloud and uploads vectors directly to Qdrant Cloud.

## One-time setup

1. Push this repository to GitHub.
2. In GitHub: **Settings → Secrets and variables → Actions → New repository secret**.
3. Add:

```text
QDRANT_URL
QDRANT_API_KEY
HF_TOKEN (optional for this public dataset, useful for Hugging Face rate limits)
```

Never put these values in the repository or frontend.

## Start the cloud build

1. Open the repository's **Actions** tab.
2. Select **Build cloud demo index**.
3. Click **Run workflow**.
4. Keep `100` for `rows_per_language`.
5. Click **Run workflow**.

The job covers all 14 MSMARCO-XI language configurations and writes to:

```text
Qdrant collection: voice_rag_demo
Embedding model: intfloat/multilingual-e5-small
```

The laptop can be shut down after the GitHub job starts. Qdrant keeps the vectors online.

## Backend settings

The deployed backend must use the same collection and embedding model:

```env
VECTOR_BACKEND=qdrant
QDRANT_COLLECTION=voice_rag_demo
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

The workflow uses a representative 100 rows per language rather than attempting to process the entire 1.37-million-row validation split on a laptop. Report this scope honestly in the demo README.
