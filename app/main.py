from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from .chunking import Document, MultiStrategyChunker
from .embedding import HashingEmbedder, SentenceTransformerEmbedder
from .guardrails import GuardrailEngine
from .harness import RAGOrchestrator
from .providers import (
    OllamaCloudAnswerGenerator,
    OpenAICompatibleAnswerGenerator,
    OpenCodeGoResponsesAnswerGenerator,
    SarvamSpeechToText,
    ElevenLabsSpeechToText,
    TemplateAnswerGenerator,
    UnavailableSpeechToText,
    get_opencode_go_api_key,
    AnthropicAnswerGenerator,
)
from .retrieval import HybridRetriever
from .schemas import PipelineRequest, PipelineResponse
from .security import SlidingWindowRateLimiter, public_error
from .vector_store import LocalVectorStore, QdrantVectorStore

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample.jsonl"
INDEX_PATH = ROOT / "storage" / "local_index.json"
FRONTEND_PATH = ROOT / "frontend" / "index.html"


class TextQuery(BaseModel):
    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, max_length=2000)
    language_code: str = Field(default="unknown", max_length=32)
    mode: str = Field(default="fast", max_length=32)


def load_jsonl_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    if not path.exists():
        return documents
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        text = str(row.get("text") or row.get("content") or row.get("passage") or "").strip()
        if not text:
            continue
        metadata = dict(row.get("metadata") or {})
        for key in ("url", "source", "language", "split"):
            if row.get(key) is not None:
                metadata.setdefault(key, row[key])
        documents.append(
            Document(
                id=str(row.get("id") or row.get("_id") or f"sample-{line_number}"),
                title=str(row.get("title") or ""),
                text=text,
                metadata=metadata,
            )
        )
    return documents


def sample_documents() -> list[Document]:
    return load_jsonl_documents(DATA_PATH)


def make_embedder():
    return HashingEmbedder(dim=384)


def build_store(embedder, documents: list[Document]):
    backend = os.getenv("VECTOR_BACKEND", "local").lower()
    configured_index = os.getenv("LOCAL_INDEX_PATH")
    msmarco_index = ROOT / "storage" / "msmarco_iix_index.json"
    local_index = Path(configured_index) if configured_index else (msmarco_index if msmarco_index.exists() else INDEX_PATH)
    if backend == "qdrant":
        store = QdrantVectorStore(
            dim=embedder.dim,
            path=os.getenv("QDRANT_PATH", str(ROOT / "storage" / "qdrant")),
            collection=os.getenv("QDRANT_COLLECTION", "voice_rag"),
            url=os.getenv("QDRANT_URL") or None,
            api_key=os.getenv("QDRANT_API_KEY") or None,
            batch_size=int(os.getenv("QDRANT_BATCH_SIZE", "64")),
            timeout_s=int(os.getenv("QDRANT_TIMEOUT_S", "120")),
        )
    else:
        if local_index.exists() and os.getenv("REBUILD_INDEX", "0") != "1":
            try:
                loaded = LocalVectorStore.load(local_index)
                if loaded.dim == embedder.dim:
                    return loaded
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        store = LocalVectorStore(dim=embedder.dim)
    chunker = MultiStrategyChunker(
        embedder=embedder,
        token_size=int(os.getenv("TOKEN_CHUNK_SIZE", "180")),
        token_overlap=int(os.getenv("TOKEN_CHUNK_OVERLAP", "36")),
        semantic_threshold=float(os.getenv("SEMANTIC_CHUNK_THRESHOLD", "0.38")),
    )
    chunks = chunker.chunk_many(documents)
    store.upsert((chunk.id, embedder.embed(chunk.text), chunk) for chunk in chunks)
    if isinstance(store, LocalVectorStore):
        store.save(local_index)
    return store


def create_orchestrator() -> RAGOrchestrator:
    documents = sample_documents()
    if not documents:
        documents = [Document(id="fallback", text="The local demo corpus is empty.")]
    embedder = make_embedder()
    store = build_store(embedder, documents)
    retriever = HybridRetriever(store=store, embedder=embedder)
    guardrails = GuardrailEngine(
        min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.025")),
        min_dense_score=float(os.getenv("MIN_DENSE_SCORE", "0.52")),
        min_grounding_overlap=float(os.getenv("MIN_GROUNDING_OVERLAP", "0.20")),
    )
    generation_key = os.getenv("GROQ_API_KEY") or os.getenv("GENERATION_API_KEY")
    ollama_key = os.getenv("OLLAMA_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    generation_provider = os.getenv("GENERATION_PROVIDER", "auto").lower()
    opencode_go_key = get_opencode_go_api_key()
    if generation_provider in {"ollama", "ollama-cloud", "ollama_cloud"} and ollama_key:
        generator = OllamaCloudAnswerGenerator(
            api_key=ollama_key,
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
            timeout_s=float(os.getenv("GENERATION_TIMEOUT_S", "30")),
        )
    elif generation_provider == "anthropic" and anthropic_key:
        generator = AnthropicAnswerGenerator(
            api_key=anthropic_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            timeout_s=float(os.getenv("GENERATION_TIMEOUT_S", "20")),
        )
    elif generation_provider in {"opencode-go", "opencode_go"} and opencode_go_key:
        generator = OpenCodeGoResponsesAnswerGenerator(
            api_key=opencode_go_key,
            model=os.getenv("OPENCODE_GO_MODEL", "gpt-5.6-luna"),
            base_url=os.getenv("OPENCODE_GO_BASE_URL"),
        )
    elif generation_provider == "auto" and opencode_go_key and not generation_key and not anthropic_key:
        generator = OpenCodeGoResponsesAnswerGenerator(
            api_key=opencode_go_key,
            model=os.getenv("OPENCODE_GO_MODEL", "gpt-5.6-luna"),
            base_url=os.getenv("OPENCODE_GO_BASE_URL"),
        )
    elif generation_provider == "auto" and anthropic_key:
        generator = AnthropicAnswerGenerator(
            api_key=anthropic_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            timeout_s=float(os.getenv("GENERATION_TIMEOUT_S", "20")),
        )
    elif generation_key:
        generator = OpenAICompatibleAnswerGenerator(
            api_key=generation_key,
            model=os.getenv("GENERATION_MODEL", "llama-3.1-8b-instant"),
            base_url=os.getenv("GENERATION_BASE_URL", "https://api.groq.com/openai/v1"),
        )
    else:
        generator = TemplateAnswerGenerator()
    stt_provider = os.getenv("STT_PROVIDER", "sarvam").lower()
    sarvam_key = os.getenv("SARVAM_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    if stt_provider == "elevenlabs" and elevenlabs_key:
        stt = ElevenLabsSpeechToText(
            api_key=elevenlabs_key,
            model=os.getenv("ELEVENLABS_MODEL", "scribe_v1"),
        )
    elif sarvam_key and sarvam_key != "your_sarvam_key_here":
        stt = SarvamSpeechToText(
            api_key=sarvam_key,
            model=os.getenv("SARVAM_MODEL", "saaras:v4"),
        )
    elif elevenlabs_key:
        stt = ElevenLabsSpeechToText(
            api_key=elevenlabs_key,
            model=os.getenv("ELEVENLABS_MODEL", "scribe_v1"),
        )
    else:
        stt = UnavailableSpeechToText()
    return RAGOrchestrator(
        retriever=retriever,
        answer_generator=generator,
        guardrails=guardrails,
        stt=stt,
        max_retries=int(os.getenv("PIPELINE_MAX_RETRIES", "2")),
        step_timeout_s=float(os.getenv("PIPELINE_STEP_TIMEOUT_S", "20")),
    )


def create_app(orchestrator: RAGOrchestrator | None = None) -> FastAPI:
    app = FastAPI(title="HH Goa Voice RAG", version="0.1.0")
    configured_origins = [origin.strip().rstrip("/") for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
    if not configured_origins:
        configured_origins = ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8001", "http://127.0.0.1:8001"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    allowed_hosts = [host.strip() for host in os.getenv("ALLOWED_HOSTS", "").split(",") if host.strip()]
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "microphone=(self), camera=()")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; script-src 'self' 'unsafe-inline' https://esm.sh https://*.esm.sh data: blob:; connect-src 'self' https: wss: ws:; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'",
        )
        if request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

    pipeline = orchestrator
    pipeline_lock = threading.Lock()

    def get_pipeline() -> RAGOrchestrator:
        nonlocal pipeline
        if pipeline is None:
            with pipeline_lock:
                if pipeline is None:
                    pipeline = create_orchestrator()
        return pipeline
    query_limiter = SlidingWindowRateLimiter(
        max_requests=int(os.getenv("RATE_LIMIT_QUERY_PER_MINUTE", "30")), window_seconds=60
    )
    voice_limiter = SlidingWindowRateLimiter(
        max_requests=int(os.getenv("RATE_LIMIT_VOICE_PER_MINUTE", "10")), window_seconds=60
    )
    slots = threading.BoundedSemaphore(max(1, int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))))
    max_audio_bytes = int(os.getenv("MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))

    def client_id(request: Request) -> str:
        return request.client.host if request.client else "anonymous"

    def rate_limit_response(limiter: SlidingWindowRateLimiter, request: Request) -> JSONResponse | None:
        if limiter.allow(client_id(request)):
            return None
        return JSONResponse(
            status_code=429,
            content={"status": "rate_limited", "error": "Too many requests. Please try again later."},
            headers={"Retry-After": "60"},
        )

    def concurrency_response() -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"status": "busy", "error": "The demo is at capacity. Please retry shortly."},
            headers={"Retry-After": "5"},
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "hh-goa-voice-rag", "pipeline_initialized": pipeline is not None}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        stt_provider = "not configured"
        if pipeline is not None:
            generator = pipeline.answer_generator
            if isinstance(generator, OllamaCloudAnswerGenerator):
                generation_provider = f"Ollama Cloud / {generator.model}"
            elif isinstance(generator, OpenCodeGoResponsesAnswerGenerator):
                generation_provider = f"OpenCode Go / {generator.model}"
            elif isinstance(generator, OpenAICompatibleAnswerGenerator):
                generation_provider = f"OpenAI-compatible / {generator.model}"
            elif isinstance(generator, AnthropicAnswerGenerator):
                generation_provider = f"Anthropic Claude / {generator.model}"
            else:
                generation_provider = "local grounded demo generator"

            stt = pipeline.stt
            if isinstance(stt, ElevenLabsSpeechToText):
                stt_provider = "ElevenLabs STT"
            elif isinstance(stt, SarvamSpeechToText):
                stt_provider = "Sarvam Saaras"
        else:
            configured_provider = os.getenv("GENERATION_PROVIDER", "auto").lower()
            if configured_provider in {"ollama", "ollama-cloud", "ollama_cloud"} or (
                configured_provider == "auto" and os.getenv("OLLAMA_API_KEY")
            ):
                generation_provider = f"Ollama Cloud / {os.getenv('OLLAMA_MODEL', 'gpt-oss:120b')}"
            elif configured_provider == "anthropic" or (
                configured_provider == "auto" and os.getenv("ANTHROPIC_API_KEY")
            ):
                generation_provider = f"Anthropic Claude / {os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')}"
            elif configured_provider in {"opencode-go", "opencode_go"}:
                generation_provider = f"OpenCode Go / {os.getenv('OPENCODE_GO_MODEL', 'gpt-5.6-luna')}"
            elif configured_provider in {"groq", "openai-compatible", "openai_compatible"}:
                generation_provider = f"OpenAI-compatible / {os.getenv('GENERATION_MODEL', 'configured')}"
            else:
                generation_provider = "local grounded demo generator"

            stt_prov = os.getenv("STT_PROVIDER", "sarvam").lower()
            if stt_prov == "elevenlabs" and os.getenv("ELEVENLABS_API_KEY"):
                stt_provider = "ElevenLabs STT"
            elif os.getenv("SARVAM_API_KEY") and os.getenv("SARVAM_API_KEY") != "your_sarvam_key_here":
                stt_provider = "Sarvam Saaras"
            elif os.getenv("ELEVENLABS_API_KEY"):
                stt_provider = "ElevenLabs STT"

        return {
            "stt_provider": stt_provider,
            "generation_provider": generation_provider,
            "vector_backend": os.getenv("VECTOR_BACKEND", "local"),
            "embedding_backend": os.getenv("EMBEDDING_BACKEND", "hash"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
            "dataset": "ai4bharat/MSMARCO-XI",
            "pipeline_initialized": pipeline is not None,
            "security": {"rate_limits": True, "max_audio_bytes": max_audio_bytes},
        }

    @app.get("/api/voice/scribe-token")
    async def get_scribe_token(request: Request):
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "ElevenLabs STT is not configured."}
            )
        url = "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers)
            if response.status_code >= 400:
                return JSONResponse(
                    status_code=502,
                    content={"status": "error", "error": f"ElevenLabs token error ({response.status_code}): {response.text}"}
                )
            data = response.json()
            token = data.get("token")
            if not token:
                return JSONResponse(
                    status_code=502,
                    content={"status": "error", "error": "Invalid token payload received."}
                )
            return {"token": token}
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "error": f"Failed to retrieve scribe token: {str(exc)}"}
            )

    @app.post("/api/query", response_model=PipelineResponse)
    def query(payload: TextQuery, request: Request):
        limited = rate_limit_response(query_limiter, request)
        if limited:
            return limited
        if not slots.acquire(blocking=False):
            return concurrency_response()
        try:
            try:
                active_pipeline = get_pipeline()
            except Exception:
                return JSONResponse(
                    status_code=503,
                    content={"status": "error", "answer": "The RAG service is still starting.", "error": public_error("pipeline_startup_failed")},
                    headers={"Retry-After": "15"},
                )
            return active_pipeline.run(
                PipelineRequest(
                    query_text=payload.query,
                    language_code=payload.language_code,
                    metadata={"mode": payload.mode},
                )
            )
        finally:
            slots.release()

    @app.post("/api/voice", response_model=PipelineResponse)
    async def voice(request: Request, audio: UploadFile = File(...), language_code: str = Form("unknown")):
        limited = rate_limit_response(voice_limiter, request)
        if limited:
            return limited
        if audio.content_type and not (audio.content_type.startswith("audio/") or audio.content_type == "application/octet-stream"):
            return JSONResponse(status_code=415, content={"status": "error", "error": "Please upload an audio file."})
        declared_length = request.headers.get("content-length")
        if declared_length and declared_length.isdigit() and int(declared_length) > max_audio_bytes + 1_000_000:
            return JSONResponse(status_code=413, content={"status": "error", "error": "Audio file is too large."})
        content = await audio.read(max_audio_bytes + 1)
        if not content:
            return JSONResponse(status_code=400, content={"status": "error", "error": "Audio file is empty."})
        if len(content) > max_audio_bytes:
            return JSONResponse(status_code=413, content={"status": "error", "error": "Audio file is too large."})
        if not slots.acquire(blocking=False):
            return concurrency_response()
        try:
            try:
                active_pipeline = get_pipeline()
            except Exception:
                return JSONResponse(
                    status_code=503,
                    content={"status": "error", "answer": "The RAG service is still starting.", "error": public_error("pipeline_startup_failed")},
                    headers={"Retry-After": "15"},
                )
            return active_pipeline.run(
                PipelineRequest(
                    audio_bytes=content,
                    audio_filename=audio.filename or "audio.webm",
                    language_code=language_code,
                )
            )
        finally:
            slots.release()

    @app.get("/runtime-config.js", response_class=FileResponse)
    def runtime_config():
        return FileResponse(
            str(FRONTEND_PATH.parent / "runtime-config.js"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/", response_class=FileResponse)
    def home():
        return str(FRONTEND_PATH)

    return app


app = create_app()
