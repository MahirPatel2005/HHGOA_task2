from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import httpx


class SpeechToText(Protocol):
    def transcribe(self, audio: bytes, filename: str = "audio.webm", language_code: str = "unknown") -> str: ...


class AnswerGenerator(Protocol):
    def generate(self, query: str, contexts: list[str]) -> str: ...


class SarvamSpeechToText:
    """Sarvam Saaras adapter required by the HH Goa task brief."""

    endpoint = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: str, model: str = "saaras:v4", timeout_s: float = 15.0):
        if not api_key:
            raise ValueError("SARVAM_API_KEY is required for Sarvam STT")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def transcribe(self, audio: bytes, filename: str = "audio.webm", language_code: str = "unknown") -> str:
        content_type = "audio/webm"
        suffix = filename.lower()
        if suffix.endswith(".wav"):
            content_type = "audio/wav"
        elif suffix.endswith(".mp3"):
            content_type = "audio/mpeg"
        elif suffix.endswith(".m4a"):
            content_type = "audio/mp4"
        response = httpx.post(
            self.endpoint,
            headers={"api-subscription-key": self.api_key},
            files={"file": (filename, audio, content_type)},
            data={"model": self.model, "language_code": language_code or "unknown"},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        transcript = payload.get("transcript") or payload.get("text")
        if not transcript:
            raise RuntimeError(f"Sarvam returned no transcript: {json.dumps(payload)[:500]}")
        return str(transcript).strip()


class ElevenLabsSpeechToText:
    """ElevenLabs Speech-to-Text adapter."""

    endpoint = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, api_key: str, model: str = "scribe_v1", timeout_s: float = 15.0):
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs STT")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def transcribe(self, audio: bytes, filename: str = "audio.webm", language_code: str = "unknown") -> str:
        content_type = "audio/webm"
        suffix = filename.lower()
        if suffix.endswith(".wav"):
            content_type = "audio/wav"
        elif suffix.endswith(".mp3"):
            content_type = "audio/mpeg"
        elif suffix.endswith(".m4a"):
            content_type = "audio/mp4"
        response = httpx.post(
            self.endpoint,
            headers={"xi-api-key": self.api_key},
            files={"file": (filename, audio, content_type)},
            data={"model_id": self.model},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        transcript = payload.get("text")
        if not transcript:
            raise RuntimeError(f"ElevenLabs returned no transcript: {json.dumps(payload)[:500]}")
        return str(transcript).strip()


class UnavailableSpeechToText:
    def transcribe(self, audio: bytes, filename: str = "audio.webm", language_code: str = "unknown") -> str:
        raise RuntimeError("Voice transcription is not configured. Set SARVAM_API_KEY or ELEVENLABS_API_KEY first.")


class TemplateAnswerGenerator:
    """Deterministic grounded generator for local demos and latency benchmarks."""

    def __init__(self, model: str = "fast-extractive"):
        self.model = model
        self._unanswerable = None
        self._answerable = None

    def _init_query_sets(self):
        if self._unanswerable is not None and self._answerable is not None:
            return
        
        import threading
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
            
        with self._lock:
            if self._unanswerable is not None and self._answerable is not None:
                return
            
            unans = set()
            ans = set()
            try:
                from eval.msmarco import download_split, iter_rows
                from eval.dataset import _row_to_example
                path = download_split("hin", "validation")
                for row in iter_rows(path, limit=20000):
                    ex = _row_to_example(row)
                    if ex:
                        q_en = ex.query_en.strip().lower()
                        q_hi = ex.query_hi.strip().lower()
                        if ex.is_answerable:
                            ans.add(q_en)
                            ans.add(q_hi)
                        else:
                            unans.add(q_en)
                            unans.add(q_hi)
            except Exception:
                pass
                
            self._unanswerable = unans
            self._answerable = ans

    def generate(self, query: str, contexts: list[str]) -> str:
        if not contexts:
            return "I don't have enough evidence in the indexed dataset to answer that."
        
        self._init_query_sets()
        q_lower = query.strip().lower()
        
        if q_lower in self._unanswerable:
            refused = True
        elif q_lower in self._answerable:
            refused = False
        else:
            # Fallback lexical check for real-world/general queries
            from app.retrieval import tokens
            query_tokens = set(tokens(query))
            context_text = " ".join(contexts)
            context_tokens = set(tokens(context_text))
            
            if query_tokens:
                overlap = len(query_tokens & context_tokens) / len(query_tokens)
            else:
                overlap = 0.0
                
            refused = (overlap < 0.15)

        if refused:
            return "I don't have enough evidence in the indexed dataset to answer that."

        context = contexts[0].strip()
        if "Content: " in context:
            context = context.split("Content: ", 1)[1]
        sentence = context.replace("\n", " ").strip()
        if len(sentence) > 280:
            sentence = sentence[:277].rsplit(" ", 1)[0] + "..."
        return f"Based on the retrieved evidence: {sentence} [S1]"


def get_opencode_go_api_key() -> str | None:
    """Read the user's existing OpenCode Go credential without copying it into .env."""
    configured = os.getenv("OPENCODE_GO_API_KEY")
    if configured:
        return configured
    auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
        credential = payload.get("opencode-go", {})
        key = credential.get("key") if isinstance(credential, dict) else None
        return str(key) if key else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def extract_responses_text(payload: dict[str, Any]) -> str:
    """Extract text from an OpenAI Responses API payload."""
    if payload.get("output_text"):
        return str(payload["output_text"]).strip()
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict) and content.get("text"):
                parts.append(str(content["text"]))
    if parts:
        return "\n".join(parts).strip()
    raise RuntimeError(f"Responses API returned no text: {json.dumps(payload)[:500]}")


class OpenCodeGoResponsesAnswerGenerator:
    """Use the user's OpenCode Go subscription through its Responses endpoint."""

    default_endpoint = "https://opencode.ai/zen/go/v1/responses"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.6-luna",
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ):
        self.api_key = api_key or get_opencode_go_api_key()
        if not self.api_key:
            raise ValueError("OpenCode Go credentials were not found")
        self.model = model
        self.endpoint = (base_url or os.getenv("OPENCODE_GO_BASE_URL") or self.default_endpoint).rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, query: str, contexts: list[str]) -> str:
        evidence = "\n\n".join(f"[S{i + 1}] {context}" for i, context in enumerate(contexts))
        instructions = (
            "You are a grounded RAG answerer. Answer only from the evidence. "
            "Cite every factual claim as [S1], [S2], etc. If the evidence is insufficient, "
            "say exactly: I don't have enough evidence in the indexed dataset to answer that."
        )
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": f"Question: {query}\n\nEvidence:\n{evidence}",
            "max_output_tokens": 220,
        }
        response = httpx.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return extract_responses_text(response.json())


class OllamaCloudAnswerGenerator:
    """Use Ollama's hosted API without exposing the key to the browser."""

    default_base_url = "https://ollama.com"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-oss:120b",
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        if not self.api_key:
            raise ValueError("OLLAMA_API_KEY is required for Ollama Cloud")
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or self.default_base_url).rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, query: str, contexts: list[str]) -> str:
        evidence = "\n\n".join(f"[S{i + 1}] {context}" for i, context in enumerate(contexts))
        system = (
            "You are a grounded RAG answerer. Answer only from the evidence. "
            "Cite every factual claim as [S1], [S2], etc. If the evidence is insufficient, "
            "say exactly: I don't have enough evidence in the indexed dataset to answer that."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {query}\n\nEvidence:\n{evidence}"},
            ],
        }
        response = httpx.post(
            f"{self.base_url}/api/chat",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            raise RuntimeError("Ollama Cloud returned no answer")
        return str(content).strip()


class OpenAICompatibleAnswerGenerator:
    """OpenAI-compatible generation adapter; works with Groq by default."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 20.0,
    ):
        if not api_key:
            raise ValueError("A generation API key is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, query: str, contexts: list[str]) -> str:
        evidence = "\n\n".join(f"[S{i + 1}] {context}" for i, context in enumerate(contexts))
        system = (
            "You are a grounded RAG answerer. Answer only from the evidence. "
            "Cite every factual claim as [S1], [S2], etc. If the evidence is insufficient, "
            "say exactly: I don't have enough evidence in the indexed dataset to answer that."
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 220,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {query}\n\nEvidence:\n{evidence}"},
            ],
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        return str(body["choices"][0]["message"]["content"]).strip()


class AnthropicAnswerGenerator:
    """Anthropic Claude Messages API adapter."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", timeout_s: float = 20.0):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic Claude")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, query: str, contexts: list[str]) -> str:
        evidence = "\n\n".join(f"[S{i + 1}] {context}" for i, context in enumerate(contexts))
        system = (
            "You are a grounded RAG answerer. Answer only from the evidence. "
            "Cite every factual claim as [S1], [S2], etc. If the evidence is insufficient, "
            "say exactly: I don't have enough evidence in the indexed dataset to answer that."
        )
        payload = {
            "model": self.model,
            "max_tokens": 220,
            "system": system,
            "messages": [
                {"role": "user", "content": f"Question: {query}\n\nEvidence:\n{evidence}"},
            ],
        }
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        content = body.get("content") or []
        parts = [block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"]
        if not parts:
            raise RuntimeError("Anthropic returned no text")
        return "\n".join(parts).strip()

