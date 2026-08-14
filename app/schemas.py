from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class PipelineRequest(BaseModel):
    """Normalized input accepted by the orchestration harness."""

    query_text: str | None = None
    audio_bytes: bytes | None = None
    audio_filename: str = "audio.webm"
    language_code: str = "unknown"
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    parent_id: str
    strategy: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineResponse(BaseModel):
    request_id: str
    status: str
    transcript: str | None = None
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refusal_reason: str | None = None
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    attempts: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
