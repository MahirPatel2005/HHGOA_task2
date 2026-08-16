from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar
from uuid import uuid4

from .guardrails import GuardrailEngine
from .providers import AnswerGenerator, SpeechToText, TemplateAnswerGenerator
from .retrieval import HybridRetriever, RetrievalResult
from .schemas import Citation, PipelineRequest, PipelineResponse
from .security import public_error

T = TypeVar("T")


class RAGOrchestrator:
    """Structured execution harness with retries, timeouts, fallbacks and output validation."""

    def __init__(
        self,
        retriever: HybridRetriever,
        answer_generator: AnswerGenerator,
        guardrails: GuardrailEngine,
        stt: SpeechToText | None = None,
        fallback_generator: AnswerGenerator | None = None,
        max_retries: int = 2,
        step_timeout_s: float = 20.0,
    ):
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.guardrails = guardrails
        self.stt = stt
        self.fallback_generator = fallback_generator or TemplateAnswerGenerator()
        self.max_retries = max(0, max_retries)
        self.step_timeout_s = step_timeout_s

    def _step(self, name: str, operation: Callable[[], T], timings: dict[str, float], attempts: dict[str, int]) -> T:
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            attempts[name] = attempt
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(operation)
                    result = future.result(timeout=self.step_timeout_s)
                timings[name] = (time.perf_counter() - started) * 1000
                return result
            except Exception as exc:  # retries are part of the harness contract
                last_error = exc
        timings[name] = (time.perf_counter() - started) * 1000
        raise RuntimeError(f"{name} failed after {attempts[name]} attempts: {last_error}") from last_error

    @staticmethod
    def _response(
        request: PipelineRequest,
        started: float,
        timings: dict[str, float],
        attempts: dict[str, int],
        **kwargs,
    ) -> PipelineResponse:
        return PipelineResponse(
            request_id=request.request_id,
            total_latency_ms=(time.perf_counter() - started) * 1000,
            stage_latency_ms=timings,
            attempts=attempts,
            **kwargs,
        )

    def run(self, request: PipelineRequest) -> PipelineResponse:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        attempts: dict[str, int] = {}
        transcript = request.query_text.strip() if request.query_text else None

        if request.audio_bytes is not None:
            if self.stt is None:
                return self._response(
                    request, started, timings, attempts, status="error", answer="Voice transcription is not configured.", error="missing_stt"
                )
            try:
                transcript = self._step(
                    "stt",
                    lambda: self.stt.transcribe(request.audio_bytes or b"", request.audio_filename, request.language_code),
                    timings,
                    attempts,
                )
            except Exception as exc:
                return self._response(
                    request, started, timings, attempts, status="error", transcript=transcript, answer="Speech transcription failed.", error=public_error("stt_failed")
                )

        query = (transcript or "").strip()
        try:
            input_decision = self._step("input_guardrails", lambda: self.guardrails.check_input(query), timings, attempts)
        except Exception as exc:
            return self._response(request, started, timings, attempts, status="error", transcript=transcript, answer="Input validation failed.", error=public_error("input_validation_failed"))
        if not input_decision.allowed:
            return self._response(
                request,
                started,
                timings,
                attempts,
                status="blocked" if input_decision.reason in {"unsafe_input", "prompt_injection"} else "abstained",
                transcript=transcript,
                answer=input_decision.message,
                refusal_reason=input_decision.reason,
            )

        try:
            results = self._step("retrieval", lambda: self.retriever.search(query, top_k=5), timings, attempts)
            retrieval_decision = self.guardrails.check_retrieval(query, results)
        except Exception as exc:
            return self._response(request, started, timings, attempts, status="error", transcript=transcript, answer="Retrieval failed.", error=public_error("retrieval_failed"))
        if not retrieval_decision.allowed:
            return self._response(
                request,
                started,
                timings,
                attempts,
                status="abstained",
                transcript=transcript,
                answer=retrieval_decision.message,
                refusal_reason=retrieval_decision.reason,
            )

        contexts = [result.chunk.text for result in results[:3]]
        mode = request.metadata.get("mode") or "fast"
        try:
            if mode == "fast":
                answer = self._step("generation", lambda: self.fallback_generator.generate(query, contexts), timings, attempts)
            else:
                answer = self._step("generation", lambda: self.answer_generator.generate(query, contexts), timings, attempts)
        except Exception as primary_error:
            try:
                answer = self._step("generation_fallback", lambda: self.fallback_generator.generate(query, contexts), timings, attempts)
            except Exception as fallback_error:
                return self._response(
                    request, started, timings, attempts, status="error", transcript=transcript, answer="Answer generation failed.", error=public_error("generation_failed")
                )

        try:
            answer_decision = self._step("answer_guardrails", lambda: self.guardrails.check_answer(answer, contexts), timings, attempts)
        except Exception as exc:
            return self._response(request, started, timings, attempts, status="error", transcript=transcript, answer="Answer validation failed.", error=public_error("answer_validation_failed"))
        if not answer_decision.allowed:
            return self._response(
                request,
                started,
                timings,
                attempts,
                status="abstained",
                transcript=transcript,
                answer=answer_decision.message,
                refusal_reason=answer_decision.reason,
            )

        citations = [
            Citation(
                chunk_id=result.chunk.id,
                parent_id=result.chunk.parent_id,
                strategy=result.chunk.strategy,
                score=round(result.score, 6),
                text=result.chunk.text,
                metadata=result.chunk.metadata,
            )
            for result in results[:3]
        ]
        return self._response(
            request,
            started,
            timings,
            attempts,
            status="answered",
            transcript=transcript,
            answer=answer,
            citations=citations,
        )
