from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_orchestrator
from app.schemas import PipelineRequest

DEFAULT_QUERIES = [
    "What is the capital of Goa?",
    "Where is Goa located?",
    "When did Goa become part of India?",
    "What is the climate in Goa?",
    "What biodiversity is found in Goa?",
    "What is retrieval augmented generation?",
    "When should a RAG system abstain?",
    "What does Sarvam Saaras do?",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower, upper = int(position), min(len(ordered) - 1, int(position) + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(values, 0.50), 3),
        "p70": round(percentile(values, 0.70), 3),
        "p100": round(max(values, default=0.0), 3),
    }


def load_queries(path: Path | None, count: int) -> list[str]:
    if path is None:
        return [DEFAULT_QUERIES[i % len(DEFAULT_QUERIES)] for i in range(count)]
    queries: list[str] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("query"):
                    queries.append(row["query"])
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                queries.append(str(row.get("query") or row.get("text") or line))
            except json.JSONDecodeError:
                queries.append(line.strip())
    return queries[:count] if count else queries


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the text RAG path")
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/benchmark.json"))
    args = parser.parse_args()

    pipeline = create_orchestrator()
    queries = load_queries(args.queries, args.count)
    for query in queries[: args.warmup]:
        pipeline.run(PipelineRequest(query_text=query))
    timings: list[float] = []
    stage_timings: dict[str, list[float]] = {"stt": [], "retrieval": [], "generation": [], "answer_guardrails": []}
    statuses: dict[str, int] = {}
    rows: list[dict] = []
    for query in queries:
        started = time.perf_counter()
        response = pipeline.run(PipelineRequest(query_text=query))
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)
        for stage_name, stage_value in response.stage_latency_ms.items():
            if stage_name in stage_timings:
                stage_timings[stage_name].append(stage_value)
        statuses[response.status] = statuses.get(response.status, 0) + 1
        rows.append({
            "query": query,
            "status": response.status,
            "total_ms": elapsed,
            "stt_ms": response.stage_latency_ms.get("stt", 0.0),
            "retrieval_ms": response.stage_latency_ms.get("retrieval", 0.0),
            "generation_ms": response.stage_latency_ms.get("generation", 0.0),
            "answer_guardrails_ms": response.stage_latency_ms.get("answer_guardrails", 0.0),
            "refusal_reason": response.refusal_reason,
            "attempts": response.attempts,
        })
    report = {
        "note": "Measured locally after warm-up; text path excludes external STT network latency.",
        "query_count": len(queries),
        "unique_query_count": len(set(queries)),
        "warmup_count": min(args.warmup, len(queries)),
        "statuses": statuses,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "generation_provider": os.getenv("GENERATION_PROVIDER", "auto"),
            "generation_model": os.getenv("OLLAMA_MODEL") or os.getenv("GENERATION_MODEL") or os.getenv("OPENCODE_GO_MODEL") or "local",
            "vector_backend": os.getenv("VECTOR_BACKEND", "local"),
            "embedding_backend": os.getenv("EMBEDDING_BACKEND", "hash"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
        },
        "total_latency_ms": latency_summary(timings),
        "stage_latency_ms": {stage: latency_summary(values) for stage, values in stage_timings.items() if values},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("query_count", "unique_query_count", "statuses", "environment", "total_latency_ms", "stage_latency_ms")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
