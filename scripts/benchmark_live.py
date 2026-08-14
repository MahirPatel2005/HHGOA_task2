from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_queries(path: Path, count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        query = str(row.get("query") or "").strip()
        if query and query not in seen:
            seen.add(query)
            unique.append(row)
    return unique[:count] if count else unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the deployed Vaani text API")
    parser.add_argument("--endpoint", default="https://vaani-voice-rag.onrender.com")
    parser.add_argument("--queries", type=Path, default=Path("benchmarks/msmarco_xi_eval.jsonl"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/live-final-results.json"))
    parser.add_argument("--max-rate-limit-waits", type=int, default=12)
    args = parser.parse_args()

    queries = load_queries(args.queries, args.count)
    rows: list[dict[str, Any]] = []
    rate_limit_waits = 0
    with httpx.Client(timeout=120.0) as client:
        index = 0
        while index < len(queries):
            item = queries[index]
            query = str(item["query"])
            started = time.perf_counter()
            try:
                response = client.post(
                    f"{args.endpoint.rstrip('/')}/api/query",
                    json={"query": query, "language_code": "unknown"},
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
            except Exception as exc:
                rows.append({"query": query, "status": "transport_error", "error": type(exc).__name__, "elapsed_ms": (time.perf_counter() - started) * 1000})
                index += 1
                continue

            if response.status_code == 429:
                rate_limit_waits += 1
                if rate_limit_waits > args.max_rate_limit_waits:
                    rows.append({"query": query, "status": "rate_limited", "http_status": 429, "elapsed_ms": elapsed_ms})
                    index += 1
                    continue
                wait_s = min(65, max(1, int(response.headers.get("Retry-After", "60"))))
                print(f"rate limited; waiting {wait_s}s before retry ({rate_limit_waits}/{args.max_rate_limit_waits})", flush=True)
                time.sleep(wait_s)
                continue

            try:
                body = response.json()
            except ValueError:
                body = {}
            rows.append({
                "query": query,
                "dataset_config": item.get("dataset_config"),
                "query_id": item.get("query_id"),
                "http_status": response.status_code,
                "status": body.get("status", "http_error"),
                "refusal_reason": body.get("refusal_reason"),
                "elapsed_ms": round(elapsed_ms, 3),
                "server_total_ms": body.get("total_latency_ms"),
                "stage_latency_ms": body.get("stage_latency_ms", {}),
                "attempts": body.get("attempts", {}),
            })
            index += 1
            if index % 10 == 0 or index == len(queries):
                print(f"completed {index}/{len(queries)}", flush=True)

    elapsed = [float(row["elapsed_ms"]) for row in rows if row.get("http_status") == 200]
    server = [float(row["server_total_ms"]) for row in rows if isinstance(row.get("server_total_ms"), (int, float))]
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    report = {
        "endpoint": args.endpoint,
        "query_count_requested": len(queries),
        "query_count_completed": len(rows),
        "unique_query_count": len({row["query"] for row in rows}),
        "rate_limit_waits": rate_limit_waits,
        "statuses": statuses,
        "client_elapsed_ms": {"p50": round(percentile(elapsed, .5), 3), "p70": round(percentile(elapsed, .7), 3), "p100": round(max(elapsed, default=0), 3)},
        "server_total_latency_ms": {"p50": round(percentile(server, .5), 3), "p70": round(percentile(server, .7), 3), "p100": round(max(server, default=0), 3)},
        "rows": rows,
        "note": "Live deployed text API; external Sarvam voice latency is not included.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("query_count_requested", "query_count_completed", "unique_query_count", "rate_limit_waits", "statuses", "client_elapsed_ms", "server_total_latency_ms")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
