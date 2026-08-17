"""Measure end-to-end retrieval latency (embed + FAISS search) against the
50ms budget defined in app/config.py.

Usage:
    python -m app.benchmark [n_queries]
"""
import statistics
import sys

import time
from app.main import make_embedder, build_store, sample_documents
from app.retrieval import HybridRetriever

LATENCY_BUDGET_MS = 50.0

class SearchResponse:
    def __init__(self, embed_ms: float, search_ms: float, total_ms: float):
        self.embed_ms = embed_ms
        self.search_ms = search_ms
        self.total_ms = total_ms

_retriever = None

def warmup():
    global _retriever
    if _retriever is None:
        documents = sample_documents()
        embedder = make_embedder()
        store = build_store(embedder, documents)
        _retriever = HybridRetriever(store=store, embedder=embedder)
    
    # Warmup query
    search("warmup")

def search(query: str, top_k: int = 5) -> SearchResponse:
    global _retriever
    if _retriever is None:
        warmup()
    
    # Measure total search time
    t_start = time.perf_counter()
    _retriever.search(query, top_k=top_k)
    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000

    # Measure embedding time separately
    t_embed_start = time.perf_counter()
    _retriever.embedder.embed(query)
    t_embed_end = time.perf_counter()
    embed_ms = (t_embed_end - t_embed_start) * 1000

    # Search time is approximately total_ms - embed_ms
    search_ms = max(0.0, total_ms - embed_ms)

    return SearchResponse(
        embed_ms=embed_ms,
        search_ms=search_ms,
        total_ms=total_ms
    )

QUERIES = [
    "What is FAISS used for?",
    "How does HNSW indexing work?",
    "What is retrieval augmented generation?",
    "Which embedding model is fast on CPU?",
    "How do you reduce RAG latency?",
    "What does efSearch control?",
    "Why normalize embeddings before indexing?",
    "What are the stages of a RAG pipeline?",
]


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("Warming up (model load + first inference)...")
    warmup()

    total_ms, embed_ms, search_ms = [], [], []
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        resp = search(query, top_k=5)
        total_ms.append(resp.total_ms)
        embed_ms.append(resp.embed_ms)
        search_ms.append(resp.search_ms)

    print(f"\nRan {n} queries\n")
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [("embed", embed_ms), ("search", search_ms), ("total", total_ms)]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nLatency budget: {LATENCY_BUDGET_MS}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= LATENCY_BUDGET_MS:
        print("PASS: within budget")
    else:
        print("FAIL: over budget -- see README 'Tuning latency' section")
        sys.exit(1)


if __name__ == "__main__":
    main()
