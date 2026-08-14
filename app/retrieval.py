from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .chunking import Chunk
from .embedding import Embedder
from .vector_store import LocalVectorStore, QdrantVectorStore

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "be", "by", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "what", "where", "which",
    "who", "why", "with", "when", "does", "do", "i", "you", "about",
}


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_score: float
    lexical_score: float
    lexical_overlap: float
    retrieval_sources: list[str] = field(default_factory=list)


class SimpleBM25:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.documents = [tokens(chunk.text) for chunk in chunks]
        self.doc_count = len(self.documents)
        self.avgdl = sum(len(doc) for doc in self.documents) / max(1, self.doc_count)
        self.doc_freq: dict[str, int] = {}
        for document in self.documents:
            for term in set(document):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def score(self, query: str) -> dict[str, float]:
        query_terms = tokens(query)
        if not query_terms:
            return {chunk.id: 0.0 for chunk in self.chunks}
        k1, b = 1.5, 0.75
        scores: dict[str, float] = {}
        for chunk, document in zip(self.chunks, self.documents):
            counts: dict[str, int] = {}
            for term in document:
                counts[term] = counts.get(term, 0) + 1
            total = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1 - b + b * len(document) / max(1.0, self.avgdl))
                total += idf * (frequency * (k1 + 1)) / denominator
            scores[chunk.id] = total
        return scores


class HybridRetriever:
    """Dense + BM25 RRF retrieval with lightweight metadata-aware reranking."""

    def __init__(self, store: LocalVectorStore | QdrantVectorStore, embedder: Embedder, rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._refresh_lexical_index()

    def _refresh_lexical_index(self) -> None:
        if hasattr(self.store, "records"):
            chunks = [chunk for _, chunk in self.store.records.values()]
        else:
            chunks = []
        self.chunks = chunks
        self.bm25 = SimpleBM25(chunks)

    def search(self, query: str, top_k: int = 5, candidate_k: int = 30) -> list[RetrievalResult]:
        if not query.strip():
            return []
        self._refresh_lexical_index()
        dense_hits = self.store.search(self.embedder.embed(query), limit=candidate_k)
        dense_rank = {chunk.id: (rank, score) for rank, (chunk, score) in enumerate(dense_hits, 1)}
        lexical_scores = self.bm25.score(query)
        lexical_ranked = sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True)
        lexical_rank = {chunk_id: (rank, score) for rank, (chunk_id, score) in enumerate(lexical_ranked, 1)}
        by_id = {chunk.id: chunk for chunk, _ in dense_hits}
        by_id.update({chunk.id: chunk for chunk in self.chunks})
        candidate_ids = set(dense_rank) | {chunk_id for chunk_id, score in lexical_ranked[:candidate_k] if score > 0}
        query_tokens = set(tokens(query))
        results: list[RetrievalResult] = []
        max_lexical = max(lexical_scores.values(), default=0.0)
        for chunk_id in candidate_ids:
            chunk = by_id[chunk_id]
            dense_rank_number, dense_score = dense_rank.get(chunk_id, (candidate_k + 1, 0.0))
            lexical_rank_number, lexical_score = lexical_rank.get(chunk_id, (candidate_k + 1, 0.0))
            fused = 0.0
            sources: list[str] = []
            if chunk_id in dense_rank:
                fused += 1.0 / (self.rrf_k + dense_rank_number)
                sources.append("dense")
            if lexical_score > 0:
                fused += 1.0 / (self.rrf_k + lexical_rank_number)
                sources.append("bm25")
            chunk_tokens = set(tokens(chunk.text))
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            normalized_lexical = lexical_score / max_lexical if max_lexical else 0.0
            final_score = fused + 0.10 * overlap + 0.05 * normalized_lexical + 0.02 * max(0.0, dense_score)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=final_score,
                    dense_score=dense_score,
                    lexical_score=lexical_score,
                    lexical_overlap=overlap,
                    retrieval_sources=sources,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: max(0, top_k)]
