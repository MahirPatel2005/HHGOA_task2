from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .chunking import Chunk
from .embedding import cosine_similarity


class LocalVectorStore:
    """Persistent local vector index used for the reproducible demo.

    It exposes the same small interface as the optional Qdrant adapter, so the
    harness does not depend on a particular storage engine.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.records: dict[str, tuple[list[float], Chunk]] = {}
        self._matrix = None
        self._chunks = []
        self._norms = None

    def _build_cache(self) -> None:
        if not self.records:
            self._matrix = None
            self._chunks = []
            self._norms = None
            return
        import numpy as np
        vectors = []
        chunks = []
        for vector, chunk in self.records.values():
            vectors.append(vector)
            chunks.append(chunk)
        self._matrix = np.array(vectors, dtype=np.float32)
        self._chunks = chunks
        norms = np.linalg.norm(self._matrix, axis=1)
        norms[norms == 0.0] = 1.0
        self._norms = norms

    def upsert(self, records: Iterable[tuple[str, list[float], Chunk]]) -> None:
        for record_id, vector, chunk in records:
            if len(vector) != self.dim:
                raise ValueError(f"Expected {self.dim}-dimensional vector, got {len(vector)}")
            self.records[record_id] = (list(vector), chunk)
        self._build_cache()

    def search(self, vector: list[float], limit: int = 10) -> list[tuple[Chunk, float]]:
        if self._matrix is None or len(self._chunks) == 0:
            self._build_cache()
            if self._matrix is None or len(self._chunks) == 0:
                return []
        import numpy as np
        query = np.array(vector, dtype=np.float32)
        q_norm = np.linalg.norm(query)
        if q_norm == 0.0:
            q_norm = 1.0
        dots = self._matrix.dot(query)
        sims = dots / (self._norms * q_norm)
        top_k = min(len(self._chunks), max(0, limit))
        if top_k == 0:
            return []
        indices = np.argsort(sims)[::-1][:top_k]
        return [(self._chunks[idx], float(sims[idx])) for idx in indices]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "records": [
                {
                    "id": record_id,
                    "vector": vector,
                    "chunk": asdict(chunk),
                }
                for record_id, (vector, chunk) in self.records.items()
            ],
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LocalVectorStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(dim=int(payload["dim"]))
        for record in payload.get("records", []):
            chunk = Chunk(**record["chunk"])
            store.records[record["id"]] = (record["vector"], chunk)
        return store


class QdrantVectorStore:
    """Optional Qdrant-backed store for the submitted deployment.

    Qdrant is kept optional so a reviewer can run the demo and tests without a
    separate server. Set VECTOR_BACKEND=qdrant and install the production
    extras to use this adapter.
    """

    def __init__(
        self,
        dim: int,
        path: str = "storage/qdrant",
        collection: str = "voice_rag",
        url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout_s: int = 120,
    ):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install qdrant-client to use the Qdrant backend") from exc
        self._models = models
        self.dim = dim
        self.collection = collection
        self.batch_size = max(1, int(batch_size))
        self.timeout_s = max(1, int(timeout_s))
        self.client = (
            QdrantClient(url=url, api_key=api_key, timeout=self.timeout_s)
            if url
            else QdrantClient(path=path)
        )
        self.records: dict[str, tuple[list[float], Chunk]] = {}
        existing = {item.name for item in self.client.get_collections().collections}
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
        # Remote retrieval queries Qdrant directly. Do not scroll every payload
        # during API startup: large collections make the first request exceed
        # hosted proxy timeouts and provide no benefit to `search()`.

    def upsert(self, records: Iterable[tuple[str, list[float], Chunk]]) -> None:
        points = []
        for record_id, vector, chunk in records:
            self.records[record_id] = (list(vector), chunk)
            points.append(
                self._models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, record_id)),
                    vector=vector,
                    payload={"record_id": record_id, "chunk": asdict(chunk)},
                )
            )
        if points:
            for start in range(0, len(points), self.batch_size):
                self.client.upsert(
                    collection_name=self.collection,
                    points=points[start : start + self.batch_size],
                    wait=True,
                )

    def search(self, vector: list[float], limit: int = 10) -> list[tuple[Chunk, float]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            (Chunk(**point.payload["chunk"]), float(point.score))
            for point in response.points
        ]

    def close(self) -> None:
        self.client.close()
