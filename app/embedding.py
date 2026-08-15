from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Protocol

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Small deterministic embedder used for the local demo and tests.

    Production ingestion can switch to BGE-M3 through the optional
    SentenceTransformerEmbedder without changing the retrieval interfaces.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def _bucket(value: str, dim: int) -> tuple[int, float]:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        index = number % dim
        sign = 1.0 if (number >> 8) & 1 else -1.0
        return index, sign

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        for token in tokens:
            index, sign = self._bucket(f"tok:{token}", self.dim)
            vector[index] += sign
            if len(token) > 2:
                for n in (2, 3):
                    for start in range(max(0, len(token) - n + 1)):
                        index, sign = self._bucket(f"char{n}:{token[start:start+n]}", self.dim)
                        vector[index] += 0.15 * sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    """Optional BGE-M3 adapter, loaded only when explicitly selected."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install the production extras to use sentence-transformers"
            ) from exc
        device = "cpu"
        print(f"Loading SentenceTransformer '{model_name}' on device '{device}'...")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
