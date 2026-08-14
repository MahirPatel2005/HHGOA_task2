from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .embedding import Embedder, cosine_similarity

_WORD_RE = re.compile(r"\S+")
_SENTENCE_RE = re.compile(r"(?<=[.!?।！？])\s+|\n+")


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    parent_id: str
    text: str
    strategy: str
    metadata: dict[str, Any]
    start: int
    end: int


def _stable_id(document_id: str, strategy: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document_id}:{strategy}:{ordinal}:{digest}"


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()]


class ParagraphChunker:
    strategy = "paragraph"

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        cursor = 0
        for ordinal, paragraph in enumerate(_paragraphs(document.text)):
            start = document.text.find(paragraph, cursor)
            start = max(start, cursor)
            end = start + len(paragraph)
            chunks.append(
                Chunk(
                    _stable_id(document.id, self.strategy, ordinal, paragraph),
                    document.id,
                    paragraph,
                    self.strategy,
                    dict(document.metadata),
                    start,
                    end,
                )
            )
            cursor = end
        return chunks


class SentenceWindowChunker:
    strategy = "sentence_window"

    def __init__(self, window_size: int = 3, overlap: int = 1):
        self.window_size = max(1, window_size)
        self.overlap = max(0, min(overlap, self.window_size - 1))

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = _sentences(document.text)
        if not sentences:
            return []
        stride = max(1, self.window_size - self.overlap)
        chunks: list[Chunk] = []
        for ordinal, start in enumerate(range(0, len(sentences), stride)):
            window = sentences[start : start + self.window_size]
            if not window:
                break
            text = " ".join(window)
            chunks.append(
                Chunk(
                    _stable_id(document.id, self.strategy, ordinal, text),
                    document.id,
                    text,
                    self.strategy,
                    dict(document.metadata),
                    start,
                    start + len(window),
                )
            )
            if start + self.window_size >= len(sentences):
                break
        return chunks


class TokenChunker:
    strategy = "token"

    def __init__(self, token_size: int = 180, overlap: int = 36):
        self.token_size = max(1, token_size)
        self.overlap = max(0, min(overlap, self.token_size - 1))

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = _WORD_RE.findall(document.text)
        if not tokens:
            return []
        stride = max(1, self.token_size - self.overlap)
        chunks: list[Chunk] = []
        for ordinal, start in enumerate(range(0, len(tokens), stride)):
            end = min(len(tokens), start + self.token_size)
            text = " ".join(tokens[start:end])
            chunks.append(
                Chunk(
                    _stable_id(document.id, self.strategy, ordinal, text),
                    document.id,
                    text,
                    self.strategy,
                    dict(document.metadata),
                    start,
                    end,
                )
            )
            if end == len(tokens):
                break
        return chunks


class SemanticChunker:
    strategy = "semantic"

    def __init__(self, embedder: Embedder, distance_threshold: float = 0.38, min_sentences: int = 1):
        self.embedder = embedder
        self.distance_threshold = distance_threshold
        self.min_sentences = max(1, min_sentences)

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = _sentences(document.text)
        if not sentences:
            return []
        groups: list[list[str]] = [[]]
        vectors = self.embedder.embed_many(sentences) if hasattr(self.embedder, "embed_many") else [self.embedder.embed(sentence) for sentence in sentences]
        previous_vector: list[float] | None = None
        for sentence, vector in zip(sentences, vectors):
            if previous_vector is not None:
                distance = 1.0 - cosine_similarity(previous_vector, vector)
                if distance >= self.distance_threshold and len(groups[-1]) >= self.min_sentences:
                    groups.append([])
            groups[-1].append(sentence)
            previous_vector = vector
        chunks: list[Chunk] = []
        sentence_cursor = 0
        for ordinal, group in enumerate(groups):
            text = " ".join(group)
            chunks.append(
                Chunk(
                    _stable_id(document.id, self.strategy, ordinal, text),
                    document.id,
                    text,
                    self.strategy,
                    dict(document.metadata),
                    sentence_cursor,
                    sentence_cursor + len(group),
                )
            )
            sentence_cursor += len(group)
        return chunks


class MetadataAwareChunker:
    strategy = "metadata"

    def chunk(self, document: Document) -> list[Chunk]:
        metadata = dict(document.metadata)
        if document.title:
            metadata.setdefault("title", document.title)
        labels = [f"{key}: {value}" for key, value in sorted(metadata.items())]
        prefix = f"Title: {document.title}\n" if document.title else ""
        prefix += ("Metadata: " + " | ".join(labels) + "\n") if labels else ""
        prefix += "Content: "
        chunks: list[Chunk] = []
        for ordinal, paragraph in enumerate(_paragraphs(document.text)):
            text = prefix + paragraph
            chunks.append(
                Chunk(
                    _stable_id(document.id, self.strategy, ordinal, text),
                    document.id,
                    text,
                    self.strategy,
                    metadata,
                    ordinal,
                    ordinal + 1,
                )
            )
        return chunks


class MultiStrategyChunker:
    """Create complementary retrieval views over the same parent document."""

    def __init__(
        self,
        embedder: Embedder,
        token_size: int = 180,
        token_overlap: int = 36,
        semantic_threshold: float = 0.38,
    ):
        self.chunkers = [
            ParagraphChunker(),
            SentenceWindowChunker(window_size=3, overlap=1),
            TokenChunker(token_size=token_size, overlap=token_overlap),
            SemanticChunker(embedder, distance_threshold=semantic_threshold),
            MetadataAwareChunker(),
        ]

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen: set[str] = set()
        for chunker in self.chunkers:
            for chunk in chunker.chunk(document):
                if chunk.id not in seen and chunk.text.strip():
                    chunks.append(chunk)
                    seen.add(chunk.id)
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk(document))
        return chunks
