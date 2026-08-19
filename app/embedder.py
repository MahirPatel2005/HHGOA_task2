import os
import numpy as np
from app.embedding import HashingEmbedder, SentenceTransformerEmbedder

_embedder = None

def get_model():
    global _embedder
    if _embedder is not None:
        return _embedder
    
    backend = os.getenv("EMBEDDING_BACKEND", "hash").lower()
    if backend == "sentence-transformers" or backend == "sentence_transformers":
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        _embedder = SentenceTransformerEmbedder(model_name)
    else:
        _embedder = HashingEmbedder(dim=384)
    return _embedder

def embed(texts: list[str]) -> np.ndarray:
    model = get_model()
    if hasattr(model, "embed_many"):
        res = model.embed_many(texts)
    else:
        res = [model.embed(t) for t in texts]
    return np.array(res, dtype=np.float32)

def embed_one(text: str) -> np.ndarray:
    model = get_model()
    res = model.embed(text)
    return np.array(res, dtype=np.float32)
