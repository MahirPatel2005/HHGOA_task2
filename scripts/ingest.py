from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.chunking import Document, MultiStrategyChunker
from app.embedding import HashingEmbedder, SentenceTransformerEmbedder
from app.vector_store import LocalVectorStore, QdrantVectorStore

MSMARCO_XI_CONFIGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"]
MSMARCO_XI_FILE_PREFIXES = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal", "mr": "mar",
    "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def msmarco_xi_parquet_url(dataset: str, config: str, split: str) -> str:
    if config not in MSMARCO_XI_FILE_PREFIXES:
        raise ValueError(f"Unsupported MSMARCO-XI config: {config}")
    if split not in {"train", "validation"}:
        raise ValueError("MSMARCO-XI split must be train or validation")
    suffix = "train" if split == "train" else "val"
    prefix = MSMARCO_XI_FILE_PREFIXES[config]
    return f"https://huggingface.co/datasets/{dataset}/resolve/main/{split}/{prefix}{suffix}.parquet"


def normalize_splits(split: str | None, splits: list[str] | None) -> list[str]:
    selected = list(splits or ([split] if split else ["train"]))
    normalized: list[str] = []
    for value in selected:
        if value not in {"train", "validation"}:
            raise ValueError("MSMARCO-XI split must be train or validation")
        if value not in normalized:
            normalized.append(value)
    return normalized


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def documents_from_row(row: dict[str, Any], row_number: int) -> list[Document]:
    dataset_config = _string(row.get("_dataset_config"))
    dataset_split = _string(row.get("_dataset_split"))
    raw_id = str(row.get("id") or row.get("_id") or row.get("doc_id") or row.get("query_id") or f"row-{row_number}")
    base_id = f"{dataset_config}:{raw_id}" if dataset_config else raw_id
    title = _string(row.get("title") or row.get("query"))
    metadata = dict(row.get("metadata") or {}) if isinstance(row.get("metadata"), dict) else {}
    if dataset_config:
        metadata.setdefault("dataset_config", dataset_config)
    if dataset_split:
        metadata.setdefault("dataset_split", dataset_split)
    for key in ("query", "query_id", "query_type", "language", "lang", "split", "url", "source", "source_lang", "target_lang"):
        if row.get(key) is not None and not isinstance(row[key], (dict, list)):
            metadata.setdefault(key, row[key])

    candidates: list[tuple[str, str]] = []
    preferred_keys = ("text", "passage", "content", "document", "context", "body", "answer")
    for key in preferred_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append((key, value.strip()))
    for key in ("passages", "positive_passages", "documents"):
        value = row.get(key)
        if isinstance(value, dict):
            # MSMARCO-XI/IIX-style schema: passages are grouped by language
            # and the translated list is the primary evidence view.
            ordered_fields = [
                field for field in value
                if "translated" in field.lower() or "passage" in field.lower() or "text" in field.lower()
            ]
            ordered_fields += [field for field in value if field not in ordered_fields]
            ordered_fields.sort(key=lambda field: (0 if "translated" in field.lower() else 1, field))
            for field in ordered_fields:
                items = value.get(field)
                if not isinstance(items, list):
                    continue
                for index, item in enumerate(items):
                    text = _string(item)
                    if text:
                        candidates.append((f"{key}-{field}-{index}", text))
                if "translated" in field.lower():
                    break
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, str) and item.strip():
                    candidates.append((f"{key}-{index}", item.strip()))
                elif isinstance(item, dict):
                    text = next((_string(item.get(field)) for field in preferred_keys if item.get(field)), "")
                    if text:
                        candidates.append((f"{key}-{index}", text))
    if not candidates:
        # Last-resort schema fallback: preserve all scalar text values, while
        # excluding query-like metadata that is not evidence itself.
        scalar_values = [_string(value) for key, value in row.items() if key not in {"id", "_id", "query"} and _string(value)]
        if scalar_values:
            candidates.append(("row", "\n".join(scalar_values)))

    documents: list[Document] = []
    seen: set[str] = set()
    for suffix, text in candidates:
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        documents.append(Document(f"{base_id}:{suffix}", normalized, title, dict(metadata)))
    return documents


def load_rows_from_file(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        yield payload


def load_msmarco_xi_sample_rows(dataset: str, config: str, split: str, limit: int) -> Iterable[dict[str, Any]]:
    """Read a small MSMARCO-XI sample directly from remote Parquet ranges using DuckDB.

    Bypasses PyArrow's ArrowNotImplementedError for nested conversions over remote ranges.
    """
    if limit <= 0:
        return
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Install duckdb: pip install duckdb") from exc
    parquet_url = msmarco_xi_parquet_url(dataset, config, split)
    columns = ["source_lang", "target_lang", "Answer", "query_id", "query_type", "passages", "Eng_Query", "Eng_Answer", "query"]
    print(f"Connecting to DuckDB to stream sample rows from {parquet_url.split('/')[-1]}...")
    import time
    for attempt in range(1, 6):
        try:
            con = duckdb.connect()
            con.execute("SET http_timeout = 5000;")
            con.execute("SET http_retries = 5;")
            schema = con.sql(f"SELECT * FROM read_parquet('{parquet_url}') LIMIT 0")
            available = set(schema.columns)
            selected = [col for col in columns if col in available]
            select_clause = ", ".join(selected)
            res = con.sql(f"SELECT {select_clause} FROM read_parquet('{parquet_url}') LIMIT {limit}")
            cols = res.columns
            rows_cursor = res.fetchall()
            for r in rows_cursor:
                row = dict(zip(cols, r))
                row["_dataset_config"] = config
                row["_dataset_split"] = split
                yield row
            break
        except Exception as ex:
            ex_str = str(ex).lower()
            if "404" in ex_str or "not found" in ex_str:
                raise ex
            if attempt == 5:
                raise RuntimeError(f"Error querying remote Parquet via DuckDB: {ex}")
            print(f"Warning: connection attempt {attempt} failed ({ex}). Retrying in 2 seconds...", file=sys.stderr)
            time.sleep(2)


def load_rows_from_huggingface(dataset: str, config: str | None, split: str, limit: int | None) -> Iterable[dict[str, Any]]:
    if dataset.lower().endswith("msmarco-xi") and config and limit is not None and limit <= 1000:
        yield from load_msmarco_xi_sample_rows(dataset, config, split, limit)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install production extras first: pip install -e '.[production]'") from exc

    if dataset.lower().endswith("msmarco-xi"):
        if not config:
            raise ValueError("MSMARCO-XI requires --config or --all-configs")
        parquet_url = msmarco_xi_parquet_url(dataset, config, split)
        stream = load_dataset(
            "parquet",
            data_files={split: parquet_url},
            split=split,
            streaming=True,
            token=os.getenv("HF_TOKEN") or None,
        )
    else:
        kwargs: dict[str, Any] = {"split": split, "streaming": True}
        if config:
            kwargs["name"] = config
        if os.getenv("HF_TOKEN"):
            kwargs["token"] = os.getenv("HF_TOKEN")
        stream = load_dataset(dataset, **kwargs)
    for index, row in enumerate(stream):
        if limit is not None and index >= limit:
            break
        item = dict(row)
        if config:
            item["_dataset_config"] = config
        item["_dataset_split"] = split
        yield item


def make_embedder(backend: str, model: str):
    if backend.lower() in {"bge", "bge-m3", "sentence-transformers"}:
        return SentenceTransformerEmbedder(model)
    return HashingEmbedder(dim=384)


def embed_chunks(embedder, chunks: list[Any]) -> list[list[float]]:
    if hasattr(embedder, "embed_many"):
        return embedder.embed_many([chunk.text for chunk in chunks])
    return [embedder.embed(chunk.text) for chunk in chunks]


def upsert_document_batch(store, chunker: MultiStrategyChunker, embedder, documents: list[Document]) -> tuple[int, int]:
    if not documents:
        return 0, 0
    chunks = chunker.chunk_many(documents)
    vectors = embed_chunks(embedder, chunks)
    store.upsert((chunk.id, vector, chunk) for chunk, vector in zip(chunks, vectors))
    return len(documents), len(chunks)


def fetch_config_worker(args_tuple):
    dataset, config, split, limit = args_tuple
    try:
        import sys
        rows = list(load_msmarco_xi_sample_rows(dataset, config, split, limit))
        return config, rows
    except Exception as ex:
        import sys
        print(f"Warning: failed to fetch configuration '{config}' ({ex}). Skipping this configuration.", file=sys.stderr)
        return config, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the multi-strategy voice-RAG index")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSONL, JSON or CSV file")
    source.add_argument("--dataset", help="Hugging Face dataset id, e.g. ai4bharat/MSMARCO-XI")
    parser.add_argument("--config", default=None, help="Optional Hugging Face dataset configuration")
    parser.add_argument("--all-configs", action="store_true", help="Index all 14 MSMARCO-XI language configurations")
    parser.add_argument("--split", default=None, help="One Hugging Face split (default: train)")
    parser.add_argument("--splits", nargs="+", default=None, help="Multiple Hugging Face splits, e.g. --splits train validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", choices=["hash", "bge-m3", "sentence-transformers"], default="hash")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--vector-backend", choices=["local", "qdrant"], default="local")
    parser.add_argument("--out-index", type=Path, default=Path("storage/msmarco_iix_index.json"))
    parser.add_argument("--qdrant-path", default="storage/qdrant")
    parser.add_argument("--qdrant-url", default=None, help="Optional Qdrant Cloud URL; defaults to QDRANT_URL")
    parser.add_argument("--qdrant-api-key", default=None, help="Optional Qdrant Cloud key; defaults to QDRANT_API_KEY")
    parser.add_argument("--qdrant-collection", default=None, help="Collection name; defaults to QDRANT_COLLECTION or voice_rag")
    parser.add_argument("--batch-size", type=int, default=32, help="Documents per embedding/index batch")
    parser.add_argument("--eval-queries", type=Path, default=None, help="Write representative query JSONL here")
    parser.add_argument("--eval-count", type=int, default=500, help="Maximum queries to write")
    args = parser.parse_args()
    try:
        selected_splits = normalize_splits(args.split, args.splits)
    except ValueError as exc:
        parser.error(str(exc))

    if args.all_configs and args.input:
        parser.error("--all-configs requires --dataset, not --input")
    if args.all_configs and args.config:
        parser.error("Use either --config or --all-configs, not both")
    if args.input:
        rows = load_rows_from_file(args.input)
        configs_used: list[str] = []
    elif args.all_configs:
        if "MSMARCO-XI" not in args.dataset.upper():
            parser.error("--all-configs currently targets ai4bharat/MSMARCO-XI")
        configs_used = list(MSMARCO_XI_CONFIGS)

        import multiprocessing
        pool_args = []
        for split in selected_splits:
            for config in MSMARCO_XI_CONFIGS:
                pool_args.append((args.dataset, config, split, args.limit))

        print(f"Spawning parallel processes to download {len(pool_args)} configurations concurrently...")
        # Enable multiprocessing spawn method to make it clean on macOS
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        with multiprocessing.Pool(processes=len(pool_args)) as pool:
            results = pool.map(fetch_config_worker, pool_args)

        combined_rows = []
        for config, rows_list in results:
            combined_rows.extend(rows_list)
        rows = combined_rows
    else:
        configs_used = [args.config] if args.config else []

        def selected_rows():
            for split in selected_splits:
                yield from load_rows_from_huggingface(args.dataset, args.config, split, args.limit)

        rows = selected_rows()

    embedder = make_embedder(args.backend, args.model)
    chunker = MultiStrategyChunker(embedder=embedder)
    if args.vector_backend == "qdrant":
        store = QdrantVectorStore(
            embedder.dim,
            path=args.qdrant_path,
            collection=args.qdrant_collection or os.getenv("QDRANT_COLLECTION", "voice_rag"),
            url=args.qdrant_url or os.getenv("QDRANT_URL") or None,
            api_key=args.qdrant_api_key or os.getenv("QDRANT_API_KEY") or None,
            batch_size=int(os.getenv("QDRANT_BATCH_SIZE", "64")),
            timeout_s=int(os.getenv("QDRANT_TIMEOUT_S", "120")),
        )
    else:
        store = LocalVectorStore(embedder.dim)

    document_batch: list[Document] = []
    eval_rows: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    rows_seen = documents_seen = chunks_seen = 0
    batch_size = max(1, args.batch_size)

    def flush_batch() -> None:
        nonlocal document_batch, documents_seen, chunks_seen
        if not document_batch:
            return
        batch_documents, batch_chunks = upsert_document_batch(store, chunker, embedder, document_batch)
        documents_seen += batch_documents
        chunks_seen += batch_chunks
        document_batch = []
        print(json.dumps({"rows": rows_seen, "documents": documents_seen, "chunks": chunks_seen}), flush=True)

    for row_number, row in enumerate(rows, 1):
        rows_seen = row_number
        query = _string(row.get("query") or row.get("Eng_Query"))
        if args.eval_queries and query and query not in seen_queries and len(eval_rows) < max(0, args.eval_count):
            seen_queries.add(query)
            eval_rows.append({
                "query": query,
                "dataset_config": row.get("_dataset_config"),
                "dataset_split": row.get("_dataset_split"),
                "query_id": row.get("query_id"),
            })
        document_batch.extend(documents_from_row(row, row_number))
        if args.limit is not None and args.input and row_number >= args.limit:
            break
        if len(document_batch) >= batch_size:
            flush_batch()
    flush_batch()

    if documents_seen == 0:
        print("No evidence documents were found in the source.", file=sys.stderr)
        return 2
    if isinstance(store, LocalVectorStore):
        store.save(args.out_index)
    if args.eval_queries:
        args.eval_queries.parent.mkdir(parents=True, exist_ok=True)
        args.eval_queries.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_rows),
            encoding="utf-8",
        )
    manifest = args.out_index.with_suffix(".manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "dataset": args.dataset or str(args.input),
                "rows": rows_seen,
                "documents": documents_seen,
                "chunks": chunks_seen,
                "strategies": ["paragraph", "sentence_window", "token", "semantic", "metadata"],
                "embedding_backend": args.backend,
                "embedding_model": args.model,
                "embedding_dim": embedder.dim,
                "vector_backend": args.vector_backend,
                "qdrant_collection": args.qdrant_collection or os.getenv("QDRANT_COLLECTION", "voice_rag"),
                "configs": configs_used,
                "splits": selected_splits,
                "eval_queries": str(args.eval_queries) if args.eval_queries else None,
                "eval_query_count": len(eval_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": rows_seen, "documents": documents_seen, "chunks": chunks_seen, "manifest": str(manifest), "eval_queries": len(eval_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
