"""
Fetches a representative subset of ai4bharat/MSMARCO-XI over plain HTTPS
using the Hugging Face **Datasets Server REST API** (`/rows`), and writes it
to data/passages.jsonl (the indexed corpus) and data/queries.jsonl (used
later to build eval/test_queries.json).

WHY THIS APPROACH (zero local dataset footprint):
This does NOT use the `datasets` Python library at all -- it never calls
`load_dataset()`, so it never invokes the Hub's parquet/arrow download
machinery and can never end up pulling gigabytes to disk, regardless of how
big the underlying dataset is (ai4bharat/MSMARCO-XI is ~55.6GB across all
its per-language configs).

Instead it talks directly to Hugging Face's hosted "Datasets Server"
(https://datasets-server.huggingface.co), the same read-only JSON API that
powers the dataset preview table on a dataset's Hub page. Each request asks
for up to 100 rows at a time (`/rows?...&offset=...&length=...`) and returns
a small JSON payload -- typically a few KB to low hundreds of KB per page.
Nothing is cached or downloaded in bulk; the only bytes that hit disk are
the passages.jsonl / queries.jsonl files this script writes itself, sized
by --max-passages, not by the dataset's total size.

FALLBACK MODE (DuckDB Remote Query):
If the Hugging Face Datasets Server returns a 500 error on the `/rows` or `/filter`
endpoints (e.g. due to ArrowNotImplementedError/TooBigRowGroupsError on large datasets),
this script automatically falls back to streaming Parquet data directly from the Hugging
Face Hub using DuckDB. DuckDB queries the remote Parquet files using HTTP Range requests,
only transferring the specific requested column ranges (a few MBs) with zero local caching.

Run:
    python data/download_dataset.py --language hin_Deva --max-passages 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent
DATASET_ID = "ai4bharat/MSMARCO-XI"
API_BASE = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100  # max rows per /rows call the API allows
DEFAULT_MAX_PASSAGES = 8000
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


def _get(session, path: str, **params: Any) -> dict:
    """GET one Datasets Server endpoint with a couple of retries. Every call
    here fetches a small JSON page -- never a data file -- so there's no
    scenario where this pulls down anything close to the full dataset."""
    url = f"{API_BASE}/{path}"
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            # 422/404 usually mean a bad config/split name -- not worth
            # retrying, surface it immediately with the server's message.
            if resp.status_code in (404, 422):
                raise RuntimeError(
                    f"{path} rejected config/split (HTTP {resp.status_code}): {resp.text[:300]}"
                )
            last_err = RuntimeError(f"{path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        except Exception as e:  # network hiccup -- back off and retry
            last_err = e
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"Datasets Server request failed after {MAX_RETRIES} attempts: {last_err}")


def discover_splits(session) -> list[dict]:
    """List every (config, split) pair the Hub-hosted viewer knows about for
    this dataset."""
    info = _get(session, "splits", dataset=DATASET_ID)
    return info.get("splits", [])


def fetch_first_page_and_schema(session, config: str, split: str) -> tuple[list[dict], list[dict]]:
    """One /rows call: gives us both the column schema (features) and the
    first page of rows in a single round trip."""
    page = _get(session, "rows", dataset=DATASET_ID, config=config, split=split, offset=0, length=PAGE_SIZE)
    return page.get("features", []), page.get("rows", [])


def extract_passages_and_query(row: dict) -> list[tuple[str, str]]:
    """Pull (query_text, passage_text) pairs out of one dataset row."""
    query_text = None
    for key in ("query", "question", "Query"):
        if row.get(key):
            query_text = row[key]
            break
    if not query_text:
        return []

    passages: list[str] = []

    raw = row.get("passages")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                passages.append(item)
            elif isinstance(item, dict):
                text = item.get("passage_text") or item.get("text") or item.get("Passage")
                if text:
                    passages.append(text)
    elif isinstance(raw, dict):
        for key in ("passage_text", "text", "Passage"):
            if isinstance(raw.get(key), list):
                passages.extend([t for t in raw[key] if t])
                break

    if not passages:
        for key in ("passage", "context", "Passage", "answer", "Answer", "answers"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                passages.append(val)
                break
            if isinstance(val, list) and val:
                for v in val:
                    if isinstance(v, str) and v.strip():
                        passages.append(v)
                        break
                break

    return [(query_text, p) for p in passages]


def extract_passages_and_query_from_parquet_row(row: dict) -> list[tuple[str, str]]:
    """Pull (query_text, passage_text) pairs out of one DuckDB Parquet row.
    Handles struct/nested column shapes specifically for MSMARCO-XI."""
    query_text = row.get("query")
    if not query_text:
        return []

    passages: list[str] = []
    passages_struct = row.get("passages") or {}
    
    # Check translated passages first
    translated = passages_struct.get("Translated_passages") or []
    for p in translated:
        if isinstance(p, str) and p.strip():
            passages.append(p)
            
    # Fallback to English passages if none translated
    if not passages:
        english = passages_struct.get("English_passages") or []
        for p in english:
            if isinstance(p, str) and p.strip():
                passages.append(p)
                
    return [(query_text, p) for p in passages]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="hi", help="Language code filter value (e.g. hin_Deva, tam_Taml, hi)")
    parser.add_argument("--max-passages", type=int, default=DEFAULT_MAX_PASSAGES)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print every available (config, split) pair for this dataset and exit.",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Sample ~1000 rows from the dataset and print the unique languages present.",
    )
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        print("Install requirements first: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    session = requests.Session()

    if args.list_configs:
        splits = discover_splits(session)
        for s in splits:
            print(f"{s.get('config')}\t{s.get('split')}")
        return

    split = args.split

    # 1. Fetch preview of default config to retrieve schema/features
    print(f"Discovering dataset schema for {DATASET_ID} config='default' split='{split}' via Datasets Server...")
    use_duckdb = False
    features = []
    first_rows = []
    try:
        features, first_rows = fetch_first_page_and_schema(session, "default", split)
    except Exception as e:
        print(f"Datasets Server `/rows` endpoint failed ({e}). Falling back to direct Parquet query via DuckDB...", file=sys.stderr)
        use_duckdb = True

    # 2. Setup DuckDB connection and URLs if falling back
    urls = []
    con = None
    if use_duckdb:
        try:
            import duckdb
        except ImportError:
            print("Error: Hugging Face Datasets Server is down (HTTP 500) and 'duckdb' is not installed.", file=sys.stderr)
            print("Please install DuckDB to enable fallback Parquet streaming:", file=sys.stderr)
            print("    .venv/bin/pip install duckdb", file=sys.stderr)
            sys.exit(1)

        print("Fetching Parquet file list from Datasets Server `/parquet` endpoint...")
        try:
            parquet_info = _get(session, "parquet", dataset=DATASET_ID)
            urls = [
                f["url"] for f in parquet_info.get("parquet_files", [])
                if f.get("config") == "default" and f.get("split") == split
            ]
            if not urls:
                raise RuntimeError(f"No parquet files found for config='default' split='{split}'")
        except Exception as ex:
            print(f"Error listing Parquet files: {ex}", file=sys.stderr)
            sys.exit(1)

        print(f"Discovered {len(urls)} Parquet files. Connecting to DuckDB...")
        con = duckdb.connect()
        try:
            desc = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{urls[0]}') LIMIT 0").fetchall()
            feature_names = [col[0] for col in desc]
        except Exception as ex:
            print(f"Error connecting/querying remote Parquet files via DuckDB: {ex}", file=sys.stderr)
            sys.exit(1)
    else:
        feature_names = [f.get("name") for f in features]

    # 3. Detect language column
    lang_col = None
    for col in ("target_lang", "language", "lang"):
        if col in feature_names:
            lang_col = col
            break

    if not lang_col:
        print(f"Warning: could not detect language column. Defaulting to 'target_lang'.", file=sys.stderr)
        lang_col = "target_lang"
    else:
        print(f"Detected language field: '{lang_col}'")

    # 4. Handle --list-languages flag
    if args.list_languages:
        if use_duckdb:
            print(f"Sampling rows across Parquet files to detect unique languages...")
            try:
                # To discover all languages (since they are ordered sequentially in Parquet files),
                # we sample 100 rows from each Parquet file and union them.
                queries = [f"(SELECT {lang_col} FROM read_parquet('{url}') LIMIT 100)" for url in urls]
                sample_query = f"SELECT {lang_col}, count(*) FROM (" + " UNION ALL ".join(queries) + f") GROUP BY {lang_col} ORDER BY count(*) DESC"
                results = con.sql(sample_query).fetchall()
                print(f"\nValues seen in sampled rows:")
                for lang, count in results:
                    print(f"  {lang:<10} ({count} rows)")
            except Exception as ex:
                print(f"Error sampling languages via DuckDB: {ex}", file=sys.stderr)
            return
        else:
            print(f"Sampling ~1000 rows from config='default' split='{split}' to list unique languages...")
            langs_seen = {}
            scanned = 0
            for offset in range(0, 1000, PAGE_SIZE):
                try:
                    if offset == 0 and first_rows:
                        rows = first_rows
                    else:
                        page = _get(session, "rows", dataset=DATASET_ID, config="default", split=split, offset=offset, length=PAGE_SIZE)
                        rows = page.get("rows", [])
                    if not rows:
                        break
                    for entry in rows:
                        row = entry.get("row", {})
                        val = row.get(lang_col)
                        if val:
                            langs_seen[val] = langs_seen.get(val, 0) + 1
                            scanned += 1
                except Exception as e:
                    print(f"Error sampling rows at offset {offset}: {e}", file=sys.stderr)
                    break

            print(f"\nValues seen in {scanned} sampled rows:")
            for lang, count in sorted(langs_seen.items(), key=lambda x: x[1], reverse=True):
                print(f"  {lang:<10} ({count} rows)")
            return

    # 5. Filter and download data
    passages_path = DATA_DIR / "passages.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"

    seen_passages: set[str] = set()
    n_written = 0
    n_scanned = 0
    empty_extractions = 0

    if use_duckdb:
        print(f"Filtering and streaming rows via DuckDB for language '{args.language}'...")
        with open(passages_path, "w", encoding="utf-8") as pf, open(queries_path, "w", encoding="utf-8") as qf:
            for url in urls:
                if n_written >= args.max_passages:
                    break
                print(f"  Querying Parquet file: {url.split('/')[-1]}...")
                try:
                    # We select slightly more rows to account for duplicate passage filtration
                    select_query = f"SELECT query, passages, {lang_col}, query_id FROM read_parquet('{url}') WHERE {lang_col} = '{args.language}' LIMIT {(args.max_passages - n_written) * 2}"
                    res = con.sql(select_query)
                    cols = res.columns
                    rows_cursor = res.fetchall()
                except Exception as ex:
                    print(f"Warning: error querying Parquet file {url}: {ex}. Skipping file.", file=sys.stderr)
                    continue

                for r in rows_cursor:
                    row = dict(zip(cols, r))
                    n_scanned += 1
                    pairs = extract_passages_and_query_from_parquet_row(row)
                    if not pairs:
                        empty_extractions += 1
                        continue

                    for i, (query_text, passage_text) in enumerate(pairs):
                        if n_written >= args.max_passages:
                            break
                        if passage_text in seen_passages:
                            continue
                        seen_passages.add(passage_text)

                        row_idx = row.get("query_id", n_scanned)
                        pid = f"{args.language}-{row_idx}-{i}"
                        qid = f"{args.language}-{row_idx}"

                        pf.write(json.dumps({
                            "passage_id": pid,
                            "text": passage_text,
                            "language": args.language,
                        }, ensure_ascii=False) + "\n")

                        qf.write(json.dumps({
                            "query_id": qid,
                            "text": query_text,
                            "gold_passage_id": pid,
                        }, ensure_ascii=False) + "\n")

                        n_written += 1
    else:
        # REST API implementation
        use_filter = True
        filter_rows = []
        where_query = f'"{lang_col}"=\'{args.language}\''
        print(f"Attempting server-side /filter with: {where_query}...")
        try:
            page = _get(session, "filter", dataset=DATASET_ID, config="default", split=split, where=where_query, offset=0, length=PAGE_SIZE)
            if "error" in page:
                raise RuntimeError(page["error"])
            filter_rows = page.get("rows", [])
            print("Server-side /filter is supported. Fetching filtered rows...")
        except Exception as e:
            print(f"Server-side /filter failed or not supported ({e}). Falling back to client-side filtering over /rows...", file=sys.stderr)
            use_filter = False

        offset = 0
        with open(passages_path, "w", encoding="utf-8") as pf, open(queries_path, "w", encoding="utf-8") as qf:
            while n_written < args.max_passages:
                if use_filter:
                    if offset == 0:
                        rows = filter_rows
                    else:
                        try:
                            page = _get(session, "filter", dataset=DATASET_ID, config="default", split=split, where=where_query, offset=offset, length=PAGE_SIZE)
                            rows = page.get("rows", [])
                        except Exception as e:
                            print(f"Error fetching filtered page at offset {offset}: {e}. Stopping.", file=sys.stderr)
                            break
                else:
                    if offset == 0 and first_rows:
                        rows = first_rows
                    else:
                        try:
                            page = _get(session, "rows", dataset=DATASET_ID, config="default", split=split, offset=offset, length=PAGE_SIZE)
                            rows = page.get("rows", [])
                        except Exception as e:
                            print(f"Error fetching page at offset {offset}: {e}. Stopping.", file=sys.stderr)
                            break

                if not rows:
                    break

                for entry in rows:
                    n_scanned += 1
                    row = entry.get("row", {})

                    if not use_filter:
                        if row.get(lang_col) != args.language:
                            continue

                    pairs = extract_passages_and_query(row)
                    if not pairs:
                        empty_extractions += 1
                        continue

                    for i, (query_text, passage_text) in enumerate(pairs):
                        if n_written >= args.max_passages:
                            break
                        if passage_text in seen_passages:
                            continue
                        seen_passages.add(passage_text)

                        row_idx = entry.get("row_idx", n_scanned)
                        pid = f"{args.language}-{row_idx}-{i}"
                        qid = f"{args.language}-{row_idx}"

                        pf.write(json.dumps({
                            "passage_id": pid,
                            "text": passage_text,
                            "language": args.language,
                        }, ensure_ascii=False) + "\n")

                        qf.write(json.dumps({
                            "query_id": qid,
                            "text": query_text,
                            "gold_passage_id": pid,
                        }, ensure_ascii=False) + "\n")

                        n_written += 1

                offset += PAGE_SIZE
                if n_written < args.max_passages and offset % (PAGE_SIZE * 20) == 0:
                    print(f"  ...scanned {n_scanned} rows, collected {n_written}/{args.max_passages} passages")

    print(f"Scanned {n_scanned} rows over the API/Parquet to collect {n_written} passages.")
    print(f"Wrote {n_written} passages -> {passages_path}")
    print(f"Wrote {n_written} queries -> {queries_path}")

    if empty_extractions and n_written == 0:
        print(
            "No passages could be extracted from any row. The field-name guesses in "
            "extract_passages_and_query() didn't match this dataset's actual schema -- "
            "look at the 'First example' fields printed above and adjust that function.",
            file=sys.stderr,
        )
        sys.exit(1)

    if n_written < args.max_passages:
        print(
            f"Warning: only found {n_written}/{args.max_passages} requested passages "
            f"in the 'default'/'{split}' split for language '{args.language}' -- "
            f"try a different --language value or lower --max-passages.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
