"""Writes a small synthetic passages.jsonl/queries.jsonl in the exact shape
data/download_dataset.py produces, for local smoke-testing the chunking ->
index -> retrieval -> harness chain without needing network access to the
HF hub or the real ai4bharat/MSMARCO-XI dataset. NOT a substitute for the
real dataset -- swap to download_dataset.py before the actual submission.

Run:
    python data/make_mock_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

MOCK_PASSAGES = [
    ("p1", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
           "It was designed by Gustave Eiffel's company and completed in 1889 as the entrance to the "
           "1889 World's Fair. It stands 330 metres tall and was the tallest man-made structure in the "
           "world for 41 years.",
     "Where is the Eiffel Tower located and when was it completed?"),
    ("p2", "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal "
           "sub-range of the Himalayas. The China-Nepal border runs across its summit point. Its elevation "
           "of 8,848.86 metres was most recently established in 2020 by a Chinese and Nepali survey.",
     "How tall is Mount Everest and where is it located?"),
    ("p3", "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, "
           "India. It was commissioned in 1631 by the Mughal emperor Shah Jahan to house the tomb of his "
           "favourite wife, Mumtaz Mahal.",
     "Who commissioned the Taj Mahal and why?"),
    ("p4", "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 "
           "individual reefs and 900 islands stretching for over 2,300 kilometres off the coast of "
           "Queensland, Australia.",
     "How long is the Great Barrier Reef?"),
    ("p5", "The Amazon rainforest is a moist broadleaf tropical rainforest that covers most of the Amazon "
           "basin of South America. This basin encompasses seven million square kilometres, of which "
           "five and a half million are covered by rainforest, mostly in Brazil.",
     "How large is the Amazon rainforest?"),
]


def main() -> None:
    passages_path = DATA_DIR / "passages.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"

    with open(passages_path, "w", encoding="utf-8") as pf, open(queries_path, "w", encoding="utf-8") as qf:
        for pid, text, query in MOCK_PASSAGES:
            pf.write(json.dumps({"passage_id": pid, "text": text, "language": "en"}, ensure_ascii=False) + "\n")
            qf.write(json.dumps({"query_id": f"q-{pid}", "text": query, "gold_passage_id": pid}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(MOCK_PASSAGES)} mock passages -> {passages_path}")
    print(f"Wrote {len(MOCK_PASSAGES)} mock queries -> {queries_path}")
    print("This is synthetic data for a local smoke test only -- use data/download_dataset.py for the real submission.")


if __name__ == "__main__":
    main()
