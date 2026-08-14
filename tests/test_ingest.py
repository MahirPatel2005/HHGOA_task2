from scripts.ingest import MSMARCO_XI_CONFIGS, documents_from_row, msmarco_xi_parquet_url, normalize_splits


def test_ingest_understands_msmarco_translated_passage_schema():
    row = {
        "query": "Where is Goa?",
        "target_lang": "hin_Deva",
        "query_id": 42,
        "passages": {
            "is_selected": [1, 0],
            "English_passages": ["Goa is in India.", "A distractor."],
            "Translated_passages": ["गोवा भारत में है।", "एक अन्य अनुच्छेद।"],
        },
    }
    documents = documents_from_row(row, 1)
    assert len(documents) == 2
    assert documents[0].text == "गोवा भारत में है।"
    assert documents[0].metadata["query"] == "Where is Goa?"


def test_all_msmarco_xi_language_configs_are_declared_and_namespace_ids():
    assert MSMARCO_XI_CONFIGS == ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"]
    row = {
        "_dataset_config": "ta",
        "query_id": 77,
        "target_lang": "tam_Taml",
        "query": "கோவா எங்கே உள்ளது?",
        "passages": {"Translated_passages": ["கோவா இந்தியாவில் உள்ளது."]},
    }
    document = documents_from_row(row, 1)[0]
    assert document.id.startswith("ta:")
    assert document.metadata["dataset_config"] == "ta"
    assert document.metadata["target_lang"] == "tam_Taml"


def test_msmarco_xi_parquet_url_maps_language_and_split_names():
    assert msmarco_xi_parquet_url("ai4bharat/MSMARCO-XI", "hi", "validation").endswith("validation/hinval.parquet")
    assert msmarco_xi_parquet_url("ai4bharat/MSMARCO-XI", "te", "validation").endswith("validation/telval.parquet")
    assert msmarco_xi_parquet_url("ai4bharat/MSMARCO-XI", "as", "train").endswith("train/asmtrain.parquet")


def test_normalize_splits_supports_full_train_and_validation_build():
    assert normalize_splits(None, ["train", "validation"]) == ["train", "validation"]
    assert normalize_splits("validation", None) == ["validation"]
