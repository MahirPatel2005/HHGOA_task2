from scripts.ingest import load_msmarco_xi_sample_rows


def test_sample_loader_symbol_is_available():
    assert callable(load_msmarco_xi_sample_rows)
