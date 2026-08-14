from scripts.benchmark import latency_summary


def test_latency_summary_reports_p50_p70_p100():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert set(summary) == {"p50", "p70", "p100"}
    assert summary["p50"] == 2.5
    assert summary["p100"] == 4.0
