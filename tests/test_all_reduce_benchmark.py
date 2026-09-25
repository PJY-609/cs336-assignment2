"""Check the aggregation used for synchronized collective latency."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("all_reduce_benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark_all_reduce.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_max_is_taken_per_iteration_before_median():
    result = benchmark.summarize([[1, 9, 1, 9], [8, 2, 8, 2]])
    assert result["iteration_max_ms"] == [8, 9, 8, 9]
    assert result["median_ms"] == 8.5
    assert result["iqr_ms"] == 1
    assert result["rank_medians_ms"] == [5, 5]


def test_single_sample_and_mismatched_rank_lengths():
    result = benchmark.summarize([[2], [3]])
    assert result["median_ms"] == 3
    assert result["iqr_ms"] == 0
    with pytest.raises(ValueError):
        benchmark.summarize([[1, 2], [3]])
