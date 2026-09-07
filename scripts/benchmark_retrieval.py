"""Synthetic CPU scoring benchmark; no dataset, model or network required."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from py.nodes.dataset_repository import DatasetEntry, DatasetIndex, DatasetRecord, _cosine, tokenize


def legacy_bm25(index, query):
    query_counts = Counter(tokenize(query))
    count = len(index.tokens) or 1
    average = sum(map(len, index.tokens)) / count
    scores = []
    for tokens in index.tokens:
        frequencies = Counter(tokens)
        score = 0.0
        for token, query_count in query_counts.items():
            frequency = frequencies.get(token, 0)
            if frequency:
                df = index.document_frequency[token]
                idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
                score += idf * (frequency * 2.5 / (frequency + 1.5 * (0.75 + 0.25 * len(tokens) / max(average, 1.0)))) * (1.0 + math.log1p(query_count))
        scores.append(score)
    return scores


def median_ms(function, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, default=5000)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.entries, args.dimensions, args.repeats) < 1:
        parser.error("entries, dimensions and repeats must be positive")
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(args.entries, args.dimensions)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query_vector = vectors[0].tolist()
    captions = ("red leather seat stitching", "blue wood door trim", "black fabric cabin metallic accent")
    record = DatasetRecord(
        dataset_name="synthetic", version="1", base_model="test", lora_name="test",
        language="en", trigger_words=[], source_path=Path("synthetic.json"),
        entries=[DatasetEntry(str(i), f"{captions[i % len(captions)]} sample {i}") for i in range(args.entries)],
    )
    lists = vectors.tolist()
    start = time.perf_counter()
    index = DatasetIndex(record, "synthetic", text_embeddings=lists)
    statistics_build_ms = (time.perf_counter() - start) * 1000
    query_text = "red leather seat metallic"
    old_dense = lambda: [_cosine(query_vector, row) for row in lists]
    new_dense = lambda: index._vector_scores(query_vector, "text")
    start = time.perf_counter()
    actual = new_dense()
    first_dense_ms = (time.perf_counter() - start) * 1000
    expected = old_dense()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(index._bm25_scores(query_text), legacy_bm25(index, query_text), atol=1e-12)
    timings = {
        "dense_python_ms": median_ms(old_dense, args.repeats),
        "dense_matrix_hot_ms": median_ms(new_dense, args.repeats),
        "bm25_scan_ms": median_ms(lambda: legacy_bm25(index, query_text), args.repeats),
        "bm25_postings_ms": median_ms(lambda: index._bm25_scores(query_text), args.repeats),
    }
    report = {
        "entries": args.entries, "dimensions": args.dimensions, "repeats": args.repeats,
        "statistics_build_ms": statistics_build_ms,
        "first_dense_including_matrix_build_ms": first_dense_ms,
        **timings,
        "dense_speedup": timings["dense_python_ms"] / max(timings["dense_matrix_hot_ms"], 1e-9),
        "bm25_speedup": timings["bm25_scan_ms"] / max(timings["bm25_postings_ms"], 1e-9),
        "max_dense_absolute_error": float(np.max(np.abs(np.asarray(actual) - expected))),
        "scope": "CPU scoring only; excludes model encoding, disk IO, MMR and generation",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
