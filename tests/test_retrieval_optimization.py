from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from py.nodes import dataset_repository as repository
from py.nodes import embedding_adapters as adapters


def make_record(root: Path, name: str = "sample") -> repository.DatasetRecord:
    directory = root / name
    directory.mkdir(exist_ok=True)
    source = directory / "dataset.json"
    source.write_text(json.dumps({"dataset_name": name}), encoding="utf-8")
    captions = ["red leather seat leather", "blue wood door trim", "black fabric cabin"]
    return repository.DatasetRecord(
        dataset_name=name, version="1", base_model="test", lora_name="test",
        language="en", trigger_words=[], source_path=source,
        entries=[repository.DatasetEntry(str(i), caption) for i, caption in enumerate(captions)],
    )


def legacy_bm25(index, query):
    query_counts = Counter(repository.tokenize(query))
    document_count = len(index.tokens) or 1
    average_length = sum(map(len, index.tokens)) / document_count
    scores = []
    for tokens in index.tokens:
        counts = Counter(tokens)
        score = 0.0
        length_factor = len(tokens) / max(average_length, 1.0)
        for token, query_count in query_counts.items():
            frequency = counts.get(token, 0)
            if frequency:
                df = index.document_frequency[token]
                idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
                score += idf * (frequency * 2.5 / max(frequency + 1.5 * (0.75 + 0.25 * length_factor), 1e-6)) * (1.0 + math.log1p(query_count))
        scores.append(score)
    return scores


class RetrievalOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = make_record(self.root)
        repository.clear_dataset_index_cache()
        self.addCleanup(repository.clear_dataset_index_cache)

    def test_postings_match_legacy_bm25(self):
        self.record.entries.extend([
            repository.DatasetEntry("zh", "\u7ea2\u8272\u76ae\u9769\u5ea7\u6905 red leather"),
            repository.DatasetEntry("empty", ""),
        ])
        index = repository.DatasetIndex(self.record, "test")
        for query in ("", "missing", "leather leather red", "wood cabin", "\u7ea2\u8272\u76ae\u9769 leather"):
            with self.subTest(query=query):
                np.testing.assert_allclose(index._bm25_scores(query), legacy_bm25(index, query), rtol=1e-14, atol=1e-14)

    def test_matrix_scores_match_legacy_and_reuse_matrix(self):
        rng = np.random.default_rng(12)
        vectors = rng.normal(size=(3, 64)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        query = vectors[0].tolist()
        data = [vectors[0].tolist(), None, vectors[2].tolist()]
        index = repository.DatasetIndex(self.record, "test", text_embeddings=data)
        expected = [repository._cosine(query, row) for row in data]
        np.testing.assert_allclose(index._vector_scores(query, "text"), expected, atol=1e-6)
        matrix = index._matrices["text"]
        self.assertEqual(matrix.dtype, np.float32)
        self.assertTrue(matrix.flags.c_contiguous)
        index._vector_scores(query, "text")
        self.assertIs(index._matrices["text"], matrix)

    def test_matrix_missing_modalities_are_zero(self):
        index = repository.DatasetIndex(self.record, "test", image_embeddings=[None] * 3)
        self.assertEqual(index._vector_scores([1.0, 0.0], "image"), [0.0] * 3)
        self.assertEqual(index._vector_scores(None, "text"), [0.0] * 3)
        self.assertEqual(index._vector_scores([1.0, 0.0], "gray"), [0.0] * 3)

    def test_matrix_rejects_invalid_vectors(self):
        cases = (
            ([[1.0]], [1.0]),
            ([[1.0], [1.0, 0.0], None], [1.0]),
            ([[float("nan")], [1.0], None], [1.0]),
            ([[1.0], [1.0], None], [1.0, 0.0]),
            ([[1.0], [1.0], None], [float("inf")]),
        )
        for vectors, query in cases:
            with self.subTest(vectors=vectors, query=query):
                index = repository.DatasetIndex(self.record, "test", text_embeddings=vectors)
                with self.assertRaises(repository.DatasetError):
                    index._vector_scores(query, "text")

    def test_seeded_retrieval_matches_scalar_reference(self):
        vectors = [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]
        index = repository.DatasetIndex(
            self.record, "test", text_embeddings=vectors, embedding_model_path="mock"
        )
        def scalar(query, kind):
            data = getattr(index, f"{kind}_embeddings")
            return [repository._cosine(query, row) for row in data] if query is not None and data else [0.0] * 3
        with patch.object(repository, "_encode_text", return_value=[1.0, 0.0]):
            for seed in range(20):
                actual, _ = index.retrieve("red leather", seed=seed, top_k=2)
                with patch.object(index, "_vector_scores", side_effect=scalar):
                    expected, _ = index.retrieve("red leather", seed=seed, top_k=2)
                self.assertEqual([row["record_id"] for row in actual], [row["record_id"] for row in expected])

    def test_hot_index_avoids_json_deserialization_and_tokenization(self):
        first = repository.get_dataset_index(self.record, self.root / "cache")
        with patch.object(repository, "_deserialize_index") as deserialize, patch.object(repository, "tokenize") as tokenize:
            second = repository.get_dataset_index(self.record, self.root / "cache")
        self.assertIs(first, second)
        deserialize.assert_not_called()
        tokenize.assert_not_called()

    def test_source_change_invalidates_memory_cache(self):
        first = repository.get_dataset_index(self.record, self.root / "cache")
        self.record.source_path.write_text('{"changed": true}', encoding="utf-8")
        second = repository.get_dataset_index(self.record, self.root / "cache")
        self.assertIsNot(first, second)
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_in_memory_caption_change_invalidates_memory_cache(self):
        first = repository.get_dataset_index(self.record, self.root / "cache")
        self.record.entries = [repository.DatasetEntry("new", "green suede")]
        second = repository.get_dataset_index(self.record, self.root / "cache")
        self.assertIsNot(first, second)
        self.assertGreater(second._bm25_scores("suede")[0], 0)

    def test_disk_cache_corruption_and_deletion_invalidate_memory(self):
        cache = self.root / "cache"
        first = repository.get_dataset_index(self.record, cache)
        path = cache / "sample.index.json"
        path.write_text("broken", encoding="utf-8")
        second = repository.get_dataset_index(self.record, cache)
        self.assertIsNot(first, second)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], 5)
        path.unlink()
        third = repository.get_dataset_index(self.record, cache)
        self.assertIsNot(second, third)
        self.assertTrue(path.is_file())

    def test_instructions_and_require_embeddings_do_not_bypass_validation(self):
        first = repository.get_dataset_index(self.record, self.root / "cache")
        second = repository.get_dataset_index(self.record, self.root / "cache", embedding_query_instruction="new")
        self.assertIsNot(first, second)
        self.assertEqual(second.query_instruction, "new")
        with self.assertRaises(repository.EmbeddingModelUnavailable):
            repository.get_dataset_index(self.record, self.root / "cache", require_embeddings=True)

    def test_hot_cache_is_lru_bounded(self):
        records = [make_record(self.root, f"record_{i}") for i in range(3)]
        indexes = [repository.get_dataset_index(record, self.root / "cache") for record in records[:2]]
        repository.get_dataset_index(records[0], self.root / "cache")
        repository.get_dataset_index(records[2], self.root / "cache")
        self.assertEqual(len(repository._INDEX_CACHE), 2)
        self.assertIs(repository.get_dataset_index(records[0], self.root / "cache"), indexes[0])
        self.assertIsNot(repository.get_dataset_index(records[1], self.root / "cache"), indexes[1])

    def test_failed_atomic_write_preserves_previous_cache_and_cleans_temp(self):
        path = self.root / "index.json"
        path.write_text("previous", encoding="utf-8")
        index = repository.DatasetIndex(self.record, "test")
        with patch.object(repository.os, "replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                repository._write_index_cache(index, path)
        self.assertEqual(path.read_text(encoding="utf-8"), "previous")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_cache_changed_during_read_is_not_remembered_under_new_digest(self):
        cache = self.root / "cache"
        repository.get_dataset_index(self.record, cache)
        repository.clear_dataset_index_cache()
        original = repository._deserialize_index
        def replace_during_read(*args):
            result = original(*args)
            (cache / "sample.index.json").write_text("broken", encoding="utf-8")
            return result
        with patch.object(repository, "_deserialize_index", side_effect=replace_during_read):
            repository.get_dataset_index(self.record, cache)
        self.assertEqual(len(repository._INDEX_CACHE), 0)
        repository.get_dataset_index(self.record, cache)
        self.assertEqual(json.loads((cache / "sample.index.json").read_text())["schema_version"], 5)

    def test_bundle_cache_preserves_model_validation_and_snapshot_identity(self):
        self.record.trigger_words = ["test"]
        vectors = [[1.0, 0.0]] * 3
        index = repository.DatasetIndex(
            self.record, repository.dataset_fingerprint(self.record),
            text_embeddings=vectors, image_embeddings=vectors, gray_embeddings=vectors,
            embedding_provider="qwen3_vl", model_signature="signature",
        )
        path = repository.write_dataset_bundle(index, self.root / "sample.iatdb")
        record = repository.load_dataset_bundle(path)
        with patch.object(repository, "detect_embedding_provider", return_value="qwen3_vl"), patch.object(
            repository, "embedding_model_signature", return_value="signature"
        ):
            first = repository.get_dataset_index(record, self.root / "cache", embedding_model_path="mock")
            self.assertIs(first, repository.get_dataset_index(record, self.root / "cache", embedding_model_path="mock"))
            fresh = repository.load_dataset_bundle(path)
            self.assertIsNot(first, repository.get_dataset_index(fresh, self.root / "cache", embedding_model_path="mock"))
            record.metadata["embedding_model_signature"] = "different"
            with self.assertRaisesRegex(repository.EmbeddingModelUnavailable, "does not match"):
                repository.get_dataset_index(record, self.root / "cache", embedding_model_path="mock")

    def test_nonfinite_disk_cache_is_rebuilt(self):
        with patch.object(repository, "_load_embedding_model"), patch.object(
            repository, "_encode_text_batch", return_value=[[1.0, 0.0]] * 3
        ) as encode, patch.object(repository, "_encode_image_paths", return_value=[]):
            cache = self.root / "cache"
            repository.get_dataset_index(self.record, cache, embedding_model_path="mock")
            path = cache / "sample.index.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["text_embeddings"][0][0] = float("nan")
            stat = path.stat()
            path.write_text(json.dumps(payload), encoding="utf-8")
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(path.stat().st_size, stat.st_size)
            rebuilt = repository.get_dataset_index(self.record, cache, embedding_model_path="mock")
        self.assertEqual(encode.call_count, 2)
        self.assertTrue(np.isfinite(rebuilt.text_embeddings).all())


class AdapterReuseTests(unittest.TestCase):
    def setUp(self):
        self.cache_patch = patch.object(adapters, "_ADAPTERS", {})
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def test_batch_size_changes_reuse_one_model(self):
        with patch.object(adapters, "detect_embedding_provider", return_value="qwen3_vl"), patch.object(
            adapters, "Qwen3VLEmbeddingAdapter"
        ) as factory:
            first = adapters.get_embedding_adapter(".", "cpu", 16)
            second = adapters.get_embedding_adapter(".", "cpu", 2)
        self.assertIs(first, second)
        factory.assert_called_once()

    def test_concurrent_loading_reuses_one_model(self):
        with patch.object(adapters, "detect_embedding_provider", return_value="qwen3_vl"), patch.object(
            adapters, "Qwen3VLEmbeddingAdapter"
        ) as factory, ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda size: adapters.get_embedding_adapter(".", "cpu", size), range(1, 9)))
        factory.assert_called_once()
        self.assertTrue(all(result is results[0] for result in results))

    def test_device_and_dimension_keep_separate_adapters(self):
        with patch.object(adapters, "detect_embedding_provider", return_value="qwen3_vl"), patch.object(
            adapters, "Qwen3VLEmbeddingAdapter"
        ) as factory:
            adapters.get_embedding_adapter(".", "cpu", 16)
            adapters.get_embedding_adapter(".", "cuda", 16)
            adapters.get_embedding_adapter(".", "cpu", 16, dimension=128)
        self.assertEqual(factory.call_count, 3)

    def test_qwen_batch_override_does_not_mutate_defaults(self):
        from unittest.mock import Mock
        adapter = object.__new__(adapters.Qwen3VLEmbeddingAdapter)
        adapters.EmbeddingAdapter.__init__(adapter, ".", "cpu", 16)
        adapter.model = Mock()
        adapter.model.encode.return_value = np.asarray([[3.0, 4.0]])
        adapter.encode_texts(["text"], batch_size=2)
        self.assertEqual(adapter.model.encode.call_args.kwargs["batch_size"], 2)
        adapter.encode_images([object()], batch_size=1)
        self.assertEqual(adapter.model.encode.call_args.kwargs["batch_size"], 1)
        adapter.encode_texts(["text"])
        self.assertEqual(adapter.model.encode.call_args.kwargs["batch_size"], 16)

    def test_repository_passes_batch_size_to_cached_adapter(self):
        with patch.object(repository, "_load_embedding_model") as loader:
            loader.return_value.encode_texts.return_value = [[1.0, 0.0]]
            repository._encode_text("mock", "text", batch_size=2)
            loader.return_value.encode_texts.assert_called_once_with(["text"], instruction="", batch_size=2)

    def test_chinese_clip_batch_override_applies_to_text_and_images(self):
        import torch
        from unittest.mock import Mock
        adapter = object.__new__(adapters.ChineseCLIPEmbeddingAdapter)
        adapters.EmbeddingAdapter.__init__(adapter, ".", "cpu", 16)
        sizes = []
        def process(**kwargs):
            count = len(kwargs.get("text", kwargs.get("images", [])))
            sizes.append(count)
            return {"features": torch.ones((count, 2))}
        adapter.processor = process
        adapter.model = Mock()
        adapter.model.get_text_features.side_effect = lambda **kwargs: kwargs["features"]
        adapter.model.get_image_features.side_effect = lambda **kwargs: kwargs["features"]
        self.assertEqual(len(adapter.encode_texts(["x"] * 5, batch_size=2)), 5)
        self.assertEqual(sizes, [2, 2, 1])
        sizes.clear()
        self.assertEqual(len(adapter.encode_images([object()] * 3, batch_size=1)), 3)
        self.assertEqual(sizes, [1, 1, 1])
        self.assertEqual(adapter.batch_size, 16)

    def test_normalization_rejects_nonfinite_and_zero_vectors(self):
        for values in ([[float("nan"), 1.0]], [[float("inf"), 1.0]], [[0.0, 0.0]]):
            with self.subTest(values=values), self.assertRaises(adapters.EmbeddingAdapterError):
                adapters._normalized_lists(values)


if __name__ == "__main__":
    unittest.main()
