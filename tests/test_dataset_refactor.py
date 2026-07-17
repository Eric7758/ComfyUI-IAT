from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from py.nodes.dataset_repository import (
    DatasetIndex,
    DatasetError,
    choose_caption,
    discover_datasets,
    get_dataset_index,
    load_dataset_record,
)
from py.nodes.llm_backends import _generate_ollama, _generate_vllm


class DatasetRepositoryTests(unittest.TestCase):
    def make_dataset(self, root: Path, with_missing_pair: bool = False) -> Path:
        dataset = root / "dataset_A"
        image_dir = dataset / "images"
        image_dir.mkdir(parents=True)
        metadata = {
            "dataset_name": "dataset_A",
            "version": "1.0",
            "base_model": "Flux.2 Klein 9B",
            "lora_name": "model_A",
            "language": "zh",
            "trigger_words": ["trigger_a"],
        }
        (dataset / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        for index, caption in enumerate(("红色产品，金属外壳，正面视图", "蓝色产品，织物表面，三分之四视图"), start=1):
            image_path = image_dir / f"{index:04d}.png"
            Image.new("RGB", (8, 8), (index * 20, index * 30, index * 40)).save(image_path)
            (image_dir / f"{index:04d}.txt").write_text(caption, encoding="utf-8")
        if with_missing_pair:
            Image.new("RGB", (8, 8), (0, 0, 0)).save(image_dir / "missing.png")
        return dataset

    def test_paired_loading_and_warnings(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.make_dataset(Path(temp), with_missing_pair=True)
            record = load_dataset_record(dataset)
            self.assertEqual(len(record.entries), 2)
            self.assertTrue(any("Missing caption" in warning for warning in record.warnings))
            self.assertEqual(record.trigger_words, ["trigger_a"])

    def make_multiview_dataset(self, root: Path, missing_control: bool = False) -> Path:
        dataset = root / "dataset_A"
        metadata = {
            "dataset_name": "dataset_A",
            "version": "1.0",
            "base_model": "Flux.2 Klein 9B",
            "lora_name": "model_A",
            "language": "zh",
            "trigger_words": ["trigger_a"],
        }
        dataset.mkdir(parents=True)
        (dataset / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
        roles = ("control1", "control2", "control3", "result")
        for role in roles:
            if missing_control and role == "control2":
                continue
            directory = dataset / f"dataset_A_{role}"
            directory.mkdir()
            for stem in ("000000", "000b00"):
                Image.new("RGB", (8, 8), (20, 30, 40)).save(directory / f"{stem}.png")
                if role == "result":
                    (directory / f"{stem}.txt").write_text(f"{stem} caption", encoding="utf-8")
        return dataset

    def test_multiview_stem_grouping_and_result_caption(self):
        with tempfile.TemporaryDirectory() as temp:
            record = load_dataset_record(self.make_multiview_dataset(Path(temp)))
        self.assertEqual(len(record.entries), 2)
        self.assertEqual(record.entries[0].record_id, "000000")
        self.assertEqual(set(record.entries[0].image_paths), {"control1", "control2", "control3", "result"})
        self.assertEqual(record.entries[0].caption, "000000 caption")
        self.assertEqual(record.entries[0].relative_image_paths["result"], "dataset_A_result/000000.png")

    def test_nested_images_role_directories_are_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.make_multiview_dataset(Path(temp))
            for role in ("control1", "control2", "control3", "result"):
                source = dataset / f"dataset_A_{role}"
                target = dataset / "images" / role
                target.parent.mkdir(exist_ok=True)
                source.rename(target)
            record = load_dataset_record(dataset)
        self.assertEqual(len(record.entries), 2)
        self.assertEqual(record.entries[0].relative_image_paths["result"], "images/result/000000.png")

    def test_retrieval_debug_includes_multiview_paths_and_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            record = load_dataset_record(self.make_multiview_dataset(Path(temp)))
            index = get_dataset_index(record, Path(temp) / "cache")
            results, debug = index.retrieve("000000", top_k=1)
        self.assertEqual(debug["reference_image_count"], 0)
        self.assertEqual(debug["selection_method"], "seeded_weighted_mmr")
        self.assertTrue(debug["candidate_pool"])
        self.assertEqual(set(results[0]["image_roles"]), {"control1", "control2", "control3", "result"})
        self.assertIn("dataset_A_result/000000.png", results[0]["image_paths"]["result"])

    def test_multiview_missing_control_is_warning_but_result_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            record = load_dataset_record(self.make_multiview_dataset(Path(temp), missing_control=True))
        self.assertEqual(len(record.entries), 2)
        self.assertTrue(any("control2" in warning for warning in record.warnings))

    def test_multiview_missing_result_or_caption_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.make_multiview_dataset(Path(temp))
            for role in ("control1", "control2", "control3"):
                Image.new("RGB", (8, 8), "white").save(dataset / f"dataset_A_{role}" / "000111.png")
            Image.new("RGB", (8, 8), "white").save(dataset / "dataset_A_result" / "000111.png")
            (dataset / "dataset_A_result" / "000111.txt").write_text("valid caption", encoding="utf-8")
            (dataset / "dataset_A_result" / "000000.txt").unlink()
            (dataset / "dataset_A_result" / "000b00.png").unlink()
            record = load_dataset_record(dataset)
        self.assertEqual(len(record.entries), 1)
        self.assertEqual(record.entries[0].record_id, "000111")
        self.assertTrue(any("caption" in warning for warning in record.warnings))

    def test_directory_requires_trigger_words(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.make_dataset(Path(temp))
            metadata_path = dataset / "dataset.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("trigger_words")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(DatasetError):
                load_dataset_record(dataset)

    def test_all_required_metadata_fields_reject_missing_or_empty_values(self):
        fields = ("dataset_name", "version", "base_model", "lora_name", "language", "trigger_words")
        for field in fields:
            for empty_value in (None, [] if field == "trigger_words" else "   "):
                with self.subTest(field=field, empty_value=empty_value), tempfile.TemporaryDirectory() as temp:
                    dataset = self.make_dataset(Path(temp))
                    metadata_path = dataset / "dataset.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if empty_value is None:
                        metadata.pop(field)
                    else:
                        metadata[field] = empty_value
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    with self.assertRaises(DatasetError):
                        load_dataset_record(dataset)

    def test_images_directory_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.make_dataset(Path(temp))
            image_dir = dataset / "images"
            renamed = dataset / "pairs"
            image_dir.rename(renamed)
            with self.assertRaisesRegex(DatasetError, "images"):
                load_dataset_record(dataset)

    def test_seeded_random_picker_is_reproducible_and_not_modulo(self):
        with tempfile.TemporaryDirectory() as temp:
            record = load_dataset_record(self.make_dataset(Path(temp)))
            first, first_index = choose_caption(record, "Random", 1)
            second, second_index = choose_caption(record, "Random", 1)
            self.assertEqual(first.record_id, second.record_id)
            self.assertEqual(first_index, second_index)
            # For two entries, seed 1 selects the second entry with Python's
            # seeded RNG, whereas the previous implementation selected index 1
            # by coincidence; seed 2 demonstrates the non-sequential mapping.
            _, seed_two_index = choose_caption(record, "Random", 2)
            self.assertEqual(seed_two_index, 0)

    def test_index_cache_and_hybrid_free_bm25_retrieval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = load_dataset_record(self.make_dataset(root))
            cache = root / "cache"
            index = get_dataset_index(record, cache)
            results, debug = index.retrieve("红色 金属", top_k=1, seed=42)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["record_id"], "images/0001")
            self.assertTrue(debug["index_version"].startswith("hybrid-v3:"))
            cached = get_dataset_index(record, cache)
            self.assertEqual(cached.fingerprint, index.fingerprint)

            caption_path = record.entries[0].image_path.with_suffix(".txt")
            caption_path.write_text("changed caption", encoding="utf-8")
            changed_record = load_dataset_record(record.source_path.parent)
            rebuilt = get_dataset_index(changed_record, cache)
            self.assertNotEqual(rebuilt.fingerprint, index.fingerprint)

    def test_exploration_retrieval_is_reproducible_and_seeded(self):
        from py.nodes.dataset_repository import DatasetEntry, DatasetRecord

        record = DatasetRecord(
            dataset_name="dataset_A",
            version="1.0",
            base_model="Flux.2 Klein 9B",
            lora_name="model_A",
            language="zh",
            trigger_words=["trigger_a"],
            entries=[
                DatasetEntry(str(index), f"黑色越野 CMF 样本 {index}")
                for index in range(24)
            ],
            source_path=Path("dataset.json"),
        )
        index = DatasetIndex(record, "fingerprint")
        first, first_debug = index.retrieve("黑色越野", top_k=4, seed=17, exploration_strength="Medium")
        repeat, repeat_debug = index.retrieve("黑色越野", top_k=4, seed=17, exploration_strength="Medium")
        alternate, _ = index.retrieve("黑色越野", top_k=4, seed=18, exploration_strength="Medium")
        self.assertEqual([item["record_id"] for item in first], [item["record_id"] for item in repeat])
        self.assertEqual(first_debug, repeat_debug)
        self.assertNotEqual(
            [item["record_id"] for item in first],
            [item["record_id"] for item in alternate],
        )
        self.assertEqual(first_debug["exploration_strength"], "Medium")

        skewed_index = DatasetIndex(
            record,
            "fingerprint",
            text_embeddings=[[1.0, 0.0]] + [[0.98, 0.2]] * 23,
            embedding_model_path="model",
        )
        with patch("py.nodes.dataset_repository._encode_text", return_value=[1.0, 0.0]):
            skewed_first, _ = skewed_index.retrieve("黑色越野", top_k=4, seed=17, exploration_strength="Medium")
            skewed_alternate, _ = skewed_index.retrieve("黑色越野", top_k=4, seed=18, exploration_strength="Medium")
        self.assertNotEqual(
            [item["record_id"] for item in skewed_first],
            [item["record_id"] for item in skewed_alternate],
        )

    def test_fingerprint_tracks_unpaired_files_for_diagnostics(self):
        from py.nodes.dataset_repository import dataset_fingerprint

        with tempfile.TemporaryDirectory() as temp:
            dataset = self.make_dataset(Path(temp))
            before = dataset_fingerprint(load_dataset_record(dataset))
            Image.new("RGB", (8, 8), "black").save(dataset / "images" / "unpaired.png")
            after = dataset_fingerprint(load_dataset_record(dataset))
        self.assertNotEqual(before, after)

    def test_fingerprint_ignores_mtime_only_changes(self):
        from py.nodes.dataset_repository import dataset_fingerprint

        with tempfile.TemporaryDirectory() as temp:
            dataset = self.make_dataset(Path(temp))
            image_path = dataset / "images" / "0001.png"
            before = dataset_fingerprint(load_dataset_record(dataset))
            stat = image_path.stat()
            os.utime(image_path, ns=(stat.st_atime_ns + 1_000_000, stat.st_mtime_ns + 1_000_000))
            after = dataset_fingerprint(load_dataset_record(dataset))
        self.assertEqual(before, after)

    @patch("py.nodes.dataset_repository._encode_image_batch")
    @patch("py.nodes.dataset_repository._encode_text_batch")
    @patch("py.nodes.dataset_repository._load_embedding_model")
    @patch("py.nodes.dataset_repository._resolve_embedding_device", return_value="cuda")
    def test_embedding_index_builds_in_batches(self, resolve_device, load_model, encode_text_batch, encode_image_batch):
        encode_text_batch.return_value = [[1.0, 0.0], [0.0, 1.0]]
        encode_image_batch.side_effect = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = load_dataset_record(self.make_dataset(root))
            index = get_dataset_index(
                record,
                root / "cache",
                embedding_model_path=str(root / "model"),
                embedding_device="auto",
                embedding_batch_size=16,
            )
        self.assertEqual(index.embedding_device, "cuda")
        encode_text_batch.assert_called_once()
        self.assertEqual(encode_image_batch.call_count, 2)
        self.assertEqual(encode_text_batch.call_args.args[3], 16)
        self.assertEqual(encode_image_batch.call_args.args[3], 16)

    @patch("py.nodes.dataset_repository._encode_image", return_value=[1.0, 0.0])
    @patch("py.nodes.dataset_repository._encode_text", return_value=[1.0, 0.0])
    def test_query_embeddings_use_index_device(self, encode_text, encode_image):
        with tempfile.TemporaryDirectory() as temp:
            record = load_dataset_record(self.make_dataset(Path(temp)))
            index = DatasetIndex(
                record,
                "fingerprint",
                text_embeddings=[[1.0, 0.0], [0.0, 1.0]],
                image_embeddings=[[1.0, 0.0], [0.0, 1.0]],
                gray_embeddings=[[1.0, 0.0], [0.0, 1.0]],
                embedding_model_path="model",
                embedding_device="cuda",
            )
            index.retrieve("query", Image.new("RGB", (4, 4)), top_k=1)
        self.assertEqual(encode_text.call_args.kwargs["device"], "cuda")
        self.assertEqual(encode_image.call_args.kwargs["device"], "cuda")

    @patch("py.nodes.dataset_repository._encode_image_batch", return_value=[[1.0, 0.0], [0.0, 1.0]])
    @patch("py.nodes.dataset_repository._encode_text", return_value=[1.0, 0.0])
    def test_multiview_query_aggregates_image_embeddings(self, encode_text, encode_image_batch):
        with tempfile.TemporaryDirectory() as temp:
            record = load_dataset_record(self.make_multiview_dataset(Path(temp)))
            index = DatasetIndex(
                record,
                "fingerprint",
                text_embeddings=[[1.0, 0.0], [0.0, 1.0]],
                image_embeddings=[[1.0, 0.0], [0.0, 1.0]],
                gray_embeddings=[[1.0, 0.0], [0.0, 1.0]],
                embedding_model_path="model",
                embedding_device="cpu",
            )
            index.retrieve("query", reference_images=[Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))], top_k=1)
        self.assertEqual(encode_image_batch.call_args.args[1].__len__(), 2)

    @patch("py.nodes.dataset_repository._encode_image_batch")
    @patch("py.nodes.dataset_repository._encode_text_batch")
    @patch("py.nodes.dataset_repository._load_embedding_model")
    @patch("py.nodes.dataset_repository._resolve_embedding_device", return_value="cpu")
    def test_multiview_index_aggregates_each_entry_group(self, resolve_device, load_model, encode_text_batch, encode_image_batch):
        encode_text_batch.return_value = [[1.0, 0.0], [0.0, 1.0]]
        encode_image_batch.side_effect = [
            [[1.0, 0.0]] * 8,
            [[0.0, 1.0]] * 8,
        ]
        with tempfile.TemporaryDirectory() as temp:
            record = load_dataset_record(self.make_multiview_dataset(Path(temp)))
            index = get_dataset_index(record, Path(temp) / "cache", embedding_model_path="model")
        self.assertEqual(len(index.image_embeddings), 2)
        self.assertEqual(index.image_embeddings[0], [1.0, 0.0])
        self.assertEqual(encode_image_batch.call_args.args[1].__len__(), 8)

    @patch("py.nodes.dataset_repository._load_embedding_model")
    @patch("py.nodes.dataset_repository._encode_text_batch", return_value=[[1.0, 0.0], [0.0, 1.0]])
    @patch("py.nodes.dataset_repository._encode_image_batch", return_value=[[1.0, 0.0]] * 8)
    @patch("py.nodes.dataset_repository._resolve_embedding_device", return_value="cpu")
    def test_multiview_cache_uses_schema_v3(self, resolve_device, encode_image_batch, encode_text_batch, load_model):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = load_dataset_record(self.make_multiview_dataset(root))
            cache_dir = root / "cache"
            get_dataset_index(record, cache_dir, embedding_model_path="model")
            payload = json.loads((cache_dir / "dataset_A.index.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(len(payload["entries"][0]["image_paths"]), 4)

    @patch("py.nodes.dataset_repository._load_embedding_model")
    @patch("py.nodes.dataset_repository._encode_text_batch", return_value=[[1.0, 0.0], [0.0, 1.0]])
    @patch("py.nodes.dataset_repository._encode_image_batch", return_value=[[1.0, 0.0], [0.0, 1.0]])
    @patch("py.nodes.dataset_repository._resolve_embedding_device", return_value="cpu")
    def test_corrupt_embedding_cache_is_rebuilt(self, resolve_device, encode_image_batch, encode_text_batch, load_model):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = load_dataset_record(self.make_dataset(root))
            cache_dir = root / "cache"
            get_dataset_index(record, cache_dir, embedding_model_path="model")
            cache_path = cache_dir / "dataset_A.index.json"
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["text_embeddings"] = [[1.0, 0.0]]
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            rebuilt = get_dataset_index(record, cache_dir, embedding_model_path="model")
        self.assertEqual(len(rebuilt.text_embeddings), 2)
        self.assertEqual(encode_text_batch.call_count, 2)

    def test_discovery_skips_index_cache_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            cache = root / ".iat_index"
            cache.mkdir()
            (cache / "fake.index.json").write_text("{}", encoding="utf-8")
            records, errors = discover_datasets(root)
            self.assertEqual(list(records), ["dataset_A"])
            self.assertFalse(any("fake" in error for error in errors))

    def test_bad_dataset_does_not_hide_valid_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            bad = root / "bad"
            bad.mkdir()
            (bad / "dataset.json").write_text("{}", encoding="utf-8")
            records, errors = discover_datasets(root)
            self.assertIn("dataset_A", records)
            self.assertTrue(errors)

    def test_duplicate_dataset_names_are_not_selectable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.make_dataset(root)
            second = root / "dataset_B"
            second.mkdir()
            shutil.copytree(first / "images", second / "images")
            (second / "dataset.json").write_text((first / "dataset.json").read_text(encoding="utf-8"), encoding="utf-8")
            records, errors = discover_datasets(root)
            self.assertNotIn("dataset_A", records)
            self.assertTrue(any("Duplicate dataset_name" in error for error in errors))


class BackendRequestTests(unittest.TestCase):
    @patch("py.nodes.llm_backends._request_json")
    def test_ollama_payload_supports_image_seed_and_keep_alive(self, request_json):
        request_json.return_value = {"message": {"content": "生成提示词"}}
        output = _generate_ollama(
            model="qwen3.5:122b",
            base_url="http://127.0.0.1:11434",
            prompt="生成一条提示词",
            images=[Image.new("RGB", (4, 4), "white")],
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
            seed=7,
            timeout=10,
            keep_alive=-1,
            think=False,
        )
        self.assertEqual(output, "生成提示词")
        payload = request_json.call_args.args[1]
        self.assertEqual(payload["model"], "qwen3.5:122b")
        self.assertEqual(payload["options"]["seed"], 7)
        self.assertEqual(payload["keep_alive"], -1)
        self.assertIn("images", payload["messages"][0])

    @patch("py.nodes.llm_backends._request_json")
    def test_vllm_payload_is_openai_compatible(self, request_json):
        request_json.return_value = {"choices": [{"message": {"content": "prompt"}}]}
        output = _generate_vllm(
            model="qwen3.5:122b",
            base_url="http://127.0.0.1:8000/v1",
            prompt="generate",
            images=None,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
            seed=9,
            timeout=10,
            api_key="",
        )
        self.assertEqual(output, "prompt")
        url = request_json.call_args.args[0]
        payload = request_json.call_args.args[1]
        self.assertEqual(url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(payload["seed"], 9)
        self.assertEqual(payload["messages"][0]["content"], "generate")

    @patch("py.nodes.llm_backends._request_json")
    def test_remote_backends_send_all_reference_images(self, request_json):
        request_json.return_value = {"message": {"content": "prompt"}}
        images = [Image.new("RGB", (4, 4), color) for color in ("red", "green", "blue", "white")]
        _generate_ollama(
            model="qwen3.5:122b",
            base_url="http://127.0.0.1:11434",
            prompt="generate",
            images=images,
            max_tokens=64,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=1,
            timeout=10,
            keep_alive=-1,
            think=False,
        )
        self.assertEqual(len(request_json.call_args.args[1]["messages"][0]["images"]), 4)

        request_json.return_value = {"choices": [{"message": {"content": "prompt"}}]}
        _generate_vllm(
            model="qwen3.5:122b",
            base_url="http://127.0.0.1:8000/v1",
            prompt="generate",
            images=images,
            max_tokens=64,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=1,
            timeout=10,
            api_key="",
        )
        content = request_json.call_args.args[1]["messages"][0]["content"]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 4)


class NodeBehaviorTests(unittest.TestCase):
    def test_dataset_nodes_import_without_torch_and_expose_split_contract(self):
        import py.nodes.qwen35_dataset_rag_nodes as module

        self.assertEqual(
            set(module.NODE_CLASS_MAPPINGS),
            {"DatasetCaptionPicker by IAT", "DatasetRAGPromptGenerator by IAT"},
        )
        picker = module.DatasetCaptionPickerNode()
        self.assertEqual(len(picker.RETURN_TYPES), 4)
        generator = module.DatasetRAGPromptGeneratorNode()
        self.assertEqual(generator.RETURN_NAMES, ("prompt", "retrieved_captions", "retrieval_debug", "dataset_metadata"))
        self.assertEqual(set(generator.INPUT_TYPES()["optional"]), {"image", "image_2", "image_3", "image_4"})
        required = generator.INPUT_TYPES()["required"]
        self.assertIn("exploration_strength", required)
        self.assertIn("variation_seed", required)

    def test_variation_plan_is_reproducible_and_keeps_hard_family_hex_pairs(self):
        from py.nodes.qwen35_dataset_rag_nodes import _build_variation_plan

        prompt = "黑色系（玄武岩黑，#1f1f1d）与棕色系（复古鞍棕，#8b4f2f）越野 CMF"
        retrieved = [
            {
                "caption": "座椅主面麂皮，中控台前饰板皮质，主辅分色",
            }
        ]
        first = _build_variation_plan(prompt, retrieved, 123, "Medium")
        repeat = _build_variation_plan(prompt, retrieved, 123, "Medium")
        alternate = _build_variation_plan(prompt, retrieved, 124, "Medium")
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, alternate)
        self.assertEqual(set(first["hard_color_families"]), {"black", "brown"})
        for assignment in first["component_assignments"]:
            self.assertEqual(assignment["hex"], first["proposed_palette"][assignment["family"]])
        assigned_families = {item["family"] for item in first["component_assignments"]}
        self.assertTrue({"black", "brown"}.issubset(assigned_families))

    def test_exploration_temperature_mapping(self):
        from py.nodes.qwen35_dataset_rag_nodes import _effective_temperature

        self.assertEqual(_effective_temperature(0.0, "Mild"), 0.15)
        self.assertEqual(_effective_temperature(0.0, "Medium"), 0.35)
        self.assertEqual(_effective_temperature(0.0, "Strong"), 0.55)
        self.assertEqual(_effective_temperature(0.2, "Strong"), 0.2)

    def test_generator_reuses_same_seed_and_exposes_variation_debug(self):
        import py.nodes.qwen35_dataset_rag_nodes as module
        from py.nodes.dataset_repository import DatasetEntry, DatasetRecord

        record = DatasetRecord(
            dataset_name="dataset_A",
            version="1.0",
            base_model="Flux.2 Klein 9B",
            lora_name="model_A",
            language="zh",
            trigger_words=["trigger_a"],
            entries=[DatasetEntry("0001", "黑色越野座椅主面麂皮")],
            source_path=Path("dataset.json"),
        )
        index_value = DatasetIndex(record, "fingerprint")
        kwargs = {
            "user_prompt": "黑色系越野内饰 CMF",
            "dataset_name": "dataset_A",
            "backend": "Ollama",
            "model_override": "qwen3.5:122b",
            "base_url_override": "http://127.0.0.1:11434",
            "retrieval_seed": 7,
            "generation_seed": 8,
            "exploration_strength": "Medium",
            "variation_seed": 9,
            "top_k": 1,
            "preserve_reference_color": False,
            "custom_instruction": "",
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "timeout_seconds": 10,
        }
        with patch.object(module, "_selected_record", return_value=record), patch.object(
            module, "dataset_fingerprint", return_value="fingerprint"
        ), patch.object(module, "get_dataset_index", return_value=index_value), patch.object(
            module, "generate_with_backend", return_value="黑色系 trigger_a 越野内饰"
        ) as generate:
            first = module.DatasetRAGPromptGeneratorNode().generate_prompt(**kwargs)
            repeat = module.DatasetRAGPromptGeneratorNode().generate_prompt(**kwargs)
        self.assertEqual(first, repeat)
        debug = json.loads(first[2])
        self.assertEqual(debug["variation_seed"], 9)
        self.assertEqual(debug["exploration_strength"], "Medium")
        self.assertEqual(debug["effective_temperature"], 0.35)
        self.assertIn("variation_plan", debug)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args.kwargs["temperature"], 0.35)
        self.assertEqual(generate.call_args.kwargs["seed"], debug["generation_seed"])

        variation_kwargs = dict(kwargs)
        variation_kwargs["variation_seed"] = 10
        with patch.object(module, "_selected_record", return_value=record), patch.object(
            module, "dataset_fingerprint", return_value="fingerprint"
        ), patch.object(module, "get_dataset_index", return_value=index_value), patch.object(
            module, "generate_with_backend", return_value="黑色系 trigger_a 越野内饰"
        ):
            variation_output = module.DatasetRAGPromptGeneratorNode().generate_prompt(**variation_kwargs)
        variation_debug = json.loads(variation_output[2])
        self.assertNotEqual(debug["composition_seed"], variation_debug["composition_seed"])
        self.assertEqual(debug["generation_seed"], variation_debug["generation_seed"])

    def test_generator_repairs_missing_hard_color_family(self):
        import py.nodes.qwen35_dataset_rag_nodes as module
        from py.nodes.dataset_repository import DatasetEntry, DatasetRecord

        record = DatasetRecord(
            dataset_name="dataset_A",
            version="1.0",
            base_model="Flux.2 Klein 9B",
            lora_name="model_A",
            language="zh",
            trigger_words=["trigger_a"],
            entries=[DatasetEntry("0001", "黑色越野")],
            source_path=Path("dataset.json"),
        )
        with patch.object(module, "_selected_record", return_value=record), patch.object(
            module, "dataset_fingerprint", return_value="fingerprint"
        ), patch.object(module, "get_dataset_index", return_value=DatasetIndex(record, "fingerprint")), patch.object(
            module, "generate_with_backend", return_value="trigger_a brown leather"
        ):
            output = module.DatasetRAGPromptGeneratorNode().generate_prompt(
                user_prompt="黑色系越野内饰",
                dataset_name="dataset_A",
                backend="Ollama",
                model_override="qwen3.5:122b",
                base_url_override="",
                retrieval_seed=1,
                generation_seed=1,
                exploration_strength="Medium",
                variation_seed=1,
                top_k=1,
                preserve_reference_color=False,
                custom_instruction="",
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.05,
                timeout_seconds=10,
            )
        self.assertIn("黑色系", output[0])
        self.assertNotIn("brown", output[0].casefold())

    def test_generator_rejects_empty_backend_output(self):
        import py.nodes.qwen35_dataset_rag_nodes as module
        from py.nodes.dataset_repository import DatasetEntry, DatasetRecord

        record = DatasetRecord(
            dataset_name="dataset_A",
            version="1.0",
            base_model="Flux.2 Klein 9B",
            lora_name="model_A",
            language="zh",
            trigger_words=["trigger_a"],
            entries=[DatasetEntry("0001", "黑色越野")],
            source_path=Path("dataset.json"),
        )
        with patch.object(module, "_selected_record", return_value=record), patch.object(
            module, "dataset_fingerprint", return_value="fingerprint"
        ), patch.object(module, "get_dataset_index", return_value=DatasetIndex(record, "fingerprint")), patch.object(
            module, "generate_with_backend", return_value="   "
        ), self.assertRaisesRegex(RuntimeError, "empty prompt"):
            module.DatasetRAGPromptGeneratorNode().generate_prompt(
                user_prompt="黑色系越野内饰",
                dataset_name="dataset_A",
                backend="Ollama",
                model_override="qwen3.5:122b",
                base_url_override="",
                retrieval_seed=1,
                generation_seed=1,
                exploration_strength="Medium",
                variation_seed=1,
                top_k=1,
                preserve_reference_color=False,
                custom_instruction="",
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.05,
                timeout_seconds=10,
            )

    def test_reference_image_collection_supports_four_and_rejects_five(self):
        import py.nodes.qwen35_dataset_rag_nodes as module
        one = object()
        with patch.object(module, "_tensor_to_pil_list", return_value=[Image.new("RGB", (4, 4))]):
            self.assertEqual(len(module._collect_reference_images(one, one, one, one)), 4)
        with self.assertRaisesRegex(DatasetError, "At most 4"):
            with patch.object(module, "_tensor_to_pil_list", return_value=[Image.new("RGB", (4, 4))]):
                module._collect_reference_images(one, one, one, one, one)

    def test_trigger_words_are_added_without_overwriting_model_output(self):
        from py.nodes.qwen35_dataset_rag_nodes import _ensure_trigger_words
        from py.nodes.dataset_repository import DatasetEntry, DatasetRecord

        record = DatasetRecord(
            dataset_name="dataset_A",
            version="1.0",
            base_model="Flux.2 Klein 9B",
            lora_name="model_A",
            language="zh",
            trigger_words=["trigger_a", "触发词"],
            entries=[DatasetEntry("0001", "caption")],
            source_path=Path("dataset.json"),
        )
        output = _ensure_trigger_words("蓝色产品，金属外壳", record)
        self.assertTrue(output.startswith("trigger_a, 触发词,"))
        self.assertIn("蓝色产品", output)


    def test_latin_trigger_requires_token_boundaries(self):
        from py.nodes.qwen35_dataset_rag_nodes import _contains_trigger

        self.assertFalse(_contains_trigger("a scarlet product", "car"))
        self.assertTrue(_contains_trigger("a car product", "car"))
        self.assertTrue(_contains_trigger("use model_A style", "model_A"))

    def test_selected_valid_dataset_ignores_unrelated_discovery_errors(self):
        import py.nodes.qwen35_dataset_rag_nodes as module

        with tempfile.TemporaryDirectory() as temp:
            helper = DatasetRepositoryTests()
            record = load_dataset_record(helper.make_dataset(Path(temp)))
            with patch.object(module, "_discover", return_value=({"dataset_A": record}, ["bad other dataset"])):
                result = module.DatasetCaptionPickerNode().pick_caption("dataset_A", "By Index", 0, 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[0], record.entries[0].caption)

    def test_node_failures_raise_instead_of_returning_prompt_text(self):
        import py.nodes.qwen35_dataset_rag_nodes as module

        generator = module.DatasetRAGPromptGeneratorNode()
        with self.assertRaisesRegex(RuntimeError, "user_prompt is required"):
            generator.generate_prompt(
                "", "dataset_A", "Ollama", "", "", 1, 1, 4, False, "", 128, 0.0, 1.0, 1.05, 10
            )
        with patch.object(module, "_discover", return_value=({}, ["invalid metadata"])):
            with self.assertRaises(DatasetError):
                module.DatasetCaptionPickerNode().pick_caption("missing", "Random", 1, 0)

    def test_is_changed_tracks_caption_edits(self):
        import py.nodes.qwen35_dataset_rag_nodes as module

        with tempfile.TemporaryDirectory() as temp:
            helper = DatasetRepositoryTests()
            dataset = helper.make_dataset(Path(temp))
            with patch.object(module, "_dataset_root", return_value=Path(temp)):
                before = module.DatasetCaptionPickerNode.IS_CHANGED("dataset_A")
                (dataset / "images" / "0001.txt").write_text("updated caption", encoding="utf-8")
                after = module.DatasetCaptionPickerNode.IS_CHANGED("dataset_A")
                generator_after = module.DatasetRAGPromptGeneratorNode.IS_CHANGED("dataset_A")
        self.assertNotEqual(before, after)
        self.assertEqual(after, generator_after)


if __name__ == "__main__":
    unittest.main()
