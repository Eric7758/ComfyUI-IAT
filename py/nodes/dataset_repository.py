from __future__ import annotations

"""Offline dataset discovery, paired caption loading, and hybrid retrieval.

The repository deliberately keeps image files outside the index.  The index stores
captions, relative paths, metadata, and optional normalized embeddings only.
"""

import hashlib
import json
import math
import os
import random
import re
import sqlite3
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .embedding_adapters import (
    EmbeddingAdapterError,
    detect_embedding_provider,
    embedding_model_signature,
    get_embedding_adapter,
    unload_embedding_adapters,
)
from .cmf_prompt import (
    SAMPLE_CLASSIFICATION_VERSION,
    SAMPLE_TYPES,
    classify_caption,
    extract_components,
    extract_materials,
    normalize_family,
    normalize_material,
)


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[^\w\s\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE)
_CONTROL_ROLE_RE = re.compile(r"(?:^|[_\-\s])control([1-3])$", re.IGNORECASE)
_IMAGE_ROLES = ("control1", "control2", "control3", "result")
_EXPLORATION_PROFILES = {
    "Mild": {"candidate_k": 16, "relevance": 0.85, "diversity": 0.15, "sampling_temperature": 0.10},
    "Medium": {"candidate_k": 16, "relevance": 0.65, "diversity": 0.35, "sampling_temperature": 0.20},
    "Strong": {"candidate_k": 16, "relevance": 0.50, "diversity": 0.50, "sampling_temperature": 0.32},
}
_BUNDLE_SCHEMA_VERSION = 2
_BUNDLE_FORMAT = "comfyui-iat-dataset"
_INDEX_SCHEMA_VERSION = 5
_QWEN_IMAGE_MAX_SIDE = 768
_INDEX_CACHE: OrderedDict = OrderedDict()
_INDEX_CACHE_LIMIT = 2
_INDEX_CACHE_LOCK = RLock()


class DatasetError(RuntimeError):
    """Base error for invalid or unavailable dataset resources."""


class EmbeddingModelUnavailable(DatasetError):
    """Raised when a configured local embedding model cannot be loaded."""


@dataclass(frozen=True)
class DatasetEntry:
    record_id: str
    caption: str
    image_path: Optional[Path] = None
    relative_image_path: str = ""
    image_paths: Dict[str, Path] = field(default_factory=dict)
    relative_image_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_types(self) -> List[str]:
        values = self.metadata.get("sample_types")
        return [str(value) for value in values] if isinstance(values, list) else []

    def grouped_image_paths(self) -> Dict[str, Path]:
        if self.image_paths:
            return dict(self.image_paths)
        if self.image_path is None:
            return {}
        return {"image": self.image_path}

    def grouped_relative_image_paths(self) -> Dict[str, str]:
        if self.relative_image_paths:
            return dict(self.relative_image_paths)
        if self.relative_image_path:
            return {"image": self.relative_image_path}
        return {}


@dataclass
class DatasetRecord:
    dataset_name: str
    version: str
    base_model: str
    lora_name: str
    language: str
    trigger_words: List[str]
    entries: List[DatasetEntry]
    source_path: Path
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    bundle_path: Optional[Path] = None
    bundled_text_embeddings: List[Optional[List[float]]] = field(default_factory=list, repr=False)
    bundled_image_embeddings: List[Optional[List[float]]] = field(default_factory=list, repr=False)
    bundled_gray_embeddings: List[Optional[List[float]]] = field(default_factory=list, repr=False)

    @property
    def captions(self) -> List[str]:
        return [entry.caption for entry in self.entries]


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalize_text(text: str) -> str:
    return _normalize_whitespace(_PUNCT_RE.sub(" ", (text or "").lower()))


def tokenize(text: str) -> List[str]:
    """Tokenize Latin words and CJK unigrams/bigrams for small Chinese datasets."""
    normalized = _normalize_text(text)
    latin = _TOKEN_RE.findall(normalized)
    cjk = _CJK_RE.findall(normalized)
    cjk_bigrams = [cjk[idx] + cjk[idx + 1] for idx in range(max(0, len(cjk) - 1))]
    return latin + cjk + cjk_bigrams


def detect_language(texts: Sequence[str]) -> str:
    joined = "\n".join(texts)
    if re.search(r"[\u3040-\u30ff]", joined):
        return "ja"
    if re.search(r"[\u3400-\u9fff]", joined):
        return "zh"
    return "en"


def _string_list(value: Any, field_name: str, path: Path, required: bool = False) -> List[str]:
    if value is None and not required:
        return []
    if not isinstance(value, list):
        raise DatasetError(f"[IAT] {path.name}: `{field_name}` must be a list of strings.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DatasetError(f"[IAT] {path.name}: `{field_name}` contains an empty/non-string item.")
        result.append(_normalize_whitespace(item))
    if required and not result:
        raise DatasetError(f"[IAT] {path.name}: `{field_name}` must contain at least one non-empty string.")
    return result


def _required_string(raw: Dict[str, Any], field_name: str, path: Path) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"[IAT] {path.name}: `{field_name}` must be a non-empty string.")
    return _normalize_whitespace(value)


def _optional_string(raw: Dict[str, Any], field_name: str, path: Path, default: str = "") -> str:
    value = raw.get(field_name, default)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DatasetError(f"[IAT] {path.name}: `{field_name}` must be a string when provided.")
    return _normalize_whitespace(value)


def _optional_enum_list(
    raw: Dict[str, Any],
    field_name: str,
    path: Path,
    allowed: Sequence[str],
) -> List[str]:
    value = raw.get(field_name)
    if value is None:
        return []
    values = _string_list(value, field_name, path, required=False)
    normalized = [item.casefold() for item in values]
    invalid = [item for item in normalized if item not in set(allowed)]
    if invalid:
        raise DatasetError(
            f"[IAT] {path.name}: `{field_name}` contains unsupported values: {', '.join(invalid)}."
        )
    return list(dict.fromkeys(normalized))


def _sample_metadata(caption: str) -> Dict[str, Any]:
    return {
        "sample_types": classify_caption(caption),
        "materials": extract_materials(caption),
        "components": extract_components(caption),
    }


def _caption_path_for_image(image_path: Path) -> Optional[Path]:
    caption_path = image_path.with_suffix(".txt")
    if not caption_path.is_file():
        # Windows datasets sometimes use an upper-case extension.
        for candidate in image_path.parent.glob(f"{image_path.stem}.*"):
            if candidate.suffix.lower() == ".txt":
                caption_path = candidate
                break
    if not caption_path.is_file():
        return None
    return caption_path


def _caption_for_image(image_path: Path) -> Optional[str]:
    caption_path = _caption_path_for_image(image_path)
    if caption_path is None:
        return None
    return _normalize_whitespace(caption_path.read_text(encoding="utf-8-sig"))


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise DatasetError(f"[IAT] Failed to parse dataset metadata `{path}`: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetError(f"[IAT] Dataset metadata `{path}` must contain a JSON object.")
    return value


def _role_from_directory(directory: Path, dataset_dir: Path) -> Optional[str]:
    try:
        relative_parts = directory.relative_to(dataset_dir).parts
    except ValueError:
        return None
    for part in reversed(relative_parts):
        normalized = part.strip().casefold()
        if normalized == "result" or normalized.endswith("_result") or normalized.endswith("-result"):
            return "result"
        match = _CONTROL_ROLE_RE.search(normalized)
        if match:
            return f"control{match.group(1)}"
    return None


def _role_directories(dataset_dir: Path, allowed_roles: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    allowed = set(allowed_roles or _IMAGE_ROLES)
    role_dirs: Dict[str, Path] = {}
    for directory in sorted((path for path in dataset_dir.rglob("*") if path.is_dir()), key=lambda path: path.as_posix().lower()):
        role = _role_from_directory(directory, dataset_dir)
        if role in allowed and role not in role_dirs:
            role_dirs[role] = directory
    return role_dirs


def _make_entry(
    dataset_dir: Path,
    record_id: str,
    caption: str,
    image_paths: Dict[str, Path],
) -> DatasetEntry:
    relative_paths = {
        role: path.relative_to(dataset_dir).as_posix()
        for role, path in sorted(image_paths.items())
    }
    primary_role = "result" if "result" in image_paths else next(iter(relative_paths), "")
    primary_path = image_paths.get(primary_role)
    return DatasetEntry(
        record_id=record_id,
        caption=caption,
        image_path=primary_path,
        relative_image_path=relative_paths.get(primary_role, ""),
        image_paths=dict(sorted(image_paths.items())),
        relative_image_paths=relative_paths,
        metadata=_sample_metadata(caption),
    )


def _build_entries_from_multiview(
    dataset_dir: Path,
    role_dirs: Dict[str, Path],
    warnings: List[str],
    caption_role: str,
) -> List[DatasetEntry]:
    groups: Dict[str, Dict[str, Path]] = {}
    for role, role_dir in role_dirs.items():
        for image_path in sorted(
            (path for path in role_dir.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
            key=lambda path: path.as_posix().lower(),
        ):
            group = groups.setdefault(image_path.stem.casefold(), {})
            if role in group:
                warnings.append(f"Duplicate `{role}` image for sample `{image_path.stem}`; kept the first file.")
                continue
            group[role] = image_path

    entries: List[DatasetEntry] = []
    for group_key in sorted(groups, key=str.casefold):
        image_paths = groups[group_key]
        record_id = image_paths.get(caption_role, next(iter(image_paths.values()))).stem
        caption_image = image_paths.get(caption_role)
        if caption_image is None:
            warnings.append(f"Missing `{caption_role}` image for sample `{record_id}`; skipped.")
            continue
        caption = _caption_for_image(caption_image)
        if not caption:
            warnings.append(
                f"Missing `{caption_role}` caption for sample `{caption_image.relative_to(dataset_dir).as_posix()}`; skipped."
            )
            continue
        for role in ("control1", "control2", "control3"):
            if role not in image_paths:
                warnings.append(f"Missing `{role}` image for sample `{record_id}`; kept result sample.")
        entries.append(_make_entry(dataset_dir, record_id, caption, image_paths))
    return entries


def _build_entries_from_directory(
    dataset_dir: Path,
    warnings: List[str],
    configured_roles: Optional[Sequence[str]] = None,
    caption_role: str = "result",
) -> List[DatasetEntry]:
    role_dirs = _role_directories(dataset_dir, configured_roles)
    if role_dirs:
        return _build_entries_from_multiview(dataset_dir, role_dirs, warnings, caption_role)

    image_dir = dataset_dir / "images"
    if not image_dir.is_dir():
        raise DatasetError(
            f"[IAT] Dataset `{dataset_dir}` is missing `images` or a recognized result directory."
        )
    root = image_dir
    entries: List[DatasetEntry] = []
    for image_path in sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda path: path.as_posix().lower(),
    ):
        caption = _caption_for_image(image_path)
        if not caption:
            warnings.append(f"Missing caption for image `{image_path.relative_to(dataset_dir).as_posix()}`; skipped.")
            continue
        relative = image_path.relative_to(dataset_dir).as_posix()
        entries.append(_make_entry(dataset_dir, Path(relative).with_suffix("").as_posix(), caption, {"image": image_path}))
    return entries


def load_dataset_record(path: Path) -> DatasetRecord:
    """Load one strict directory dataset with paired image/caption files."""
    path = Path(path)
    if not path.is_dir():
        raise DatasetError(f"[IAT] Dataset path must be a directory: `{path}`")
    source_path = path / "dataset.json"
    if not source_path.is_file():
        raise DatasetError(f"[IAT] Dataset directory `{path}` is missing dataset.json.")
    raw = _load_json(source_path)

    dataset_name = _required_string(raw, "dataset_name", source_path)
    version = _required_string(raw, "version", source_path)
    base_model = _required_string(raw, "base_model", source_path)
    lora_name = _required_string(raw, "lora_name", source_path)
    language = _required_string(raw, "language", source_path).lower()
    if language not in {"zh", "en", "ja"}:
        raise DatasetError(f"[IAT] {source_path.name}: `language` must be one of zh, en, or ja.")
    trigger_words = _string_list(raw.get("trigger_words"), "trigger_words", source_path, required=True)
    dataset_kind = _optional_string(raw, "dataset_kind", source_path, "cmf") or "cmf"
    style_id = _optional_string(raw, "style_id", source_path)
    scope = (_optional_string(raw, "scope", source_path, "full_cabin") or "full_cabin").casefold()
    if scope not in set(SAMPLE_TYPES):
        raise DatasetError(f"[IAT] {source_path.name}: `scope` must be one of {', '.join(SAMPLE_TYPES)}.")
    declared_sample_types = _optional_enum_list(raw, "sample_types", source_path, SAMPLE_TYPES)
    supported_material_values = _string_list(raw.get("supported_materials"), "supported_materials", source_path)
    supported_materials: List[str] = []
    for value in supported_material_values:
        normalized_material = normalize_material(value)
        if normalized_material is None:
            raise DatasetError(f"[IAT] {source_path.name}: unsupported material `{value}`.")
        if normalized_material not in supported_materials:
            supported_materials.append(normalized_material)
    supported_color_values = _string_list(
        raw.get("supported_color_families"), "supported_color_families", source_path
    )
    supported_color_families: List[str] = []
    for value in supported_color_values:
        normalized_family = normalize_family(value)
        if normalized_family is None:
            raise DatasetError(f"[IAT] {source_path.name}: unsupported color family `{value}`.")
        if normalized_family not in supported_color_families:
            supported_color_families.append(normalized_family)
    configured_roles = _string_list(raw.get("image_roles"), "image_roles", source_path, required=False)
    if configured_roles:
        configured_roles = [role.casefold() for role in configured_roles]
        invalid_roles = [role for role in configured_roles if role not in _IMAGE_ROLES]
        if invalid_roles:
            raise DatasetError(
                f"[IAT] {source_path.name}: `image_roles` contains unsupported roles: {', '.join(invalid_roles)}."
            )
        configured_roles = list(dict.fromkeys(configured_roles))
    caption_role = raw.get("caption_role", "result")
    if not isinstance(caption_role, str) or not caption_role.strip():
        raise DatasetError(f"[IAT] {source_path.name}: `caption_role` must be a non-empty string when provided.")
    caption_role = _normalize_whitespace(caption_role).casefold()
    if caption_role not in _IMAGE_ROLES:
        raise DatasetError(f"[IAT] {source_path.name}: `caption_role` must be one of {', '.join(_IMAGE_ROLES)}.")

    warnings: List[str] = []
    entries = _build_entries_from_directory(path, warnings, configured_roles or None, caption_role)
    if not entries:
        raise DatasetError(f"[IAT] Dataset `{dataset_name}` has no valid image/caption entries.")

    entry_type_counts: Dict[str, int] = {}
    for entry in entries:
        for sample_type in entry.sample_types:
            entry_type_counts[sample_type] = entry_type_counts.get(sample_type, 0) + 1
    derived_sample_types = sorted(set(entry_type_counts) | set(declared_sample_types))
    metadata = {
        "dataset_name": dataset_name,
        "version": version,
        "base_model": base_model,
        "lora_name": lora_name,
        "language": language,
        "trigger_words": trigger_words,
        "dataset_kind": dataset_kind,
        "style_id": style_id,
        "scope": scope,
        "sample_types": derived_sample_types,
        "sample_classification_version": SAMPLE_CLASSIFICATION_VERSION,
        "declared_sample_types": declared_sample_types,
        "sample_type_counts": entry_type_counts,
        "supported_materials": supported_materials,
        "supported_color_families": supported_color_families,
        "image_roles": configured_roles,
        "caption_role": caption_role,
        "entry_count": len(entries),
        "source_path": str(source_path),
        "warnings": warnings,
    }
    return DatasetRecord(
        dataset_name=dataset_name,
        version=version,
        base_model=base_model,
        lora_name=lora_name,
        language=language,
        trigger_words=trigger_words,
        entries=entries,
        source_path=source_path,
        warnings=warnings,
        metadata=metadata,
    )


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    import numpy as np

    values = np.asarray(vector, dtype="<f4")
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise DatasetError("[IAT] Cannot store an empty or non-finite embedding vector.")
    return values.tobytes(order="C")


def _vector_from_blob(value: bytes, dimension: int, label: str) -> List[float]:
    import numpy as np

    if not isinstance(value, bytes) or len(value) != dimension * 4:
        raise DatasetError(f"[IAT] Compiled dataset has an invalid {label} vector payload.")
    vector = np.frombuffer(value, dtype="<f4")
    if vector.size != dimension or not np.isfinite(vector).all():
        raise DatasetError(f"[IAT] Compiled dataset has an invalid {label} vector.")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        raise DatasetError(f"[IAT] Compiled dataset has a zero-length {label} vector.")
    return (vector / norm).tolist()


def _read_bundle_metadata(connection: sqlite3.Connection, path: Path) -> Dict[str, Any]:
    try:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.Error as exc:
        raise DatasetError(f"[IAT] Compiled dataset `{path}` is missing its metadata table: {exc}") from exc
    metadata: Dict[str, Any] = {}
    for key, value in rows:
        try:
            metadata[str(key)] = json.loads(value)
        except Exception as exc:
            raise DatasetError(f"[IAT] Compiled dataset `{path}` has invalid metadata `{key}`: {exc}") from exc
    if metadata.get("format") != _BUNDLE_FORMAT or metadata.get("schema_version") != _BUNDLE_SCHEMA_VERSION:
        raise DatasetError(
            f"[IAT] Unsupported compiled dataset format/schema in `{path}`: "
            f"{metadata.get('format')!r}/{metadata.get('schema_version')!r}."
        )
    return metadata


def load_dataset_bundle(path: Path) -> DatasetRecord:
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() != ".iatdb":
        raise DatasetError(f"[IAT] Compiled dataset must be an existing .iatdb file: `{path}`")
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise DatasetError(f"[IAT] Could not open compiled dataset `{path}`: {exc}") from exc
    try:
        integrity = connection.execute("PRAGMA quick_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise DatasetError(f"[IAT] Compiled dataset `{path}` failed SQLite integrity check: {integrity}")
        metadata = _read_bundle_metadata(connection, path)
        dimension = int(metadata.get("embedding_dimension") or 0)
        if dimension <= 0:
            raise DatasetError(f"[IAT] Compiled dataset `{path}` has an invalid embedding dimension.")
        try:
            rows = connection.execute(
                "SELECT ordinal, record_id, caption, entry_metadata, text_embedding, image_embedding, gray_embedding "
                "FROM chunks ORDER BY ordinal"
            ).fetchall()
        except sqlite3.Error as exc:
            raise DatasetError(f"[IAT] Compiled dataset `{path}` is missing its chunks table: {exc}") from exc
    finally:
        connection.close()

    expected_count = int(metadata.get("entry_count") or 0)
    if not rows or len(rows) != expected_count:
        raise DatasetError(
            f"[IAT] Compiled dataset `{path}` expected {expected_count} chunks but contains {len(rows)}."
        )
    entries: List[DatasetEntry] = []
    text_embeddings: List[Optional[List[float]]] = []
    image_embeddings: List[Optional[List[float]]] = []
    gray_embeddings: List[Optional[List[float]]] = []
    seen_ids = set()
    for expected_ordinal, row in enumerate(rows):
        ordinal, record_id, caption, entry_metadata_blob, text_blob, image_blob, gray_blob = row
        if ordinal != expected_ordinal:
            raise DatasetError(f"[IAT] Compiled dataset `{path}` has non-contiguous chunk ordinals.")
        if not isinstance(record_id, str) or not record_id.strip() or record_id in seen_ids:
            raise DatasetError(f"[IAT] Compiled dataset `{path}` has an empty or duplicate record_id.")
        if not isinstance(caption, str) or not caption.strip():
            raise DatasetError(f"[IAT] Compiled dataset `{path}` has an empty caption for `{record_id}`.")
        try:
            entry_metadata = json.loads(entry_metadata_blob)
        except Exception as exc:
            raise DatasetError(f"[IAT] Compiled dataset `{path}` has invalid metadata for `{record_id}`: {exc}") from exc
        if not isinstance(entry_metadata, dict):
            raise DatasetError(f"[IAT] Compiled dataset `{path}` has non-object metadata for `{record_id}`.")
        sample_types = entry_metadata.get("sample_types")
        if not isinstance(sample_types, list) or not sample_types or any(
            str(value) not in SAMPLE_TYPES for value in sample_types
        ):
            raise DatasetError(f"[IAT] Compiled dataset `{path}` has invalid sample types for `{record_id}`.")
        seen_ids.add(record_id)
        entries.append(
            DatasetEntry(
                record_id=record_id,
                caption=_normalize_whitespace(caption),
                metadata={
                    **entry_metadata,
                    "sample_types": list(dict.fromkeys(str(value) for value in sample_types)),
                    "materials": [
                        str(value) for value in entry_metadata.get("materials", [])
                        if isinstance(value, str) and value.strip()
                    ],
                    "components": [
                        str(value) for value in entry_metadata.get("components", [])
                        if isinstance(value, str) and value.strip()
                    ],
                },
            )
        )
        text_embeddings.append(_vector_from_blob(text_blob, dimension, "text"))
        image_embeddings.append(_vector_from_blob(image_blob, dimension, "image"))
        gray_embeddings.append(_vector_from_blob(gray_blob, dimension, "grayscale image"))

    record_metadata = {
        "dataset_name": _required_string(metadata, "dataset_name", path),
        "version": _required_string(metadata, "version", path),
        "base_model": _required_string(metadata, "base_model", path),
        "lora_name": _required_string(metadata, "lora_name", path),
        "language": _required_string(metadata, "language", path).lower(),
        "trigger_words": _string_list(metadata.get("trigger_words"), "trigger_words", path, required=True),
        "dataset_kind": str(metadata.get("dataset_kind") or "cmf"),
        "style_id": str(metadata.get("style_id") or ""),
        "scope": str(metadata.get("scope") or "full_cabin"),
        "sample_types": list(metadata.get("sample_types") or []),
        "sample_classification_version": int(metadata.get("sample_classification_version") or 1),
        "declared_sample_types": list(metadata.get("declared_sample_types") or []),
        "sample_type_counts": dict(metadata.get("sample_type_counts") or {}),
        "supported_materials": list(metadata.get("supported_materials") or []),
        "supported_color_families": list(metadata.get("supported_color_families") or []),
        "entry_count": len(entries),
        "source_path": str(path),
        "compiled": True,
        "embedding_provider": str(metadata.get("embedding_provider") or ""),
        "embedding_dimension": dimension,
        "embedding_model_signature": str(metadata.get("embedding_model_signature") or ""),
        "embedding_query_instruction": str(metadata.get("embedding_query_instruction") or ""),
        "embedding_document_instruction": str(metadata.get("embedding_document_instruction") or ""),
        "source_fingerprint": str(metadata.get("source_fingerprint") or ""),
        "warnings": [],
    }
    if record_metadata["language"] not in {"zh", "en", "ja"}:
        raise DatasetError(f"[IAT] Compiled dataset `{path}` has an unsupported language.")
    return DatasetRecord(
        dataset_name=record_metadata["dataset_name"],
        version=record_metadata["version"],
        base_model=record_metadata["base_model"],
        lora_name=record_metadata["lora_name"],
        language=record_metadata["language"],
        trigger_words=record_metadata["trigger_words"],
        entries=entries,
        source_path=path,
        warnings=[],
        metadata=record_metadata,
        bundle_path=path,
        bundled_text_embeddings=text_embeddings,
        bundled_image_embeddings=image_embeddings,
        bundled_gray_embeddings=gray_embeddings,
    )


def dataset_bundle_is_current(index: "DatasetIndex", output_path: Path) -> bool:
    output_path = Path(output_path)
    if not output_path.is_file():
        return False
    try:
        connection = sqlite3.connect(f"{output_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            metadata = _read_bundle_metadata(connection, output_path)
        finally:
            connection.close()
    except Exception:
        return False
    expected_dimension = len(index.text_embeddings[0]) if index.text_embeddings else 0
    return all(
        (
            metadata.get("source_fingerprint") == index.fingerprint,
            metadata.get("embedding_provider") == index.embedding_provider,
            metadata.get("embedding_model_signature") == index.model_signature,
            int(metadata.get("embedding_dimension") or 0) == expected_dimension,
            metadata.get("embedding_query_instruction") == index.query_instruction,
            metadata.get("embedding_document_instruction") == index.document_instruction,
            int(metadata.get("sample_classification_version") or 0) == SAMPLE_CLASSIFICATION_VERSION,
            int(metadata.get("entry_count") or 0) == len(index.record.entries),
        )
    )


def dataset_bundle_matches_source(
    record: DatasetRecord,
    output_path: Path,
    model_path: str,
    provider: str,
    dimension: int,
    query_instruction: str,
    document_instruction: str,
) -> bool:
    output_path = Path(output_path)
    if not output_path.is_file():
        return False
    try:
        resolved_provider = detect_embedding_provider(model_path, provider)
        signature = embedding_model_signature(model_path, resolved_provider)
        source_fingerprint = dataset_fingerprint(record)
        connection = sqlite3.connect(f"{output_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            metadata = _read_bundle_metadata(connection, output_path)
        finally:
            connection.close()
    except Exception:
        return False
    stored_dimension = int(metadata.get("embedding_dimension") or 0)
    return all(
        (
            metadata.get("dataset_name") == record.dataset_name,
            metadata.get("source_fingerprint") == source_fingerprint,
            metadata.get("embedding_provider") == resolved_provider,
            metadata.get("embedding_model_signature") == signature,
            not dimension or stored_dimension == int(dimension),
            metadata.get("embedding_query_instruction") == query_instruction,
            metadata.get("embedding_document_instruction") == document_instruction,
            int(metadata.get("sample_classification_version") or 0) == SAMPLE_CLASSIFICATION_VERSION,
            int(metadata.get("entry_count") or 0) == len(record.entries),
        )
    )


def write_dataset_bundle(index: "DatasetIndex", output_path: Path) -> Path:
    output_path = Path(output_path).resolve()
    entries = index.record.entries
    vector_sets = (index.text_embeddings, index.image_embeddings, index.gray_embeddings)
    if not entries or any(len(vectors) != len(entries) for vectors in vector_sets):
        raise DatasetError("[IAT] A compiled dataset requires complete text, image, and grayscale vectors.")
    if any(vector is None for vectors in vector_sets for vector in vectors):
        raise DatasetError("[IAT] A compiled dataset cannot contain missing vectors.")
    dimension = len(index.text_embeddings[0] or [])
    if dimension <= 0 or any(len(vector or []) != dimension for vectors in vector_sets for vector in vectors):
        raise DatasetError("[IAT] A compiled dataset requires one consistent non-zero embedding dimension.")
    if not index.model_signature or not index.embedding_provider:
        raise DatasetError("[IAT] A compiled dataset requires a resolved embedding model identity.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    metadata = {
        "format": _BUNDLE_FORMAT,
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "dataset_name": index.record.dataset_name,
        "version": index.record.version,
        "base_model": index.record.base_model,
        "lora_name": index.record.lora_name,
        "language": index.record.language,
        "trigger_words": index.record.trigger_words,
        "entry_count": len(entries),
        "source_fingerprint": index.fingerprint,
        "embedding_provider": index.embedding_provider,
        "embedding_dimension": dimension,
        "embedding_model_signature": index.model_signature,
        "embedding_query_instruction": index.query_instruction,
        "embedding_document_instruction": index.document_instruction,
    }
    for key in (
        "dataset_kind",
        "style_id",
        "scope",
        "sample_types",
        "sample_classification_version",
        "declared_sample_types",
        "sample_type_counts",
        "supported_materials",
        "supported_color_families",
    ):
        if key in index.record.metadata:
            metadata[key] = index.record.metadata[key]
    try:
        connection = sqlite3.connect(str(temp_path))
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE chunks ("
                "ordinal INTEGER PRIMARY KEY, record_id TEXT NOT NULL UNIQUE, caption TEXT NOT NULL, "
                "entry_metadata TEXT NOT NULL, "
                "text_embedding BLOB NOT NULL, image_embedding BLOB NOT NULL, gray_embedding BLOB NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [(key, json.dumps(value, ensure_ascii=False, separators=(",", ":"))) for key, value in metadata.items()],
            )
            connection.executemany(
                "INSERT INTO chunks(ordinal, record_id, caption, entry_metadata, text_embedding, image_embedding, gray_embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        ordinal,
                        entry.record_id,
                        entry.caption,
                        json.dumps(
                            entry.metadata or _sample_metadata(entry.caption),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        _vector_to_blob(index.text_embeddings[ordinal] or []),
                        _vector_to_blob(index.image_embeddings[ordinal] or []),
                        _vector_to_blob(index.gray_embeddings[ordinal] or []),
                    )
                    for ordinal, entry in enumerate(entries)
                ],
            )
            connection.execute(f"PRAGMA user_version={_BUNDLE_SCHEMA_VERSION}")
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise DatasetError(f"[IAT] New compiled dataset failed integrity check: {integrity}")
        finally:
            connection.close()
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return output_path


def discover_datasets(root: Path) -> Tuple[Dict[str, DatasetRecord], List[str]]:
    """Discover only directories containing a canonical ``dataset.json``."""
    root = Path(root)
    records: Dict[str, DatasetRecord] = {}
    duplicate_names = set()
    errors: List[str] = []
    if not root.is_dir():
        return records, [f"[IAT] Dataset root does not exist: `{root}`"]

    candidates = sorted(
        (directory for directory in root.rglob("*") if directory.is_dir() and (directory / "dataset.json").is_file()),
        key=lambda path: path.as_posix().lower(),
    )

    for candidate in candidates:
        try:
            record = load_dataset_record(candidate)
        except Exception as exc:
            errors.append(str(exc))
            continue
        if record.dataset_name in duplicate_names:
            errors.append(
                f"[IAT] Duplicate dataset_name `{record.dataset_name}` in `{record.source_path}`."
            )
            continue
        if record.dataset_name in records:
            errors.append(
                f"[IAT] Duplicate dataset_name `{record.dataset_name}` in `{records[record.dataset_name].source_path}` and `{record.source_path}`."
            )
            records.pop(record.dataset_name, None)
            duplicate_names.add(record.dataset_name)
            continue
        records[record.dataset_name] = record

    bundle_records: Dict[str, DatasetRecord] = {}
    duplicate_bundle_names = set()
    for bundle_path in sorted(root.rglob("*.iatdb"), key=lambda path: path.as_posix().lower()):
        try:
            bundle_record = load_dataset_bundle(bundle_path)
        except Exception as exc:
            errors.append(str(exc))
            continue
        name = bundle_record.dataset_name
        if name in duplicate_bundle_names:
            errors.append(f"[IAT] Duplicate compiled dataset_name `{name}` in `{bundle_path}`.")
            continue
        if name in bundle_records:
            errors.append(
                f"[IAT] Duplicate compiled dataset_name `{name}` in "
                f"`{bundle_records[name].source_path}` and `{bundle_path}`."
            )
            bundle_records.pop(name, None)
            duplicate_bundle_names.add(name)
            continue
        bundle_records[name] = bundle_record

    # A compiled bundle is the portable runtime artifact and intentionally takes
    # precedence over its colocated authoring directory.
    records.update(bundle_records)
    return records, errors


def choose_caption(record: DatasetRecord, mode: str, seed: int, index: int = 0) -> Tuple[DatasetEntry, int]:
    if not record.entries:
        raise DatasetError(f"[IAT] Dataset `{record.dataset_name}` has no captions.")
    if mode == "Random":
        selected_index = random.Random(int(seed)).randrange(len(record.entries))
    elif mode == "By Index":
        selected_index = int(index) % len(record.entries)
    else:
        selected_index = int(seed) % len(record.entries)
    return record.entries[selected_index], selected_index


def dataset_fingerprint(record: DatasetRecord) -> str:
    # Hash content rather than mtimes so the same dataset version remains stable
    # after a touch/copy operation while still invalidating changed image bytes.
    digest = hashlib.sha256()
    if record.bundle_path is not None:
        try:
            with record.bundle_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise DatasetError(f"[IAT] Could not read compiled dataset `{record.bundle_path}`: {exc}") from exc
        return digest.hexdigest()
    dataset_dir = record.source_path.parent
    digest.update(record.source_path.name.encode("utf-8"))
    digest.update(record.source_path.read_bytes())
    tracked_suffixes = _IMAGE_SUFFIXES | {".txt"}
    for path in sorted(
        (
            item
            for item in dataset_dir.rglob("*")
            if item.is_file() and item.suffix.lower() in tracked_suffixes and item != record.source_path
        ),
        key=lambda item: item.as_posix().lower(),
    ):
        relative = path.relative_to(dataset_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise DatasetError(f"[IAT] Could not read dataset file `{path}` while computing its fingerprint: {exc}") from exc
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"
    if safe != value:
        safe = f"{safe}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"
    return safe


def _cosine(left: Optional[Sequence[float]], right: Optional[Sequence[float]]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    value = sum(float(a) * float(b) for a, b in zip(left, right))
    return max(-1.0, min(1.0, value))


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-8:
        return [0.0 if abs(high) < 1e-8 else 1.0 for _ in scores]
    return [(value - low) / (high - low) for value in scores]


def _mean_vector(vectors: Sequence[Optional[Sequence[float]]]) -> Optional[List[float]]:
    usable = [list(vector) for vector in vectors if vector]
    if not usable:
        return None
    dimension = len(usable[0])
    if dimension == 0 or any(len(vector) != dimension for vector in usable):
        return None
    values = [sum(float(vector[index]) for vector in usable) / len(usable) for index in range(dimension)]
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        return [0.0] * dimension
    return [value / norm for value in values]


def _exploration_profile(value: str) -> Tuple[str, Dict[str, float]]:
    normalized = (value or "Medium").strip().title()
    if normalized not in _EXPLORATION_PROFILES:
        normalized = "Medium"
    return normalized, _EXPLORATION_PROFILES[normalized]


def _seeded_weighted_choice(
    candidates: Sequence[int],
    utilities: Sequence[float],
    rng: random.Random,
    sampling_temperature: float,
) -> int:
    if not candidates:
        raise DatasetError("[IAT] Cannot choose from an empty retrieval candidate set.")
    scale = max(float(sampling_temperature), 1e-6)
    maximum = max(utilities)
    weights = [math.exp((utility - maximum) / scale) for utility in utilities]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return candidates[rng.randrange(len(candidates))]
    target = rng.random() * total
    for candidate, weight in zip(candidates, weights):
        target -= weight
        if target <= 0.0:
            return candidate
    return candidates[-1]


class DatasetIndex:
    def __init__(
        self,
        record: DatasetRecord,
        fingerprint: str,
        text_embeddings: Optional[List[Optional[List[float]]]] = None,
        image_embeddings: Optional[List[Optional[List[float]]]] = None,
        gray_embeddings: Optional[List[Optional[List[float]]]] = None,
        embedding_model_path: str = "",
        embedding_device: str = "cpu",
        embedding_provider: str = "auto",
        embedding_batch_size: int = 1,
        embedding_dimension: int = 0,
        query_instruction: str = "",
        document_instruction: str = "",
        model_signature: str = "",
        warnings: Optional[List[str]] = None,
    ):
        self.record = record
        self.fingerprint = fingerprint
        self.text_embeddings = text_embeddings or []
        self.image_embeddings = image_embeddings or []
        self.gray_embeddings = gray_embeddings or []
        self.embedding_model_path = embedding_model_path
        self.embedding_device = embedding_device
        self.embedding_provider = embedding_provider
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self.embedding_dimension = max(0, int(embedding_dimension))
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.model_signature = model_signature
        self.warnings = list(warnings or [])
        self.tokens = [tokenize(entry.caption) for entry in record.entries]
        self.document_frequency: Dict[str, int] = {}
        for tokens in self.tokens:
            for token in set(tokens):
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1
        # Precompute the existing BM25 formula; queries only visit matching postings.
        document_count = len(self.tokens) or 1
        average_length = sum(map(len, self.tokens)) / document_count
        postings: Dict[str, list] = {}
        for ordinal, tokens in enumerate(self.tokens):
            length_factor = len(tokens) / max(average_length, 1.0)
            for token, frequency in Counter(tokens).items():
                df = self.document_frequency[token]
                idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
                weight = idf * (frequency * 2.5 / (frequency + 1.5 * (0.75 + 0.25 * length_factor)))
                postings.setdefault(token, []).append((ordinal, weight))
        self._postings = {
            token: (np.asarray([row[0] for row in rows], dtype=np.intp),
                    np.asarray([row[1] for row in rows], dtype=np.float64))
            for token, rows in postings.items()
        }
        self._matrices: Dict[str, np.ndarray] = {}

    @property
    def version(self) -> str:
        return f"hybrid-v5:{self.fingerprint[:12]}"

    def _bm25_scores(self, query: str) -> List[float]:
        scores = np.zeros(len(self.record.entries), dtype=np.float64)
        for token, count in Counter(tokenize(query)).items():
            posting = self._postings.get(token)
            if posting is not None:
                ordinals, weights = posting
                scores[ordinals] += weights * (1.0 + math.log1p(count))
        return scores.tolist()

    def _vector_scores(self, query: Optional[Sequence[float]], kind: str) -> List[float]:
        vectors = getattr(self, f"{kind}_embeddings")
        count = len(self.record.entries)
        if query is None or not vectors:
            return [0.0] * count
        matrix = self._matrices.get(kind)
        if matrix is None:
            if len(vectors) != count:
                raise DatasetError(f"[IAT] {kind} vector count does not match dataset entries.")
            dimension = next((len(vector) for vector in vectors if vector is not None), 0)
            matrix = np.zeros((count, dimension), dtype=np.float32)
            for ordinal, vector in enumerate(vectors):
                if vector is None:
                    continue
                if len(vector) != dimension or not dimension:
                    raise DatasetError(f"[IAT] Inconsistent {kind} embedding dimensions.")
                matrix[ordinal] = vector
            if not np.isfinite(matrix).all():
                raise DatasetError(f"[IAT] Non-finite {kind} embeddings.")
            self._matrices[kind] = matrix
        if not matrix.shape[1]:
            return [0.0] * count
        query_vector = np.asarray(query, dtype=np.float32)
        if query_vector.shape != (matrix.shape[1],) or not np.isfinite(query_vector).all():
            raise DatasetError(f"[IAT] Invalid query vector for {kind} embeddings.")
        return np.clip(matrix @ query_vector, -1.0, 1.0).tolist()

    def _similarity_to_selected(self, left: int, right: int) -> float:
        if self.text_embeddings and left < len(self.text_embeddings) and right < len(self.text_embeddings):
            return max(0.0, _cosine(self.text_embeddings[left], self.text_embeddings[right]))
        left_tokens, right_tokens = set(self.tokens[left]), set(self.tokens[right])
        return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)

    def _entry_materials(self, index: int) -> List[str]:
        entry = self.record.entries[index]
        values = entry.metadata.get("materials") if isinstance(entry.metadata, dict) else None
        if not isinstance(values, list) or not values:
            values = extract_materials(entry.caption)
        normalized: List[str] = []
        for value in values:
            material = normalize_material(value)
            if material and material not in normalized:
                normalized.append(material)
        return normalized

    def _material_coverage_indices(
        self,
        ranked_candidates: Sequence[int],
        required_materials: Sequence[str],
    ) -> Tuple[List[int], Dict[str, List[str]]]:
        """Reserve the best available sample for each requested material.

        The semantic rank remains the primary ordering signal.  This only
        prevents a heavily represented material from consuming the entire
        small reference set.
        """
        normalized = []
        for value in required_materials:
            material = normalize_material(value)
            if material and material not in normalized:
                normalized.append(material)
        if not normalized:
            return [], {}

        candidates_by_material: Dict[str, List[int]] = {material: [] for material in normalized}
        for index in ranked_candidates:
            entry_materials = set(self._entry_materials(index))
            for material in normalized:
                if material in entry_materials:
                    candidates_by_material[material].append(index)

        reserved: List[int] = []
        for material in normalized:
            material_candidates = candidates_by_material[material]
            preferred = [
                index for index in material_candidates
                if "full_cabin" not in (
                    self.record.entries[index].sample_types
                    or classify_caption(self.record.entries[index].caption)
                )
            ]
            for index in preferred + material_candidates:
                if index not in reserved:
                    reserved.append(index)
                    break
        return reserved, {
            material: [self.record.entries[index].record_id for index in indexes[:3]]
            for material, indexes in candidates_by_material.items()
        }

    def retrieve(
        self,
        query: str,
        reference_image: Any = None,
        reference_images: Optional[Sequence[Any]] = None,
        preserve_reference_color: bool = False,
        top_k: int = 4,
        candidate_k: int = 16,
        seed: int = 0,
        exploration_strength: str = "Medium",
        required_materials: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        top_k = max(1, min(8, int(top_k)))
        exploration_name, profile = _exploration_profile(exploration_strength)
        candidate_k = max(top_k, min(int(candidate_k), int(profile["candidate_k"])))
        references = list(reference_images or [])
        if reference_image is not None:
            references.insert(0, reference_image)
        bm25 = self._bm25_scores(query)
        text_query = (
            _encode_text(
                self.embedding_model_path,
                query,
                device=self.embedding_device,
                provider=self.embedding_provider,
                batch_size=self.embedding_batch_size,
                dimension=self.embedding_dimension,
                instruction=self.query_instruction,
            )
            if self.embedding_model_path
            else None
        )
        image_query = None
        if references and self.embedding_model_path:
            if len(references) == 1:
                image_query = _encode_image(
                    self.embedding_model_path,
                    references[0],
                    grayscale=not preserve_reference_color,
                    device=self.embedding_device,
                    provider=self.embedding_provider,
                    batch_size=self.embedding_batch_size,
                    dimension=self.embedding_dimension,
                    instruction=self.query_instruction,
                )
            else:
                image_query = _mean_vector(
                    _encode_image_batch(
                        self.embedding_model_path,
                        references,
                        self.embedding_device,
                        min(self.embedding_batch_size, len(references)),
                        grayscale=not preserve_reference_color,
                        provider=self.embedding_provider,
                        dimension=self.embedding_dimension,
                        instruction=self.query_instruction,
                    )
                )

        text_scores = self._vector_scores(text_query, "text")
        image_scores = self._vector_scores(image_query, "image" if preserve_reference_color else "gray")

        if references and self.embedding_model_path:
            weights = {"image": 0.45, "text": 0.35, "bm25": 0.20}
        elif self.embedding_model_path:
            weights = {"image": 0.0, "text": 0.65, "bm25": 0.35}
        else:
            weights = {"image": 0.0, "text": 0.0, "bm25": 1.0}
        normalized_bm25 = _normalize_scores(bm25)
        normalized_text = _normalize_scores(text_scores)
        normalized_image = _normalize_scores(image_scores)
        combined = [
            weights["image"] * normalized_image[idx]
            + weights["text"] * normalized_text[idx]
            + weights["bm25"] * normalized_bm25[idx]
            for idx in range(len(self.record.entries))
        ]
        # The seed controls only deterministic sampling within the relevant pool.
        # It never changes the semantic scores or allows candidates outside the pool.
        tie_rng = random.Random(int(seed))
        tie_breakers = {idx: tie_rng.random() for idx in range(len(combined))}
        ranked_candidates = sorted(
            range(len(combined)),
            key=lambda idx: (combined[idx], tie_breakers[idx], self.record.entries[idx].record_id),
            reverse=True,
        )
        candidates = ranked_candidates[:candidate_k]
        candidate_pool = list(candidates)

        material_reserved, material_candidates = self._material_coverage_indices(
            ranked_candidates,
            required_materials or (),
        )
        selected: List[int] = []
        # Keep one complete-cabin reference when there is room after reserving
        # material examples.  Detail/color-material samples then complement it.
        if material_reserved and len(material_reserved) < top_k:
            full_cabin = next(
                (
                    index for index in ranked_candidates
                    if "full_cabin" in (
                        self.record.entries[index].sample_types
                        or classify_caption(self.record.entries[index].caption)
                    )
                    and index not in material_reserved
                ),
                None,
            )
            if full_cabin is None and str(self.record.metadata.get("scope") or "").casefold() == "full_cabin":
                # Some legacy CMF captions describe only the palette while the
                # dataset metadata still guarantees full-cabin result images.
                full_cabin = next(
                    (index for index in ranked_candidates if index not in material_reserved),
                    None,
                )
            if full_cabin is not None:
                material_reserved.append(full_cabin)
        selected.extend(material_reserved[:top_k])
        candidates = [index for index in candidates if index not in selected]
        while candidates and len(selected) < top_k:
            utilities = [
                profile["relevance"] * combined[idx]
                - (
                    profile["diversity"]
                    * max((self._similarity_to_selected(idx, other) for other in selected), default=0.0)
                )
                for idx in candidates
            ]
            best = _seeded_weighted_choice(
                candidates,
                utilities,
                tie_rng,
                profile["sampling_temperature"],
            )
            selected.append(best)
            candidates.remove(best)

        results = []
        for rank, idx in enumerate(selected, start=1):
            entry = self.record.entries[idx]
            results.append(
                {
                    "rank": rank,
                    "record_id": entry.record_id,
                    "caption": entry.caption,
                    "image_path": entry.relative_image_path,
                    "image_paths": entry.grouped_relative_image_paths(),
                    "image_roles": list(entry.grouped_relative_image_paths()),
                    "sample_types": entry.sample_types,
                    "entry_metadata": dict(entry.metadata),
                    "material_tags": self._entry_materials(idx),
                    "component_tags": list(entry.metadata.get("components") or []),
                    "score": round(float(combined[idx]), 6),
                    "components": {
                        "bm25": round(float(normalized_bm25[idx]), 6),
                        "text_embedding": round(float(normalized_text[idx]), 6),
                        "image_embedding": round(float(normalized_image[idx]), 6),
                    },
                }
            )
        selected_ranks = {idx: rank for rank, idx in enumerate(selected, start=1)}
        debug = {
            "index_version": self.version,
            "embedding_model_path": self.embedding_model_path,
            "embedding_provider": self.embedding_provider,
            "embedding_dimension": self.embedding_dimension,
            "embedding_device": self.embedding_device,
            "reference_image_used": bool(references),
            "reference_image_count": len(references),
            "reference_color_preserved": bool(preserve_reference_color),
            "weights": weights,
            "candidate_k": candidate_k,
            "selected_k": len(results),
            "retrieval_seed": int(seed),
            "exploration_strength": exploration_name,
            "relevance_weight": profile["relevance"],
            "diversity_weight": profile["diversity"],
            "sampling_temperature": profile["sampling_temperature"],
            "selection_method": "seeded_weighted_mmr",
            "ranking_source": "hybrid_score_then_seeded_mmr",
            "material_coverage": {
                "requested": [
                    normalize_material(value) for value in (required_materials or ())
                    if normalize_material(value)
                ],
                "reserved_record_ids": [self.record.entries[index].record_id for index in material_reserved],
                "candidate_record_ids": material_candidates,
                "used": sorted({
                    material
                    for index in selected
                    for material in self._entry_materials(index)
                }),
                "missing": sorted({
                    normalize_material(value)
                    for value in (required_materials or ())
                    if normalize_material(value)
                } - {
                    material
                    for index in selected
                    for material in self._entry_materials(index)
                }),
            },
            "candidate_pool": [
                {
                    "candidate_rank": rank,
                    "record_id": self.record.entries[idx].record_id,
                    "selected_rank": selected_ranks.get(idx),
                    "score": round(float(combined[idx]), 6),
                    "tie_breaker": round(float(tie_breakers[idx]), 6),
                    "components": {
                        "bm25": round(float(normalized_bm25[idx]), 6),
                        "text_embedding": round(float(normalized_text[idx]), 6),
                        "image_embedding": round(float(normalized_image[idx]), 6),
                    },
                }
                for rank, idx in enumerate(candidate_pool, start=1)
            ],
            "warnings": self.warnings + self.record.warnings,
        }
        return results, debug


def _resolve_embedding_device(device: str) -> str:
    try:
        import torch

        normalized = (device or "cpu").strip().lower()
        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "cuda" and not torch.cuda.is_available():
            raise EmbeddingModelUnavailable("[IAT] Embedding device is cuda, but CUDA is not available.")
        if normalized not in {"cpu", "cuda"}:
            raise EmbeddingModelUnavailable(f"[IAT] Unsupported embedding device: `{device}`")
        return normalized
    except EmbeddingModelUnavailable:
        raise
    except Exception as exc:
        raise EmbeddingModelUnavailable(f"[IAT] Could not resolve embedding device: {exc}") from exc


def _load_embedding_model(
    model_path: str,
    device: str = "cpu",
    batch_size: int = 1,
    provider: str = "auto",
    dimension: int = 0,
):
    if not str(model_path or "").strip():
        raise EmbeddingModelUnavailable("[IAT] Embedding model path is not configured.")
    try:
        return get_embedding_adapter(
            model_path,
            _resolve_embedding_device(device),
            batch_size,
            provider=provider,
            dimension=dimension,
        )
    except EmbeddingAdapterError as exc:
        raise EmbeddingModelUnavailable(f"[IAT] {exc}") from exc


def _encode_text(
    model_path: str,
    text: str,
    device: str = "cpu",
    provider: str = "auto",
    batch_size: int = 1,
    dimension: int = 0,
    instruction: str = "",
) -> Optional[List[float]]:
    if not text:
        return None
    try:
        adapter = _load_embedding_model(model_path, device, batch_size, provider, dimension)
        return adapter.encode_texts([text], instruction=instruction, batch_size=batch_size)[0]
    except (EmbeddingModelUnavailable, EmbeddingAdapterError) as exc:
        raise EmbeddingModelUnavailable(f"[IAT] Failed to encode text with local embedding model: {exc}") from exc


def _encode_image(
    model_path: str,
    image: Any,
    grayscale: bool = True,
    device: str = "cpu",
    provider: str = "auto",
    batch_size: int = 1,
    dimension: int = 0,
    instruction: str = "",
) -> Optional[List[float]]:
    if image is None:
        return None
    try:
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("reference image must be a PIL image")
        prepared = image.convert("RGB")
        if provider == "qwen3_vl" and max(prepared.size) > _QWEN_IMAGE_MAX_SIDE:
            scale = _QWEN_IMAGE_MAX_SIDE / max(prepared.size)
            prepared = prepared.resize(
                (max(1, round(prepared.width * scale)), max(1, round(prepared.height * scale))),
                Image.Resampling.LANCZOS,
            )
        if grayscale:
            prepared = prepared.convert("L").convert("RGB")
        adapter = _load_embedding_model(model_path, device, batch_size, provider, dimension)
        return adapter.encode_images([prepared], instruction=instruction, batch_size=batch_size)[0]
    except (EmbeddingModelUnavailable, EmbeddingAdapterError) as exc:
        raise EmbeddingModelUnavailable(f"[IAT] Failed to encode image with local embedding model: {exc}") from exc


def _encode_text_batch(
    model_path: str,
    texts: Sequence[str],
    device: str,
    batch_size: int,
    provider: str = "auto",
    dimension: int = 0,
    instruction: str = "",
) -> List[List[float]]:
    try:
        adapter = _load_embedding_model(model_path, device, batch_size, provider, dimension)
        return adapter.encode_texts(texts, instruction=instruction, batch_size=batch_size)
    except (EmbeddingModelUnavailable, EmbeddingAdapterError) as exc:
        raise EmbeddingModelUnavailable(f"[IAT] Failed to encode text batch: {exc}") from exc


def _encode_image_batch(
    model_path: str,
    images: Sequence[Any],
    device: str,
    batch_size: int,
    grayscale: bool,
    provider: str = "auto",
    dimension: int = 0,
    instruction: str = "",
) -> List[List[float]]:
    try:
        from PIL import Image

        prepared = []
        for image in images:
            value = image.convert("RGB")
            if provider == "qwen3_vl" and max(value.size) > _QWEN_IMAGE_MAX_SIDE:
                scale = _QWEN_IMAGE_MAX_SIDE / max(value.size)
                value = value.resize(
                    (max(1, round(value.width * scale)), max(1, round(value.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            if grayscale:
                value = value.convert("L").convert("RGB")
            prepared.append(value)
        adapter = _load_embedding_model(model_path, device, batch_size, provider, dimension)
        return adapter.encode_images(prepared, instruction=instruction, batch_size=batch_size)
    except (EmbeddingModelUnavailable, EmbeddingAdapterError) as exc:
        raise EmbeddingModelUnavailable(f"[IAT] Failed to encode image batch: {exc}") from exc


def _encode_image_paths(
    model_path: str,
    image_paths: Sequence[Path],
    device: str,
    batch_size: int,
    grayscale: bool,
    provider: str,
    dimension: int,
    instruction: str,
) -> List[List[float]]:
    from PIL import Image

    vectors: List[List[float]] = []
    for start in range(0, len(image_paths), batch_size):
        images = []
        try:
            for image_path in image_paths[start : start + batch_size]:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
            vectors.extend(
                _encode_image_batch(
                    model_path,
                    images,
                    device,
                    batch_size,
                    grayscale=grayscale,
                    provider=provider,
                    dimension=dimension,
                    instruction=instruction,
                )
            )
        finally:
            for image in images:
                image.close()
    return vectors


def _serialize_index(index: DatasetIndex) -> Dict[str, Any]:
    return {
        "schema_version": _INDEX_SCHEMA_VERSION,
        "fingerprint": index.fingerprint,
        "embedding_model_path": index.embedding_model_path,
        "embedding_provider": index.embedding_provider,
        "embedding_dimension": index.embedding_dimension,
        "query_instruction": index.query_instruction,
        "document_instruction": index.document_instruction,
        "model_signature": index.model_signature,
        "text_embeddings": index.text_embeddings,
        "image_embeddings": index.image_embeddings,
        "gray_embeddings": index.gray_embeddings,
        "entries": [
            {
                "record_id": entry.record_id,
                "caption": entry.caption,
                "image_path": entry.relative_image_path,
                "image_paths": entry.grouped_relative_image_paths(),
                "metadata": dict(entry.metadata),
            }
            for entry in index.record.entries
        ],
    }


def _deserialize_index(
    payload: Dict[str, Any],
    record: DatasetRecord,
    fingerprint: str,
    embedding_model_path: str,
    embedding_device: str,
    embedding_provider: str,
    embedding_batch_size: int,
    embedding_dimension: int,
    query_instruction: str,
    document_instruction: str,
    model_signature: str,
) -> Optional[DatasetIndex]:
    if payload.get("schema_version") != _INDEX_SCHEMA_VERSION or payload.get("fingerprint") != fingerprint:
        return None
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(record.entries):
        return None
    for expected, cached in zip(record.entries, entries):
        if (
            expected.record_id != cached.get("record_id")
            or expected.caption != cached.get("caption")
            or expected.grouped_relative_image_paths() != (cached.get("image_paths") or {})
            or expected.metadata != (cached.get("metadata") or {})
        ):
            return None
    cached_model_path = str(payload.get("embedding_model_path") or "")
    if cached_model_path != str(embedding_model_path or ""):
        return None
    if str(payload.get("embedding_provider") or "auto") != embedding_provider:
        return None
    if int(payload.get("embedding_dimension") or 0) != int(embedding_dimension or 0):
        return None
    if str(payload.get("query_instruction") or "") != query_instruction:
        return None
    if str(payload.get("document_instruction") or "") != document_instruction:
        return None
    if str(payload.get("model_signature") or "") != model_signature:
        return None
    text_embeddings = payload.get("text_embeddings") or []
    image_embeddings = payload.get("image_embeddings") or []
    gray_embeddings = payload.get("gray_embeddings") or []
    if cached_model_path:
        expected_count = len(record.entries)
        if any(
            not isinstance(vectors, list) or len(vectors) != expected_count
            for vectors in (text_embeddings, image_embeddings, gray_embeddings)
        ):
            return None
        if any(vector is None for vector in text_embeddings):
            return None
        dimensions: Optional[int] = None
        for vectors in (text_embeddings, image_embeddings, gray_embeddings):
            for vector in vectors:
                if vector is None:
                    continue
                if not isinstance(vector, list) or not vector:
                    return None
                if dimensions is None:
                    dimensions = len(vector)
                if len(vector) != dimensions or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector
                ):
                    return None
    return DatasetIndex(
        record,
        fingerprint,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        gray_embeddings=gray_embeddings,
        embedding_model_path=cached_model_path,
        embedding_device=embedding_device,
        embedding_provider=embedding_provider,
        embedding_batch_size=embedding_batch_size,
        embedding_dimension=embedding_dimension,
        query_instruction=query_instruction,
        document_instruction=document_instruction,
        model_signature=model_signature,
    )


def clear_dataset_index_cache() -> None:
    """Release CPU index state without unloading embedding models."""
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


def _cache_content_revision(path: Path) -> Optional[str]:
    # Same-size rewrites can share a filesystem timestamp. Hash the bytes so a
    # hot index cannot hide a modified/corrupt disk cache; skip JSON decoding.
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _write_index_cache(index: DatasetIndex, cache_path: Path) -> Optional[str]:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(_serialize_index(index), stream, ensure_ascii=False)
        revision = _cache_content_revision(temporary)
        os.replace(temporary, cache_path)
        return revision
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_dataset_index(
    record: DatasetRecord,
    cache_dir: Path,
    embedding_model_path: str = "",
    require_embeddings: bool = False,
    embedding_device: str = "cpu",
    embedding_batch_size: int = 16,
    embedding_provider: str = "auto",
    embedding_dimension: int = 0,
    embedding_query_instruction: str = "",
    embedding_document_instruction: str = "",
) -> DatasetIndex:
    fingerprint = dataset_fingerprint(record)
    resolved_device = _resolve_embedding_device(embedding_device) if embedding_model_path else "cpu"
    resolved_provider = (embedding_provider or "auto").strip().lower().replace("-", "_")
    model_signature = ""
    if embedding_model_path:
        try:
            resolved_provider = detect_embedding_provider(embedding_model_path, resolved_provider)
            model_signature = embedding_model_signature(embedding_model_path, resolved_provider)
        except EmbeddingAdapterError as exc:
            if record.bundle_path is not None:
                raise EmbeddingModelUnavailable(f"[IAT] {exc}") from exc
            # Unit tests can mock the loader with a synthetic path; real loads still fail below.
            model_signature = hashlib.sha256(str(embedding_model_path).encode("utf-8")).hexdigest()

    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{_safe_name(record.dataset_name)}.index.json"
    tracked_path = record.bundle_path or cache_path
    # Include in-memory metadata as well as source bytes: callers may supply a
    # modified record, and bundle compatibility checks must not be bypassed.
    record_signature = hashlib.sha256(json.dumps(
        {
            "metadata": record.metadata,
            "warnings": record.warnings,
            "entries": [
                (entry.record_id, entry.caption, entry.grouped_relative_image_paths(), entry.metadata)
                for entry in record.entries
            ],
        }, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    identity = (
        str(record.source_path.resolve()), str(tracked_path.resolve()), fingerprint, record_signature,
        str(embedding_model_path), resolved_provider, model_signature, resolved_device,
        int(embedding_batch_size), int(embedding_dimension),
        embedding_query_instruction, embedding_document_instruction, bool(require_embeddings),
        # A loaded bundle is a snapshot. Do not reuse its matrices for a newly
        # decoded snapshot merely because captions/metadata happen to match.
        tuple(id(vectors) for vectors in (
            record.bundled_text_embeddings, record.bundled_image_embeddings, record.bundled_gray_embeddings
        )) if record.bundle_path is not None else (),
    )

    def cache_revision() -> Optional[str]:
        return fingerprint if record.bundle_path is not None else _cache_content_revision(cache_path)

    memory_key = (identity, cache_revision())
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(memory_key)
        if cached is not None:
            _INDEX_CACHE.move_to_end(memory_key)
            return cached

    def remember(index: DatasetIndex, revision: Optional[str]) -> DatasetIndex:
        if revision is None or revision != cache_revision():
            return index
        key = (identity, revision)
        with _INDEX_CACHE_LOCK:
            _INDEX_CACHE[key] = index
            _INDEX_CACHE.move_to_end(key)
            while len(_INDEX_CACHE) > _INDEX_CACHE_LIMIT:
                _INDEX_CACHE.popitem(last=False)
        return index

    if record.bundle_path is not None:
        if not embedding_model_path:
            raise EmbeddingModelUnavailable(
                "[IAT] Compiled datasets require the same local embedding model used during compilation."
            )
        metadata = record.metadata
        expected_provider = str(metadata.get("embedding_provider") or "")
        expected_signature = str(metadata.get("embedding_model_signature") or "")
        expected_dimension = int(metadata.get("embedding_dimension") or 0)
        stored_query_instruction = str(metadata.get("embedding_query_instruction") or "")
        stored_document_instruction = str(metadata.get("embedding_document_instruction") or "")
        if resolved_provider != expected_provider:
            raise EmbeddingModelUnavailable(
                f"[IAT] Compiled dataset requires embedding provider `{expected_provider}`, got `{resolved_provider}`."
            )
        if model_signature != expected_signature:
            raise EmbeddingModelUnavailable(
                "[IAT] Configured embedding model does not match the model used to compile this dataset."
            )
        if embedding_dimension and int(embedding_dimension) != expected_dimension:
            raise EmbeddingModelUnavailable(
                f"[IAT] Compiled dataset uses {expected_dimension}-dimensional embeddings, "
                f"but config requests {embedding_dimension}."
            )
        if embedding_query_instruction and embedding_query_instruction != stored_query_instruction:
            raise EmbeddingModelUnavailable(
                "[IAT] Query instruction differs from the instruction stored in the compiled dataset."
            )
        if embedding_document_instruction and embedding_document_instruction != stored_document_instruction:
            raise EmbeddingModelUnavailable(
                "[IAT] Document instruction differs from the instruction stored in the compiled dataset."
            )
        return remember(DatasetIndex(
            record,
            fingerprint,
            text_embeddings=record.bundled_text_embeddings,
            image_embeddings=record.bundled_image_embeddings,
            gray_embeddings=record.bundled_gray_embeddings,
            embedding_model_path=str(embedding_model_path),
            embedding_device=resolved_device,
            embedding_provider=resolved_provider,
            embedding_batch_size=embedding_batch_size,
            embedding_dimension=expected_dimension,
            query_instruction=stored_query_instruction,
            document_instruction=stored_document_instruction,
            model_signature=model_signature,
        ), fingerprint)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        try:
            cache_bytes = cache_path.read_bytes()
            read_revision = hashlib.sha256(cache_bytes).hexdigest()
            cached = _deserialize_index(
                json.loads(cache_bytes),
                record,
                fingerprint,
                embedding_model_path,
                resolved_device,
                resolved_provider,
                embedding_batch_size,
                embedding_dimension,
                embedding_query_instruction,
                embedding_document_instruction,
                model_signature,
            )
            if cached is not None:
                if require_embeddings and not cached.text_embeddings:
                    raise EmbeddingModelUnavailable("[IAT] Dataset index has no embeddings; configure a local Chinese CLIP model.")
                return remember(cached, read_revision)
        except EmbeddingModelUnavailable:
            raise
        except Exception:
            pass

    warnings: List[str] = []
    text_embeddings: List[Optional[List[float]]] = []
    image_embeddings: List[Optional[List[float]]] = []
    gray_embeddings: List[Optional[List[float]]] = []
    if embedding_model_path:
        batch_size = max(1, int(embedding_batch_size))
        _load_embedding_model(
            embedding_model_path,
            resolved_device,
            batch_size,
            resolved_provider,
            embedding_dimension,
        )
        text_embeddings = _encode_text_batch(
            embedding_model_path,
            [entry.caption for entry in record.entries],
            resolved_device,
            batch_size,
            provider=resolved_provider,
            dimension=embedding_dimension,
            instruction=embedding_document_instruction,
        )
        image_paths: List[Path] = []
        image_counts: List[int] = []
        image_embeddings = [None] * len(record.entries)
        gray_embeddings = [None] * len(record.entries)
        for idx, entry in enumerate(record.entries):
            entry_images = entry.grouped_image_paths()
            if not entry_images:
                image_counts.append(0)
                continue
            count = 0
            for image_path in entry_images.values():
                image_paths.append(image_path)
                count += 1
            image_counts.append(count)
        rgb_vectors = _encode_image_paths(
            embedding_model_path,
            image_paths,
            resolved_device,
            batch_size,
            grayscale=False,
            provider=resolved_provider,
            dimension=embedding_dimension,
            instruction=embedding_document_instruction,
        )
        gray_vectors = _encode_image_paths(
            embedding_model_path,
            image_paths,
            resolved_device,
            batch_size,
            grayscale=True,
            provider=resolved_provider,
            dimension=embedding_dimension,
            instruction=embedding_document_instruction,
        )
        offset = 0
        for idx, count in enumerate(image_counts):
            if not count:
                continue
            image_embeddings[idx] = _mean_vector(rgb_vectors[offset : offset + count])
            gray_embeddings[idx] = _mean_vector(gray_vectors[offset : offset + count])
            offset += count
    else:
        warnings.append("Embedding model path is empty; using offline BM25 only.")

    index = DatasetIndex(
        record,
        fingerprint,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        gray_embeddings=gray_embeddings,
        embedding_model_path=str(embedding_model_path or ""),
        embedding_device=resolved_device,
        embedding_provider=resolved_provider,
        embedding_batch_size=embedding_batch_size,
        embedding_dimension=embedding_dimension,
        query_instruction=embedding_query_instruction,
        document_instruction=embedding_document_instruction,
        model_signature=model_signature,
        warnings=warnings,
    )
    if require_embeddings and not text_embeddings:
        raise EmbeddingModelUnavailable("[IAT] Embedding model path is not configured; set datasets.embedding_model_path for hybrid retrieval.")
    written_revision = None
    try:
        written_revision = _write_index_cache(index, cache_path)
    except Exception as exc:
        index.warnings.append(f"Could not write index cache `{cache_path}`: {exc}")
    return remember(index, written_revision)


def dataset_metadata(record: DatasetRecord) -> Dict[str, Any]:
    return dict(record.metadata)
