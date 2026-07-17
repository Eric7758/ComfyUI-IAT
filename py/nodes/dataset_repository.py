from __future__ import annotations

"""Offline dataset discovery, paired caption loading, and hybrid retrieval.

The repository deliberately keeps image files outside the index.  The index stores
captions, relative paths, metadata, and optional normalized embeddings only.
"""

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
_EMBEDDING_MODELS: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


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

    metadata = {
        "dataset_name": dataset_name,
        "version": version,
        "base_model": base_model,
        "lora_name": lora_name,
        "language": language,
        "trigger_words": trigger_words,
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
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"


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
        warnings: Optional[List[str]] = None,
    ):
        self.record = record
        self.fingerprint = fingerprint
        self.text_embeddings = text_embeddings or []
        self.image_embeddings = image_embeddings or []
        self.gray_embeddings = gray_embeddings or []
        self.embedding_model_path = embedding_model_path
        self.embedding_device = embedding_device
        self.warnings = list(warnings or [])
        self.tokens = [tokenize(entry.caption) for entry in record.entries]
        self.document_frequency: Dict[str, int] = {}
        for tokens in self.tokens:
            for token in set(tokens):
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

    @property
    def version(self) -> str:
        return f"hybrid-v3:{self.fingerprint[:12]}"

    def _bm25_scores(self, query: str) -> List[float]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return [0.0] * len(self.record.entries)
        query_counts: Dict[str, int] = {}
        for token in query_tokens:
            query_counts[token] = query_counts.get(token, 0) + 1
        document_count = len(self.tokens) or 1
        average_length = sum(len(tokens) for tokens in self.tokens) / document_count if self.tokens else 1.0
        scores = []
        for tokens in self.tokens:
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            length_factor = len(tokens) / max(average_length, 1.0)
            score = 0.0
            for token, query_count in query_counts.items():
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency.get(token, 0)
                idf = math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                numerator = frequency * 2.5
                denominator = frequency + 1.5 * (0.75 + 0.25 * length_factor)
                score += idf * (numerator / max(denominator, 1e-6)) * (1.0 + math.log1p(query_count))
            scores.append(score)
        return scores

    def _similarity_to_selected(self, left: int, right: int) -> float:
        if self.text_embeddings and left < len(self.text_embeddings) and right < len(self.text_embeddings):
            return max(0.0, _cosine(self.text_embeddings[left], self.text_embeddings[right]))
        left_tokens, right_tokens = set(self.tokens[left]), set(self.tokens[right])
        return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)

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
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        top_k = max(1, min(8, int(top_k)))
        exploration_name, profile = _exploration_profile(exploration_strength)
        candidate_k = max(top_k, min(int(candidate_k), int(profile["candidate_k"])))
        references = list(reference_images or [])
        if reference_image is not None:
            references.insert(0, reference_image)
        bm25 = self._bm25_scores(query)
        text_query = (
            _encode_text(self.embedding_model_path, query, device=self.embedding_device)
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
                )
            else:
                image_query = _mean_vector(
                    _encode_image_batch(
                        self.embedding_model_path,
                        references,
                        self.embedding_device,
                        min(16, len(references)),
                        grayscale=not preserve_reference_color,
                    )
                )

        text_scores = [_cosine(text_query, vector) for vector in self.text_embeddings] if text_query else [0.0] * len(self.record.entries)
        image_vectors = self.image_embeddings if preserve_reference_color else self.gray_embeddings
        image_scores = [_cosine(image_query, vector) for vector in image_vectors] if image_query else [0.0] * len(self.record.entries)

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

        selected: List[int] = []
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


def _load_embedding_model(model_path: str, device: str = "cpu"):
    normalized = str(Path(model_path).expanduser())
    if not normalized:
        raise EmbeddingModelUnavailable("[IAT] Embedding model path is not configured.")
    resolved_device = _resolve_embedding_device(device)
    cache_key = (normalized, resolved_device)
    if cache_key in _EMBEDDING_MODELS:
        return _EMBEDDING_MODELS[cache_key]
    path = Path(normalized)
    if not path.is_dir():
        raise EmbeddingModelUnavailable(f"[IAT] Local embedding model does not exist: `{path}`")
    try:
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        processor = ChineseCLIPProcessor.from_pretrained(str(path), local_files_only=True)
        model = ChineseCLIPModel.from_pretrained(str(path), local_files_only=True)
        model.eval()
        model.to(resolved_device)
    except Exception as exc:
        raise EmbeddingModelUnavailable(
            f"[IAT] Failed to load local Chinese CLIP embedding model `{path}` without downloading: {exc}"
        ) from exc
    _EMBEDDING_MODELS[cache_key] = (processor, model)
    return processor, model


def _as_float_list(tensor: Any) -> List[float]:
    values = tensor.detach().float().cpu()
    norm = values.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return (values / norm)[0].tolist()


def _as_float_lists(tensor: Any) -> List[List[float]]:
    values = tensor.detach().float().cpu()
    norm = values.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return (values / norm).tolist()


def _move_inputs(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _encode_text(model_path: str, text: str, device: str = "cpu") -> Optional[List[float]]:
    if not text:
        return None
    try:
        import torch
        resolved_device = _resolve_embedding_device(device)
        processor, model = _load_embedding_model(model_path, resolved_device)
        inputs = _move_inputs(processor(text=[text], padding=True, return_tensors="pt"), resolved_device)
        with torch.inference_mode():
            features = model.get_text_features(**inputs)
        return _as_float_list(features)
    except EmbeddingModelUnavailable:
        raise
    except Exception as exc:
        raise EmbeddingModelUnavailable(f"[IAT] Failed to encode text with local embedding model: {exc}") from exc


def _encode_image(model_path: str, image: Any, grayscale: bool = True, device: str = "cpu") -> Optional[List[float]]:
    if image is None:
        return None
    try:
        import torch
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("reference image must be a PIL image")
        prepared = image.convert("L").convert("RGB") if grayscale else image.convert("RGB")
        resolved_device = _resolve_embedding_device(device)
        processor, model = _load_embedding_model(model_path, resolved_device)
        inputs = _move_inputs(processor(images=[prepared], return_tensors="pt"), resolved_device)
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
        return _as_float_list(features)
    except EmbeddingModelUnavailable:
        raise
    except Exception as exc:
        raise EmbeddingModelUnavailable(f"[IAT] Failed to encode image with local embedding model: {exc}") from exc


def _encode_text_batch(model_path: str, texts: Sequence[str], device: str, batch_size: int) -> List[List[float]]:
    import torch

    resolved_device = _resolve_embedding_device(device)
    processor, model = _load_embedding_model(model_path, resolved_device)
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        inputs = processor(text=list(texts[start : start + batch_size]), padding=True, return_tensors="pt")
        with torch.inference_mode():
            features = model.get_text_features(**_move_inputs(inputs, resolved_device))
        vectors.extend(_as_float_lists(features))
    return vectors


def _encode_image_batch(
    model_path: str,
    images: Sequence[Any],
    device: str,
    batch_size: int,
    grayscale: bool,
) -> List[List[float]]:
    import torch

    resolved_device = _resolve_embedding_device(device)
    processor, model = _load_embedding_model(model_path, resolved_device)
    vectors: List[List[float]] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        prepared = [image.convert("L").convert("RGB") if grayscale else image.convert("RGB") for image in batch]
        inputs = processor(images=prepared, return_tensors="pt")
        with torch.inference_mode():
            features = model.get_image_features(**_move_inputs(inputs, resolved_device))
        vectors.extend(_as_float_lists(features))
    return vectors


def _serialize_index(index: DatasetIndex) -> Dict[str, Any]:
    return {
        "schema_version": 3,
        "fingerprint": index.fingerprint,
        "embedding_model_path": index.embedding_model_path,
        "text_embeddings": index.text_embeddings,
        "image_embeddings": index.image_embeddings,
        "gray_embeddings": index.gray_embeddings,
        "entries": [
            {
                "record_id": entry.record_id,
                "caption": entry.caption,
                "image_path": entry.relative_image_path,
                "image_paths": entry.grouped_relative_image_paths(),
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
) -> Optional[DatasetIndex]:
    if payload.get("schema_version") != 3 or payload.get("fingerprint") != fingerprint:
        return None
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(record.entries):
        return None
    for expected, cached in zip(record.entries, entries):
        if (
            expected.record_id != cached.get("record_id")
            or expected.caption != cached.get("caption")
            or expected.grouped_relative_image_paths() != (cached.get("image_paths") or {})
        ):
            return None
    cached_model_path = str(payload.get("embedding_model_path") or "")
    if cached_model_path != str(embedding_model_path or ""):
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
                if len(vector) != dimensions or any(not isinstance(value, (int, float)) for value in vector):
                    return None
    return DatasetIndex(
        record,
        fingerprint,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        gray_embeddings=gray_embeddings,
        embedding_model_path=cached_model_path,
        embedding_device=embedding_device,
    )


def get_dataset_index(
    record: DatasetRecord,
    cache_dir: Path,
    embedding_model_path: str = "",
    require_embeddings: bool = False,
    embedding_device: str = "cpu",
    embedding_batch_size: int = 16,
) -> DatasetIndex:
    fingerprint = dataset_fingerprint(record)
    resolved_device = _resolve_embedding_device(embedding_device) if embedding_model_path else "cpu"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_safe_name(record.dataset_name)}.index.json"
    if cache_path.is_file():
        try:
            cached = _deserialize_index(
                json.loads(cache_path.read_text(encoding="utf-8")),
                record,
                fingerprint,
                embedding_model_path,
                resolved_device,
            )
            if cached is not None:
                if require_embeddings and not cached.text_embeddings:
                    raise EmbeddingModelUnavailable("[IAT] Dataset index has no embeddings; configure a local Chinese CLIP model.")
                return cached
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
        _load_embedding_model(embedding_model_path, resolved_device)
        text_embeddings = _encode_text_batch(
            embedding_model_path,
            [entry.caption for entry in record.entries],
            resolved_device,
            batch_size,
        )
        from PIL import Image

        images: List[Any] = []
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
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
                count += 1
            image_counts.append(count)
        rgb_vectors = _encode_image_batch(embedding_model_path, images, resolved_device, batch_size, grayscale=False)
        gray_vectors = _encode_image_batch(embedding_model_path, images, resolved_device, batch_size, grayscale=True)
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
        warnings=warnings,
    )
    if require_embeddings and not text_embeddings:
        raise EmbeddingModelUnavailable("[IAT] Embedding model path is not configured; set datasets.embedding_model_path for hybrid retrieval.")
    try:
        cache_path.write_text(json.dumps(_serialize_index(index), ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        index.warnings.append(f"Could not write index cache `{cache_path}`: {exc}")
    return index


def dataset_metadata(record: DatasetRecord) -> Dict[str, Any]:
    return dict(record.metadata)
