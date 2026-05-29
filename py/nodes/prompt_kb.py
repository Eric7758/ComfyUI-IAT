from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[^\w\s\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE)
_COLOR_FAMILY_KEYWORDS = {
    "black_gray": ["黑", "灰", "银", "炭", "岩灰", "水泥灰", "火山灰", "冷灰"],
    "white_beige": ["白", "米", "沙", "卡其", "浅灰", "浅沙", "米白"],
    "brown": ["棕", "褐", "咖", "焦糖", "赭", "马鞍", "泥岩", "大地", "陶土"],
    "green": ["绿", "橄榄", "森林", "松针", "苔藓", "军绿", "鼠尾草"],
    "blue": ["蓝", "海军蓝", "深海蓝", "暗夜蓝", "亮蓝", "冰蓝", "青"],
    "yellow_orange": ["黄", "橙", "芥末", "琥珀", "暖金", "金铜", "沙漠黄"],
    "red_purple": ["红", "砖红", "玫瑰粉", "紫"],
}
_ACCENT_FAMILIES = {"green", "blue", "yellow_orange", "red_purple"}


@dataclass(frozen=True)
class PromptKBRecord:
    dataset_name: str
    version: str
    trigger_words: List[str]
    captions: List[str]
    source_path: Path


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    return _normalize_whitespace(text)


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text)
    latin_tokens = _TOKEN_RE.findall(normalized)
    cjk_chars = _CJK_RE.findall(normalized)
    return latin_tokens + cjk_chars


def _detect_language_hint(texts: List[str]) -> str:
    joined = "\n".join(texts)
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", joined):
        if re.search(r"[\u3040-\u30ff]", joined):
            return "ja"
        return "zh"
    if re.search(r"[a-zA-Z]", joined):
        return "en"
    return "en"


def _validate_string_list(value, field_name: str, path: Path) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"[IAT] {path.name}: `{field_name}` must be a non-empty list.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"[IAT] {path.name}: `{field_name}` items must be non-empty strings.")
        result.append(item.strip())
    return result


def load_prompt_kb_record(path: Path) -> PromptKBRecord:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"[IAT] {path.name}: root JSON value must be an object.")

    dataset_name = raw.get("dataset_name")
    version = raw.get("version")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError(f"[IAT] {path.name}: `dataset_name` must be a non-empty string.")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"[IAT] {path.name}: `version` must be a non-empty string.")

    trigger_words = _validate_string_list(raw.get("trigger_words"), "trigger_words", path)
    captions = _validate_string_list(raw.get("captions"), "captions", path)

    return PromptKBRecord(
        dataset_name=dataset_name.strip(),
        version=version.strip(),
        trigger_words=trigger_words,
        captions=captions,
        source_path=path,
    )


def discover_prompt_kb_records(kb_root: Path) -> Tuple[Dict[str, PromptKBRecord], List[str]]:
    records: Dict[str, PromptKBRecord] = {}
    if not kb_root.is_dir():
        return records, []

    errors: List[str] = []
    for path in sorted(kb_root.glob("*.json")):
        try:
            record = load_prompt_kb_record(path)
        except Exception as exc:
            errors.append(str(exc))
            continue
        if record.dataset_name in records:
            errors.append(
                f"[IAT] Duplicate dataset_name `{record.dataset_name}` in "
                f"{records[record.dataset_name].source_path.name} and {path.name}."
            )
            continue
        records[record.dataset_name] = record
    return records, errors


def get_prompt_kb_dataset_options(kb_root: Path) -> List[str]:
    records, _errors = discover_prompt_kb_records(kb_root)
    return sorted(records.keys())


def _build_query_weights(query: str) -> Dict[str, float]:
    tokens = _tokenize(query)
    if not tokens:
        return {}
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    max_count = max(counts.values())
    return {token: 0.5 + (count / max_count) for token, count in counts.items()}


def _score_caption(query_weights: Dict[str, float], caption: str) -> float:
    caption_tokens = _tokenize(caption)
    if not caption_tokens or not query_weights:
        return 0.0

    caption_counts: Dict[str, int] = {}
    for token in caption_tokens:
        caption_counts[token] = caption_counts.get(token, 0) + 1

    overlap = 0.0
    for token, q_weight in query_weights.items():
        if token in caption_counts:
            overlap += q_weight * (1.0 + math.log1p(caption_counts[token]))

    diversity_bonus = min(len(set(caption_tokens)) / 24.0, 1.0)
    length_penalty = abs(len(caption_tokens) - 32) / 64.0
    return overlap + diversity_bonus - length_penalty


def retrieve_top_captions(query: str, record: PromptKBRecord, top_n: int) -> List[str]:
    query_weights = _build_query_weights(query)
    top_n = max(1, int(top_n))
    if not query_weights:
        return record.captions[:top_n]

    ranked = sorted(
        record.captions,
        key=lambda caption: (_score_caption(query_weights, caption), caption),
        reverse=True,
    )
    return ranked[:top_n]


def _caption_color_families(caption: str):
    families = set()
    for family, keywords in _COLOR_FAMILY_KEYWORDS.items():
        if any(keyword in caption for keyword in keywords):
            families.add(family)
    return families


def select_palette_reference_captions(record: PromptKBRecord, seed: int, count: int = 2) -> List[str]:
    if not record.captions:
        return []

    count = max(1, int(count))
    colorful = []
    fallback = []
    for idx, caption in enumerate(record.captions):
        families = _caption_color_families(caption)
        score = len(families) * 10 + len(families & _ACCENT_FAMILIES) * 20
        item = (score, idx, caption)
        if score > 0 and families & _ACCENT_FAMILIES:
            colorful.append(item)
        else:
            fallback.append(item)

    pool = colorful if colorful else (fallback if fallback else [(0, idx, cap) for idx, cap in enumerate(record.captions)])
    pool = sorted(pool, key=lambda item: (item[0], item[1]), reverse=True)
    if not pool:
        return []

    selected = []
    used_indexes = set()
    start = int(seed) % len(pool)
    for offset in range(len(pool)):
        item = pool[(start + offset) % len(pool)]
        if item[1] in used_indexes:
            continue
        used_indexes.add(item[1])
        selected.append(item[2])
        if len(selected) >= count:
            break
    return selected


def select_caption_by_seed(record: PromptKBRecord, seed: int) -> str:
    if not record.captions:
        raise ValueError(f"[IAT] Dataset `{record.dataset_name}` has no captions.")
    index = int(seed) % len(record.captions)
    return record.captions[index]


def resolve_output_language(language: str, record: PromptKBRecord) -> str:
    if language == "English":
        return "en"
    if language == "中文":
        return "zh"
    return _detect_language_hint(record.captions)
