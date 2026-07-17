from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from .dataset_repository import (
    DatasetError,
    EmbeddingModelUnavailable,
    DatasetRecord,
    choose_caption,
    dataset_fingerprint,
    dataset_metadata,
    discover_datasets,
    get_dataset_index,
)
from .llm_backends import BackendError, generate_with_backend
_CFG = getattr(sys.modules.get("comfyui_iat_config"), "data", {}) or {}
_CFG_PATH = Path(getattr(sys.modules.get("comfyui_iat_config"), "path", Path(__file__).resolve().parents[2] / "config.yaml"))
_MODEL_CFG = (_CFG.get("model") or {}) if isinstance(_CFG, dict) else {}
_DATASET_CFG = (_CFG.get("datasets") or {}) if isinstance(_CFG, dict) else {}
_LLM_CFG = (_CFG.get("llm") or {}) if isinstance(_CFG, dict) else {}
_OLLAMA_CFG = (_CFG.get("ollama") or {}) if isinstance(_CFG, dict) else {}
_VLLM_CFG = (_CFG.get("vllm") or {}) if isinstance(_CFG, dict) else {}

_BACKEND_OPTIONS = ["Ollama", "vLLM", "Local"]
_SELECTION_OPTIONS = ["Random", "Sequential", "By Index"]
_EXPLORATION_OPTIONS = ["Mild", "Medium", "Strong"]
_LANGUAGE_OPTIONS = ["Auto", "中文", "English"]
_LOCAL_MODEL_OPTIONS = [
    "Qwen3.5-0.8B",
    "Qwen3.5-2B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "Qwen3.5-27B",
    "Qwen3.6-35B-A3B",
]
_ATTENTION_OPTIONS = ["SDPA", "FlashAttention-2", "Eager"]
_DEFAULT_ATTENTION_BACKEND = str((_CFG.get("runtime") or {}).get("default_attention_backend") or "SDPA")
if _DEFAULT_ATTENTION_BACKEND not in _ATTENTION_OPTIONS:
    _DEFAULT_ATTENTION_BACKEND = "SDPA"
_DATASET_ROOT = str(_DATASET_CFG.get("root") or "").strip()
_EMBEDDING_MODEL_PATH = str(_DATASET_CFG.get("embedding_model_path") or "").strip()
_INDEX_CACHE_DIR = str(_DATASET_CFG.get("index_cache_dir") or "").strip()
_EMBEDDING_DEVICE = str(_DATASET_CFG.get("embedding_device") or "cpu").strip()
_EMBEDDING_BATCH_SIZE = int(_DATASET_CFG.get("embedding_batch_size") or 16)
_DEFAULT_BACKEND = str(_LLM_CFG.get("default_backend") or "Ollama")
if _DEFAULT_BACKEND not in _BACKEND_OPTIONS:
    _DEFAULT_BACKEND = "Ollama"


def _resolve_config_path(value: str, fallback: Path) -> Path:
    if not value:
        return fallback
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_CFG_PATH.parent / path).resolve()
    return path


def _dataset_root() -> Path:
    return _resolve_config_path(_DATASET_ROOT, _CFG_PATH.parent / "datasets")


def _index_cache_root() -> Path:
    if _INDEX_CACHE_DIR:
        return _resolve_config_path(_INDEX_CACHE_DIR, _dataset_root() / ".iat_index")
    return _dataset_root() / ".iat_index"


def _embedding_model_path() -> str:
    if not _EMBEDDING_MODEL_PATH:
        return ""
    return str(_resolve_config_path(_EMBEDDING_MODEL_PATH, _CFG_PATH.parent / "models" / "embeddings"))


def _discover() -> tuple[Dict[str, DatasetRecord], List[str]]:
    return discover_datasets(_dataset_root())


def _selected_record(dataset_name: str) -> DatasetRecord:
    records, errors = _discover()
    record = records.get(dataset_name)
    if record is not None:
        return record
    details = f"\nDiscovery diagnostics:\n" + "\n".join(errors) if errors else ""
    raise DatasetError(
        f"[IAT] Dataset `{dataset_name}` was not found or is invalid under `{_dataset_root()}`.{details}"
    )


def _dataset_change_token(dataset_name: str) -> str:
    try:
        return dataset_fingerprint(_selected_record(dataset_name))
    except DatasetError as exc:
        return f"invalid:{dataset_name}:{exc}"


def _dataset_options() -> List[str]:
    records, errors = _discover()
    options = sorted(records.keys())
    if not options:
        return ["__NO_DATASET_FOUND__"]
    return options


def _tensor_to_pil_list(image: Any) -> List[Image.Image]:
    if image is None:
        return []
    try:
        dimensions = image.dim()
        if dimensions == 3:
            image = image.unsqueeze(0)
        elif dimensions != 4:
            raise ValueError(f"expected IMAGE tensor with 3 or 4 dimensions, got {dimensions}")
        return [
            Image.fromarray((item.cpu().numpy() * 255.0).clip(0, 255).astype("uint8")).convert("RGB")
            for item in image
        ]
    except Exception as exc:
        raise DatasetError(f"[IAT] Could not convert IMAGE input to PIL: {exc}") from exc


def _generation_images(images: Sequence[Image.Image], preserve_color: bool) -> Optional[List[Image.Image]]:
    if not images:
        return None
    prepared = []
    for image in images:
        current = image.convert("RGB")
        if not preserve_color:
            current = current.convert("L").convert("RGB")
        prepared.append(current)
    return prepared


def _collect_reference_images(*inputs: Any) -> List[Image.Image]:
    images: List[Image.Image] = []
    for value in inputs:
        images.extend(_tensor_to_pil_list(value))
    if len(images) > 4:
        raise DatasetError("[IAT] At most 4 reference images are supported (image through image_4).")
    return images


def _sanitize_prompt(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"\s*\[(?:cite\s*:\s*\d+|\d+)\]\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:最终提示词|提示词|Prompt|Final prompt)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_trigger(prompt: str, trigger: str) -> bool:
    normalized_prompt = prompt.casefold()
    normalized_trigger = trigger.strip().casefold()
    if not normalized_trigger:
        return True
    if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", normalized_trigger):
        return normalized_trigger in normalized_prompt
    return re.search(rf"(?<!\w){re.escape(normalized_trigger)}(?!\w)", normalized_prompt) is not None


def _ensure_trigger_words(prompt: str, record: DatasetRecord) -> str:
    result = _sanitize_prompt(prompt)
    missing = [word for word in record.trigger_words if not _contains_trigger(result, word)]
    if missing:
        result = ", ".join(missing + ([result] if result else []))
    return result


def _output_language(record: DatasetRecord, selection: str) -> str:
    if selection == "中文":
        return "zh"
    if selection == "English":
        return "en"
    return record.language


def _backend_defaults(backend: str) -> Dict[str, Any]:
    if backend == "Ollama":
        return {
            "model": str(_OLLAMA_CFG.get("model") or _LLM_CFG.get("default_model") or "qwen3.5:122b"),
            "base_url": str(_OLLAMA_CFG.get("base_url") or "http://127.0.0.1:11434"),
        }
    if backend == "vLLM":
        return {
            "model": str(_VLLM_CFG.get("model") or _LLM_CFG.get("default_model") or "qwen3.5:122b"),
            "base_url": str(_VLLM_CFG.get("base_url") or "http://127.0.0.1:8000/v1"),
        }
    default_variant = str(_MODEL_CFG.get("default_variant") or "Qwen3.5-2B")
    if default_variant not in _LOCAL_MODEL_OPTIONS:
        default_variant = "Qwen3.5-2B"
    return {"model": default_variant, "base_url": ""}


_COLOR_FAMILY_TERMS = {
    "black": ("black", "黑色系", "黑色", "炭黑", "玄武岩黑"),
    "brown": ("brown", "棕色系", "棕色", "鞍棕", "驼棕"),
    "gray": ("gray", "grey", "灰色系", "灰色", "砾岩灰", "岩层灰"),
    "white": ("white", "白色系", "白色", "象牙白"),
    "blue": ("blue", "蓝色系", "蓝色"),
    "green": ("green", "绿色系", "绿色"),
    "red": ("red", "红色系", "红色"),
    "orange": ("orange", "橙色系", "橙色"),
    "purple": ("purple", "紫色系", "紫色", "紫罗兰"),
    "yellow": ("yellow", "黄色系", "黄色", "芥末黄"),
    "cyan": ("cyan", "teal", "青色系", "青色", "蓝绿色"),
    "beige": ("beige", "米色系", "米色", "沙色"),
}
_FAMILY_BASE_HEX = {
    "black": "#1f1f1d",
    "brown": "#8b4f2f",
    "gray": "#8a8178",
    "white": "#d9d4c9",
    "blue": "#405a73",
    "green": "#526b52",
    "red": "#7b3730",
    "orange": "#a96735",
    "purple": "#665a78",
    "yellow": "#b28a3b",
    "cyan": "#477b7b",
    "beige": "#b6a182",
    "custom": "#808080",
}
_MATERIAL_TERMS = (
    "麂皮", "皮质", "皮革", "织物", "编织", "金属", "木纹", "橡胶", "塑料", "玻璃",
    "suede", "leather", "fabric", "woven", "metal", "wood", "rubber",
)
_COMPONENT_TERMS = (
    "座椅主面", "座椅侧翼", "座椅", "中控台面上层", "中控台前饰板", "中控台", "门板",
    "扶手", "方向盘", "仪表台", "饰条", "地毯", "seat center", "seat bolsters",
    "dashboard", "center console", "door panel", "armrest", "steering wheel", "trim",
)
_RELATION_TERMS = (
    "同色统一", "主辅分色", "局部撞色", "低对比", "高对比", "局部强调", "连续延展",
    "tone on tone", "two-tone", "contrast", "accent", "low contrast",
)
_EXPLORATION_GENERATION_TEMPERATURE = {"Mild": 0.15, "Medium": 0.35, "Strong": 0.55}
_EXPLORATION_COLOR_SHIFT = {"Mild": 8, "Medium": 18, "Strong": 28}


def _derive_seed(*parts: Any) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _matching_terms(text: str, terms: Sequence[str]) -> List[str]:
    normalized = (text or "").casefold()
    matches = [term for term in terms if term.casefold() in normalized]
    return [
        term
        for term in matches
        if not any(
            term != other and term.casefold() in other.casefold()
            for other in matches
        )
    ]


def _matching_families(text: str) -> List[str]:
    normalized = (text or "").casefold()
    return [family for family, terms in _COLOR_FAMILY_TERMS.items() if any(term.casefold() in normalized for term in terms)]


def _hex_codes(text: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"#[0-9a-fA-F]{6}", text or "")))


def _shift_hex(value: str, rng: random.Random, maximum_shift: int = 18) -> str:
    try:
        channels = [int(value[index : index + 2], 16) for index in (1, 3, 5)]
    except (TypeError, ValueError):
        channels = [31, 31, 29]
    shift = rng.randint(-maximum_shift, maximum_shift)
    jitter = max(1, maximum_shift // 4)
    adjusted = [max(0, min(255, channel + shift + rng.randint(-jitter, jitter))) for channel in channels]
    return "#" + "".join(f"{channel:02x}" for channel in adjusted)


def _normalize_exploration_strength(value: str) -> str:
    normalized = (value or "Medium").strip().title()
    return normalized if normalized in _EXPLORATION_GENERATION_TEMPERATURE else "Medium"


def _build_variation_plan(
    user_prompt: str,
    retrieved: Sequence[Dict[str, Any]],
    seed: int,
    exploration_strength: str,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    exploration_name = _normalize_exploration_strength(exploration_strength)
    captions = "\n".join(str(item.get("caption") or "") for item in retrieved)
    user_families = _matching_families(user_prompt)
    sample_families = _matching_families(captions)
    has_unrecognized_color_hint = bool(
        re.search(r"(?:色系|色调|颜色|色彩|色|colou?r)", user_prompt or "", flags=re.IGNORECASE)
    )
    if not user_families and has_unrecognized_color_hint:
        families = ["custom"]
    else:
        families = list(dict.fromkeys(user_families + sample_families)) or ["black", "brown", "gray"]
    hard_families = [family for family in dict.fromkeys(user_families) if family in families]
    source_hexes = _hex_codes(user_prompt + "\n" + captions)
    user_hexes = _hex_codes(user_prompt)
    source_family_hexes = {
        family: value for family, value in zip(user_families, user_hexes)
    }
    if "custom" in families and user_hexes:
        source_family_hexes["custom"] = user_hexes[0]
    palette: Dict[str, str] = {}
    for family in families:
        palette[family] = _shift_hex(
            source_family_hexes.get(family, _FAMILY_BASE_HEX[family]),
            rng,
            _EXPLORATION_COLOR_SHIFT[exploration_name],
        )

    component_pool = list(dict.fromkeys(_matching_terms(captions + "\n" + user_prompt, _COMPONENT_TERMS)))
    material_pool = list(dict.fromkeys(_matching_terms(captions + "\n" + user_prompt, _MATERIAL_TERMS)))
    relation_pool = list(dict.fromkeys(_matching_terms(captions + "\n" + user_prompt, _RELATION_TERMS)))
    if not component_pool:
        component_pool = ["座椅主面", "座椅侧翼", "中控台面上层", "中控台前饰板"]
    if not material_pool:
        material_pool = ["麂皮", "皮质", "织物"]
    if not relation_pool:
        relation_pool = ["主辅分色", "同色统一", "局部强调"]
    rng.shuffle(component_pool)
    rng.shuffle(material_pool)
    rng.shuffle(relation_pool)
    selected_components = component_pool[: min(4, len(component_pool))]
    assignments = []
    family_keys = list(palette)
    assignment_families = [rng.choice(family_keys) for _ in selected_components]
    if len(selected_components) >= len(hard_families):
        replacement_slots = list(range(len(selected_components)))
        rng.shuffle(replacement_slots)
        for family, slot in zip(hard_families, replacement_slots):
            assignment_families[slot] = family
    for index, component in enumerate(selected_components):
        family = assignment_families[index]
        assignments.append(
            {
                "component": component,
                "family": family,
                "hex": palette[family],
                "material": material_pool[index % len(material_pool)],
            }
        )
    return {
        "exploration_strength": exploration_name,
        "hard_color_families": hard_families,
        "color_families": families,
        "proposed_palette": palette,
        "source_family_hexes": source_family_hexes,
        "component_assignments": assignments,
        "relationship": rng.choice(relation_pool),
        "source_hexes": source_hexes[:12],
        "novel_combination_required": True,
    }


def _effective_temperature(requested: float, exploration_strength: str) -> float:
    strength = _normalize_exploration_strength(exploration_strength)
    value = float(requested)
    if not math.isfinite(value) or value <= 0.0:
        value = _EXPLORATION_GENERATION_TEMPERATURE[strength]
    return max(0.0, min(1.5, value))


def _remove_conflicting_color_families(prompt: str, required_families: Sequence[str]) -> str:
    if not required_families:
        return prompt
    result = prompt
    required = set(required_families)
    for family, terms in _COLOR_FAMILY_TERMS.items():
        if family in required:
            continue
        for term in sorted(terms, key=len, reverse=True):
            if re.search(r"[A-Za-z]", term):
                result = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", "", result, flags=re.IGNORECASE)
            else:
                result = result.replace(term, "")
    return re.sub(r"\s+([,，])", r"\1", re.sub(r"([,，])\s+([,，])", r"\1", result)).strip(" ,，")


def _ensure_color_families(prompt: str, user_prompt: str, language: str) -> str:
    result = _sanitize_prompt(prompt)
    required = _matching_families(user_prompt)
    result = _remove_conflicting_color_families(result, required)
    present = set(_matching_families(result))
    missing = [family for family in required if family not in present]
    if not missing:
        return result
    labels = {
        "black": "黑色系" if language == "zh" else "black color family",
        "brown": "棕色系" if language == "zh" else "brown color family",
        "gray": "灰色系" if language == "zh" else "gray color family",
        "white": "白色系" if language == "zh" else "white color family",
        "blue": "蓝色系" if language == "zh" else "blue color family",
        "green": "绿色系" if language == "zh" else "green color family",
        "red": "红色系" if language == "zh" else "red color family",
        "orange": "橙色系" if language == "zh" else "orange color family",
        "purple": "紫色系" if language == "zh" else "purple color family",
        "yellow": "黄色系" if language == "zh" else "yellow color family",
        "cyan": "青色系" if language == "zh" else "cyan color family",
        "beige": "米色系" if language == "zh" else "beige color family",
    }
    prefix = ", ".join(labels[family] for family in missing)
    return ", ".join(part for part in (prefix, result) if part)


def _build_generation_instruction(
    *,
    record: DatasetRecord,
    user_prompt: str,
    retrieved: List[Dict[str, Any]],
    language: str,
    custom_instruction: str,
    preserve_reference_color: bool,
    variation_plan: Optional[Dict[str, Any]] = None,
    exploration_strength: str = "Medium",
    effective_temperature: float = 0.35,
) -> str:
    examples = "\n".join(f"{item['rank']}. {item['caption']}" for item in retrieved)
    trigger_words = ", ".join(record.trigger_words) or "(none declared)"
    variation_text = json.dumps(variation_plan or {}, ensure_ascii=False, separators=(",", ":"))
    variation_rules = (
        f"Deterministic exploration: {exploration_strength}; effective temperature: {effective_temperature:.2f}.\n"
        f"Seeded CMF variation plan: {variation_text}\n"
        "The user's stated color family is a hard constraint. Exact color names and HEX values are adjustable unless the user explicitly says they are fixed. "
        "Use the plan as a direction, create a novel component-color-material combination, and do not copy a retrieved caption as a whole. "
        "Components may share one color/material; do not force every component to be a different color.\n"
    )
    if language == "zh":
        return variation_rules + (
            "你是 Flux.2 Klein 9B LoRA 训练集提示词工程师。请只输出一条可以直接用于生图的正向提示词，不要解释。\n\n"
            f"用户要求（硬约束，必须满足）：{user_prompt}\n"
            f"数据集：{record.dataset_name} v{record.version}\n"
            f"基础模型：{record.base_model or 'Flux.2 Klein 9B'}\n"
            f"LoRA 元数据（只用于提示词触发词，不要生成 LoRA 权重语法）：{record.lora_name or '(未声明)'}\n"
            f"必须包含的触发词：{trigger_words}\n"
            "数据集样本是软约束：学习其词序、术语、描述密度、标注格式、材质表达和视觉训练分布，但不要照抄任何样本。"
            "用户明确指定的主体、数量、动作、构图、颜色、材质和风格要求优先于数据集偏好。"
            f"参考图颜色策略：{'可以使用参考图颜色' if preserve_reference_color else '只参考主体、结构、比例、视角和构图，不继承参考图颜色与材质'}。\n"
            "要求：保持用户硬约束；生成一条完整的新提示词；自动包含所有触发词；不要输出负面提示词、脚注、引用、编辑指令或 LoRA 权重；不要复述说明文字。\n"
            f"额外指令：{custom_instruction or '无'}\n\n"
            f"检索到的训练样本：\n{examples}"
        )
    return variation_rules + (
        "You are a Flux.2 Klein 9B LoRA training-caption prompt engineer. Output exactly one ready-to-use positive image prompt and nothing else.\n\n"
        f"User requirements (hard constraints): {user_prompt}\n"
        f"Dataset: {record.dataset_name} v{record.version}\n"
        f"Base model: {record.base_model or 'Flux.2 Klein 9B'}\n"
        f"LoRA metadata: {record.lora_name or '(not declared)'}\n"
        f"Required trigger words: {trigger_words}\n"
        "Treat retrieved captions as soft constraints: learn their wording order, terminology, density, annotation grammar, material vocabulary, and visual training distribution, but never copy a caption verbatim. "
        "User-specified subject, count, action, composition, color, material, and style requirements always win. "
        f"Reference image policy: {'colors may be preserved' if preserve_reference_color else 'use only subject, structure, proportions, viewpoint, and composition; do not inherit source colors or materials'}.\n"
        "Output one complete new positive prompt, include every required trigger word, and do not output negative prompts, citations, editing instructions, explanations, or LoRA weight syntax.\n"
        f"Extra instruction: {custom_instruction or 'none'}\n\n"
        f"Retrieved training captions:\n{examples}"
    )


class DatasetCaptionPickerNode:
    @classmethod
    def INPUT_TYPES(cls):
        options = _dataset_options()
        return {
            "required": {
                "dataset_name": (options, {"default": options[0]}),
                "selection_mode": (_SELECTION_OPTIONS, {"default": "Random"}),
                "selection_seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("caption", "caption_index", "trigger_words", "dataset_metadata")
    FUNCTION = "pick_caption"
    CATEGORY = "IAT/Dataset"

    @classmethod
    def IS_CHANGED(cls, dataset_name, **kwargs):
        return _dataset_change_token(dataset_name)

    def pick_caption(self, dataset_name, selection_mode, selection_seed, index):
        record = _selected_record(dataset_name)
        try:
            entry, selected_index = choose_caption(record, selection_mode, selection_seed, index)
        except Exception as exc:
            raise RuntimeError(f"[IAT] Caption selection failed: {exc}") from exc
        metadata = dataset_metadata(record)
        metadata["selected_record_id"] = entry.record_id
        metadata["selected_image_path"] = entry.relative_image_path
        metadata["selected_image_paths"] = entry.grouped_relative_image_paths()
        metadata["selected_image_roles"] = list(entry.grouped_relative_image_paths())
        return (
            entry.caption,
            selected_index,
            ", ".join(record.trigger_words),
            json.dumps(metadata, ensure_ascii=False),
        )


class DatasetRAGPromptGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        options = _dataset_options()
        return {
            "required": {
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "dataset_name": (options, {"default": options[0]}),
                "backend": (_BACKEND_OPTIONS, {"default": _DEFAULT_BACKEND}),
                "model_override": ("STRING", {"default": ""}),
                "base_url_override": ("STRING", {"default": ""}),
                "retrieval_seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1}),
                "generation_seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1}),
                "exploration_strength": (_EXPLORATION_OPTIONS, {"default": "Medium"}),
                "variation_seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1}),
                "top_k": ("INT", {"default": 4, "min": 1, "max": 8}),
                "preserve_reference_color": ("BOOLEAN", {"default": False}),
                "custom_instruction": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0}),
                "timeout_seconds": ("INT", {"default": int(_LLM_CFG.get("timeout_seconds") or 300), "min": 5, "max": 900}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "retrieved_captions", "retrieval_debug", "dataset_metadata")
    FUNCTION = "generate_prompt"
    CATEGORY = "IAT/Dataset"

    @classmethod
    def IS_CHANGED(cls, dataset_name, **kwargs):
        return _dataset_change_token(dataset_name)

    def generate_prompt(
        self,
        user_prompt,
        dataset_name,
        backend,
        model_override,
        base_url_override,
        retrieval_seed,
        generation_seed,
        top_k,
        preserve_reference_color,
        custom_instruction,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        timeout_seconds,
        exploration_strength="Medium",
        variation_seed=1,
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
    ):
        if not (user_prompt or "").strip():
            raise RuntimeError("[IAT] user_prompt is required.")

        record = _selected_record(dataset_name)

        try:
            reference_images = _collect_reference_images(image, image_2, image_3, image_4)
            record_fingerprint = dataset_fingerprint(record)
            prompt_text = (user_prompt or "").strip()
            dataset_identity = (record.dataset_name, record.version, record_fingerprint)
            effective_retrieval_seed = _derive_seed(
                retrieval_seed,
                "retrieval",
                dataset_identity,
                prompt_text,
            )
            effective_composition_seed = _derive_seed(
                variation_seed,
                "composition",
                dataset_identity,
                prompt_text,
            )
            effective_temperature = _effective_temperature(float(temperature), exploration_strength)
            index = get_dataset_index(
                record,
                _index_cache_root(),
                embedding_model_path=_embedding_model_path(),
                require_embeddings=bool(_embedding_model_path()),
                embedding_device=_EMBEDDING_DEVICE,
                embedding_batch_size=_EMBEDDING_BATCH_SIZE,
            )
            retrieved, debug = index.retrieve(
                (user_prompt or "").strip(),
                reference_images=reference_images,
                preserve_reference_color=bool(preserve_reference_color),
                top_k=top_k,
                seed=effective_retrieval_seed,
                exploration_strength=exploration_strength,
            )
            variation_plan = _build_variation_plan(
                prompt_text,
                retrieved,
                effective_composition_seed,
                exploration_strength,
            )
            debug["variation_seed"] = int(variation_seed)
            debug["requested_retrieval_seed"] = int(retrieval_seed)
            debug["requested_generation_seed"] = int(generation_seed)
            debug["retrieval_seed"] = effective_retrieval_seed
            debug["composition_seed"] = effective_composition_seed
            debug["effective_temperature"] = effective_temperature
            debug["variation_plan"] = variation_plan
            debug["dataset_root"] = str(_dataset_root())

            defaults = _backend_defaults(backend)
            model = (model_override or "").strip() or defaults["model"]
            base_url = (base_url_override or "").strip() or defaults["base_url"]
            effective_generation_seed = _derive_seed(
                generation_seed,
                "generation",
                dataset_identity,
                prompt_text,
                backend,
                model,
            )
            debug["generation_seed"] = effective_generation_seed
            language = _output_language(record, "Auto")
            generation_prompt = _build_generation_instruction(
                record=record,
                user_prompt=prompt_text,
                retrieved=retrieved,
                language=language,
                custom_instruction=(custom_instruction or "").strip(),
                preserve_reference_color=bool(preserve_reference_color),
                variation_plan=variation_plan,
                exploration_strength=exploration_strength,
                effective_temperature=effective_temperature,
            )
            output = generate_with_backend(
                backend=backend,
                model=model,
                base_url=base_url,
                prompt=generation_prompt,
                images=_generation_images(reference_images, bool(preserve_reference_color)),
                max_tokens=max_tokens,
                temperature=effective_temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=effective_generation_seed,
                timeout=timeout_seconds,
                local_device=str(_MODEL_CFG.get("device") or "cuda"),
                local_attention_backend=_DEFAULT_ATTENTION_BACKEND,
                keep_local_model_loaded=True,
                ollama_keep_alive=_OLLAMA_CFG.get("keep_alive", -1),
                ollama_think=bool(_OLLAMA_CFG.get("think", False)),
                vllm_api_key=str(_VLLM_CFG.get("api_key") or ""),
            )
            if not (output or "").strip():
                raise BackendError("[IAT] Generation backend returned an empty prompt.")
            final_prompt = _ensure_color_families(output, prompt_text, language)
            final_prompt = _ensure_trigger_words(final_prompt, record)
            if not final_prompt:
                raise BackendError("[IAT] Generation backend returned an empty prompt.")
            return (
                final_prompt,
                json.dumps(retrieved, ensure_ascii=False),
                json.dumps(debug, ensure_ascii=False),
                json.dumps(dataset_metadata(record), ensure_ascii=False),
            )
        except (DatasetError, EmbeddingModelUnavailable, BackendError) as exc:
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            raise RuntimeError(f"[IAT] Dataset RAG failed: {exc}") from exc


NODE_CLASS_MAPPINGS = {
    "DatasetCaptionPicker by IAT": DatasetCaptionPickerNode,
    "DatasetRAGPromptGenerator by IAT": DatasetRAGPromptGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DatasetCaptionPicker by IAT": "Dataset Caption Picker（IAT）",
    "DatasetRAGPromptGenerator by IAT": "Dataset RAG Prompt Generator（IAT）",
}
