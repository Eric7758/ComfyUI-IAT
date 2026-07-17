from __future__ import annotations

import json
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


def _build_generation_instruction(
    *,
    record: DatasetRecord,
    user_prompt: str,
    retrieved: List[Dict[str, Any]],
    language: str,
    custom_instruction: str,
    preserve_reference_color: bool,
) -> str:
    examples = "\n".join(f"{item['rank']}. {item['caption']}" for item in retrieved)
    trigger_words = ", ".join(record.trigger_words) or "(none declared)"
    if language == "zh":
        return (
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
    return (
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
                seed=retrieval_seed,
            )
            debug["retrieval_seed"] = int(retrieval_seed)
            debug["generation_seed"] = int(generation_seed)
            debug["dataset_root"] = str(_dataset_root())

            defaults = _backend_defaults(backend)
            model = (model_override or "").strip() or defaults["model"]
            base_url = (base_url_override or "").strip() or defaults["base_url"]
            language = _output_language(record, "Auto")
            generation_prompt = _build_generation_instruction(
                record=record,
                user_prompt=(user_prompt or "").strip(),
                retrieved=retrieved,
                language=language,
                custom_instruction=(custom_instruction or "").strip(),
                preserve_reference_color=bool(preserve_reference_color),
            )
            output = generate_with_backend(
                backend=backend,
                model=model,
                base_url=base_url,
                prompt=generation_prompt,
                images=_generation_images(reference_images, bool(preserve_reference_color)),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=generation_seed,
                timeout=timeout_seconds,
                local_device=str(_MODEL_CFG.get("device") or "cuda"),
                local_attention_backend=_DEFAULT_ATTENTION_BACKEND,
                keep_local_model_loaded=True,
                ollama_keep_alive=_OLLAMA_CFG.get("keep_alive", -1),
                ollama_think=bool(_OLLAMA_CFG.get("think", False)),
                vllm_api_key=str(_VLLM_CFG.get("api_key") or ""),
            )
            final_prompt = _ensure_trigger_words(output, record)
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
