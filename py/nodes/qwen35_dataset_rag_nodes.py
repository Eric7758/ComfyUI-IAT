from __future__ import annotations

import sys
from pathlib import Path
import re

from PIL import Image

from .prompt_kb import (
    discover_prompt_kb_records,
    get_prompt_kb_dataset_options,
    resolve_output_language,
    select_palette_reference_captions,
    select_caption_by_seed,
    retrieve_top_captions,
)
from .qwen35_runtime import (
    ATTENTION_OPTIONS,
    DEFAULT_ATTENTION_BACKEND,
    DEVICE_OPTIONS,
    VL_MODEL_CANDIDATES,
    VL_MODEL_LABEL_TO_VARIANT,
    VL_MODEL_OPTIONS_GROUPED,
    generate_vision_text,
    unload_all_models,
)

_CFG = getattr(sys.modules.get("comfyui_iat_config"), "data", {}) or {}
_MODEL_CFG = (_CFG.get("model") or {}) if isinstance(_CFG, dict) else {}
_DEFAULT_VARIANT = _MODEL_CFG.get("default_variant", "Qwen3.5-2B")
_DEFAULT_DEVICE = _MODEL_CFG.get("device", "cuda")
_LANGUAGE_OPTIONS = ["Auto", "English", "中文"]
_GENERATION_MODE_OPTIONS = ["Dataset RAG", "Direct Caption"]
_KB_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "prompts_kb"

DATASET_KB_RECORDS, DATASET_KB_ERRORS = discover_prompt_kb_records(_KB_ROOT)


def _tensor_to_pil_list(image):
    if image is None:
        return []
    if image.dim() == 3:
        image = image.unsqueeze(0)

    pil_images = []
    for idx in range(image.shape[0]):
        array = (image[idx].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        pil_images.append(Image.fromarray(array))
    return pil_images


def _to_vl_variant(selection: str) -> str:
    return VL_MODEL_LABEL_TO_VARIANT.get(selection, selection)


def _to_achromatic_reference(images):
    achromatic_images = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        gray = image.convert("L").convert("RGB")
        achromatic_images.append(gray)
    return achromatic_images


def _build_retrieval_prompt(language: str) -> str:
    if language == "zh":
        return (
            "请只输出一段用于检索训练集标注样本的中性视觉描述。"
            "聚焦主体、分件关系、结构、体量、视角、构图、场景与风格关键词。"
            "忽略原图色彩，不要根据原图颜色做描述。"
            "不要写解释、不要分点、不要加入训练集触发词。"
        )
    return (
        "Output one neutral visual retrieval description only. "
        "Focus on subject, part breakdown, structure, massing, view, composition, scene, and style keywords. "
        "Ignore the source image colors. Do not describe its existing color palette. "
        "Do not explain. Do not add dataset trigger words."
    )


def _sanitize_output_prompt(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"\s*\[cite:\s*\d+\s*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[\s*\d+\s*\]\s*$", "", cleaned)
    cleaned = re.sub(r"^(?:最终提示词|提示词|Prompt)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _build_generation_prompt(
    *,
    record,
    language: str,
    include_trigger_words: bool,
    custom_instruction: str,
    palette_captions,
    top_captions,
) -> str:
    trigger_text = ", ".join(record.trigger_words) if include_trigger_words else "(disabled)"
    palette_block = "\n".join(f"P{idx + 1}. {caption}" for idx, caption in enumerate(palette_captions))
    caption_block = "\n".join(f"{idx + 1}. {caption}" for idx, caption in enumerate(top_captions))
    if language == "zh":
        return (
            "你是 LoRA 训练集提示词迁移工程师。"
            "任务：基于输入图像的主体、结构、构图和使用场景，生成一条全新的正向生成提示词，"
            "并严格贴合指定数据集的标注风格。\n\n"
            f"数据集名称：{record.dataset_name}\n"
            f"数据集版本：{record.version}\n"
            f"触发词：{trigger_text}\n"
            "色彩/材质参考原则：优先采用下方“CMF参考样本”里的配色、材质、纹理和饰件组合，不要默认回到黑灰色系。\n"
            "要求：\n"
            "1. 只输出一条最终提示词，不要解释。\n"
            "2. 学习 few-shot 样本的词序、术语密度、标签语法、段落节奏和风格描述方式。\n"
            "3. 保持整体格式和句式组织方式稳定，但必须改写其中具体的色彩、材质、纹理、饰件与氛围描述。\n"
            "3.1 如果 few-shot 样本呈现出固定句式骨架或分号分隔格式，优先继承这种格式骨架，但把具体 CMF 内容重写为新的方案。\n"
            "4. 不要把结果写成“为这张图”“将其改为”这类编辑指令，要直接输出可用于生成的新 prompt。\n"
            "5. 输入图仅用于提供造型、分件、比例、构图和使用场景参考；不要继承原图颜色方案。\n"
            "6. 色彩组合、材质搭配、纹理方向和饰件细节优先从数据集样本风格中提炼，并重新组合出新的 CMF 方案。\n"
            "7. 不要复述输入图已有的 CMF 细节；若用户要求新风格、新颜色或新材质，必须生成与输入图不同的颜色、材质、拼色关系、纹理和饰件描述。\n"
            "8. 当数据集中存在更鲜明的拼色、撞色或亮色样本时，优先从这些样本提炼色彩组合，而不是退回保守的黑灰棕默认答案。\n"
            "9. 可以参考输入图的主体类别与构图，但视觉风格、CMF 和气质可以整体重设计，重点是探索更符合数据集风格的新描述。\n"
            "10. 如果输入图是内饰/产品/界面等设计图，优先输出适合该对象的可落地 CMF 方案，而不是对原图做高相似复述。\n"
            "11. 输出语言使用中文，除非样本中的英文术语更自然。\n"
            "12. 不要输出负向提示词、引用标记、脚注、[cite: n]、LoRA 权重。\n"
            f"13. 额外指令：{custom_instruction or '无'}\n\n"
            "CMF参考样本：\n"
            f"{palette_block}\n\n"
            "Few-shot 样本：\n"
            f"{caption_block}"
        )
    return (
        "You are a prompt engineer for LoRA training-style prompt transfer. "
        "Generate one brand-new positive generation prompt from the input image semantics while matching the dataset annotation style.\n\n"
        f"Dataset: {record.dataset_name}\n"
        f"Version: {record.version}\n"
        f"Trigger words: {trigger_text}\n"
        "Color/material rule: prefer the color combinations, material pairings, textures, and trim cues from the CMF reference samples below instead of defaulting back to generic dark off-road palettes.\n"
        "Requirements:\n"
        "1. Output one final prompt only.\n"
        "2. Mimic the few-shot examples' wording order, term density, label grammar, structural rhythm, and style description patterns.\n"
        "3. Keep the overall format and sentence organization stable, but change the concrete colors, materials, textures, trim details, and mood descriptors.\n"
        "3.1 If the few-shot examples use a stable sentence template or semicolon-separated format, keep that structural skeleton while rewriting the actual CMF content.\n"
        "4. Do not write an editing instruction such as 'redesign this image' or 'for this image'; output a ready-to-use generation prompt directly.\n"
        "5. Use the input image only as a reference for form, part breakdown, proportions, composition, and use case; do not inherit its source color palette.\n"
        "6. Derive color combinations, material pairings, texture directions, and trim accents from the dataset style and recombine them into a new CMF proposal.\n"
        "7. If the user asks for a new style, new colors, or new materials, you must replace the source image CMF with different color stories, material pairings, texture treatments, and trim details.\n"
        "8. When the dataset contains brighter, contrasting, or more distinctive colorways, prefer those combinations over conservative black/gray/brown defaults.\n"
        "9. For interior, product, or interface design inputs, prioritize a plausible new CMF proposal instead of a high-similarity restatement of the source image.\n"
        "10. Do not copy any example verbatim.\n"
        "11. Output in English.\n"
        "12. Do not include negative prompt text, citations, footnotes, [cite: n], or LoRA weights.\n"
        f"13. Extra instruction: {custom_instruction or 'none'}\n\n"
        "CMF reference samples:\n"
        f"{palette_block}\n\n"
        "Few-shot examples:\n"
        f"{caption_block}"
    )


class Qwen35DatasetRAGReversePromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        default_variant = (
            _DEFAULT_VARIANT if _DEFAULT_VARIANT in VL_MODEL_CANDIDATES else list(VL_MODEL_CANDIDATES.keys())[0]
        )
        default_variant_label = next(
            (key for key, value in VL_MODEL_LABEL_TO_VARIANT.items() if value == default_variant),
            VL_MODEL_OPTIONS_GROUPED[0],
        )
        default_device = _DEFAULT_DEVICE if _DEFAULT_DEVICE in DEVICE_OPTIONS else "cuda"
        default_attention_backend = (
            DEFAULT_ATTENTION_BACKEND if DEFAULT_ATTENTION_BACKEND in ATTENTION_OPTIONS else "SDPA"
        )
        dataset_options = get_prompt_kb_dataset_options(_KB_ROOT)
        dataset_default = dataset_options[0] if dataset_options else "__NO_DATASET_FOUND__"

        return {
            "required": {
                "model_variant": (VL_MODEL_OPTIONS_GROUPED, {"default": default_variant_label}),
                "device": (DEVICE_OPTIONS, {"default": default_device}),
                "attention_backend": (ATTENTION_OPTIONS, {"default": default_attention_backend}),
                "dataset_name": (
                    dataset_options if dataset_options else [dataset_default],
                    {
                        "default": dataset_default,
                        "tooltip": "Knowledge-base dataset to mimic. Refresh requires ComfyUI restart.",
                    },
                ),
                "generation_mode": (
                    _GENERATION_MODE_OPTIONS,
                    {
                        "default": "Dataset RAG",
                        "tooltip": "Dataset RAG uses Qwen with retrieval. Direct Caption skips language-model generation and returns one dataset caption by seed.",
                    },
                ),
                "language": (_LANGUAGE_OPTIONS, {"default": "Auto"}),
                "few_shot_count": ("INT", {"default": 6, "min": 1, "max": 16}),
                "include_trigger_words": ("BOOLEAN", {"default": True}),
                "custom_instruction": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 256, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "reverse_prompt_with_dataset_rag"
    CATEGORY = "IAT/Qwen3.5"

    def reverse_prompt_with_dataset_rag(
        self,
        model_variant,
        device,
        attention_backend,
        dataset_name,
        generation_mode,
        language,
        few_shot_count,
        include_trigger_words,
        custom_instruction,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        image,
    ):
        pil_images = _tensor_to_pil_list(image)
        if len(pil_images) == 0:
            return ("[IAT] Please connect one IMAGE input for dataset RAG reverse prompt.",)

        if DATASET_KB_ERRORS:
            return ("\n".join(DATASET_KB_ERRORS),)

        record = DATASET_KB_RECORDS.get(dataset_name)
        if record is None:
            return (
                f"[IAT] Dataset `{dataset_name}` was not found under `{_KB_ROOT}`. "
                "Restart ComfyUI after adding JSON files.",
            )

        if generation_mode == "Direct Caption":
            return (_sanitize_output_prompt(select_caption_by_seed(record, seed)),)

        model_variant = _to_vl_variant(model_variant)
        output_language = resolve_output_language(language, record)
        achromatic_images = _to_achromatic_reference(pil_images)
        retrieval_description = generate_vision_text(
            variant=model_variant,
            device=device,
            attention_backend=attention_backend,
            images=achromatic_images,
            text_prompt=_build_retrieval_prompt(output_language),
            max_tokens=min(max_tokens, 192),
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=seed,
        )

        top_captions = retrieve_top_captions(retrieval_description, record, few_shot_count)
        palette_captions = select_palette_reference_captions(record, seed=seed, count=2)
        final_prompt = generate_vision_text(
            variant=model_variant,
            device=device,
            attention_backend=attention_backend,
            images=achromatic_images,
            text_prompt=_build_generation_prompt(
                record=record,
                language=output_language,
                include_trigger_words=include_trigger_words,
                custom_instruction=(custom_instruction or "").strip(),
                palette_captions=palette_captions,
                top_captions=top_captions,
            ),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        if not keep_model_loaded:
            unload_all_models()
        return (_sanitize_output_prompt(final_prompt),)


NODE_CLASS_MAPPINGS = {
    "Qwen35DatasetRAGReversePrompt by IAT": Qwen35DatasetRAGReversePromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35DatasetRAGReversePrompt by IAT": "Qwen3.5 数据集RAG反推提示词（IAT）",
}
