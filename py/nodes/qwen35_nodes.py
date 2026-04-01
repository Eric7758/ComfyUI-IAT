import re
import sys

from PIL import Image

from .qwen35_runtime import (
    ATTENTION_OPTIONS,
    DEFAULT_ATTENTION_BACKEND,
    DEVICE_OPTIONS,
    QUANT_OPTIONS,
    TEXT_MODEL_CANDIDATES,
    VL_MODEL_CANDIDATES,
    generate_text,
    generate_vision_text,
    unload_all_models,
)

_CFG = getattr(sys.modules.get("comfyui_iat_config"), "data", {}) or {}
_MODEL_CFG = (_CFG.get("model") or {}) if isinstance(_CFG, dict) else {}
_DEFAULT_VARIANT = _MODEL_CFG.get("default_variant", "Qwen3.5-2B")
_DEFAULT_QUANT = _MODEL_CFG.get("quantization", "None (FP16/BF16)")
_DEFAULT_DEVICE = _MODEL_CFG.get("device", "auto")
TRANSLATION_TARGET_OPTIONS = ["English", "Chinese"]

PROMPT_STYLES = {
    "Enhance": "Expand and enrich this prompt with vivid details while keeping the original intent.",
    "Refine": "Rewrite this prompt to be clear, concise, and production-ready for image generation.",
    "Creative Rewrite": "Rewrite this prompt creatively with stronger visual storytelling.",
    "Detailed Visual": "Convert this prompt into a highly detailed visual description for text-to-image models.",
}

REVERSE_PRESETS = {
    "Detailed Description": "Describe this image in detail and output an English image generation prompt only.",
    "Prompt Reverse": "Infer a compact production-grade positive prompt from this image. English only.",
    "Style Focus": "Describe style, camera, lighting, color, composition, and materials as a reusable generation prompt.",
}

KONTEXT_SYSTEM_PROMPT = """You are a prompt engineer specialized in image editing prompts.
Rewrite user requests into clean English prompts that are explicit, editable, and production-ready.
Rules:
1) Be concrete and visual, avoid vague words.
2) Preserve identity and composition when user requires consistency.
3) For replacement tasks, use exact text/object names.
4) Output only one final English prompt, no explanations.
"""


def _tensor_to_pil_list(image):
    if image is None:
        return []
    if image.dim() == 3:
        image = image.unsqueeze(0)

    pil_images = []
    for i in range(image.shape[0]):
        array = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        pil_images.append(Image.fromarray(array))
    return pil_images


def _collect_pil_images(*image_inputs):
    merged = []
    for img in image_inputs:
        merged.extend(_tensor_to_pil_list(img))
    return merged


def _detect_language(text: str):
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", text):
        return "ja" if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text) else "zh"
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return None


class Qwen35PromptEnhancerNode:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取默认模型，如果不存在则使用第一个可用模型
        default_variant = _DEFAULT_VARIANT if _DEFAULT_VARIANT in TEXT_MODEL_CANDIDATES else list(TEXT_MODEL_CANDIDATES.keys())[0]
        default_quant = _DEFAULT_QUANT if _DEFAULT_QUANT in QUANT_OPTIONS else "None (FP16/BF16)"
        default_device = _DEFAULT_DEVICE if _DEFAULT_DEVICE in DEVICE_OPTIONS else "auto"
        default_attention_backend = DEFAULT_ATTENTION_BACKEND if DEFAULT_ATTENTION_BACKEND in ATTENTION_OPTIONS else "Auto"
        
        return {
            "required": {
                "model_variant": (list(TEXT_MODEL_CANDIDATES.keys()), {"default": default_variant}),
                "quantization": (QUANT_OPTIONS, {"default": default_quant}),
                "device": (DEVICE_OPTIONS, {"default": default_device}),
                "attention_backend": (ATTENTION_OPTIONS, {"default": default_attention_backend}),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "enhancement_style": (list(PROMPT_STYLES.keys()), {"default": "Enhance"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 256, "min": 32, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ENHANCED_OUTPUT",)
    FUNCTION = "enhance_prompt"
    CATEGORY = "IAT/Qwen3.5"

    def enhance_prompt(
        self,
        model_variant,
        quantization,
        device,
        attention_backend,
        prompt_text,
        enhancement_style,
        custom_system_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
    ):
        system_prompt = (custom_system_prompt or "").strip() or PROMPT_STYLES.get(enhancement_style, PROMPT_STYLES["Enhance"])
        user_prompt = (prompt_text or "").strip() or "Describe a cinematic scene in rich visual detail."

        text = generate_text(
            variant=model_variant,
            quantization=quantization,
            device=device,
            attention_backend=attention_backend,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        if not keep_model_loaded:
            unload_all_models()
        return (text,)


class Qwen35ReversePromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取默认模型，如果不存在则使用第一个可用模型
        default_variant = _DEFAULT_VARIANT if _DEFAULT_VARIANT in VL_MODEL_CANDIDATES else list(VL_MODEL_CANDIDATES.keys())[0]
        default_quant = _DEFAULT_QUANT if _DEFAULT_QUANT in QUANT_OPTIONS else "None (FP16/BF16)"
        default_device = _DEFAULT_DEVICE if _DEFAULT_DEVICE in DEVICE_OPTIONS else "auto"
        default_attention_backend = DEFAULT_ATTENTION_BACKEND if DEFAULT_ATTENTION_BACKEND in ATTENTION_OPTIONS else "Auto"
        
        return {
            "required": {
                "model_variant": (list(VL_MODEL_CANDIDATES.keys()), {"default": default_variant}),
                "quantization": (QUANT_OPTIONS, {"default": default_quant}),
                "device": (DEVICE_OPTIONS, {"default": default_device}),
                "attention_backend": (ATTENTION_OPTIONS, {"default": default_attention_backend}),
                "preset_prompt": (list(REVERSE_PRESETS.keys()), {"default": "Detailed Description"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 192, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "reverse_prompt"
    CATEGORY = "IAT/Qwen3.5"

    def reverse_prompt(
        self,
        model_variant,
        quantization,
        device,
        attention_backend,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
    ):
        pil_images = _collect_pil_images(image, image_2, image_3, image_4)
        if len(pil_images) == 0:
            return ("[IAT] Please connect at least one IMAGE input for reverse prompt.",)

        text_prompt = (custom_prompt or "").strip() or REVERSE_PRESETS.get(preset_prompt, REVERSE_PRESETS["Detailed Description"])

        text = generate_vision_text(
            variant=model_variant,
            quantization=quantization,
            device=device,
            attention_backend=attention_backend,
            images=pil_images,
            text_prompt=text_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        if not keep_model_loaded:
            unload_all_models()
        return (text,)


class QwenTranslatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取默认模型，如果不存在则使用第一个可用模型
        default_variant = _DEFAULT_VARIANT if _DEFAULT_VARIANT in TEXT_MODEL_CANDIDATES else list(TEXT_MODEL_CANDIDATES.keys())[0]
        default_quant = _DEFAULT_QUANT if _DEFAULT_QUANT in QUANT_OPTIONS else "None (FP16/BF16)"
        default_device = _DEFAULT_DEVICE if _DEFAULT_DEVICE in DEVICE_OPTIONS else "auto"
        default_attention_backend = DEFAULT_ATTENTION_BACKEND if DEFAULT_ATTENTION_BACKEND in ATTENTION_OPTIONS else "Auto"
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Please input text to translate."}),
                "target_language": (TRANSLATION_TARGET_OPTIONS, {"default": "English"}),
                "model_variant": (list(TEXT_MODEL_CANDIDATES.keys()), {"default": default_variant}),
                "quantization": (QUANT_OPTIONS, {"default": default_quant}),
                "device": (DEVICE_OPTIONS, {"default": default_device}),
                "attention_backend": (ATTENTION_OPTIONS, {"default": default_attention_backend}),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.5}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate"
    CATEGORY = "IAT/Qwen3.5"

    def translate(self, text, target_language, model_variant, quantization, device, attention_backend, max_tokens, temperature, keep_model_loaded, seed):
        src = (text or "").strip()
        if not src:
            return ("",)

        lang = _detect_language(src)
        target_code = "en" if target_language == "English" else "zh"
        if lang == target_code:
            return (src,)

        target_prompt = "natural English" if target_code == "en" else "自然流畅的中文"
        response = generate_text(
            variant=model_variant,
            quantization=quantization,
            device=device,
            attention_backend=attention_backend,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate the user text to {target_prompt}. Return only the translation.",
                },
                {"role": "user", "content": src},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=seed,
        )

        if not keep_model_loaded:
            unload_all_models()
        return (response.strip(),)


class QwenKontextTranslatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取默认模型，如果不存在则使用第一个可用模型
        default_variant = _DEFAULT_VARIANT if _DEFAULT_VARIANT in TEXT_MODEL_CANDIDATES else list(TEXT_MODEL_CANDIDATES.keys())[0]
        default_quant = _DEFAULT_QUANT if _DEFAULT_QUANT in QUANT_OPTIONS else "None (FP16/BF16)"
        default_device = _DEFAULT_DEVICE if _DEFAULT_DEVICE in DEVICE_OPTIONS else "auto"
        default_attention_backend = DEFAULT_ATTENTION_BACKEND if DEFAULT_ATTENTION_BACKEND in ATTENTION_OPTIONS else "Auto"
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Please input editing instruction."}),
                "model_variant": (list(TEXT_MODEL_CANDIDATES.keys()), {"default": default_variant}),
                "quantization": (QUANT_OPTIONS, {"default": default_quant}),
                "device": (DEVICE_OPTIONS, {"default": default_device}),
                "attention_backend": (ATTENTION_OPTIONS, {"default": default_attention_backend}),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("optimized_prompt",)
    FUNCTION = "optimize_prompt"
    CATEGORY = "IAT/Qwen3.5"

    def optimize_prompt(self, text, model_variant, quantization, device, attention_backend, max_tokens, temperature, keep_model_loaded, seed):
        src = (text or "").strip()
        if not src:
            return ("",)

        response = generate_text(
            variant=model_variant,
            quantization=quantization,
            device=device,
            attention_backend=attention_backend,
            messages=[
                {"role": "system", "content": KONTEXT_SYSTEM_PROMPT},
                {"role": "user", "content": src},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            seed=seed,
        )

        if "```" in response:
            parts = response.split("```")
            response = parts[1] if len(parts) > 1 else parts[0]

        if not keep_model_loaded:
            unload_all_models()
        return (response.strip(),)


NODE_CLASS_MAPPINGS = {
    "Qwen35PromptEnhancer by IAT": Qwen35PromptEnhancerNode,
    "Qwen35ReversePrompt by IAT": Qwen35ReversePromptNode,
    "QwenTranslator by IAT": QwenTranslatorNode,
    "QwenKontextTranslator by IAT": QwenKontextTranslatorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35PromptEnhancer by IAT": "Qwen3.5 提示词增强器（IAT）",
    "Qwen35ReversePrompt by IAT": "Qwen3.5 反推提示词（IAT）",
    "QwenTranslator by IAT": "Qwen 翻译器（IAT）",
    "QwenKontextTranslator by IAT": "Qwen 编辑提示词优化（IAT）",
}
