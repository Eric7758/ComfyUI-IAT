import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
except Exception:
    from modelscope import snapshot_download as ms_snapshot_download

import folder_paths


QWEN35_MODELS = {
    "Qwen3.5-0.8B": [
        "Qwen/Qwen3.5-0.8B-Instruct",
        "Qwen/Qwen3.5-0.8B",
    ],
    "Qwen3.5-2B": [
        "Qwen/Qwen3.5-2B-Instruct",
        "Qwen/Qwen3.5-2B",
    ],
    "Qwen3.5-4B": [
        "Qwen/Qwen3.5-4B-Instruct",
        "Qwen/Qwen3.5-4B",
    ],
    "Qwen3.5-9B": [
        "Qwen/Qwen3.5-9B-Instruct",
        "Qwen/Qwen3.5-9B",
    ],
    "Qwen3.5-27B": [
        "Qwen/Qwen3.5-27B-Instruct",
        "Qwen/Qwen3.5-27B",
    ],
    "Qwen3.5-35B": [
        "Qwen/Qwen3.5-35B-A3B-Instruct",
        "Qwen/Qwen3.5-35B-A3B",
    ],
}

QUANT_OPTIONS = ["None (FP16/BF16)", "8-bit", "4-bit"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu", "mps"]

PROMPT_STYLES = {
    "Enhance": "Expand and enrich this prompt with vivid details while keeping the original intent.",
    "Refine": "Rewrite this prompt to be clear, concise, and production-ready for image generation.",
    "Creative Rewrite": "Rewrite this prompt creatively with stronger visual storytelling.",
    "Detailed Visual": "Convert this prompt into a highly detailed visual description for text-to-image models.",
}

REVERSE_PRESETS = {
    "Detailed Description": "Describe this image in detail. Output a high-quality image generation prompt in English only.",
    "Prompt Reverse": "Infer a compact, production-grade positive prompt from this image. English only.",
    "Style Focus": "Describe style, camera, lighting, color, composition, and materials as a reusable generation prompt.",
}

_MODEL_CACHE = {
    "text": {"model": None, "tokenizer": None, "signature": None},
    "vl": {"model": None, "tokenizer": None, "processor": None, "signature": None},
}


def _get_llm_dir() -> Path:
    llm_paths = []
    if hasattr(folder_paths, "folder_names_and_paths") and "LLM" in folder_paths.folder_names_and_paths:
        llm_paths = folder_paths.get_folder_paths("LLM")
    if llm_paths:
        llm_dir = Path(llm_paths[0])
    else:
        llm_dir = Path(folder_paths.models_dir) / "LLM"
    llm_dir.mkdir(parents=True, exist_ok=True)
    return llm_dir


def _model_local_dir(repo_id: str) -> Path:
    return _get_llm_dir() / repo_id.split("/")[-1]


def _has_weights(model_dir: Path) -> bool:
    patterns = ("*.safetensors", "*.bin", "*.pt")
    return any(model_dir.glob(pattern) for pattern in patterns)


def _download_from_modelscope(repo_id: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        ms_snapshot_download(
            model_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        ms_snapshot_download(repo_id, cache_dir=str(local_dir.parent))
    return local_dir


def _ensure_model(model_name: str) -> Path:
    candidates = QWEN35_MODELS.get(model_name)
    if not candidates:
        raise ValueError(f"[IAT] Unsupported Qwen3.5 model: {model_name}")

    last_error = None
    for repo_id in candidates:
        target = _model_local_dir(repo_id)
        if target.exists() and _has_weights(target):
            return target
        try:
            downloaded = _download_from_modelscope(repo_id, target)
            if _has_weights(downloaded):
                print(f"[IAT] Downloaded model from ModelScope: {repo_id}")
                return downloaded
        except Exception as exc:
            last_error = exc
            print(f"[IAT] ModelScope download failed for {repo_id}: {exc}")
    raise RuntimeError(f"[IAT] Unable to download model {model_name} from ModelScope: {last_error}")


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return "cpu"
    return device


def _dtype_for_device(device: str):
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _quant_config(quantization: str):
    if quantization == "None (FP16/BF16)":
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError("[IAT] bitsandbytes is required for 4-bit/8-bit quantization.")
    if quantization == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "4-bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    raise ValueError(f"[IAT] Unsupported quantization: {quantization}")


def _apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _clear_cache(kind: str):
    cache = _MODEL_CACHE[kind]
    for key in list(cache.keys()):
        cache[key] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_text_model(model_name: str, quantization: str, device: str):
    model_dir = _ensure_model(model_name)
    run_device = _resolve_device(device)
    signature = (str(model_dir), quantization, run_device)
    cache = _MODEL_CACHE["text"]
    if cache["model"] is not None and cache["signature"] == signature:
        return cache["model"], cache["tokenizer"], run_device

    _clear_cache("text")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    qcfg = _quant_config(quantization)
    load_kwargs = {"trust_remote_code": True}
    if qcfg is not None:
        load_kwargs["quantization_config"] = qcfg
        if run_device == "cuda":
            load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = _dtype_for_device(run_device)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs).eval()
    if qcfg is None:
        model.to(run_device)

    cache["model"] = model
    cache["tokenizer"] = tokenizer
    cache["signature"] = signature
    return model, tokenizer, run_device


def _load_vl_model(model_name: str, quantization: str, device: str):
    model_dir = _ensure_model(model_name)
    run_device = _resolve_device(device)
    signature = (str(model_dir), quantization, run_device)
    cache = _MODEL_CACHE["vl"]
    if cache["model"] is not None and cache["signature"] == signature:
        return cache["model"], cache["tokenizer"], cache["processor"], run_device

    _clear_cache("vl")
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    qcfg = _quant_config(quantization)
    load_kwargs = {"trust_remote_code": True}
    if qcfg is not None:
        load_kwargs["quantization_config"] = qcfg
        if run_device == "cuda":
            load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = _dtype_for_device(run_device)
    model = AutoModelForVision2Seq.from_pretrained(str(model_dir), **load_kwargs).eval()
    if qcfg is None:
        model.to(run_device)

    cache["model"] = model
    cache["tokenizer"] = tokenizer
    cache["processor"] = processor
    cache["signature"] = signature
    return model, tokenizer, processor, run_device


def _tensor_to_pil(image):
    if image is None:
        return None
    if image.dim() == 4:
        image = image[0]
    np_img = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


class Qwen35PromptEnhancerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(QWEN35_MODELS.keys()), {"default": "Qwen3.5-4B"}),
                "quantization": (QUANT_OPTIONS, {"default": "None (FP16/BF16)"}),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
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
        model_name,
        quantization,
        device,
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
        torch.manual_seed(seed)
        model, tokenizer, run_device = _load_text_model(model_name, quantization, device)
        system_prompt = custom_system_prompt.strip() or PROMPT_STYLES.get(enhancement_style, PROMPT_STYLES["Enhance"])
        user_prompt = (prompt_text or "").strip() or "Describe a cinematic scene in rich visual detail."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = _apply_chat_template(tokenizer, messages)
        model_inputs = tokenizer([prompt], return_tensors="pt")
        model_device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device(run_device)
        model_inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in model_inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "top_p": top_p,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        output_ids = model.generate(**model_inputs, **generate_kwargs)
        generated = output_ids[0][model_inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        if not keep_model_loaded:
            _clear_cache("text")
        return (text,)


class Qwen35ReversePromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(QWEN35_MODELS.keys()), {"default": "Qwen3.5-4B"}),
                "quantization": (QUANT_OPTIONS, {"default": "None (FP16/BF16)"}),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "preset_prompt": (list(REVERSE_PRESETS.keys()), {"default": "Detailed Description"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "reverse_prompt"
    CATEGORY = "IAT/Qwen3.5"

    def reverse_prompt(
        self,
        model_name,
        quantization,
        device,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        image=None,
    ):
        if image is None:
            return ("[IAT] Please connect an IMAGE input for reverse prompt.",)

        torch.manual_seed(seed)
        model, tokenizer, processor, run_device = _load_vl_model(model_name, quantization, device)
        text_prompt = (custom_prompt or "").strip() or REVERSE_PRESETS.get(preset_prompt, REVERSE_PRESETS["Detailed Description"])
        pil_image = _tensor_to_pil(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        processed = processor(text=chat, images=[pil_image], return_tensors="pt")
        model_device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device(run_device)
        model_inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in processed.items()}

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "top_p": top_p,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        output_ids = model.generate(**model_inputs, **generate_kwargs)
        input_len = model_inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        if not keep_model_loaded:
            _clear_cache("vl")
        return (text,)


NODE_CLASS_MAPPINGS = {
    "Qwen35PromptEnhancer by IAT": Qwen35PromptEnhancerNode,
    "Qwen35ReversePrompt by IAT": Qwen35ReversePromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35PromptEnhancer by IAT": "Qwen3.5 Prompt Enhancer by IAT",
    "Qwen35ReversePrompt by IAT": "Qwen3.5 Reverse Prompt by IAT",
}



