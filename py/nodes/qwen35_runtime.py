from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Dict, List

import torch
import transformers
from packaging import version
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
    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except Exception:
        ms_snapshot_download = None

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except Exception:
    hf_snapshot_download = None

import folder_paths

MIN_TRANSFORMERS_FOR_QWEN35 = "4.57.0"

TEXT_MODEL_CANDIDATES: Dict[str, List[str]] = {
    "Qwen3.5-Latest": [
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-0.8B",
    ],
    "Qwen3.5-0.8B": ["Qwen/Qwen3.5-0.8B"],
    "Qwen3.5-9B": ["Qwen/Qwen3.5-9B"],
    "Qwen3.5-4B": ["Qwen/Qwen3.5-4B"],
    "Qwen3.5-2B": ["Qwen/Qwen3.5-2B"],
}

VL_MODEL_CANDIDATES: Dict[str, List[str]] = {
    "Qwen3.5-Latest": [
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-0.8B",
    ],
    "Qwen3.5-0.8B": ["Qwen/Qwen3.5-0.8B"],
    "Qwen3.5-9B": ["Qwen/Qwen3.5-9B"],
    "Qwen3.5-4B": ["Qwen/Qwen3.5-4B"],
    "Qwen3.5-2B": ["Qwen/Qwen3.5-2B"],
}

QUANT_OPTIONS = ["None (FP16/BF16)", "8-bit", "4-bit"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu", "mps"]

_MODEL_CACHE = {
    "text": {"signature": None, "model": None, "tokenizer": None},
    "vl": {"signature": None, "model": None, "tokenizer": None, "processor": None},
}


def _check_transformers_support() -> None:
    current = getattr(transformers, "__version__", "0.0.0")
    if version.parse(current) < version.parse(MIN_TRANSFORMERS_FOR_QWEN35):
        raise RuntimeError(
            f"[IAT] transformers>={MIN_TRANSFORMERS_FOR_QWEN35} is required for Qwen3.5. Current version: {current}."
        )


def _get_llm_dir() -> Path:
    llm_paths = []
    if hasattr(folder_paths, "folder_names_and_paths") and "LLM" in folder_paths.folder_names_and_paths:
        llm_paths = folder_paths.get_folder_paths("LLM")
    llm_dir = Path(llm_paths[0]) if llm_paths else Path(folder_paths.models_dir) / "LLM"
    llm_dir.mkdir(parents=True, exist_ok=True)
    return llm_dir


def _normalize_local_name(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    if name.endswith("-Instruct"):
        name = name[: -len("-Instruct")]
    return name


def _model_local_dir(repo_id: str) -> Path:
    return _get_llm_dir() / _normalize_local_name(repo_id)


def _has_weights(model_dir: Path) -> bool:
    for pat in ("*.safetensors", "*.bin", "*.pt"):
        if next(model_dir.glob(pat), None) is not None:
            return True
    return False


def _has_vl_processor_files(model_dir: Path) -> bool:
    required_any = (
        "processor_config.json",
        "preprocessor_config.json",
        "image_processor_config.json",
        "feature_extractor_config.json",
    )
    return any((model_dir / name).exists() for name in required_any)


def _download_from_modelscope(repo_id: str, local_dir: Path) -> bool:
    if ms_snapshot_download is None:
        return False
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        ms_snapshot_download(model_id=repo_id, local_dir=str(local_dir))
    except TypeError:
        ms_snapshot_download(repo_id, cache_dir=str(local_dir.parent))
    return True


def _download_from_hf(repo_id: str, local_dir: Path) -> bool:
    if hf_snapshot_download is None:
        return False
    local_dir.mkdir(parents=True, exist_ok=True)
    hf_snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return True


def ensure_model(variant: str, mode: str) -> Path:
    model_map = TEXT_MODEL_CANDIDATES if mode == "text" else VL_MODEL_CANDIDATES
    candidates = model_map.get(variant)
    if not candidates:
        raise ValueError(f"[IAT] Unsupported Qwen3.5 variant: {variant}")

    errors: List[str] = []
    for repo_id in candidates:
        target = _model_local_dir(repo_id)
        if target.exists() and _has_weights(target):
            if mode == "vl" and not _has_vl_processor_files(target):
                print(f"[IAT] Skip invalid VL model dir (missing processor files): {target}")
            else:
                if repo_id != candidates[0]:
                    print(f"[IAT] Using fallback repository: {repo_id}")
                return target

        for source_name, downloader in (("ModelScope", _download_from_modelscope), ("HuggingFace", _download_from_hf)):
            try:
                ok = downloader(repo_id, target)
                if not ok:
                    continue
                if _has_weights(target):
                    if mode == "vl" and not _has_vl_processor_files(target):
                        errors.append(f"{repo_id}: missing VL processor files after {source_name} download")
                        continue
                    print(f"[IAT] Downloaded {repo_id} from {source_name} -> {target}")
                    return target
                errors.append(f"{repo_id}: no model weights after {source_name} download")
            except Exception as exc:
                errors.append(f"{repo_id}: {source_name} download failed: {exc}")

    raise RuntimeError(
        "[IAT] Failed to prepare Qwen3.5 model. "
        f"Variant={variant}, mode={mode}, tried={', '.join(candidates)}. "
        f"Details: {' | '.join(errors[-6:])}"
    )


def resolve_device(device: str) -> str:
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
    if device == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device in {"cuda", "mps"} else torch.float32


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


def _assert_quant_applied(model, quantization: str, mode: str):
    if quantization == "8-bit" and not bool(getattr(model, "is_loaded_in_8bit", False)):
        raise RuntimeError(f"[IAT] {mode} quantization mismatch: requested 8-bit but model is not 8-bit.")
    if quantization == "4-bit" and not bool(getattr(model, "is_loaded_in_4bit", False)):
        raise RuntimeError(f"[IAT] {mode} quantization mismatch: requested 4-bit but model is not 4-bit.")


def apply_chat_template(tokenizer, messages) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def apply_vl_chat_template(processor, conversation) -> str:
    try:
        return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)


def _strip_thinking_content(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>[\\s\\S]*?</think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _load_tokenizer(model_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as exc:
        print(f"[IAT] Fast tokenizer load failed, fallback to slow tokenizer: {exc}")
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=False)


def _clear_cache(kind: str) -> None:
    cache = _MODEL_CACHE[kind]
    for key in list(cache.keys()):
        cache[key] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_all_models() -> None:
    _clear_cache("text")
    _clear_cache("vl")


def load_text_model(variant: str, quantization: str, device: str):
    _check_transformers_support()
    model_dir = ensure_model(variant, mode="text")
    run_device = resolve_device(device)
    signature = (str(model_dir), quantization, run_device)

    cache = _MODEL_CACHE["text"]
    if cache["model"] is not None and cache["signature"] == signature:
        return cache["model"], cache["tokenizer"], run_device

    _clear_cache("text")
    tokenizer = _load_tokenizer(model_dir)

    qcfg = _quant_config(quantization)
    kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}

    if qcfg is not None:
        if run_device != "cuda":
            raise RuntimeError("[IAT] 4-bit/8-bit quantization requires CUDA device.")
        kwargs["quantization_config"] = qcfg
    else:
        kwargs["dtype"] = _dtype_for_device(run_device)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs).eval()
    if qcfg is None:
        model.to(run_device)
    else:
        _assert_quant_applied(model, quantization, mode="Text")

    print(f"[IAT] Text model loaded: variant={variant}, device={run_device}, quant={quantization}")

    cache["signature"] = signature
    cache["model"] = model
    cache["tokenizer"] = tokenizer
    return model, tokenizer, run_device


def load_vl_model(variant: str, quantization: str, device: str):
    _check_transformers_support()
    model_dir = ensure_model(variant, mode="vl")
    run_device = resolve_device(device)
    signature = (str(model_dir), quantization, run_device)

    cache = _MODEL_CACHE["vl"]
    if cache["model"] is not None and cache["signature"] == signature:
        return cache["model"], cache["tokenizer"], cache["processor"], run_device

    _clear_cache("vl")
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer = _load_tokenizer(model_dir)

    qcfg = _quant_config(quantization)
    kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}

    if qcfg is not None:
        if run_device != "cuda":
            raise RuntimeError("[IAT] 4-bit/8-bit quantization requires CUDA device.")
        kwargs["quantization_config"] = qcfg
    else:
        kwargs["dtype"] = _dtype_for_device(run_device)

    model = AutoModelForVision2Seq.from_pretrained(str(model_dir), **kwargs).eval()
    if qcfg is None:
        model.to(run_device)
    else:
        _assert_quant_applied(model, quantization, mode="VL")

    print(f"[IAT] VL model loaded: variant={variant}, device={run_device}, quant={quantization}")

    cache["signature"] = signature
    cache["model"] = model
    cache["tokenizer"] = tokenizer
    cache["processor"] = processor
    return model, tokenizer, processor, run_device


def generate_text(
    *,
    variant: str,
    quantization: str,
    device: str,
    messages,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    print("[IAT] [Text] Loading model...")
    model, tokenizer, run_device = load_text_model(variant, quantization, device)
    prompt = apply_chat_template(tokenizer, messages)

    model_inputs = tokenizer([prompt], return_tensors="pt")
    model_device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device(run_device)
    model_inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in model_inputs.items()}

    print("[IAT] [Text] Generating (thinking disabled)...")
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = max(temperature, 1e-5)

    output_ids = model.generate(**model_inputs, **gen_kwargs)
    generated = output_ids[0][model_inputs["input_ids"].shape[-1] :]
    return _strip_thinking_content(tokenizer.decode(generated, skip_special_tokens=True))


def generate_vision_text(
    *,
    variant: str,
    quantization: str,
    device: str,
    images,
    text_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    print("[IAT] [VL] Loading model...")
    model, tokenizer, processor, run_device = load_vl_model(variant, quantization, device)

    if not isinstance(images, list):
        images = [images]
    images = [img for img in images if img is not None]
    if len(images) == 0:
        raise ValueError("[IAT] generate_vision_text requires at least one image.")

    print(f"[IAT] [VL] Reading images: {len(images)}")
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": text_prompt})

    conversation = [{"role": "user", "content": content}]
    chat = apply_vl_chat_template(processor, conversation)
    processed = processor(text=chat, images=images, return_tensors="pt")

    model_device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device(run_device)
    model_inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in processed.items()}

    print("[IAT] [VL] Generating (thinking disabled)...")
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = max(temperature, 1e-5)

    output_ids = model.generate(**model_inputs, **gen_kwargs)
    generated = output_ids[0][model_inputs["input_ids"].shape[-1] :]
    return _strip_thinking_content(tokenizer.decode(generated, skip_special_tokens=True))
