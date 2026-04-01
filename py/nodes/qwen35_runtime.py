from __future__ import annotations

import gc
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

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

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None

MIN_TRANSFORMERS_FOR_QWEN35 = "4.57.0"

# Qwen3.5 Dense Models for Consumer GPUs
TEXT_MODEL_CANDIDATES: Dict[str, List[str]] = {
    "Qwen3.5-0.8B": ["Qwen/Qwen3.5-0.8B"],
    "Qwen3.5-2B": ["Qwen/Qwen3.5-2B"],
    "Qwen3.5-4B": ["Qwen/Qwen3.5-4B"],
    "Qwen3.5-9B": ["Qwen/Qwen3.5-9B"],
    "Qwen3.5-27B": ["Qwen/Qwen3.5-27B"],
}

VL_MODEL_CANDIDATES: Dict[str, List[str]] = {
    "Qwen3.5-0.8B": ["Qwen/Qwen3.5-0.8B"],
    "Qwen3.5-2B": ["Qwen/Qwen3.5-2B"],
    "Qwen3.5-4B": ["Qwen/Qwen3.5-4B"],
    "Qwen3.5-9B": ["Qwen/Qwen3.5-9B"],
    "Qwen3.5-27B": ["Qwen/Qwen3.5-27B"],
}

QUANT_OPTIONS = ["None (FP16/BF16)", "8-bit", "4-bit"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu", "mps"]

# 模型缓存 - 避免重复加载
_MODEL_CACHE = {
    "text": {"signature": None, "model": None, "tokenizer": None},
    "vl": {"signature": None, "model": None, "tokenizer": None, "processor": None},
}

# 性能优化配置
_ATTN_IMPLEMENTATION = None  # 自动检测最佳注意力实现

def _get_optimal_attn_implementation():
    """自动检测最佳的注意力实现"""
    global _ATTN_IMPLEMENTATION
    if _ATTN_IMPLEMENTATION is not None:
        return _ATTN_IMPLEMENTATION
    
    # 检查 Flash Attention 2 是否可用
    try:
        import flash_attn
        _ATTN_IMPLEMENTATION = "flash_attention_2"
        print("[IAT] 使用 Flash Attention 2 加速")
        return _ATTN_IMPLEMENTATION
    except ImportError:
        pass
    
    # 检查是否支持 SDPA (Scaled Dot Product Attention)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        _ATTN_IMPLEMENTATION = "sdpa"
        print("[IAT] 使用 SDPA (Scaled Dot Product Attention) 加速")
        return _ATTN_IMPLEMENTATION
    
    _ATTN_IMPLEMENTATION = "eager"
    print("[IAT] 使用标准注意力实现")
    return _ATTN_IMPLEMENTATION


def _log_major(message: str):
    print(f"\033[92m[IAT] {message}\033[0m")


def _log_info(message: str):
    print(f"[IAT] {message}")


def _log_warning(message: str):
    print(f"\033[93m[IAT] 警告: {message}\033[0m")


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


def _get_offload_dir() -> Path:
    """获取CPU卸载目录，用于显存不足时"""
    offload_dir = _get_llm_dir() / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    return offload_dir


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
                _log_info(f"跳过无效多模态模型目录（缺少处理器文件）: {target}")
            else:
                if repo_id != candidates[0]:
                    _log_info(f"使用候选回退仓库: {repo_id}")
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
                    _log_major(f"模型已下载: {repo_id} <- {source_name}")
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


def _cuda_max_memory_map(quantization: str = None, reserve_gib: int = 4):
    """
    计算每个GPU的最大可用显存
    
    Args:
        quantization: 量化模式，用于调整预留显存
        reserve_gib: 基础预留显存（GB）
    """
    if not torch.cuda.is_available():
        return None
    
    # 根据量化模式调整预留显存
    if quantization == "4-bit":
        reserve_gib = max(2, reserve_gib - 2)  # 4-bit量化需要更少预留
    elif quantization == "8-bit":
        reserve_gib = max(3, reserve_gib - 1)
    
    max_memory = {}
    for idx in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(idx).total_memory // (1024**3)
        # 预留更多显存给系统和其他进程
        usable_gib = max(1, int(total_gib) - reserve_gib)
        max_memory[idx] = f"{usable_gib}GiB"
        _log_info(f"GPU {idx}: 总显存 {total_gib}GB, 可用 {usable_gib}GB")
    
    return max_memory


def _prepare_cuda_for_load():
    """准备CUDA环境，清理缓存"""
    if not torch.cuda.is_available():
        return
    try:
        if model_management is not None:
            model_management.unload_all_models()
            model_management.soft_empty_cache()
    except Exception as exc:
        _log_info(f"model_management 清理跳过: {exc}")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # 显示当前显存状态
    for idx in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(idx) / (1024**3)
        reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        _log_info(f"GPU {idx} 清理后: 已分配 {allocated:.2f}GB, 预留 {reserved:.2f}GB")


def _quant_config(quantization: str):
    """创建量化配置"""
    if quantization == "None (FP16/BF16)":
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError("[IAT] bitsandbytes is required for 4-bit/8-bit quantization. Install: pip install bitsandbytes")
    
    if quantization == "8-bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # 阈值，超过此值的outliers保持FP16
            llm_int8_has_fp16_weight=False,
        )
    
    if quantization == "4-bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normal Float 4，比FP4更好
            bnb_4bit_use_double_quant=True,  # 嵌套量化，进一步减少显存
            bnb_4bit_compute_dtype=torch.float16,  # 计算时使用的dtype
        )
    
    raise ValueError(f"[IAT] Unsupported quantization: {quantization}")


def _assert_quant_applied(model, quantization: str, mode: str):
    """验证量化是否正确应用"""
    if quantization == "8-bit":
        is_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
        if not is_8bit:
            _log_warning(f"{mode} 量化未生效: 请求8-bit但模型未以8-bit加载")
        return is_8bit
    
    if quantization == "4-bit":
        is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
        if not is_4bit:
            _log_warning(f"{mode} 量化未生效: 请求4-bit但模型未以4-bit加载")
        return is_4bit
    
    return True


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
        _log_info(f"快速 tokenizer 加载失败，回退慢速 tokenizer: {exc}")
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=False)


def _clear_cache(kind: str) -> None:
    """清理模型缓存"""
    cache = _MODEL_CACHE[kind]
    for key in list(cache.keys()):
        cache[key] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_all_models() -> None:
    """卸载所有模型"""
    _clear_cache("text")
    _clear_cache("vl")
    _log_major("所有模型已卸载")


def _get_model_loading_kwargs(quantization: str, device: str, model_dir: Path):
    """
    获取模型加载参数
    
    优化点：
    1. 使用最佳的注意力实现
    2. 优化量化配置
    3. 合理的device_map和max_memory配置
    4. 支持CPU卸载
    """
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": _get_optimal_attn_implementation(),
    }
    
    qcfg = _quant_config(quantization)
    
    if qcfg is not None:
        # 量化模式
        if device != "cuda":
            raise RuntimeError("[IAT] 4-bit/8-bit quantization requires CUDA device.")
        
        kwargs["quantization_config"] = qcfg
        kwargs["torch_dtype"] = torch.float16  # 量化时使用float16
        
        # 关键优化：使用更好的device_map配置
        max_memory = _cuda_max_memory_map(quantization)
        if max_memory:
            kwargs["max_memory"] = max_memory
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = "auto"
        
        # 添加offload文件夹配置，允许CPU卸载
        kwargs["offload_folder"] = str(_get_offload_dir())
        
    else:
        # 非量化模式
        kwargs["torch_dtype"] = _dtype_for_device(device)
        if device == "cuda":
            # 非量化时，小模型直接加载到GPU，大模型使用auto
            kwargs["device_map"] = "auto"
            max_memory = _cuda_max_memory_map(quantization)
            if max_memory:
                kwargs["max_memory"] = max_memory
            kwargs["offload_folder"] = str(_get_offload_dir())
    
    return kwargs, qcfg


def load_text_model(variant: str, quantization: str, device: str):
    """加载文本模型，带缓存机制"""
    _check_transformers_support()
    model_dir = ensure_model(variant, mode="text")
    run_device = resolve_device(device)
    
    # 创建缓存签名
    signature = (str(model_dir), quantization, run_device)
    
    # 检查缓存
    cache = _MODEL_CACHE["text"]
    if cache["model"] is not None and cache["signature"] == signature:
        _log_info(f"使用缓存的文本模型: {variant}")
        return cache["model"], cache["tokenizer"], run_device
    
    # 清理旧缓存
    _clear_cache("text")
    
    if run_device == "cuda":
        _prepare_cuda_for_load()
    
    # 加载tokenizer
    tokenizer = _load_tokenizer(model_dir)
    
    # 获取加载参数
    kwargs, qcfg = _get_model_loading_kwargs(quantization, run_device, model_dir)
    
    _log_major(f"正在加载文本模型: {variant} | 量化: {quantization} | 注意力: {kwargs.get('attn_implementation', 'default')}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    
    # 验证量化
    if qcfg is not None:
        _assert_quant_applied(model, quantization, "Text")
    
    # 设置为评估模式
    model.eval()
    
    # 编译模型以加速（仅CUDA，且非量化）
    if run_device == "cuda" and qcfg is None and hasattr(torch, "compile"):
        try:
            _log_info("尝试编译模型以加速推理...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            _log_info(f"模型编译失败（不影响使用）: {e}")
    
    _log_major(f"文本模型加载完成 | 规格={variant} | 设备={run_device} | 量化={quantization}")
    
    # 更新缓存
    cache["signature"] = signature
    cache["model"] = model
    cache["tokenizer"] = tokenizer
    
    return model, tokenizer, run_device


def load_vl_model(variant: str, quantization: str, device: str):
    """加载视觉语言模型，带缓存机制"""
    _check_transformers_support()
    model_dir = ensure_model(variant, mode="vl")
    run_device = resolve_device(device)
    
    # 创建缓存签名
    signature = (str(model_dir), quantization, run_device)
    
    # 检查缓存
    cache = _MODEL_CACHE["vl"]
    if cache["model"] is not None and cache["signature"] == signature:
        _log_info(f"使用缓存的多模态模型: {variant}")
        return cache["model"], cache["tokenizer"], cache["processor"], run_device
    
    # 清理旧缓存
    _clear_cache("vl")
    
    if run_device == "cuda":
        _prepare_cuda_for_load()
    
    # 加载processor和tokenizer
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer = _load_tokenizer(model_dir)
    
    # 获取加载参数
    kwargs, qcfg = _get_model_loading_kwargs(quantization, run_device, model_dir)
    
    _log_major(f"正在加载多模态模型: {variant} | 量化: {quantization} | 注意力: {kwargs.get('attn_implementation', 'default')}")
    
    # 加载模型
    model = AutoModelForVision2Seq.from_pretrained(str(model_dir), **kwargs)
    
    # 验证量化
    if qcfg is not None:
        _assert_quant_applied(model, quantization, "VL")
    
    # 设置为评估模式
    model.eval()
    
    _log_major(f"多模态模型加载完成 | 规格={variant} | 设备={run_device} | 量化={quantization}")
    
    # 更新缓存
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
    """生成文本，优化推理速度"""
    torch.manual_seed(seed)
    
    # 加载模型（使用缓存）
    model, tokenizer, run_device = load_text_model(variant, quantization, device)
    
    # 应用chat template
    prompt = apply_chat_template(tokenizer, messages)
    
    # 编码输入
    model_inputs = tokenizer([prompt], return_tensors="pt")
    
    # 获取模型所在设备
    if hasattr(model, "device"):
        model_device = model.device
    elif hasattr(model, "parameters"):
        try:
            model_device = next(model.parameters()).device
        except:
            model_device = torch.device(run_device)
    else:
        model_device = torch.device(run_device)
    
    # 移动输入到模型设备
    model_inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in model_inputs.items()}
    
    # 生成参数优化
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,  # 使用KV缓存加速
    }
    
    if temperature > 0:
        gen_kwargs["temperature"] = max(temperature, 1e-5)
    
    # 使用torch.inference_mode()加速推理
    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, **gen_kwargs)
    
    # 解码输出
    generated = output_ids[0][model_inputs["input_ids"].shape[-1]:]
    result = _strip_thinking_content(tokenizer.decode(generated, skip_special_tokens=True))
    
    return result


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
    """生成视觉文本，优化推理速度"""
    torch.manual_seed(seed)
    
    # 加载模型（使用缓存）
    model, tokenizer, processor, run_device = load_vl_model(variant, quantization, device)
    
    # 处理图像
    if not isinstance(images, list):
        images = [images]
    images = [img for img in images if img is not None]
    if len(images) == 0:
        raise ValueError("[IAT] generate_vision_text requires at least one image.")
    
    # 构建对话
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": text_prompt})
    conversation = [{"role": "user", "content": content}]
    
    # 应用chat template
    chat = apply_vl_chat_template(processor, conversation)
    
    # 处理输入
    processed = processor(text=chat, images=images, return_tensors="pt")
    
    # 获取模型所在设备
    if hasattr(model, "device"):
        model_device = model.device
    elif hasattr(model, "parameters"):
        try:
            model_device = next(model.parameters()).device
        except:
            model_device = torch.device(run_device)
    else:
        model_device = torch.device(run_device)
    
    # 移动输入到模型设备
    model_inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in processed.items()}
    
    # 生成参数优化
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,  # 使用KV缓存加速
    }
    
    if temperature > 0:
        gen_kwargs["temperature"] = max(temperature, 1e-5)
    
    # 使用torch.inference_mode()加速推理
    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, **gen_kwargs)
    
    # 解码输出
    generated = output_ids[0][model_inputs["input_ids"].shape[-1]:]
    result = _strip_thinking_content(tokenizer.decode(generated, skip_special_tokens=True))
    
    return result
