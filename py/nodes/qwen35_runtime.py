from __future__ import annotations

import gc
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
try:
    from transformers.utils import is_flash_attn_2_available
except Exception:
    def is_flash_attn_2_available():
        return False

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

_CFG = getattr(sys.modules.get("comfyui_iat_config"), "data", {}) or {}
_RUNTIME_CFG = (_CFG.get("runtime") or {}) if isinstance(_CFG, dict) else {}
_LOGGING_CFG = (_CFG.get("logging") or {}) if isinstance(_CFG, dict) else {}

MIN_TRANSFORMERS_FOR_QWEN35 = "4.57.0"
QWEN35_MODEL_TYPE = "qwen3_5"
TRANSFORMERS_UPGRADE_SOURCES = (
    "git+https://github.com/huggingface/transformers.git",
    "https://github.com/huggingface/transformers/archive/refs/heads/main.zip",
)

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

QUANT_OPTIONS = ["无", "8-bit", "4-bit"]
DEVICE_OPTIONS = ["cuda", "cpu"]
ATTENTION_OPTIONS = [
    "SDPA",
    "FlashAttention-2",
    "Eager",
]
_ATTENTION_BACKEND_TO_IMPL = {
    "SDPA": "sdpa",
    "FlashAttention-2": "flash_attention_2",
    "Eager": "eager",
}
_ATTENTION_BACKEND_ALIASES = {
    "auto": "SDPA",
    "sdpa": "SDPA",
    "flash": "FlashAttention-2",
    "flash_attention_2": "FlashAttention-2",
    "flash attention 2": "FlashAttention-2",
    "flash_attention_3": "FlashAttention-2",
    "flash attention 3": "FlashAttention-2",
    "flash_attention_4": "FlashAttention-2",
    "flash attention 4": "FlashAttention-2",
    "flex_attention": "SDPA",
    "flex attention": "SDPA",
    "eager": "Eager",
    "default": "SDPA",
    "transformers default": "SDPA",
}

# 模型缓存 - 避免重复加载
_MODEL_CACHE = {
    "text": {"signature": None, "model": None, "tokenizer": None},
    "vl": {"signature": None, "model": None, "tokenizer": None, "processor": None},
}

# 性能优化配置
_ATTN_IMPLEMENTATION = None  # 自动检测最佳注意力实现
_ATTN_IMPLEMENTATION_RESOLVED = False
_TRANSFORMERS_UPGRADE_ATTEMPTED = False


def _cfg_bool(name: str, default: bool) -> bool:
    value = _RUNTIME_CFG.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _cfg_logging_bool(name: str, default: bool) -> bool:
    value = _LOGGING_CFG.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


AUTO_UPGRADE_TRANSFORMERS = _cfg_bool("auto_upgrade_transformers", True)
PREFER_OPTIMIZED_ATTENTION = _cfg_bool("prefer_optimized_attention", True)
ENABLE_TORCH_COMPILE = _cfg_bool("enable_torch_compile", False)
DEFAULT_ATTENTION_BACKEND = "SDPA"
VERBOSE_LOGGING = _cfg_logging_bool("verbose", False)


def _get_optimal_attn_implementation(device: str) -> Optional[str]:
    """自动检测最佳的注意力实现"""
    global _ATTN_IMPLEMENTATION, _ATTN_IMPLEMENTATION_RESOLVED
    if not PREFER_OPTIMIZED_ATTENTION or device != "cuda":
        return None
    if _ATTN_IMPLEMENTATION_RESOLVED:
        return _ATTN_IMPLEMENTATION

    _ATTN_IMPLEMENTATION_RESOLVED = True

    def _check_availability(fn, label: str) -> bool:
        try:
            return bool(fn())
        except Exception as exc:
            _log_info(f"{label} 可用性检测失败，已跳过: {exc}")
            return False

    # 优先使用 transformers 自带的 availability 检测，避免 flash_attn 包名映射异常
    if _check_availability(is_flash_attn_2_available, "Flash Attention 2"):
        _ATTN_IMPLEMENTATION = "flash_attention_2"
        _log_info("使用 Flash Attention 2 加速")
        return _ATTN_IMPLEMENTATION

    # 检查是否支持 SDPA (Scaled Dot Product Attention)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        _ATTN_IMPLEMENTATION = "sdpa"
        _log_info("使用 SDPA (Scaled Dot Product Attention) 加速")
        return _ATTN_IMPLEMENTATION

    _ATTN_IMPLEMENTATION = None
    _log_info("使用 Transformers 默认注意力实现")
    return None


def _log_major(message: str):
    print(f"\033[92m[IAT] {message}\033[0m")


def _log_info(message: str):
    if VERBOSE_LOGGING:
        print(f"[IAT] {message}")


def _log_warning(message: str):
    print(f"\033[93m[IAT] 警告: {message}\033[0m")


def _normalize_attention_backend(attention_backend: Optional[str]) -> str:
    if attention_backend in ATTENTION_OPTIONS:
        return attention_backend
    alias = _ATTENTION_BACKEND_ALIASES.get(str(attention_backend or "").strip().lower())
    if alias in ATTENTION_OPTIONS:
        return alias
    return DEFAULT_ATTENTION_BACKEND


def _resolve_attention_backend(attention_backend: Optional[str], device: str) -> Tuple[Optional[str], str, bool]:
    backend = _normalize_attention_backend(attention_backend)
    return _ATTENTION_BACKEND_TO_IMPL.get(backend), backend, False


DEFAULT_ATTENTION_BACKEND = _normalize_attention_backend(_RUNTIME_CFG.get("default_attention_backend", "SDPA"))


def _supports_qwen35_architecture() -> bool:
    try:
        transformers.AutoConfig.for_model(QWEN35_MODEL_TYPE)
        return True
    except Exception:
        pass

    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        return QWEN35_MODEL_TYPE in CONFIG_MAPPING
    except Exception:
        return False


def _manual_transformers_upgrade_command() -> str:
    return f'"{sys.executable}" -m pip install --upgrade {TRANSFORMERS_UPGRADE_SOURCES[0]}'


def _tail_text(text: str, lines: int = 8) -> str:
    chunks = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return " | ".join(chunks[-lines:])


def _attempt_transformers_upgrade() -> Tuple[bool, str]:
    _log_major("检测到当前 transformers 缺少 Qwen3.5 架构支持，尝试自动升级源码版本")
    errors: List[str] = []

    for source in TRANSFORMERS_UPGRADE_SOURCES:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", source]
        _log_info(f"执行依赖升级: {' '.join(cmd)}")
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            continue

        if completed.returncode == 0:
            return True, source

        summary = _tail_text(completed.stderr or completed.stdout)
        errors.append(f"{source}: {summary or f'pip exited with code {completed.returncode}'}")

    return False, " | ".join(errors[-4:])


def _check_transformers_support() -> None:
    global _TRANSFORMERS_UPGRADE_ATTEMPTED
    current = getattr(transformers, "__version__", "0.0.0")
    issues: List[str] = []

    if version.parse(current) < version.parse(MIN_TRANSFORMERS_FOR_QWEN35):
        issues.append(f"requires transformers>={MIN_TRANSFORMERS_FOR_QWEN35} (current: {current})")
    if not _supports_qwen35_architecture():
        issues.append(f"current transformers build does not register the '{QWEN35_MODEL_TYPE}' architecture")

    if not issues:
        return

    manual_command = _manual_transformers_upgrade_command()
    detail = "; ".join(issues)

    if AUTO_UPGRADE_TRANSFORMERS and not _TRANSFORMERS_UPGRADE_ATTEMPTED:
        _TRANSFORMERS_UPGRADE_ATTEMPTED = True
        upgraded, upgrade_detail = _attempt_transformers_upgrade()
        if upgraded:
            raise RuntimeError(
                "[IAT] Current transformers build cannot load Qwen3.5 "
                f"({detail}). The plugin has upgraded transformers automatically from {upgrade_detail}. "
                "Please restart ComfyUI and run the node again."
            )
        raise RuntimeError(
            "[IAT] Current transformers build cannot load Qwen3.5 "
            f"({detail}). Automatic upgrade failed. Run `{manual_command}` and restart ComfyUI. "
            f"Details: {upgrade_detail}"
        )

    auto_upgrade_note = ""
    if not AUTO_UPGRADE_TRANSFORMERS:
        auto_upgrade_note = " Automatic upgrade is disabled in config.yaml."
    raise RuntimeError(
        "[IAT] Current transformers build cannot load Qwen3.5 "
        f"({detail}).{auto_upgrade_note} Run `{manual_command}` and restart ComfyUI."
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
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if device not in {"cuda", "cpu"}:
        return "cpu"
    return device


def _dtype_for_device(device: str):
    if device == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device == "cuda" else torch.float32


def _variant_size_billions(variant: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)B", variant or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _primary_cuda_total_memory_gib() -> Optional[float]:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return None


def _available_system_memory_gib() -> Optional[int]:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return max(1, int((pages * page_size) / (1024**3)))
    except Exception:
        return None


def _cpu_offload_memory_budget_gib() -> Optional[int]:
    available = _available_system_memory_gib()
    if available is None:
        return None
    return max(4, int(available * 0.8))


def _should_use_auto_device_map(variant: str, quantization: str, device: str) -> bool:
    if device != "cuda":
        return False
    if quantization != "无":
        return True
    size_b = _variant_size_billions(variant)
    if size_b is None:
        return True
    if size_b >= 27.0:
        return True
    if size_b >= 9.0:
        total_gib = _primary_cuda_total_memory_gib()
        return total_gib is None or total_gib < 23.0
    return False


def _dtype_kwarg_name() -> str:
    return "dtype" if version.parse(getattr(transformers, "__version__", "0.0.0")) >= version.parse("5.0.0") else "torch_dtype"


def _set_model_dtype(kwargs: dict, dtype) -> None:
    kwargs[_dtype_kwarg_name()] = dtype


def _has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


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


def _auto_device_max_memory(variant: str, quantization: str):
    max_memory = _cuda_max_memory_map(quantization)
    if max_memory is None:
        return None
    cpu_budget = _cpu_offload_memory_budget_gib()
    if cpu_budget is not None:
        max_memory["cpu"] = f"{cpu_budget}GiB"
        _log_info(f"CPU 卸载预算: {cpu_budget}GiB")
    size_b = _variant_size_billions(variant)
    if size_b is not None and size_b >= 27.0:
        _log_warning("Qwen3.5-27B 在 FP16/BF16 下会启用自动分配和 CPU 卸载；24GB 单卡更推荐 4-bit/8-bit。")
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
    if quantization == "无":
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


def _is_attention_compat_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "attn_implementation",
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
        "flash attention 2",
        "flash attention 3",
        "flash attention 4",
        "flash_attn",
        "sdpa",
        "scaled_dot_product_attention",
    )
    return any(marker in message for marker in markers)


def _load_pretrained_with_fallback(
    loader,
    model_dir: Path,
    kwargs: dict,
    label: str,
    attention_backend: str,
    allow_attn_fallback: bool,
):
    try:
        return loader.from_pretrained(str(model_dir), **kwargs)
    except Exception as exc:
        attn_implementation = kwargs.get("attn_implementation")
        if attn_implementation and _is_attention_compat_error(exc):
            if not allow_attn_fallback:
                raise RuntimeError(
                    f"[IAT] {label} 指定的注意力实现 `{attention_backend}` 当前环境不可用或不兼容: {exc}"
                ) from exc
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("attn_implementation", None)
            _log_warning(
                f"{label} 在注意力实现 `{attention_backend}` 上加载失败，回退到 Transformers 默认实现。"
            )
            _log_info(f"{label} 注意力回退原因: {exc}")
            return loader.from_pretrained(str(model_dir), **retry_kwargs)
        raise


def _get_model_loading_kwargs(variant: str, quantization: str, device: str, attention_backend: Optional[str]):
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
    }

    attn_implementation, resolved_attention_backend, allow_attn_fallback = _resolve_attention_backend(attention_backend, device)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    
    qcfg = _quant_config(quantization)
    
    if qcfg is not None:
        # 量化模式
        if device != "cuda":
            raise RuntimeError("[IAT] 4-bit/8-bit quantization requires CUDA device.")
        if not _has_accelerate():
            raise RuntimeError("[IAT] 4-bit/8-bit quantization with automatic device placement requires accelerate. Install: pip install accelerate")
        
        kwargs["quantization_config"] = qcfg
        _set_model_dtype(kwargs, torch.float16)  # 量化时使用float16
        
        # 关键优化：使用更好的device_map配置
        max_memory = _auto_device_max_memory(variant, quantization)
        if max_memory:
            kwargs["max_memory"] = max_memory
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = "auto"
        
        # 添加offload文件夹配置，允许CPU卸载
        kwargs["offload_folder"] = str(_get_offload_dir())
        kwargs["offload_state_dict"] = True
        
    else:
        # 非量化模式
        _set_model_dtype(kwargs, _dtype_for_device(device))
        if device == "cuda":
            if _should_use_auto_device_map(variant, quantization, device) and _has_accelerate():
                # accelerate 可用时，允许自动放置和溢出到 CPU
                kwargs["device_map"] = "auto"
                max_memory = _auto_device_max_memory(variant, quantization)
                if max_memory:
                    kwargs["max_memory"] = max_memory
                kwargs["offload_folder"] = str(_get_offload_dir())
                kwargs["offload_state_dict"] = True
                _log_info("大模型启用 auto device_map + CPU offload。")
            elif _should_use_auto_device_map(variant, quantization, device):
                _log_info("模型较大但未检测到 accelerate，回退为常规 CUDA 加载。")
            elif (_variant_size_billions(variant) or 0) >= 9.0:
                _log_info("9B 模型在 24GB 级显卡上默认使用单卡直载，优先保证稳定性和响应速度。")
            else:
                _log_info("小模型默认使用常规 CUDA 加载，避免 auto device_map 带来的额外开销。")
    
    return kwargs, qcfg, resolved_attention_backend, allow_attn_fallback


def load_text_model(variant: str, quantization: str, device: str, attention_backend: Optional[str] = None):
    """加载文本模型，带缓存机制"""
    _check_transformers_support()
    model_dir = ensure_model(variant, mode="text")
    run_device = resolve_device(device)
    resolved_attention_backend = _normalize_attention_backend(attention_backend)
    
    # 创建缓存签名
    signature = (str(model_dir), quantization, run_device, resolved_attention_backend)
    
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
    kwargs, qcfg, resolved_attention_backend, allow_attn_fallback = _get_model_loading_kwargs(
        variant, quantization, run_device, resolved_attention_backend
    )
    
    _log_major(f"正在加载文本模型: {variant} | 量化: {quantization} | 注意力: {resolved_attention_backend}")
    
    # 加载模型
    model = _load_pretrained_with_fallback(
        AutoModelForCausalLM,
        model_dir,
        kwargs,
        "文本模型",
        resolved_attention_backend,
        allow_attn_fallback,
    )
    
    # 验证量化
    if qcfg is not None:
        _assert_quant_applied(model, quantization, "Text")
    
    # 设置为评估模式
    model.eval()
    if qcfg is None and not hasattr(model, "hf_device_map"):
        model.to(run_device)
    
    # 编译模型以加速（仅CUDA，且非量化）
    if ENABLE_TORCH_COMPILE and run_device == "cuda" and qcfg is None and hasattr(torch, "compile"):
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


def load_vl_model(variant: str, quantization: str, device: str, attention_backend: Optional[str] = None):
    """加载视觉语言模型，带缓存机制"""
    _check_transformers_support()
    model_dir = ensure_model(variant, mode="vl")
    run_device = resolve_device(device)
    resolved_attention_backend = _normalize_attention_backend(attention_backend)
    
    # 创建缓存签名
    signature = (str(model_dir), quantization, run_device, resolved_attention_backend)
    
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
    kwargs, qcfg, resolved_attention_backend, allow_attn_fallback = _get_model_loading_kwargs(
        variant, quantization, run_device, resolved_attention_backend
    )
    
    _log_major(f"正在加载多模态模型: {variant} | 量化: {quantization} | 注意力: {resolved_attention_backend}")
    
    # 加载模型
    model = _load_pretrained_with_fallback(
        AutoModelForVision2Seq,
        model_dir,
        kwargs,
        "多模态模型",
        resolved_attention_backend,
        allow_attn_fallback,
    )
    
    # 验证量化
    if qcfg is not None:
        _assert_quant_applied(model, quantization, "VL")
    
    # 设置为评估模式
    model.eval()
    if qcfg is None and not hasattr(model, "hf_device_map"):
        model.to(run_device)
    
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
    attention_backend: Optional[str],
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
    model, tokenizer, run_device = load_text_model(variant, quantization, device, attention_backend)
    
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
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,  # 使用KV缓存加速
    }
    
    if temperature > 0:
        gen_kwargs["temperature"] = max(temperature, 1e-5)
        gen_kwargs["top_p"] = top_p
    
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
    attention_backend: Optional[str],
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
    model, tokenizer, processor, run_device = load_vl_model(variant, quantization, device, attention_backend)
    
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
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,  # 使用KV缓存加速
    }
    
    if temperature > 0:
        gen_kwargs["temperature"] = max(temperature, 1e-5)
        gen_kwargs["top_p"] = top_p
    
    # 使用torch.inference_mode()加速推理
    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, **gen_kwargs)
    
    # 解码输出
    generated = output_ids[0][model_inputs["input_ids"].shape[-1]:]
    result = _strip_thinking_content(tokenizer.decode(generated, skip_special_tokens=True))
    
    return result
