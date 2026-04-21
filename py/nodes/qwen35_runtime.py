from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

try:
    import fcntl
except Exception:
    fcntl = None

import folder_paths

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None

_CFG = getattr(sys.modules.get("comfyui_iat_config"), "data", {}) or {}
_RUNTIME_CFG = (_CFG.get("runtime") or {}) if isinstance(_CFG, dict) else {}
_LOGGING_CFG = (_CFG.get("logging") or {}) if isinstance(_CFG, dict) else {}

MIN_TRANSFORMERS_FOR_QWEN35 = "5.2.0"
QWEN35_MODEL_TYPE = "qwen3_5"
# 模型映射说明：
# 1) 仅保留 Qwen 官方原版仓库。
# 2) 不提供量化/GGUF 变体选项。
_BASE_MODEL_REPOS: Dict[str, str] = {
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
}

def _build_text_model_candidates() -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for model_name, official_repo in _BASE_MODEL_REPOS.items():
        # 仅保留官方原版模型选项
        result[model_name] = [official_repo]
    return result


def _build_vl_model_candidates() -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for model_name, official_repo in _BASE_MODEL_REPOS.items():
        # 仅保留官方原版模型选项
        result[model_name] = [official_repo]
    return result


TEXT_MODEL_CANDIDATES: Dict[str, List[str]] = _build_text_model_candidates()
VL_MODEL_CANDIDATES: Dict[str, List[str]] = _build_vl_model_candidates()


def _grouped_options(model_map: Dict[str, List[str]]):
    grouped_labels: List[str] = []
    label_to_variant: Dict[str, str] = {}

    def _variant_size_key(variant: str):
        m = re.search(r"(\d+(?:\.\d+)?)B", variant or "")
        if not m:
            return float("inf")
        try:
            return float(m.group(1))
        except ValueError:
            return float("inf")

    # 仅保留原始模型名展示，并按参数规模从小到大排序。
    variants = sorted(model_map.keys(), key=lambda x: (_variant_size_key(x), x))
    for variant in variants:
        grouped_labels.append(variant)
        label_to_variant[variant] = variant
    return grouped_labels, label_to_variant


TEXT_MODEL_OPTIONS_GROUPED, TEXT_MODEL_LABEL_TO_VARIANT = _grouped_options(TEXT_MODEL_CANDIDATES)
VL_MODEL_OPTIONS_GROUPED, VL_MODEL_LABEL_TO_VARIANT = _grouped_options(VL_MODEL_CANDIDATES)


def resolve_model_variant(selection: str, mode: str = "text") -> str:
    """将分组展示标签解析为真实模型variant。"""
    if mode == "vl":
        mapping = VL_MODEL_LABEL_TO_VARIANT
        candidates = VL_MODEL_CANDIDATES
    else:
        mapping = TEXT_MODEL_LABEL_TO_VARIANT
        candidates = TEXT_MODEL_CANDIDATES
    return mapping.get(selection, selection if selection in candidates else selection)
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


PREFER_OPTIMIZED_ATTENTION = _cfg_bool("prefer_optimized_attention", True)
ENABLE_TORCH_COMPILE = _cfg_bool("enable_torch_compile", False)
OFFLINE_ONLY = _cfg_bool("offline_only", False)
DEFAULT_ATTENTION_BACKEND = "SDPA"
VERBOSE_LOGGING = _cfg_logging_bool("verbose", False)
DOWNLOAD_RETRY_TIMES = 2
DOWNLOAD_RETRY_DELAY_SECONDS = 1.0
DOWNLOAD_LOCK_TIMEOUT_SECONDS = 300


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


def _raise_runtime_error(code: str, message: str, detail: Optional[str] = None, cause: Optional[Exception] = None):
    """统一错误出口：对用户简洁，对日志可追踪。"""
    trace_id = uuid.uuid4().hex[:8]
    if detail:
        _log_warning(f"[{code}][{trace_id}] {detail}")
    if cause is not None:
        _log_warning(f"[{code}][{trace_id}] cause={repr(cause)}")
    raise RuntimeError(f"[IAT:{code}][{trace_id}] {message}")


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
    return f'"{sys.executable}" -m pip install --upgrade "transformers>={MIN_TRANSFORMERS_FOR_QWEN35}"'


def _check_transformers_support() -> None:
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
    _raise_runtime_error(
        "E5001",
        "Transformers 版本或架构不满足要求。",
        detail=f"{detail}; 修复命令: {manual_command}",
    )


def _get_llm_base_dirs() -> List[Path]:
    # 优先使用 diffusion_models，兼容读取 unet 目录中的同名模型。
    # 写入和下载仍默认落在 diffusion_models。
    roots = [
        Path(folder_paths.models_dir) / "diffusion_models",
        Path(folder_paths.models_dir) / "unet",
    ]
    unique: List[Path] = []
    seen: Set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        root.mkdir(parents=True, exist_ok=True)
        unique.append(root)
    return unique


def _get_primary_llm_dir() -> Path:
    return _get_llm_base_dirs()[0]


def _get_offload_dir() -> Path:
    """获取CPU卸载目录，用于显存不足时"""
    offload_dir = _get_primary_llm_dir() / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    return offload_dir


def _normalize_local_name(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    if name.endswith("-Instruct"):
        name = name[: -len("-Instruct")]
    return name


def _model_local_dir(repo_id: str) -> Path:
    return _get_primary_llm_dir() / _normalize_local_name(repo_id)


def _model_local_dirs(repo_id: str) -> List[Path]:
    name = _normalize_local_name(repo_id)
    return [base / name for base in _get_llm_base_dirs()]


def _model_state_file(model_dir: Path) -> Path:
    return model_dir / ".iat.model_state.json"


def _model_lock_file(model_dir: Path) -> Path:
    return model_dir / ".iat.download.lock"


def _load_model_state(model_dir: Path) -> dict:
    state_path = _model_state_file(model_dir)
    if not state_path.exists():
        return {}
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_model_state(model_dir: Path, state: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    state_path = _model_state_file(model_dir)
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, state_path)


def _clear_model_state(model_dir: Path) -> None:
    try:
        _model_state_file(model_dir).unlink(missing_ok=True)
    except Exception:
        pass


@contextmanager
def _model_dir_lock(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _model_lock_file(model_dir)
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        if fcntl is None:
            yield
            return

        start = time.time()
        acquired = False
        while time.time() - start < DOWNLOAD_LOCK_TIMEOUT_SECONDS:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(0.2)
            except Exception as exc:
                _raise_runtime_error("E2003", "模型下载锁获取失败。", detail=str(exc), cause=exc)

        if not acquired:
            _raise_runtime_error(
                "E2004",
                "模型下载锁等待超时。",
                detail=f"path={lock_path}; timeout={DOWNLOAD_LOCK_TIMEOUT_SECONDS}s",
            )
        try:
            yield
        finally:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _has_weights(model_dir: Path) -> bool:
    for pat in ("*.safetensors", "*.bin", "*.pt"):
        if next(model_dir.glob(pat), None) is not None:
            return True
    return False


def _expected_weight_files_from_index(model_dir: Path) -> Set[str]:
    expected: Set[str] = set()
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx = model_dir / name
        if not idx.exists():
            continue
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
        except Exception:
            continue
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if isinstance(weight_map, dict):
            for file_name in weight_map.values():
                if isinstance(file_name, str) and file_name.strip():
                    expected.add(file_name.strip())
    return expected


def _missing_weight_files(model_dir: Path) -> Set[str]:
    expected = _expected_weight_files_from_index(model_dir)
    if not expected:
        return set()
    return {name for name in expected if not (model_dir / name).exists()}


def _has_local_model_artifacts(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    try:
        return any(not path.name.startswith(".iat.") for path in model_dir.iterdir())
    except Exception:
        return False


def _is_model_complete(model_dir: Path, repo_id: str, mode: str) -> bool:
    if not model_dir.exists():
        return False
    if not _has_weights(model_dir):
        return False
    if _missing_weight_files(model_dir):
        return False
    if mode == "vl" and not _has_vl_processor_files(model_dir):
        return False

    state = _load_model_state(model_dir)
    if (
        state.get("repo_id") == repo_id
        and state.get("mode") == mode
        and state.get("complete") is True
    ):
        return True

    # 兼容旧目录：即使没有状态文件，只要完整即补写状态并视为可用。
    _save_model_state(
        model_dir,
        {
            "repo_id": repo_id,
            "mode": mode,
            "complete": True,
            "updated_at": int(time.time()),
        },
    )
    return True


def _has_vl_processor_files(model_dir: Path) -> bool:
    required_any = (
        "processor_config.json",
        "preprocessor_config.json",
        "image_processor_config.json",
        "feature_extractor_config.json",
    )
    return any((model_dir / name).exists() for name in required_any)


def _warn_incomplete_local_model(model_dir: Path, repo_id: str, mode: str) -> None:
    problems: List[str] = []
    if not _has_weights(model_dir):
        problems.append("missing weights")
    missing = _missing_weight_files(model_dir)
    if missing:
        problems.append(f"missing shards ({len(missing)})")
    if mode == "vl" and not _has_vl_processor_files(model_dir):
        problems.append("missing processor files")
    summary = ", ".join(problems) if problems else "validation incomplete"
    _log_warning(
        f"本地模型目录校验未通过，仍继续尝试加载：{repo_id} @ {model_dir} ({summary})"
    )


def _raise_model_load_error(
    *,
    stage: str,
    variant: str,
    mode: str,
    model_dir: Path,
    exc: Exception,
) -> None:
    _raise_runtime_error(
        "E2002",
        f"本地模型加载失败：{variant}（{mode}）",
        detail=f"stage={stage}; model_dir={model_dir}; offline_only={OFFLINE_ONLY}; cause={repr(exc)}",
        cause=exc,
    )


def _download_from_hf(repo_id: str, local_dir: Path) -> bool:
    if hf_snapshot_download is None:
        return False
    local_dir.mkdir(parents=True, exist_ok=True)
    hf_snapshot_download(repo_id=repo_id, local_dir=str(local_dir), resume_download=True)
    return True


def _download_from_modelscope(repo_id: str, local_dir: Path) -> bool:
    if ms_snapshot_download is None:
        return False
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        # ModelScope 新接口
        ms_snapshot_download(model_id=repo_id, local_dir=str(local_dir))
    except TypeError:
        # 兼容旧接口
        ms_snapshot_download(repo_id, cache_dir=str(local_dir.parent))
    return True


def _download_with_retry(downloader_name: str, downloader, repo_id: str, local_dir: Path) -> bool:
    last_exc: Optional[Exception] = None
    for i in range(1, DOWNLOAD_RETRY_TIMES + 1):
        try:
            ok = downloader(repo_id, local_dir)
            if ok:
                return True
        except Exception as exc:
            last_exc = exc
            _log_info(f"{downloader_name} 下载失败（第 {i}/{DOWNLOAD_RETRY_TIMES} 次）: {repo_id} -> {exc}")
            if i < DOWNLOAD_RETRY_TIMES:
                time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
    if last_exc is not None:
        raise last_exc
    return False


def ensure_model(variant: str, mode: str) -> Path:
    model_map = TEXT_MODEL_CANDIDATES if mode == "text" else VL_MODEL_CANDIDATES
    candidates = model_map.get(variant)
    if not candidates:
        _raise_runtime_error("E1001", f"不支持的模型选项：{variant}")

    errors: List[str] = []
    for repo_id in candidates:
        model_dirs = _model_local_dirs(repo_id)
        for existing_dir in model_dirs:
            if _is_model_complete(existing_dir, repo_id, mode):
                if repo_id != candidates[0]:
                    _log_info(f"使用候选回退仓库: {repo_id}")
                return existing_dir

        local_dirs = [d for d in model_dirs if _has_local_model_artifacts(d)]
        if local_dirs:
            target = next((d for d in local_dirs if _has_weights(d)), local_dirs[0])
            _warn_incomplete_local_model(target, repo_id, mode)
            return target

        if OFFLINE_ONLY:
            errors.append(f"{repo_id}: offline_only enabled and no local model artifacts found")
            continue

        target = _model_local_dir(repo_id)
        with _model_dir_lock(target):
            if _is_model_complete(target, repo_id, mode):
                if repo_id != candidates[0]:
                    _log_info(f"使用候选回退仓库: {repo_id}")
                return target

            if target.exists() and _has_weights(target):
                missing = _missing_weight_files(target)
                if missing:
                    _log_warning(f"检测到模型分片缺失，将自动补齐：{repo_id} 缺少 {len(missing)} 个文件")
                _clear_model_state(target)

            # 标准模型下载策略：ModelScope 优先，HuggingFace 兜底
            for source_name, downloader in (
                ("ModelScope", _download_from_modelscope),
                ("HuggingFace", _download_from_hf),
            ):
                try:
                    ok = _download_with_retry(source_name, downloader, repo_id, target)
                    if not ok:
                        continue
                    if _is_model_complete(target, repo_id, mode):
                        _log_major(f"模型已下载: {repo_id} <- {source_name}")
                        return target

                    missing_after = _missing_weight_files(target)
                    if missing_after:
                        errors.append(f"{repo_id}: still missing shards after {source_name} ({len(missing_after)} files)")
                        continue
                    if mode == "vl" and not _has_vl_processor_files(target):
                        errors.append(f"{repo_id}: missing VL processor files after {source_name} download")
                        continue
                    errors.append(f"{repo_id}: validation failed after {source_name} download")
                except Exception as exc:
                    errors.append(f"{repo_id}: {source_name} download failed: {exc}")
            if _has_local_model_artifacts(target):
                _warn_incomplete_local_model(target, repo_id, mode)
                return target

    _raise_runtime_error(
        "E2001",
        f"模型下载或校验失败：{variant}（{mode}）",
        detail=f"tried={', '.join(candidates)}; details={' | '.join(errors[-6:])}",
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


def _should_use_auto_device_map(variant: str, device: str) -> bool:
    if device != "cuda":
        return False
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


def _cuda_max_memory_map(reserve_gib: int = 4):
    """
    计算每个GPU的最大可用显存
    
    Args:
        reserve_gib: 基础预留显存（GB）
    """
    if not torch.cuda.is_available():
        return None
    
    max_memory = {}
    for idx in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(idx).total_memory // (1024**3)
        # 预留更多显存给系统和其他进程
        usable_gib = max(1, int(total_gib) - reserve_gib)
        max_memory[idx] = f"{usable_gib}GiB"
        _log_info(f"GPU {idx}: 总显存 {total_gib}GB, 可用 {usable_gib}GB")
    
    return max_memory


def _auto_device_max_memory(variant: str):
    max_memory = _cuda_max_memory_map()
    if max_memory is None:
        return None
    cpu_budget = _cpu_offload_memory_budget_gib()
    if cpu_budget is not None:
        max_memory["cpu"] = f"{cpu_budget}GiB"
        _log_info(f"CPU 卸载预算: {cpu_budget}GiB")
    size_b = _variant_size_billions(variant)
    if size_b is not None and size_b >= 27.0:
        _log_warning("Qwen3.5-27B 在 FP16/BF16 下会启用自动分配和 CPU 卸载；建议使用更大显存或多卡环境。")
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
    common_kwargs = {"trust_remote_code": True}
    if OFFLINE_ONLY:
        common_kwargs["local_files_only"] = True
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), **common_kwargs)
    except Exception as exc:
        _log_info(f"快速 tokenizer 加载失败，回退慢速 tokenizer: {exc}")
        slow_kwargs = dict(common_kwargs)
        slow_kwargs["use_fast"] = False
        return AutoTokenizer.from_pretrained(str(model_dir), **slow_kwargs)


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


def _get_model_loading_kwargs(variant: str, device: str, attention_backend: Optional[str]):
    """
    获取模型加载参数
    
    优化点：
    1. 使用最佳的注意力实现
    2. 合理的device_map和max_memory配置
    3. 支持CPU卸载
    """
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if OFFLINE_ONLY:
        kwargs["local_files_only"] = True

    attn_implementation, resolved_attention_backend, allow_attn_fallback = _resolve_attention_backend(attention_backend, device)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    
    _set_model_dtype(kwargs, _dtype_for_device(device))
    if device == "cuda":
        if _should_use_auto_device_map(variant, device) and _has_accelerate():
            # accelerate 可用时，允许自动放置和溢出到 CPU
            kwargs["device_map"] = "auto"
            max_memory = _auto_device_max_memory(variant)
            if max_memory:
                kwargs["max_memory"] = max_memory
            kwargs["offload_folder"] = str(_get_offload_dir())
            kwargs["offload_state_dict"] = True
            _log_info("大模型启用 auto device_map + CPU offload。")
        elif _should_use_auto_device_map(variant, device):
            _log_info("模型较大但未检测到 accelerate，回退为常规 CUDA 加载。")
        elif (_variant_size_billions(variant) or 0) >= 9.0:
            _log_info("9B 模型在 24GB 级显卡上默认使用单卡直载，优先保证稳定性和响应速度。")
        else:
            _log_info("小模型默认使用常规 CUDA 加载，避免 auto device_map 带来的额外开销。")

    return kwargs, resolved_attention_backend, allow_attn_fallback


def load_text_model(variant: str, device: str, attention_backend: Optional[str] = None):
    """加载文本模型，带缓存机制"""
    run_device = resolve_device(device)
    resolved_attention_backend = _normalize_attention_backend(attention_backend)
    _check_transformers_support()
    model_dir = ensure_model(variant, mode="text")
    
    # 创建缓存签名
    signature = (str(model_dir), run_device, resolved_attention_backend)
    
    # 检查缓存
    cache = _MODEL_CACHE["text"]
    if cache["model"] is not None and cache["signature"] == signature:
        _log_info(f"使用缓存的文本模型: {variant}")
        return cache["model"], cache["tokenizer"], run_device
    
    # 清理旧缓存
    _clear_cache("text")
    
    if run_device == "cuda":
        _prepare_cuda_for_load()

    # 加载tokenizer（Transformers 路径）
    try:
        tokenizer = _load_tokenizer(model_dir)
    except Exception as exc:
        _raise_model_load_error(
            stage="tokenizer",
            variant=variant,
            mode="text",
            model_dir=model_dir,
            exc=exc,
        )
    
    # 获取加载参数
    kwargs, resolved_attention_backend, allow_attn_fallback = _get_model_loading_kwargs(
        variant, run_device, resolved_attention_backend
    )
    
    _log_major(f"正在加载文本模型: {variant} | 注意力: {resolved_attention_backend}")
    
    # 加载模型
    try:
        model = _load_pretrained_with_fallback(
            AutoModelForCausalLM,
            model_dir,
            kwargs,
            "文本模型",
            resolved_attention_backend,
            allow_attn_fallback,
        )
    except Exception as exc:
        _raise_model_load_error(
            stage="text_model",
            variant=variant,
            mode="text",
            model_dir=model_dir,
            exc=exc,
        )
    
    # 设置为评估模式
    model.eval()
    if not hasattr(model, "hf_device_map"):
        model.to(run_device)
    
    # 编译模型以加速（仅CUDA）
    if ENABLE_TORCH_COMPILE and run_device == "cuda" and hasattr(torch, "compile"):
        try:
            _log_info("尝试编译模型以加速推理...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            _log_info(f"模型编译失败（不影响使用）: {e}")
    
    _log_major(f"文本模型加载完成 | 规格={variant} | 设备={run_device}")
    
    # 更新缓存
    cache["signature"] = signature
    cache["model"] = model
    cache["tokenizer"] = tokenizer
    
    return model, tokenizer, run_device


def load_vl_model(variant: str, device: str, attention_backend: Optional[str] = None):
    """加载视觉语言模型，带缓存机制"""
    _check_transformers_support()
    model_dir = ensure_model(variant, mode="vl")
    run_device = resolve_device(device)
    resolved_attention_backend = _normalize_attention_backend(attention_backend)
    
    # 创建缓存签名
    signature = (str(model_dir), run_device, resolved_attention_backend)
    
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
    processor_kwargs = {"trust_remote_code": True}
    if OFFLINE_ONLY:
        processor_kwargs["local_files_only"] = True
    try:
        processor = AutoProcessor.from_pretrained(str(model_dir), **processor_kwargs)
    except Exception as exc:
        _raise_model_load_error(
            stage="processor",
            variant=variant,
            mode="vl",
            model_dir=model_dir,
            exc=exc,
        )
    try:
        tokenizer = _load_tokenizer(model_dir)
    except Exception as exc:
        _raise_model_load_error(
            stage="tokenizer",
            variant=variant,
            mode="vl",
            model_dir=model_dir,
            exc=exc,
        )
    
    # 获取加载参数
    kwargs, resolved_attention_backend, allow_attn_fallback = _get_model_loading_kwargs(
        variant, run_device, resolved_attention_backend
    )
    
    _log_major(f"正在加载多模态模型: {variant} | 注意力: {resolved_attention_backend}")
    
    # 加载模型
    try:
        model = _load_pretrained_with_fallback(
            AutoModelForVision2Seq,
            model_dir,
            kwargs,
            "多模态模型",
            resolved_attention_backend,
            allow_attn_fallback,
        )
    except Exception as exc:
        _raise_model_load_error(
            stage="vl_model",
            variant=variant,
            mode="vl",
            model_dir=model_dir,
            exc=exc,
        )
    
    # 设置为评估模式
    model.eval()
    if not hasattr(model, "hf_device_map"):
        model.to(run_device)
    
    _log_major(f"多模态模型加载完成 | 规格={variant} | 设备={run_device}")
    
    # 更新缓存
    cache["signature"] = signature
    cache["model"] = model
    cache["tokenizer"] = tokenizer
    cache["processor"] = processor
    
    return model, tokenizer, processor, run_device


def generate_text(
    *,
    variant: str,
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
    model, tokenizer, run_device = load_text_model(variant, device, attention_backend)

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
    model, tokenizer, processor, run_device = load_vl_model(variant, device, attention_backend)
    
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
