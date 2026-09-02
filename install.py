from __future__ import annotations

import importlib
import os
import subprocess
import sys

from packaging import version

QWEN35_MODEL_TYPE = "qwen3_5"
MIN_TRANSFORMERS_FOR_QWEN35 = "5.2.0"


def _run(cmd: list[str]) -> int:
    """执行命令并打印日志。"""
    print(f"[IAT] Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=os.path.dirname(os.path.realpath(__file__)))


def _load_transformers():
    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        return None, f"transformers import failed: {exc}"
    return transformers, ""


def _supports_qwen35_architecture(transformers) -> bool:
    try:
        transformers.AutoConfig.for_model(QWEN35_MODEL_TYPE)
        return True
    except Exception:
        pass

    try:
        cfg_auto = importlib.import_module("transformers.models.auto.configuration_auto")
        config_mapping = getattr(cfg_auto, "CONFIG_MAPPING", {})
        if QWEN35_MODEL_TYPE in config_mapping:
            return True
    except Exception:
        pass

    return False


def _manual_transformers_upgrade_command() -> str:
    return f'"{sys.executable}" -m pip install --upgrade "transformers>={MIN_TRANSFORMERS_FOR_QWEN35}"'


def _ensure_transformers_support() -> int:
    """检查 transformers 版本和架构支持，不做自动升级。"""
    transformers, import_error = _load_transformers()
    if transformers is None:
        print(f"[IAT][E5001] {import_error}")
        print(f"[IAT][E5001] 修复命令: {_manual_transformers_upgrade_command()}")
        return 1

    current = getattr(transformers, "__version__", "0.0.0")
    issues: list[str] = []

    if version.parse(current) < version.parse(MIN_TRANSFORMERS_FOR_QWEN35):
        issues.append(f"requires transformers>={MIN_TRANSFORMERS_FOR_QWEN35} (current: {current})")
    if not _supports_qwen35_architecture(transformers):
        issues.append(f"current transformers build does not register '{QWEN35_MODEL_TYPE}'")

    if not issues:
        print(f"[IAT] transformers {current} supports {QWEN35_MODEL_TYPE}.")
        return 0

    print(
        "[IAT][E5001] Transformers 版本或架构不满足要求："
        f" {'; '.join(issues)}"
    )
    print(f"[IAT][E5001] 修复命令: {_manual_transformers_upgrade_command()}")
    return 1


def main() -> int:
    req = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    if not os.path.isfile(req):
        print("[IAT] requirements.txt not found, skipping dependency install.")
        return 0

    # 使用当前解释器，避免与 ComfyUI 运行时 Python 不一致。
    cmd = [sys.executable, "-m", "pip", "install", "-r", req]
    print(f"[IAT] Installing dependencies with Python: {sys.executable}")
    if _run(cmd) != 0:
        return 1
    return _ensure_transformers_support()


if __name__ == "__main__":
    raise SystemExit(main())
