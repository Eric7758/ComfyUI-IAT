from __future__ import annotations

import importlib
import os
import subprocess
import sys

QWEN35_MODEL_TYPE = "qwen3_5"
TRANSFORMERS_UPGRADE_SOURCES = (
    "git+https://github.com/huggingface/transformers.git",
    "https://github.com/huggingface/transformers/archive/refs/heads/main.zip",
)


def _run(cmd: list[str]) -> int:
    print(f"[IAT] Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=os.path.dirname(os.path.realpath(__file__)))


def _supports_qwen35_architecture() -> tuple[bool, str]:
    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        return False, f"transformers import failed: {exc}"

    current = getattr(transformers, "__version__", "unknown")
    try:
        transformers.AutoConfig.for_model(QWEN35_MODEL_TYPE)
        return True, current
    except Exception:
        pass

    try:
        cfg_auto = importlib.import_module("transformers.models.auto.configuration_auto")
        config_mapping = getattr(cfg_auto, "CONFIG_MAPPING", {})
        if QWEN35_MODEL_TYPE in config_mapping:
            return True, current
    except Exception:
        pass

    return False, current


def _ensure_transformers_support() -> int:
    supported, detail = _supports_qwen35_architecture()
    if supported:
        print(f"[IAT] transformers {detail} already supports {QWEN35_MODEL_TYPE}.")
        return 0

    print(
        "[IAT] Installed transformers build does not expose "
        f"{QWEN35_MODEL_TYPE}. Attempting source upgrade."
    )
    for source in TRANSFORMERS_UPGRADE_SOURCES:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", source]
        if _run(cmd) == 0:
            supported_after, new_detail = _supports_qwen35_architecture()
            if supported_after:
                print(f"[IAT] transformers {new_detail} now supports {QWEN35_MODEL_TYPE}.")
                return 0

            print(
                "[IAT] Source upgrade finished, but this process could not re-verify "
                f"{QWEN35_MODEL_TYPE}. Restart ComfyUI if the old package is still cached."
            )
            return 0

    print(
        "[IAT] Failed to install a transformers build with qwen3_5 support. "
        f"Please run: {sys.executable} -m pip install --upgrade {TRANSFORMERS_UPGRADE_SOURCES[0]}"
    )
    return 1


def main() -> int:
    req = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    if not os.path.isfile(req):
        print("[IAT] requirements.txt not found, skipping dependency install.")
        return 0

    # Use the current interpreter to avoid mismatching ComfyUI's Python.
    cmd = [sys.executable, "-m", "pip", "install", "-r", req]
    print(f"[IAT] Installing dependencies with Python: {sys.executable}")
    if _run(cmd) != 0:
        return 1
    return _ensure_transformers_support()


if __name__ == "__main__":
    raise SystemExit(main())
